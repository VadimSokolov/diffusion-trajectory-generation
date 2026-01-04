#!/usr/bin/env python3
"""
CSDI: Conditional Score-based Diffusion for Vehicle Trajectory Generation

Based on "CSDI: Conditional Score-based Diffusion Models for Probabilistic 
Time Series Imputation" (Tashiro et al., NeurIPS 2021)

Architecture:
- Score-based diffusion with transformer backbone
- Conditional on features (avg_speed, duration, max_speed)
- Handles variable-length sequences via masking
- Built-in boundary condition enforcement

Key Features:
- Transformer attention for long-range dependencies
- Parallel computation (no autoregressive generation)
- High sample diversity from diffusion process
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import glob
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from datetime import datetime
import math

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "max_length": 512,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,           # Increased from 4 for better conditioning
    "d_ff": 1024,            # Increased from 512 for more capacity
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 1e-4,              # Slightly lower for stability
    "epochs": 200,           # Increased from 100 for better convergence
    "speed_limit": 40.0,     # Max speed ~39 m/s in data, 40 is good ceiling
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Diffusion parameters
    "diffusion_steps": 200,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "schedule": "cosine",    # Cosine schedule often better than linear
    # Smoothness loss weight (penalizes jerky predictions)
    "smoothness_weight": 0.1,
}


# ============================================================================
# Dataset
# ============================================================================

class TrajectoryDataset(Dataset):
    """Load micro-trip CSVs for training."""
    
    def __init__(self, data_path, max_len=CONFIG["max_length"], 
                 min_duration=50, max_duration=1000, limit_files=None):
        self.max_len = max_len
        self.data = []
        self.conditions = []
        self.masks = []  # 1 = valid data, 0 = padding
        
        # Find all trip files
        files = glob.glob(os.path.join(data_path, "**", "results_trip_*.csv"), recursive=True)
        if limit_files:
            files = files[:limit_files]
        
        print(f"Loading {len(files)} trajectory files...")
        
        for f in tqdm(files, desc="Loading"):
            try:
                df = pd.read_csv(f)
                if "speedMps" not in df.columns:
                    continue
                
                speed = df["speedMps"].values.astype(np.float32)
                duration = len(speed)
                
                # Filter by duration
                if duration < min_duration or duration > max_duration:
                    continue
                
                avg_speed = np.mean(speed)
                max_speed = np.max(speed)
                
                # Normalize speed to [-1, 1]
                speed_norm = (speed / CONFIG["speed_limit"]) * 2 - 1
                speed_norm = np.clip(speed_norm, -1, 1)
                
                # Create mask
                mask = np.ones(max_len, dtype=np.float32)
                
                # Pad or truncate
                if duration > max_len:
                    speed_pad = speed_norm[:max_len]
                else:
                    speed_pad = np.pad(speed_norm, (0, max_len - duration), 
                                      mode='constant', constant_values=-1)
                    mask[duration:] = 0
                
                self.data.append(speed_pad)
                self.masks.append(mask)
                
                # Normalize conditions
                cond = np.array([
                    avg_speed / 30.0,
                    duration / 1000.0,
                    max_speed / 40.0,
                ], dtype=np.float32)
                self.conditions.append(cond)
                
            except Exception as e:
                continue
        
        self.data = torch.tensor(np.stack(self.data), dtype=torch.float32)
        self.conditions = torch.tensor(np.stack(self.conditions), dtype=torch.float32)
        self.masks = torch.tensor(np.stack(self.masks), dtype=torch.float32)
        
        print(f"Loaded {len(self.data)} valid trajectories")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.conditions[idx], self.masks[idx]


# ============================================================================
# Diffusion Utilities
# ============================================================================

class DiffusionSchedule:
    """Linear or cosine noise schedule."""
    
    def __init__(self, num_steps, beta_start=0.0001, beta_end=0.02, schedule="linear"):
        self.num_steps = num_steps
        
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule == "cosine":
            steps = torch.arange(num_steps + 1) / num_steps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            self.betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            self.betas = torch.clamp(self.betas, 0.0001, 0.02)
        
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
    
    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        return self
    
    def add_noise(self, x, t):
        """Add noise to x at timestep t."""
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        noise = torch.randn_like(x)
        noisy_x = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
        return noisy_x, noise


# ============================================================================
# Model Components
# ============================================================================

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder with condition injection."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, key_padding_mask=mask)
        x = x + self.dropout(h)
        
        # Feed-forward
        h = self.norm2(x)
        x = x + self.ff(h)
        
        return x


class CSDITransformer(nn.Module):
    """
    CSDI Score Network with Transformer backbone.
    
    Predicts the noise added to the input at each diffusion timestep.
    """
    
    def __init__(self, d_model=256, n_heads=8, n_layers=4, d_ff=512,
                 dropout=0.1, max_len=512, cond_dim=3, num_diffusion_steps=100):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Input projection (noisy trajectory)
        self.input_proj = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # Diffusion timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Condition embedding
        self.cond_emb = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection (predict noise)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
    
    def forward(self, x_noisy, t, cond, mask=None):
        """
        Predict noise from noisy input.
        
        Args:
            x_noisy: (B, L) noisy trajectory
            t: (B,) diffusion timesteps
            cond: (B, cond_dim) conditions
            mask: (B, L) padding mask (1=valid, 0=padding)
        
        Returns:
            noise_pred: (B, L) predicted noise
        """
        B, L = x_noisy.shape
        
        # Project input
        h = x_noisy.unsqueeze(-1)  # (B, L, 1)
        h = self.input_proj(h)  # (B, L, d_model)
        
        # Add positional encoding
        h = h + self.pos_emb[:, :L]
        
        # Add time embedding (broadcast over sequence)
        t_emb = self.time_emb(t)  # (B, d_model)
        h = h + t_emb.unsqueeze(1)
        
        # Add condition embedding (broadcast over sequence)
        c_emb = self.cond_emb(cond)  # (B, d_model)
        h = h + c_emb.unsqueeze(1)
        
        # Create attention mask (invert: True = ignore)
        attn_mask = None
        if mask is not None:
            attn_mask = (1 - mask).bool()
        
        # Transformer layers
        for layer in self.layers:
            h = layer(h, mask=attn_mask)
        
        # Output projection
        noise_pred = self.output_proj(h).squeeze(-1)  # (B, L)
        
        return noise_pred


# ============================================================================
# CSDI Sampler
# ============================================================================

class CSDISampler:
    """DDPM-style sampler with boundary condition enforcement."""
    
    def __init__(self, model, schedule, device):
        self.model = model
        self.schedule = schedule
        self.device = device
    
    @torch.no_grad()
    def sample(self, cond, lengths, n_steps=None, boundary_ramp=3, smooth_kernel=9):
        """
        Generate trajectories via reverse diffusion.
        
        Args:
            cond: (B, cond_dim) conditions
            lengths: (B,) target lengths
            n_steps: number of denoising steps (None = use all)
            boundary_ramp: length of boundary smoothing ramp
            smooth_kernel: kernel size for Gaussian smoothing (0 to disable)
        
        Returns:
            samples: (B, max_len) generated trajectories
        """
        self.model.eval()
        B = cond.shape[0]
        L = CONFIG["max_length"]
        
        # Start from pure noise
        x = torch.randn(B, L, device=self.device)
        
        # Create mask
        mask = torch.zeros(B, L, device=self.device)
        for i, length in enumerate(lengths):
            mask[i, :int(length)] = 1
        
        # Boundary constraint: fix start and end to -1 (zero speed normalized)
        boundary_value = -1.0
        
        # Reverse diffusion
        steps = list(range(self.schedule.num_steps))[::-1]
        if n_steps is not None:
            step_size = max(1, len(steps) // n_steps)
            steps = steps[::step_size]
        
        print(f"  Starting reverse diffusion ({len(steps)} steps)...")
        for step_num, t_idx in enumerate(steps):
            t = torch.full((B,), t_idx, device=self.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model(x, t, cond, mask)
            
            # DDPM update
            alpha = self.schedule.alphas[t_idx]
            alpha_bar = self.schedule.alpha_bars[t_idx]
            beta = self.schedule.betas[t_idx]
            
            # Mean prediction
            mean = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred
            )
            
            # Add noise (except last step)
            if t_idx > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean
            
            # Enforce boundaries (inpainting-style)
            for i, length in enumerate(lengths):
                length = int(length)
                # Start boundary
                x[i, 0] = boundary_value
                # End boundary
                if length < L:
                    x[i, length-1] = boundary_value
                    x[i, length:] = boundary_value
                else:
                    x[i, -1] = boundary_value
            
            # Progress logging every 50 steps
            if (step_num + 1) % 50 == 0 or step_num == len(steps) - 1:
                print(f"    Step {step_num + 1}/{len(steps)} complete")
        
        print("  Diffusion complete, applying post-processing...")
        
        # Apply Gaussian smoothing to reduce jitter
        if smooth_kernel > 0:
            x = self._smooth_trajectory(x, lengths, kernel_size=smooth_kernel)
            print(f"    Smoothing applied (kernel={smooth_kernel})")
        
        # Apply smooth ramps at boundaries
        x = self._apply_ramps(x, lengths, boundary_ramp)
        print(f"    Boundary ramps applied (ramp={boundary_ramp})")
        
        # Mask padding
        x = x * mask + boundary_value * (1 - mask)
        
        return x
    
    def _apply_ramps(self, x, lengths, ramp_len):
        """Apply cosine ramps at boundaries."""
        B, L = x.shape
        
        for i in range(B):
            length = int(lengths[i])
            if length <= 2 * ramp_len:
                continue
            
            # Convert to 0-1 scale for ramping
            traj = (x[i, :length] + 1) / 2  # [-1, 1] -> [0, 1]
            
            # Only apply start ramp if not already near zero
            if traj[ramp_len].item() > 0.05:  # Only ramp if speed > ~2 m/s
                ramp_up = (1 - torch.cos(torch.linspace(0, math.pi/2, ramp_len, device=x.device)))
                traj[:ramp_len] *= ramp_up
            else:
                traj[:ramp_len] = 0  # Already low, just zero it
            
            # Only apply end ramp if not already near zero
            if traj[length-ramp_len-1].item() > 0.05:
                ramp_down = torch.cos(torch.linspace(0, math.pi/2, ramp_len, device=x.device))
                traj[length-ramp_len:length] *= ramp_down
            else:
                traj[length-ramp_len:length] = 0
            
            # Convert back
            x[i, :length] = traj * 2 - 1
        
        return x
    
    def _smooth_trajectory(self, x, lengths, kernel_size=5):
        """Apply Gaussian smoothing to reduce jitter while preserving dynamics."""
        B, L = x.shape
        
        # Create Gaussian kernel
        sigma = kernel_size / 4.0
        kernel = torch.exp(-torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size, device=x.device)**2 / (2*sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)
        
        for i in range(B):
            length = int(lengths[i])
            if length <= kernel_size:
                continue
            
            # Extract valid trajectory
            traj = x[i, :length].unsqueeze(0).unsqueeze(0)  # (1, 1, L)
            
            # Pad for conv
            pad = kernel_size // 2
            traj_padded = F.pad(traj, (pad, pad), mode='replicate')
            
            # Apply smoothing
            smoothed = F.conv1d(traj_padded, kernel).squeeze()
            
            # Preserve boundary values
            x[i, :length] = smoothed
        
        return x


# ============================================================================
# Training
# ============================================================================

def train(args):
    """Train the CSDI model."""
    print("=" * 60)
    print("CSDI Trajectory Generator - Training")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Epochs: {args.epochs}")
    print(f"Diffusion steps: {CONFIG['diffusion_steps']}")
    
    # Dataset
    dataset = TrajectoryDataset(
        args.data_path,
        max_len=CONFIG["max_length"],
        limit_files=args.limit_files
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Model
    model = CSDITransformer(
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"],
        d_ff=CONFIG["d_ff"],
        dropout=CONFIG["dropout"],
        max_len=CONFIG["max_length"],
        num_diffusion_steps=CONFIG["diffusion_steps"],
    ).to(CONFIG["device"])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Diffusion schedule
    schedule = DiffusionSchedule(
        CONFIG["diffusion_steps"],
        CONFIG["beta_start"],
        CONFIG["beta_end"],
        CONFIG["schedule"],
    ).to(CONFIG["device"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x, cond, mask in pbar:
            x = x.to(CONFIG["device"])
            cond = cond.to(CONFIG["device"])
            mask = mask.to(CONFIG["device"])
            
            B = x.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, CONFIG["diffusion_steps"], (B,), device=CONFIG["device"])
            
            # Add noise
            x_noisy, noise = schedule.add_noise(x, t)
            
            # Predict noise
            noise_pred = model(x_noisy, t, cond, mask)
            
            # Loss (only on valid positions)
            mse_loss = F.mse_loss(noise_pred * mask, noise * mask, reduction='sum')
            mse_loss = mse_loss / mask.sum()
            
            # Temporal smoothness loss: penalize jerky noise predictions
            # This encourages the model to predict smooth noise patterns
            diff = noise_pred[:, 1:] - noise_pred[:, :-1]
            diff_mask = mask[:, 1:] * mask[:, :-1]  # Only valid transitions
            smoothness_loss = (diff ** 2 * diff_mask).sum() / diff_mask.sum().clamp(min=1)
            
            # Combined loss with smoothness weight from config
            smoothness_weight = CONFIG.get("smoothness_weight", 0.1)
            loss = mse_loss + smoothness_weight * smoothness_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = total_loss / n_batches
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"csdi_ckpt_ep{epoch+1}.pt")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "csdi_best.pt")
    
    torch.save(model.state_dict(), "csdi_final.pt")
    print(f"Training complete. Best loss: {best_loss:.4f}")


# ============================================================================
# Generation
# ============================================================================

def generate(args):
    """Generate synthetic trajectories."""
    print("=" * 60)
    print("CSDI Trajectory Generator - Generation")
    print("=" * 60)
    
    # Load model
    model = CSDITransformer(
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"],
        d_ff=CONFIG["d_ff"],
        max_len=CONFIG["max_length"],
        num_diffusion_steps=CONFIG["diffusion_steps"],
    ).to(CONFIG["device"])
    
    model.load_state_dict(torch.load(args.model_path, map_location=CONFIG["device"]))
    model.eval()
    
    # Diffusion schedule
    schedule = DiffusionSchedule(
        CONFIG["diffusion_steps"],
        CONFIG["beta_start"],
        CONFIG["beta_end"],
        CONFIG["schedule"],
    ).to(CONFIG["device"])
    
    sampler = CSDISampler(model, schedule, CONFIG["device"])
    
    # Generate conditions by sampling from real data distribution
    n_samples = args.n_samples
    
    # Load real data to get condition distribution
    print(f"Loading real data for condition sampling from: {args.data_path}")
    real_conditions = []
    search_pattern = os.path.join(args.data_path, "**", "results_trip_*.csv")
    print(f"Search pattern: {search_pattern}")
    files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(files)} files")
    for f in files[:min(len(files), 2000)]:  # Sample from subset for speed
        try:
            df = pd.read_csv(f)
            if "speedMps" in df.columns:
                speed = df["speedMps"].values
                if len(speed) >= 50:
                    real_conditions.append({
                        'avg_speed': np.mean(speed),
                        'duration': len(speed),
                        'max_speed': np.max(speed)
                    })
        except:
            continue
    
    if len(real_conditions) > 0:
        print(f"Loaded {len(real_conditions)} real conditions for sampling")
        # Sample with replacement from real conditions
        sampled_idx = np.random.choice(len(real_conditions), n_samples, replace=True)
        sampled = [real_conditions[i] for i in sampled_idx]
        
        avg_speeds = torch.tensor([c['avg_speed'] / 30.0 for c in sampled])
        durations = torch.tensor([min(c['duration'], CONFIG["max_length"]) for c in sampled])
        max_speeds = torch.tensor([c['max_speed'] / 40.0 for c in sampled])
    else:
        print("Warning: Could not load real data, using uniform sampling")
        avg_speeds = torch.rand(n_samples) * 0.8 + 0.2
        durations = torch.randint(50, 500, (n_samples,))
        max_speeds = avg_speeds + torch.rand(n_samples) * 0.3
    
    cond = torch.stack([
        avg_speeds.float(),
        durations.float() / 1000.0,
        max_speeds.float(),
    ], dim=1).to(CONFIG["device"])
    
    lengths = durations.to(CONFIG["device"])
    
    print(f"Generating {n_samples} trajectories...")
    print(f"  Diffusion steps: {CONFIG['diffusion_steps']}")
    print(f"  Smoothing kernel: {args.smooth_kernel}")
    print(f"  Boundary ramp: 3 (conditional)")
    print(f"  Condition stats:")
    print(f"    Avg speed range: {avg_speeds.min()*30:.1f} - {avg_speeds.max()*30:.1f} m/s")
    print(f"    Duration range: {durations.min().item()} - {durations.max().item()} s")
    
    trajectories = sampler.sample(cond, lengths, smooth_kernel=args.smooth_kernel)
    print("Diffusion sampling complete!")
    
    # Clip normalized output to valid range, then denormalize
    trajectories = trajectories.cpu().numpy()
    trajectories = np.clip(trajectories, -1, 1)  # Ensure within normalized range
    trajectories = (trajectories + 1) / 2 * CONFIG["speed_limit"]  # [-1,1] -> [0, speed_limit]
    trajectories = np.clip(trajectories, 0, CONFIG["speed_limit"])  # Safety clip
    
    # Log post-processing stats
    valid_speeds = [trajectories[i, :int(durations[i])] for i in range(n_samples)]
    all_speeds = np.concatenate(valid_speeds)
    all_accels = np.concatenate([np.diff(s) for s in valid_speeds])
    print(f"Post-processing stats:")
    print(f"  Speed - Mean: {np.mean(all_speeds):.2f}, Std: {np.std(all_speeds):.2f}, Max: {np.max(all_speeds):.2f} m/s")
    print(f"  Accel - Mean: {np.mean(all_accels):.4f}, Std: {np.std(all_accels):.4f} m/s²")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    df = pd.DataFrame(trajectories)
    df["length"] = durations.numpy()
    df["avg_speed_target"] = avg_speeds.numpy() * 30
    df.to_csv(f"csdi_synthetic_{timestamp}.csv", index=False)
    
    # Save as pickle for evaluation
    with open(f"csdi_synthetic_{timestamp}.pkl", "wb") as f:
        pickle.dump({
            "trajectories": [trajectories[i, :int(durations[i])] for i in range(n_samples)],
            "conditions": cond.cpu().numpy(),
            "durations": durations.numpy(),
        }, f)
    
    # Plot samples
    plt.figure(figsize=(15, 10))
    for i in range(min(12, n_samples)):
        plt.subplot(3, 4, i+1)
        length = int(durations[i])
        plt.plot(trajectories[i, :length])
        plt.title(f"L={length}, V={avg_speeds[i]*30:.1f}")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (m/s)")
        plt.ylim(0, 40)
    plt.tight_layout()
    plt.savefig(f"csdi_samples_{timestamp}.png", dpi=150)
    print(f"Saved samples plot: csdi_samples_{timestamp}.png")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--data_path", type=str, default="../../data/Microtrips")
    parser.add_argument("--model_path", type=str, default="csdi_best.pt")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--limit_files", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--smooth_kernel", type=int, default=7, help="Gaussian smoothing kernel size (0=disable)")
    parser.add_argument("--diffusion_steps", type=int, default=CONFIG["diffusion_steps"])
    
    args = parser.parse_args()
    
    # Override config with args
    CONFIG["diffusion_steps"] = args.diffusion_steps
    
    if args.train:
        train(args)
    if args.generate:
        generate(args)
    
    if not (args.train or args.generate):
        print("Usage: python csdi_trajectory.py --train or --generate")

