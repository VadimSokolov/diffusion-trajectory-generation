import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import glob
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

# --- Configuration & Hyperparameters ---
CONFIG = {
    "speed_limit": 40.0,  # Max speed approximately observed (m/s)
    "accel_limit": 5.0,   # Max accel approximately observed (m/s^2)
    "max_length": 512,    # Pad/Crop trajectories to this length
    "batch_size": 256,    # Balanced for update frequency
    "lr": 1e-4,
    "epochs": 100,        # Default, can be overridden
    "timesteps": 1000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "embed_dim": 256,     # Increased model capacity (A100 optimized)
    # --- Physics Loss Parameters (from CSDI, reduced for stable fine-tuning) ---
    "accel_distribution_weight": 0.02,  # Reduced from 0.05 for stability
    "jerk_penalty_weight": 0.01,        # Reduced from 0.02 for stability
    "target_accel_std": 0.6,            # m/s² (from real data analysis)
    "jerk_threshold": 2.0,              # m/s³ max comfortable jerk
    "accel_threshold": 2.5,             # m/s² max acceleration
    "decel_threshold": 4.0,             # m/s² max deceleration
}

# --- 1. Data Loading & Preprocessing ---

class TrajectoryDataset(Dataset):
    def __init__(self, data_path, max_len=CONFIG["max_length"], limit_files=None, save_stats=True, cache_file="data/dataset_cache.pt"):
        # Check for cache
        if os.path.exists(cache_file):
            print(f"Loading dataset from cache: {cache_file}")
            cache = torch.load(cache_file)
            self.data = cache["data"]
            self.conditions = cache["conditions"]
            self.max_len = cache["max_len"]
            print(f"Dataset loaded: {len(self.data)} trajectories (Cached).")
            # If stats are needed, we assume they are already saved in real_data_stats.pkl from the run that created the cache
            return
            
        self.files = glob.glob(os.path.join(data_path, "results_trip_*.csv"))
        if len(self.files) == 0:
             # Try recursive or slightly different pattern if needed, or exact path
             self.files = glob.glob(os.path.join(data_path, "**", "results_trip_*.csv"), recursive=True)
        
        if limit_files is not None:
            self.files = self.files[:limit_files]

        self.max_len = max_len
        self.data = []
        self.conditions = [] # avg_speed, duration
        self.real_speeds = [] # For stats
        self.real_accels = [] # For stats

        print(f"Loading {len(self.files)} files from {data_path}...")
        for f in tqdm(self.files, desc="Loading CSVs"):
            try:
                df = pd.read_csv(f)
                if "speedMps" not in df.columns:
                    continue
                
                speed = df["speedMps"].values.astype(np.float32)
                
                # Basic validation: ensure non-empty
                if len(speed) < 10: 
                    continue

                # Calculate Controls
                duration = len(speed)
                avg_speed = np.mean(speed)
                
                # Accel calculation: diff
                accel = np.diff(speed, prepend=speed[0])
                
                # Normalize Speed: [0, max] -> [-1, 1]
                speed_norm = (speed / CONFIG["speed_limit"]) * 2.0 - 1.0
                # Normalize Accel: [-max, max] -> [-1, 1] (roughly)
                accel_norm = np.clip(accel / CONFIG["accel_limit"], -1.0, 1.0)
                
                # Padding / Truncating
                if duration > max_len:
                    speed_pad = speed_norm[:max_len]
                    accel_pad = accel_norm[:max_len]
                else:
                    speed_pad = np.pad(speed_norm, (0, max_len - duration), mode='constant', constant_values=-1.0)
                    accel_pad = np.pad(accel_norm, (0, max_len - duration), mode='constant', constant_values=0.0)
                
                # Stack channels: (2, L)
                multi_channel = np.stack([speed_pad, accel_pad], axis=0)
                self.data.append(multi_channel)
                
                # Normalize Conditions 
                cond = np.array([avg_speed / 30.0, duration / 3000.0], dtype=np.float32)
                self.conditions.append(cond)

                if save_stats:
                    self.real_speeds.append(speed)
                    self.real_accels.append(accel)

            except Exception as e:
                print(f"Error reading {f}: {e}")

        self.data = torch.tensor(np.stack(self.data), dtype=torch.float32) # (N, 2, L)
        self.conditions = torch.tensor(np.stack(self.conditions), dtype=torch.float32)  # (N, 2)
        print(f"Dataset loaded: {len(self.data)} valid trajectories.")
        
        # Save Cache
        torch.save({
            "data": self.data, 
            "conditions": self.conditions,
            "max_len": self.max_len
        }, cache_file)
        print(f"Saved dataset cache to {cache_file}")

        if save_stats and len(self.real_speeds) > 0:
            import pickle
            all_speeds = np.concatenate(self.real_speeds)
            all_accels = np.concatenate(self.real_accels)
            
            # NEW: Save metadata for each trip (avg_speed, duration)
            avg_speeds_list = [np.mean(s) for s in self.real_speeds]
            durations_list = [len(s) for s in self.real_speeds]
            
            stats = {
                "speed_mean": np.mean(all_speeds),
                "speed_std": np.std(all_speeds),
                "accel_mean": np.mean(all_accels),
                "accel_std": np.std(all_accels),
                "speeds": self.real_speeds, # Save raw for histograms (might be large, but useful for 500 subset)
                "accels": self.real_accels,
                "avg_speeds": avg_speeds_list,  # NEW: Per-trip metadata
                "durations": durations_list      # NEW: Per-trip metadata
            }
            with open("data/real_data_stats.pkl", "wb") as f:
                pickle.dump(stats, f)
            print("Saved data/real_data_stats.pkl")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.conditions[idx]

# --- 2. Model Architecture (Conditional 1D UNet) ---

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, cond_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding (Additive)
        self.time_emb = nn.Linear(time_dim, out_channels)
        
        # Condition embedding (FiLM: Scale & Shift)
        # Projects to 2 * out_channels
        self.cond_emb = nn.Linear(cond_dim, out_channels * 2)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t, c):
        h = self.conv1(self.act1(self.norm1(x)))
        
        # 1. Add Time Embedding (Standard DDPM)
        h += self.time_emb(t)[:, :, None]
        
        # 2. Apply FiLM Conditioning (Scale & Shift)
        # (Batch, 2*Out) -> Scale, Shift
        c_emb = self.cond_emb(c)[:, :, None]
        scale, shift = c_emb.chunk(2, dim=1)
        
        # FiLM: h = h * (1 + scale) + shift
        h = h * (1.0 + scale) + shift
        
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)

class Unet1D(nn.Module):
    def __init__(self, in_channels=2, dim=CONFIG["embed_dim"], cond_dim=2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )
        
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

        time_dim = dim * 4
        
        # Downsample
        self.conv_in = nn.Conv1d(in_channels, dim, 3, padding=1)
        self.down1 = ResidualBlock(dim, dim, time_dim, time_dim)
        self.down2 = ResidualBlock(dim, dim * 2, time_dim, time_dim)
        self.down3 = ResidualBlock(dim * 2, dim * 4, time_dim, time_dim)
        
        # Middle
        self.mid1 = ResidualBlock(dim * 4, dim * 4, time_dim, time_dim)
        self.mid2 = ResidualBlock(dim * 4, dim * 4, time_dim, time_dim)
        
        # Upsample
        self.up1 = ResidualBlock(dim * 8, dim * 2, time_dim, time_dim)
        self.up2 = ResidualBlock(dim * 4, dim, time_dim, time_dim)
        self.up3 = ResidualBlock(dim * 2, dim, time_dim, time_dim)
        
        self.conv_out = nn.Conv1d(dim, in_channels, 3, padding=1)
        
        self.down_pool = nn.MaxPool1d(2)
        self.up_sample = nn.Upsample(scale_factor=2)

    def forward(self, x, t, cond):
        t = self.time_mlp(t)
        c = self.cond_mlp(cond)
        
        x = self.conv_in(x)
        
        d1 = self.down1(x, t, c)
        x = self.down_pool(d1)
        d2 = self.down2(x, t, c)
        x = self.down_pool(d2)
        d3 = self.down3(x, t, c)
        x = self.down_pool(d3)
        
        x = self.mid1(x, t, c)
        x = self.mid2(x, t, c)
        
        x = self.up_sample(x)
        x = torch.cat((x, d3), dim=1)
        x = self.up1(x, t, c)
        
        x = self.up_sample(x)
        x = torch.cat((x, d2), dim=1)
        x = self.up2(x, t, c)
        
        x = self.up_sample(x)
        x = torch.cat((x, d1), dim=1)
        x = self.up3(x, t, c)
        
        return self.conv_out(x)

# --- 3. Diffusion Logic (DDPM) ---

class DiffusionUtils:
    def __init__(self, timesteps=CONFIG["timesteps"]):
        self.timesteps = timesteps
        self.beta = torch.linspace(1e-4, 0.02, timesteps).to(CONFIG["device"])
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat)

    def noise_images(self, x, t):
        sqrt_alpha_hat = self.sqrt_alpha_hat[t][:, None, None]
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t][:, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.timesteps, size=(n,)).to(CONFIG["device"])

    def sample_physics(self, model, n_samples, cond, cfg_scale=7.5, target_lengths=None, physics_guide_fn=None):
        """
        Sample with Inference-Time Physics Guidance.
        physics_guide_fn: A function f(x) -> loss, used to steer generation.
                         Useful for vehicle-specific constraints (Bus vs Car).
        """
        model.eval()
        with torch.no_grad():
            x = torch.randn((n_samples, 2, CONFIG["max_length"])).to(CONFIG["device"])
            
            # Inpainting mask setup (same as standard sample)
            mask = torch.zeros_like(x)
            gt_constraint = torch.zeros_like(x)
            gt_constraint[:, 0, :] = -1.0
            
            if target_lengths is not None:
                for i, l in enumerate(target_lengths):
                    l = int(l)
                    mask[i, :, 0] = 1.0
                    if l < CONFIG["max_length"]:
                        mask[i, :, l-1] = 1.0
                        mask[i, :, l:] = 1.0
                    else:
                        mask[i, :, CONFIG["max_length"]-1] = 1.0
            else:
                 mask[:, :, 0] = 1.0
                 mask[:, :, -1] = 1.0

            # Sampling Loop with Gradient Guidance
            for i in tqdm(reversed(range(1, self.timesteps)), desc="Physics Sampling", position=0):
                t = (torch.ones(n_samples) * i).long().to(CONFIG["device"])
                
                # 1. Enable grad on x to compute physics gradients
                x = x.detach().requires_grad_(True)
                
                # 2. CFG Prediction
                if cfg_scale > 0:
                     cond_zero = torch.zeros_like(cond).to(CONFIG["device"]) 
                     predicted_noise = model(x, t, cond)  
                     uncond_predicted_noise = model(x, t, cond_zero)
                     predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                else:
                     predicted_noise = model(x, t, cond)
                
                # 3. Physics Guidance Term
                # If a guidance function is provided (e.g., max_accel constraint), 
                # compute gradient of loss w.r.t x and subtract from noise
                if physics_guide_fn is not None:
                    # Estimate x0 (clean) from current noisy x
                    alpha_bar = self.alpha_hat[t][:, None, None]
                    x0_hat = (x - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
                    
                    # Compute physics loss on x0 estimate
                    phys_loss = physics_guide_fn(x0_hat)
                    
                    # Compute gradient: grad_x (loss)
                    param_grad = torch.autograd.grad(phys_loss.sum(), x)[0]
                    
                    # Guidance scale (needs tuning, usually small)
                    guidance_scale = 20.0 
                    
                    # Modify noise prediction to move AWAY from high loss regions
                    # effective_noise = predicted_noise + scale * sigma * grad_score
                    # Here we subtract gradient from noise prediction (since noise predicts error)
                    # Implementation heuristic: noise = noise - sqrt(1-alpha_bar) * grad
                    predicted_noise = predicted_noise - torch.sqrt(1 - alpha_bar) * guidance_scale * param_grad
                
                # Detach for step
                x = x.detach()
                predicted_noise = predicted_noise.detach()
                
                # 4. Step
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
                # 5. Inpainting (re-apply constraints)
                if i > 1:
                     noise_for_constraint = torch.randn_like(x)
                     x_known = self.sqrt_alpha_hat[i-1] * gt_constraint + self.sqrt_one_minus_alpha_hat[i-1] * noise_for_constraint
                else:
                     x_known = gt_constraint
                x = x * (1. - mask) + x_known * mask
        
        # Final cleanup (same as standard)
        with torch.no_grad():
            x = torch.clamp(x, -1.0, 1.0)
            x[:, 0, :] = (x[:, 0, :] + 1.0) / 2.0 * CONFIG["speed_limit"]
            # x[:, 0, :] = torch.clamp(x[:, 0, :], min=0.0, max=38.5) # REMOVED
            x[:, 1, :] = x[:, 1, :] * CONFIG["accel_limit"]
            x_np = x.cpu().numpy()
            # for i in range(n_samples):
            #      x_np[i, 0, :] = gaussian_filter1d(x_np[i, 0, :], sigma=1.0)
            #      x_np[i, 1, :] = gaussian_filter1d(x_np[i, 1, :], sigma=1.0)
            return torch.from_numpy(x_np).to(CONFIG["device"])

    def sample(self, model, n_samples, cond, cfg_scale=7.5, target_lengths=None):
        with torch.no_grad():
            x = torch.randn((n_samples, 2, CONFIG["max_length"])).to(CONFIG["device"])
            
            # --- Inpainting Setup ---
            mask = torch.zeros_like(x)
            
            # Target image for inpainting:
            # Channel 0 (Speed): 0 m/s -> -1.0
            # Channel 1 (Accel): 0 m/s^2 -> 0.0
            gt_constraint = torch.zeros_like(x)
            gt_constraint[:, 0, :] = -1.0 # Speed baseline
            gt_constraint[:, 1, :] = 0.0  # Accel baseline
            
            if target_lengths is not None:
                for i, l in enumerate(target_lengths):
                    l = int(l)
                    # Constraint: Start is 0
                    mask[i, :, 0] = 1.0
                    # Constraint: End is 0
                    if l < CONFIG["max_length"]:
                        mask[i, :, l-1] = 1.0
                        # Constraint: Everything AFTER end is padding
                        mask[i, :, l:] = 1.0
                        # For padding, speed should be -1.0 (denorm to 0)
                    else:
                        mask[i, :, CONFIG["max_length"]-1] = 1.0
            else:
                mask[:, :, 0] = 1.0
                mask[:, :, -1] = 1.0

            for i in tqdm(reversed(range(1, self.timesteps)), desc="Sampling", position=0):
                t = (torch.ones(n_samples) * i).long().to(CONFIG["device"])
                
                # CFG Prediction
                if cfg_scale > 0:
                    # Unconditional: input zeros for condition
                    cond_zero = torch.zeros_like(cond).to(CONFIG["device"]) 
                    predicted_noise = model(x, t, cond)  
                    uncond_predicted_noise = model(x, t, cond_zero)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                else:
                    predicted_noise = model(x, t, cond)


                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # Standard DDPM Update
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
                # --- The Inpainting Step ---
                # We enforce x to match the noisy version of the Constraint (Zero) at the masked locations.
                
                # 1. Get the "known" state at timestep i-1 (which is just noise added to -1.0)
                # Note: We technically just stepped from i -> i-1. 
                # So we need q(x_{i-1} | x_0=ZERO)
                if i > 1:
                    noise_for_constraint = torch.randn_like(x)
                    x_known = self.sqrt_alpha_hat[i-1] * gt_constraint + self.sqrt_one_minus_alpha_hat[i-1] * noise_for_constraint
                else:
                    x_known = gt_constraint
                
                # 2. Blend: Keep generated parts (1-mask), Replace constrained parts (mask)
                x = x * (1. - mask) + x_known * mask

        # Constraint: Clamp to valid range [-1, 1] to prevent exploding gradients/outliers
        x = torch.clamp(x, -1.0, 1.0)

        # Denormalize channels
        # Speed: [-1, 1] -> [0, speed_limit]
        x[:, 0, :] = (x[:, 0, :] + 1.0) / 2.0 * CONFIG["speed_limit"]
        # x[:, 0, :] = torch.clamp(x[:, 0, :], min=0.0, max=38.5) # REMOVED: Redundant and restrictive.
        
        # Accel: [-1, 1] -> [-limit, limit]
        x[:, 1, :] = x[:, 1, :] * CONFIG["accel_limit"]
        
        # Post-Processing: Smoothing
        # Apply Gaussian filter to smooth out high-freq noise (accel tails)
        # Sigma 1.0 is mild but effective for 1D trajectories
        x_np = x.cpu().numpy()
        for i in range(n_samples):
             # Smooth speed
             # x_np[i, 0, :] = gaussian_filter1d(x_np[i, 0, :], sigma=1.0)
             pass
             # Smooth accel
             # x_np[i, 1, :] = gaussian_filter1d(x_np[i, 1, :], sigma=1.0)
        
        return torch.from_numpy(x_np).to(CONFIG["device"]) # Returns (N, 2, L)

# --- 4. Training Loop ---

def train(args):
    dataset = TrajectoryDataset(args.data_path, limit_files=args.limit_files)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = Unet1D().to(CONFIG["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    diffusion = DiffusionUtils()
    loss_fn = nn.MSELoss()
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        # pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        steps = 0
        for x, cond in dataloader:
            x, cond = x.to(CONFIG["device"]), cond.to(CONFIG["device"])
            
            t = diffusion.sample_timesteps(x.shape[0])
            x_t, noise = diffusion.noise_images(x, t)
            
            if np.random.random() < 0.1:
                cond = torch.zeros_like(cond)
                
            predicted_noise = model(x_t, t, cond)
            loss = loss_fn(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | MSE: {epoch_loss/steps:.5f}")
            torch.save(model.state_dict(), f"data/diffusion_ckpt_ep{epoch+1}.pt")
    
    torch.save(model.state_dict(), "data/diffusion_final.pt")
    print("Training Complete.")

def train_pid(args):
    """
    Physics-Informed Diffusion (PID) Training Loop.
    Includes additional loss terms for Distance Consistency, Smoothness, and Constraints.
    """
    dataset = TrajectoryDataset(args.data_path, limit_files=args.limit_files)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = Unet1D().to(CONFIG["device"])
    
    # --- Fine-tuning support: Load checkpoint if provided ---
    if hasattr(args, 'model_path') and args.model_path and os.path.exists(args.model_path):
        print(f"Loading checkpoint for fine-tuning: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=CONFIG["device"]))
    else:
        print("Training from scratch (no checkpoint provided)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    diffusion = DiffusionUtils()
    mse_loss_fn = nn.MSELoss()
    
    print("Starting PID training (Physics-Informed Diffusion)...")
    print(f"DEBUG: Epochs: {args.epochs}, Batch Size: {args.batch_size}, Dataset Len: {len(dataset)}")

    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        
        for x, cond in dataloader:
            x, cond = x.to(CONFIG["device"]), cond.to(CONFIG["device"])
            
            t = diffusion.sample_timesteps(x.shape[0])
            x_t, noise = diffusion.noise_images(x, t)
            
            # Classifier-Free Guidance Training (10% drop)
            if np.random.random() < 0.1:
                cond_in = torch.zeros_like(cond)
            else:
                cond_in = cond
                
            predicted_noise = model(x_t, t, cond_in)
            
            # --- 1. Standard MSE Loss ---
            mse_loss = mse_loss_fn(noise, predicted_noise)
            
            # --- 2. Physics-Based Losses ---
            # Estimate denoised trajectory x0 from prediction
            # x0 = (xt - sqrt(1-alpha_bar)*eps) / sqrt(alpha_bar)
            alpha_bar = diffusion.alpha_hat[t][:, None, None]
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
            
            # Denormalize x0 for physical calculations
            # Speed: x[:, 0] in [-1, 1] -> [0, 40]
            # Accel: x[:, 1] in [-1, 1] -> [-5, 5]
            speed_norm = x0_pred[:, 0, :]
            accel_norm = x0_pred[:, 1, :]
            
            speed_phys = (speed_norm + 1.0) / 2.0 * CONFIG["speed_limit"]
            accel_phys = accel_norm * CONFIG["accel_limit"]
            
            # A. Distance Consistency Loss (Normalized)
            # Original L_dist = MSE(dist_pred, dist_target) explodes because dist ~ 60,000 m.
            # Fix: Compare Average Speeds (scale ~20 m/s) instead.
            # dist_pred = sum(speed_phys)
            # dist_target = target_avg_speed * target_duration
            # avg_speed_pred = dist_pred / target_duration
            
            # Avoid div by zero (target_duration min is ~60/3000 -> small)
            # target_duration is in seconds (cond * 3000)
            
            # Integral of predicted speed
            dist_pred = torch.sum(torch.clamp(speed_phys, min=0), dim=1) # (B,)
            
            # Restore target_duration and target_avg_speed definitions
            target_duration = cond[:, 1] * 3000.0
            target_avg_speed = cond[:, 0] * 30.0 # m/s
            
            avg_speed_pred = dist_pred / (target_duration + 1.0) # (B,)
            
            # Note: target_avg_speed is target MEAN speed. 
            # avg_speed_pred should match target_avg_speed.
            
            loss_dist = F.mse_loss(avg_speed_pred, target_avg_speed)
            
            # loss_dist is now (m/s)^2. error ~ 5 m/s -> loss ~ 25.
            # Base MSE is ~1.0. 
            # Weight 0.1 -> Contribution ~2.5. Safe.
            
            # B. Smoothness/Jerk Loss & Asymmetric Acceleration
            # Real driving: vehicles brake stronger than they can accelerate.
            
            # Penalize values outside thresholds (in physical space)
            accel_threshold_norm = CONFIG["accel_threshold"] / CONFIG["accel_limit"]  # ~0.5
            decel_threshold_norm = CONFIG["decel_threshold"] / CONFIG["accel_limit"]  # ~0.8
            
            accel_pos = F.relu(accel_norm)
            accel_neg = F.relu(-accel_norm)
            
            loss_asym = (torch.mean(F.relu(accel_pos - accel_threshold_norm)**2) + 
                         torch.mean(F.relu(accel_neg - decel_threshold_norm)**2))
            
            # Smoothness (Jerk) - penalize rapid changes in acceleration
            jerk = torch.diff(accel_norm, dim=1)
            jerk_threshold_norm = CONFIG["jerk_threshold"] / CONFIG["accel_limit"]  # ~0.4
            jerk_excess = F.relu(torch.abs(jerk) - jerk_threshold_norm)
            loss_jerk = torch.mean(jerk_excess ** 2)
            
            # --- CSDI-style Accel Distribution Variance Loss ---
            # Penalize if variance of acceleration is too wide (encourage concentration around 0)
            target_var_norm = (CONFIG["target_accel_std"] / CONFIG["accel_limit"]) ** 2  # ~0.0144
            accel_var = torch.var(accel_norm, dim=1).mean()  # Mean variance across batch
            loss_accel_dist = F.relu(accel_var - target_var_norm)
            
            # C. Boundary Constraint Loss
            # Start and End speed should be 0 (-1 in normalized space)
            loss_boundary = (torch.mean((speed_norm[:, 0] + 1.0)**2) + 
                             torch.mean((speed_norm[:, -1] + 1.0)**2))
            
            # Combine Losses (v1.3: Full CSDI-style physics weights)
            # MSE: 1.0 (base)
            # Dist: 0.0 (Rely on conditioning + post-proc)
            # Asym: 0.03 
            # Jerk: 0.02 (from CONFIG["jerk_penalty_weight"])
            # Accel Dist: 0.05 (from CONFIG["accel_distribution_weight"])
            # Boundary: 0.0 (Rely on inference-time inpainting)
            total_loss = (mse_loss + 
                          0.0 * loss_dist + 
                          0.03 * loss_asym + 
                          CONFIG["jerk_penalty_weight"] * loss_jerk + 
                          CONFIG["accel_distribution_weight"] * loss_accel_dist +
                          0.0 * loss_boundary)

            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            steps += 1
            
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss/steps:.5f} (MSE base: {mse_loss.item():.5f})")
            torch.save(model.state_dict(), f"{args.save_dir}/diffusion_ckpt_ep{epoch+1}.pt")
            print(f"Epoch {epoch+1}/{args.epochs} | MSE: {epoch_loss/steps:.5f}", flush=True)
            torch.save(model.state_dict(), f"{args.save_dir}/diffusion_ckpt_ep{epoch+1}.pt")
            
    torch.save(model.state_dict(), "data/diffusion_final_pid.pt")
    print("Training Complete.", flush=True)

# --- 5. Generation / Inference ---

def generate(args):
    print("Generating trajectories...")
    model = Unet1D().to(CONFIG["device"])
    model.load_state_dict(torch.load(args.model_path, map_location=CONFIG["device"])) 
    model.eval()
    
    diffusion = DiffusionUtils()
    
    # Generate N random samples
    n_samples = args.n_samples
    
    # --- IMPROVED CONDITION SAMPLING ---
    # Instead of Uniform [5, 35], sample from Real Data Distribution to match density.
    import pickle
    stats_path = "data/real_data_stats.pkl"
    if os.path.exists(stats_path):
        print(f"Loading real data stats from {stats_path} for Condition Sampling...")
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
            
            # Try to load pre-saved metadata (NEW format)
            if "avg_speeds" in stats and "durations" in stats:
                print(f"Using pre-saved metadata: {len(stats['avg_speeds'])} trips.")
                real_avg_speeds = np.array(stats["avg_speeds"])
                real_lens = np.array(stats["durations"])
            # Fallback: Reconstruct from raw speeds (OLD format)
            elif "speeds" in stats:
                print("Reconstructing metadata from raw speeds...")
                real_avg_speeds = []
                real_lens = []
                for s in stats["speeds"]:
                    if len(s) > 10:
                        real_avg_speeds.append(np.mean(s))
                        real_lens.append(len(s))
                real_avg_speeds = np.array(real_avg_speeds)
                real_lens = np.array(real_lens)
            else:
                print("Warning: real_data_stats.pkl malformed. Using fallback Uniform.")
                real_avg_speeds = None
                real_lens = None
            
            if real_avg_speeds is not None and len(real_avg_speeds) > 0:
                print(f"Sampling conditions from {len(real_avg_speeds)} real trips.")
                
                # --- CSDI-style Weighted Condition Sampling ---
                # Boost probability for high-speed trips to fill the tail.
                speed_boost_power = getattr(args, 'speed_boost', 1.5)  # 1.0 = uniform, 2.0 = strong boost
                weights = np.power(real_avg_speeds + 1, speed_boost_power)  # +1 to avoid zero
                weights = weights / weights.sum()  # Normalize to probability
                
                # Sample with weighted replacement
                indices = np.random.choice(len(real_avg_speeds), n_samples, replace=True, p=weights)
                
                # Log the effect
                original_mean = np.mean(real_avg_speeds)
                sampled_mean = np.mean(real_avg_speeds[indices])
                print(f"  Weighted sampling (power={speed_boost_power}): original mean={original_mean:.1f} → sampled mean={sampled_mean:.1f} m/s")
                
                # Add noise to prevent overfitting
                target_speeds = torch.tensor(real_avg_speeds[indices]).float() + torch.randn(n_samples) * 0.5
                
                # Keep bias correction minimal since weighted sampling handles the tail
                mask_high = target_speeds > 20.0
                target_speeds[mask_high] *= 1.02  # Reduced from 1.03
                
                target_speeds = torch.clamp(target_speeds, 2.0, 42.0)
                
                target_lengths = torch.tensor(real_lens[indices]).float()
                target_lengths = torch.clamp(target_lengths, 60, CONFIG["max_length"])


            else:
                print("Warning: real_data_stats.pkl empty. Using fallback Uniform.")
                target_lengths = torch.randint(low=60, high=CONFIG["max_length"], size=(n_samples,))
                target_speeds = torch.rand(n_samples) * 30.0 + 5.0
    else:
        print("Warning: real_data_stats.pkl not found. Using fallback Uniform.")
        target_lengths = torch.randint(low=60, high=CONFIG["max_length"], size=(n_samples,))
        target_speeds = torch.rand(n_samples) * 30.0 + 5.0
    
    # Normalize conditions
    c_speed = target_speeds / 30.0
    c_dur = target_lengths / 3000.0
    cond = torch.stack([c_speed, c_dur], dim=1).float().to(CONFIG["device"])
    
    samples = diffusion.sample(model, n_samples=n_samples, cond=cond, cfg_scale=args.cfg_scale, target_lengths=target_lengths)
    
    # Save to CSV (Extract speed channel)
    output_speed = samples[:, 0, :].cpu().numpy()
    # --- Post-Processing: Tail Stretching (REMOVED) ---
    # With fixed real_data_stats.pkl (full dataset), model generates correct tail naturally.
    # Hack removed.
            
    # Clamp to physical max to be safe
    output_speed = np.clip(output_speed, 0.0, 40.0)

    # Enforce integral of speed matches target distance exactly
    print("Applying Global Distance Constraint...")
    for i in range(n_samples):
        l = int(target_lengths[i])
        current_speed = output_speed[i, :l]
        
        # Calculate current distance (integral of speed, dt=1s)
        current_dist = np.sum(current_speed)
        
        # Target distance = target_avg_speed * duration
        # Note: target_speeds[i] is the target AVERAGE speed
        target_dist = target_speeds[i].item() * l
        
        # Calculate scaling factor
        if current_dist > 1.0: # Avoid division by zero
            k = target_dist / current_dist
            
            # Apply scaling
            # We clip k to avoid extreme scaling (e.g. if model generated 0 speed)
            k = np.clip(k, 0.5, 2.0) 
            
            output_speed[i, :l] *= k
            
            # Enforce exact 0 at boundaries
            output_speed[i, 0] = 0.0
            output_speed[i, l-1] = 0.0
            
            # Also scale acceleration to keep consistency?
            # a_new = d(v_new)/dt = d(k*v)/dt = k * a
            # So yes, we should scale acceleration too.
            samples[i, 1, :l] *= k
            
    # Update output_speed after scaling for saving
    # samples tensor is already updated in-place for accel
    
    # Final Safety Clamp (Post-scaling)
    output_speed = np.clip(output_speed, 0.0, 40.0)

    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%b%d_%I%M%p").lower()
    
    df_out = pd.DataFrame(output_speed)
    df_out["target_len"] = target_lengths.cpu().numpy()
    df_out["target_spd"] = target_speeds.cpu().numpy()
    
    if args.output_file:
        csv_path = args.output_file
    else:
        csv_path = f"{args.save_dir}/synthetic_trajectories_{timestamp}.csv"
        
    df_out.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    
    # Plot
    # plt.figure(figsize=(12, 8))
    # plt.subplot(2, 1, 1)
    # for i in range(n_samples):
    #     l = int(target_lengths[i])
    #     plt.plot(output_speed[i, :l], label=f"V={target_speeds[i]:.1f}" if i%5==0 else "")
    # plt.title("Synthetic Trajectories (Speed Channel)")
    # plt.ylabel("Speed (m/s)")
    # plt.legend()
    #
    # plt.subplot(2, 1, 2)
    # output_accel = samples[:, 1, :].cpu().numpy()
    # for i in range(n_samples):
    #     l = int(target_lengths[i])
    #     plt.plot(output_accel[i, :l])
    # plt.title("Synthetic Trajectories (Accel Channel)")
    # plt.ylabel("Accel (m/s^2)")
    # plt.xlabel("Time (s)")
    # 
    # plt.tight_layout()
    # plot_path = f"report/validation_plot_{timestamp}.png"
    # plt.savefig(plot_path)
    # print(f"Saved {plot_path}")

    # --- NEW: Grid Plot (10x5) for better visibility ---
    if n_samples >= 50:
        print("Generating 5x10 grid plot...")
        fig, axes = plt.subplots(5, 10, figsize=(25, 15))
        axes = axes.flatten()
        for i in range(50):
            l = int(target_lengths[i])
            axes[i].plot(output_speed[i, :l], color='blue', linewidth=1)
            axes[i].set_title(f"Target: {target_speeds[i]:.1f} m/s", fontsize=8)
            axes[i].set_ylim(0, 40)
            axes[i].grid(True, alpha=0.3)
            if i >= 45: axes[i].set_xlabel("Time (s)", fontsize=8)
            if i % 5 == 0: axes[i].set_ylabel("Speed", fontsize=8)
        
        plt.tight_layout()
        grid_plot_path = f"report/validation_grid_{timestamp}.png"
        plt.savefig(grid_plot_path)
        print(f"Saved grid plot: {grid_plot_path}")

def plot_csv(csv_path):
    print(f"Re-plotting trajectories from {csv_path}...")
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    # Extract metadata if available
    if "target_len" in df.columns and "target_spd" in df.columns:
        target_lengths = df["target_len"].values
        target_speeds = df["target_spd"].values
        traj_cols = [c for c in df.columns if c not in ["target_len", "target_spd"]]
        output_speed = df[traj_cols].values
    else:
        # Fallback for old format
        output_speed = df.values
        target_lengths = [len(row) for row in output_speed]
        target_speeds = [np.mean(row) for row in output_speed]

    n_samples = len(output_speed)
    n_plot = min(50, n_samples)
    
    timestamp = os.path.basename(csv_path).replace("synthetic_trajectories_", "").replace(".csv", "")
    if not timestamp:
        timestamp = "replot"

    print(f"Generating 5x10 grid plot for {n_plot} samples...")
    fig, axes = plt.subplots(5, 10, figsize=(25, 15))
    axes = axes.flatten()
    for i in range(n_plot):
        l = int(target_lengths[i])
        axes[i].plot(output_speed[i, :l], color='blue', linewidth=1)
        axes[i].set_title(f"Target: {target_speeds[i]:.1f} m/s", fontsize=8)
        axes[i].set_ylim(0, 40)
        axes[i].grid(True, alpha=0.3)
        if i >= 45: axes[i].set_xlabel("Time (s)", fontsize=8)
        if i % 5 == 0: axes[i].set_ylabel("Speed", fontsize=8)
    
    # Hide unused axes
    for j in range(n_plot, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    grid_plot_path = f"report/validation_grid_{timestamp}.png"
    plt.savefig(grid_plot_path)
    print(f"Saved grid plot: {grid_plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--pid', action='store_true', help="Train with Physics-Informed Diffusion Loss")
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--data_path', type=str, default='../../artifacts')
    parser.add_argument('--model_path', type=str, default='diffusion_final.pt')
    parser.add_argument('--save_dir', type=str, default='.', help="Directory to save checkpoints")
    parser.add_argument('--epochs', type=int, default=CONFIG["epochs"])
    parser.add_argument('--batch_size', type=int, default=CONFIG["batch_size"])
    parser.add_argument('--lr', type=float, default=CONFIG["lr"])
    parser.add_argument('--limit_files', type=int, default=None, help="Limit number of training files")
    parser.add_argument('--cfg_scale', type=float, default=7.5, help="Classifier-Free Guidance Scale")
    parser.add_argument('--n_samples', type=int, default=500, help="Number of samples to generate")
    parser.add_argument('--output_file', type=str, default=None, help="Explicit output filename for generated CSV")
    parser.add_argument('--speed_boost', type=float, default=1.5, help="Power for weighted condition sampling (1.0=uniform, 2.0=strong)")
    parser.add_argument('--plot_csv', type=str, default=None, help="Re-plot trajectories from a CSV file")

    
    args = parser.parse_args()
    
    # Override config from args
    CONFIG["epochs"] = args.epochs
    CONFIG["batch_size"] = args.batch_size
    CONFIG["lr"] = args.lr

    if args.pid:
        train_pid(args)
    elif args.train:
        # Pass limit_files to train function, need to update train signature
        train(args)
    if args.generate:
        generate(args)
    if args.plot_csv:
        plot_csv(args.plot_csv)
    
    if not (args.train or args.pid or args.generate or args.plot_csv):
        print("Please specify --train, --pid, --generate, or --plot_csv")


# ------------------------------------------------------------------------------
# Example: How to use sample_physics for future vehicle types
# ------------------------------------------------------------------------------
# def bus_physics_constraint(x0):
#     """Example custom constraint for a BUS (Heavy Vehicle)."""
#     # 1. Power Limit: Max accel decreases as speed increases
#     # P = m * a * v => a_max = P_max / (m * v)
#     speed = (x0[:, 0, :] + 1) / 2 * 40.0
#     accel = x0[:, 1, :] * 5.0
#     
#     power_limit_accel = 200000 / (15000 * (speed + 1.0)) # Dummy values
#     accel_excess = F.relu(accel - power_limit_accel)
#     
#     # 2. Comfort: Strict jerk limit for passengers
#     jerk = torch.diff(accel, dim=1)
#     jerk_excess = F.relu(torch.abs(jerk) - 1.0)
#     
#     return torch.mean(accel_excess**2) + torch.mean(jerk_excess**2)
#
# To generate bus trajectories:
# samples = diffusion.sample_physics(model, 10, cond, physics_guide_fn=bus_physics_constraint)
# ------------------------------------------------------------------------------
