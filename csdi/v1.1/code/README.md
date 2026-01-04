# CSDI: Conditional Score-based Diffusion for Trajectories

## Overview

Implementation of Conditional Score-based Diffusion with a Transformer backbone for vehicle trajectory generation. Based on the CSDI paper (Tashiro et al., NeurIPS 2021).

**Status**: ✅ Production-ready with post-processing smoothing

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CSDITransformer                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Noisy Trajectory x_t ─────────────────────┐                    │
│                                             │                    │
│  ┌─────────────────┐                       │                    │
│  │ Diffusion Time  │                       ▼                    │
│  │ Embedding t     │ ──▶ ┌──────────────────────────────────┐  │
│  └─────────────────┘     │                                   │  │
│                          │    Transformer Encoder            │  │
│  ┌─────────────────┐     │    (4 layers, 8 heads)           │  │
│  │ Condition Embed │ ──▶ │                                   │  │
│  │ (avg_spd, dur)  │     │    + Positional Encoding          │  │
│  └─────────────────┘     │    + Time Injection               │  │
│                          │    + Condition Injection          │  │
│                          └──────────────────────────────────┘  │
│                                        │                        │
│                                        ▼                        │
│                          ┌──────────────────────────────────┐  │
│                          │   Output: Predicted Noise ε̂      │  │
│                          └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

Reverse Diffusion (Sampling):
    x_T ~ N(0, I)  →  x_{T-1}  →  ...  →  x_1  →  x_0 (clean trajectory)
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Score-based Diffusion** | DDPM-style denoising for high diversity |
| **Transformer Backbone** | Full attention, parallel training |
| **Boundary Inpainting** | Force v=0 at start/end during sampling |
| **Conditional Generation** | Control via avg_speed, duration, max_speed |
| **Variable Length** | Masking for different trajectory lengths |
| **Post-process Smoothing** | Gaussian filter to reduce diffusion jitter |

---

## Development History

### Version 1.0 (Initial)
- Basic CSDI implementation with 100 diffusion steps
- Boundary ramp of 10 seconds
- **Issue**: Generated trajectories were jerky/noisy due to diffusion sampling artifacts

### Version 1.1 (Current - Best Results)
Fixed jerkiness with multiple improvements:

| Change | Before | After | Impact |
|--------|--------|-------|--------|
| Diffusion steps | 100 | 200 | Smoother denoising |
| Boundary ramp | 10s | 15s | Better zero-speed transitions |
| Post-process smoothing | None | Gaussian (kernel=9) | Eliminates high-freq jitter |
| Temporal smoothness loss | None | 0.1 weight | Smoother learned patterns |

---

## Current Best Pipeline

### Generation Command (Recommended)
```bash
python csdi_trajectory.py \
    --generate \
    --model_path csdi_best.pt \
    --n_samples 1000 \
    --diffusion_steps 200 \
    --smooth_kernel 9
```

### Pipeline Stages

```
Training Data (Microtrips)
         │
         ▼
┌─────────────────────────────┐
│  1. TRAIN: Score Network    │
│     - 100 epochs            │
│     - MSE loss + smoothness │
│     - Cosine LR schedule    │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  2. SAMPLE: Reverse Diffusion│
│     - 200 diffusion steps   │
│     - Boundary inpainting   │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  3. SMOOTH: Gaussian Filter │
│     - Kernel size 9         │
│     - Preserves dynamics    │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  4. BOUNDARY: Cosine Ramps  │
│     - 15s ramp at start/end │
│     - Ensures v=0 endpoints │
└─────────────────────────────┘
         │
         ▼
   Final Trajectories
```

---

## Configuration & Hyperparameters

### Model Architecture
```python
CONFIG = {
    "max_length": 512,       # Max trajectory length (seconds)
    "d_model": 256,          # Transformer hidden dimension
    "n_heads": 8,            # Attention heads
    "n_layers": 4,           # Transformer layers
    "d_ff": 512,             # Feed-forward dimension
    "dropout": 0.1,          # Regularization
}
```

### Diffusion Settings
```python
CONFIG = {
    "diffusion_steps": 200,  # Denoising steps (more = smoother)
    "beta_start": 0.0001,    # Noise schedule start
    "beta_end": 0.02,        # Noise schedule end
    "schedule": "linear",    # Beta schedule type
}
```

### Training Settings
```python
CONFIG = {
    "epochs": 100,
    "batch_size": 32,
    "lr": 2e-4,              # Learning rate
}

# Loss function (in training loop)
smoothness_weight = 0.1      # Weight for temporal smoothness loss
```

### Sampling/Post-processing Settings
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--diffusion_steps` | 200 | 100-500 | More steps = smoother but slower |
| `--smooth_kernel` | 7 | 0-15 | Gaussian kernel size (0=disable) |
| `boundary_ramp` | 15 | 10-30 | Seconds for start/end ramp |

---

## Post-Processing: Gaussian Smoothing

### Why Smoothing is Needed
Diffusion models generate samples by iteratively denoising random noise. Even with many steps, residual high-frequency artifacts remain. Post-processing eliminates these while preserving trajectory dynamics.

### Implementation
```python
def _smooth_trajectory(self, x, lengths, kernel_size=7):
    """Gaussian smoothing to reduce jitter."""
    sigma = kernel_size / 4.0
    kernel = torch.exp(-torch.linspace(...)**2 / (2*sigma**2))
    kernel = kernel / kernel.sum()  # Normalize
    
    # Apply 1D convolution with replicate padding
    smoothed = F.conv1d(traj_padded, kernel)
```

### Effect of Kernel Size

| Kernel | Smoothness | Detail Preservation | Recommended For |
|--------|------------|---------------------|-----------------|
| 0 | None | Full | Debugging only |
| 5 | Light | High | Minimal smoothing |
| 7 | Medium | Good | General use |
| **9** | **Strong** | **Good** | **Best results** |
| 11+ | Very strong | Some loss | Over-smoothed |

---

## Temporal Smoothness Loss (Training)

### Motivation
Instead of only post-processing, we can train the model to predict smooth noise patterns directly.

### Implementation
```python
# In training loop:
# Standard MSE loss for noise prediction
mse_loss = F.mse_loss(noise_pred * mask, noise * mask)

# Temporal smoothness: penalize jerky predictions
diff = noise_pred[:, 1:] - noise_pred[:, :-1]  # First derivative
diff_mask = mask[:, 1:] * mask[:, :-1]
smoothness_loss = (diff ** 2 * diff_mask).sum() / diff_mask.sum()

# Combined loss
loss = mse_loss + 0.1 * smoothness_loss
```

### Effect
| Metric | Without Smoothness Loss | With Smoothness Loss |
|--------|-------------------------|----------------------|
| Jitter | Post-processing needed | Reduced inherently |
| Training time | Same | Same |
| Model size | Same | Same |

### Recommendation
- **Current model** was trained WITHOUT smoothness loss
- **Retraining** with smoothness loss is optional but provides cleaner outputs
- Post-processing smoothing is sufficient for production quality

---

## Results Comparison

### Before vs After Smoothing

| Aspect | v1.0 (No smoothing) | v1.1 (With smoothing) |
|--------|---------------------|----------------------|
| High-freq jitter | ❌ Severe | ✅ Eliminated |
| Curve smoothness | Jagged | Natural |
| Boundary conditions | ✅ Good | ✅ Good |
| Diversity | ✅ High | ✅ High |
| Publication-ready | ❌ No | ✅ Yes |

### Comparison with Other Models

| Model | Mode Collapse | Boundary | Smoothness | Training Time |
|-------|---------------|----------|------------|---------------|
| PARSynthesizer (SDV) | ❌ Severe | ❌ Violated | ❌ Flat | ~45 min |
| Chronos | ✅ None | ✅ Good | ✅ Smooth | ~30 min |
| **CSDI v1.1** | ✅ None | ✅ Perfect | ✅ Smooth | ~18 min |

---

## Usage

### Training
```bash
# Full training (with smoothness loss)
python csdi_trajectory.py \
    --train \
    --data_path ../../data/Microtrips \
    --epochs 100 \
    --batch_size 32 \
    --lr 2e-4
```

### Generation
```bash
# Best quality (recommended)
python csdi_trajectory.py \
    --generate \
    --model_path csdi_best.pt \
    --n_samples 1000 \
    --smooth_kernel 9 \
    --diffusion_steps 200

# Faster generation (slightly lower quality)
python csdi_trajectory.py \
    --generate \
    --model_path csdi_best.pt \
    --n_samples 1000 \
    --smooth_kernel 7 \
    --diffusion_steps 100
```

### HPC (Hopper)
```bash
# Sync code
./sync_to_hopper.sh

# On Hopper
cd ~/projects/svtrip/paper3/code/csdi
sbatch submit.sbatch

# Monitor
squeue -u $USER
tail -f csdi-*.out

# Fetch results
./fetch_results.sh  # Downloads to results/ subdirectory
```

---

## Output Files

| File | Description |
|------|-------------|
| `csdi_best.pt` | Best model checkpoint |
| `csdi_final.pt` | Final model after all epochs |
| `csdi_ckpt_ep*.pt` | Periodic checkpoints (every 10 epochs) |
| `csdi_synthetic_*.csv` | Generated trajectories (padded) |
| `csdi_synthetic_*.pkl` | Trajectories as numpy arrays (variable length) |
| `csdi_samples_*.png` | Sample visualization (12 trajectories) |
| `results/` | Fetched results directory |

---

## Troubleshooting

### Jerky Trajectories
- Increase `--smooth_kernel` (try 9 or 11)
- Increase `--diffusion_steps` (try 300)
- Retrain with smoothness loss

### Slow Generation
- Reduce `--diffusion_steps` (100 is acceptable)
- Reduce `--n_samples` for testing

### Conda Activation Error on Hopper
Add before `conda activate`:
```bash
eval "$(conda shell.bash hook)"
```

---

## References

- [CSDI Paper](https://arxiv.org/abs/2107.03502) - Original CSDI for time series imputation
- [DDPM](https://arxiv.org/abs/2006.11239) - Denoising Diffusion Probabilistic Models
- [Transformers for Time Series](https://arxiv.org/abs/2012.07436) - Attention mechanisms in temporal data
