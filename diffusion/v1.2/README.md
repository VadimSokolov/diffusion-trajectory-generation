# Diffusion Model v1.2 Snapshot

## Overview

v1.2 represents a **Physics-Informed Diffusion (PID)** model fine-tuned with CSDI-style physics losses to improve acceleration distribution and speed gap matching. This is an incremental improvement over v1.1, focusing on realistic trajectory generation.

## Evolution from v1.1

| Aspect | v1.1 | v1.2 |
| :--- | :--- | :--- |
| **Physics Losses** | Basic asymmetric accel | + Accel Distribution Variance, + Jerk Penalty |
| **Training** | Single-stage | Fine-tuned from v1.1 with reduced LR (1e-5) |
| **Generation** | Uniform condition sampling | Weighted sampling (`speed_boost=1.5`) |
| **WD Speed** | ~0.70 | **0.50** (30% improvement) |
| **WD Accel** | ~0.13 | **0.10** (23% improvement) |

### Key Changes

1. **Accel Distribution Variance Loss**: Penalizes if acceleration variance exceeds 0.6 m/s² (from CSDI).
2. **Jerk Penalty**: Penalizes rapid changes in acceleration (threshold: 2.0 m/s³).
3. **Weighted Condition Sampling**: High-speed trips are oversampled (`speed_boost_power=1.5`) to fill the distribution tail.
4. **Reduced Physics Weights**: `accel_distribution_weight=0.02`, `jerk_penalty_weight=0.01` for stable fine-tuning.

---

## Architecture

```
Model: Unet1D (Conditional Time-Dependent U-Net)
├── Input: (B, 2, 512) - [Speed, Accel] trajectories, normalized to [-1, 1]
├── Condition: (B, 2) - [avg_speed/30, duration/3000]
├── Time Embedding: Sinusoidal + MLP
├── Encoder: 4 ResBlocks with self-attention
├── Decoder: 4 ResBlocks with skip connections
└── Output: Noise prediction (B, 2, 512)

Parameters: ~43M
Embedding Dim: 256
Max Length: 512 timesteps
```

### Diffusion Process

- **Timesteps**: 1000
- **Noise Schedule**: Linear (β: 1e-4 → 0.02)
- **Sampling**: DDPM with CFG (scale=3.0)

---

## Training Settings

```python
CONFIG = {
    "speed_limit": 40.0,           # m/s
    "accel_limit": 5.0,            # m/s²
    "max_length": 512,
    "batch_size": 256,
    "lr": 1e-5,                    # Reduced for fine-tuning
    "epochs": 300,                 # Fine-tuned from v1.1
    "timesteps": 1000,
    "embed_dim": 256,
    # Physics Loss Parameters
    "accel_distribution_weight": 0.02,
    "jerk_penalty_weight": 0.01,
    "target_accel_std": 0.6,       # m/s²
    "jerk_threshold": 2.0,         # m/s³
    "accel_threshold": 2.5,        # m/s²
    "decel_threshold": 4.0,        # m/s²
}
```

### Loss Function

```
L_total = L_mse + 0.03 * L_asym + 0.01 * L_jerk + 0.02 * L_accel_dist
```

---

## Generation Settings

```bash
python diffusion_trajectory.py --generate \
    --model_path v1.2/diffusion_ckpt_ep300.pt \
    --cfg_scale 3.0 \
    --speed_boost 1.5 \
    --n_samples 500
```

### Weighted Condition Sampling

```python
weights = np.power(avg_speeds + 1, speed_boost_power)  # 1.5 default
indices = np.random.choice(len(avg_speeds), n_samples, replace=True, p=weights/weights.sum())
```

---

## Evaluation Results (v1.3v2 ep300)

| Metric | Real | Synthetic |
| :--- | :--- | :--- |
| Mean Speed (m/s) | 16.92 | 18.16 |
| Max Speed (m/s) | 39.56 | 38.32 |
| Mean Duration (s) | 304.0 | 261.9 |

| Distribution Metric | Value |
| :--- | :--- |
| WD Speed | 0.4964 |
| WD Accel | 0.1017 |
| WD VSP | 1.9991 |
| Boundary Violation | 0.00% |

---

## Files in This Snapshot

```
v1.2/
├── README.md                           # This file
├── diffusion_trajectory.py             # Main script (frozen at v1.2)
├── diffusion_ckpt_ep300.pt             # Model weights (175 MB)
├── evaluation_report_v1.3v2_ep300.md   # Evaluation report
├── evaluation_plots_v1.3v2_ep300.png   # Distribution plots
└── bin/
    └── eval.sh                         # Evaluation script
```

---

## Reproduction

```bash
# On Hopper cluster
cd ~/projects/svtrip/diffusion

# Evaluate
sbatch bin/eval.sh --model_path v1.2/diffusion_ckpt_ep300.pt --cfg_scale 3.0 --suffix _v1.2_repro

# Generate trajectories
python diffusion_trajectory.py --generate \
    --model_path v1.2/diffusion_ckpt_ep300.pt \
    --cfg_scale 3.0 \
    --n_samples 500
```
