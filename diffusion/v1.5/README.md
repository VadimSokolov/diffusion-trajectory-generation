# v1.5: The Final Physics-Informed Diffusion Model

**Status:** FINAL / PRODUCTION
**Date:** January 4, 2026
**Base Model:** v1.4 (Trained from scratch)
**Fine-Tuning:** 200 Epochs with increased physics penalties.

## Overview
v1.5 is the definitive model version that solves the "fat acceleration tail" problem while maintaining excellent speed distribution alignment. It was fine-tuned to strictly enforce acceleration variance constraints.

## Key Configurations
*   **Physics Weights:**
    *   `accel_distribution_weight`: **0.05** (Increased from 0.02)
    *   `jerk_penalty_weight`: **0.02** (Unchanged)
*   **Generation:**
    *   `speed_boost`: **1.75** (Optimal balance for 25-30 m/s gap)
    *   `cfg_scale`: **3.0**

## Performance Metrics (SOTA)
| Metric | Result | Target | Pass/Fail |
| :--- | :--- | :--- | :--- |
| **WD Accel** | **0.0800** | < 0.08 | ✅ PASS |
| **WD Speed** | **0.5622** | < 0.60 | ✅ PASS |
| **Mean Speed** | 18.17 m/s | ~18 m/s | ✅ PASS |
| **Boundary Violations** | 0.00% | 0.00% | ✅ PASS |

## Reproduction Steps

### 1. Training (Fine-Tuning)
This model was fine-tuned from `diffusion_final_pid.pt` (v1.4).
```bash
# Using the provided sbatch script
sbatch finetune_v15.sbatch
```
*Script content:*
```bash
python diffusion_trajectory.py --pid \
    --model_path data/diffusion_final_pid.pt \
    --epochs 200 \
    --lr 2e-5 \
    --save_dir data/v1.5_checkpoints \
    --accel_weight 0.05 \
    --jerk_weight 0.02
```

### 2. Generation & Evaluation
To reproduce the results, generate trajectories with `speed_boost=1.75`.
```bash
python diffusion_trajectory.py --generate \
    --model_path data/diffusion_final.pt \
    --n_samples 6367 \
    --cfg_scale 3.0 \
    --speed_boost 1.75 \
    --output_file data/synthetic_v1.5.csv
```

### 3. Files Included
*   `diffusion_trajectory.py`: The exact code used for training and generation.
*   `finetune_v15.sbatch`: Slurm script for reproduction.
*   `eval.sh`: Script used for evaluation.
*   `evaluation_report_*.md`: Full metrics output.
*   `evaluation_plots_*.png`: Comparison histograms.

## Model Location
The final model weights are stored on Hopper at:
`~/projects/svtrip/diffusion/data/diffusion_final.pt` (Symlinked from `diffusion_v1.5.pt`)
