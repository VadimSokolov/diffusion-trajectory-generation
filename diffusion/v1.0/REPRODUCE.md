# Reproducibility Snapshot v1.0 - Tail Fix

This directory contains the code and model used to generate the `tail_fix` results, which achieved the best performance thus far.

## Summary of Results (Reproduction 5197869)
### Distribution Metrics
- **WD Speed:** 1.6029
- **WD Accel:** 0.0284
- **WD VSP:** 1.0129
- **WD SAFD (2D):** 0.0003
- **MMD:** 0.0105
- **Discriminative Score:** 0.1067
- **TSTR MAE:** 0.3051
- **KS VSP:** 0.0823

### Kinematic Metrics
- **LDLJ (Smoothness) Real:** -11.0779
- **LDLJ (Smoothness) Syn:** -11.0277
- **Boundary Violation Rate:** 0.00% (Threshold: 0.1 m/s)

### Summary Statistics
| Metric | Real | Synthetic |
| --- | --- | --- |
| Mean Speed (m/s) | 17.00 | 17.34 |
| Max Speed (m/s) | 38.83 | 37.90 |
| Mean Duration (s) | 317.4 | 247.8 |

## Components
- **Model:** `data/diffusion_final.pt` (Standard Diffusion model trained for 5000 epochs).
- **Logic:** `diffusion_trajectory.py` includes the post-processing "Tail Stretching" and "Global Distance Constraint" logic.
- **Config:** Standard hyperparameters (1000 timesteps, CFG scale 1.0).

## How to Reproduce

### 1. Generate Synthetic Trajectories
Run the generation script using the v1.0 model:
```bash
python diffusion_trajectory.py --generate --model_path data/diffusion_final.pt --n_samples 500 --cfg_scale 1.0
```
*Note: This will save a timestamped CSV in `data/` and both a validation plot and a **10x5 grid plot** in `report/`.*

### 2. Re-plot Existing Data
If you already have a CSV and want to generate a new grid plot:
```bash
bash bin/replot.sh data/synthetic_trajectories_XXX.csv
```

## Remote Reproduction (Hopper Cluster)

I have also included a `reproduce.sh` script to automate the remote execution:
1. `cd v1.0`
2. `bash reproduce.sh` (Syncs files and submits SLURM job)
3. Wait for the job to complete.
4. Uncomment the `rsync` lines in `reproduce.sh` to pull results back, or use:
   ```bash
   rsync -avz hopper:~/projects/svtrip/diffusion/v1.0/report/ ./report/
   ```
- `bin/`: Reproduction shell scripts.
- `data/`: Contains `diffusion_final.pt` and `real_data_stats.pkl`.
- `report/`: Destination for generated plots and reports.
- `*.py`: Version-locked source code.
