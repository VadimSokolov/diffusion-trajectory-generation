# Diffusion v1.5 - U-Net Based Trajectory Generation

Standard U-Net diffusion model for conditional vehicle speed trajectory generation.

## Model Architecture

- **Type**: U-Net with time embeddings and conditional inputs
- **Input**: Boundary conditions (start/end speeds, duration, distance)
- **Output**: Speed trajectory sequence
- **Performance**: WD Speed=0.56, WD Accel=0.080, SAFD WD=0.0005

## Files

- `diffusion_trajectory.py` - Main training script
- `evaluate_distribution.py` - Evaluation metrics
- `v1.5/` - Final model checkpoints and results
- `README.md` - Original documentation

## Requirements

```bash
pip install torch numpy pandas matplotlib scipy
```

## Usage

### Generate Synthetic Trajectories

```bash
python3 diffusion_trajectory.py \
  --model_path v1.5/model/diffusion_final.pt \
  --data_path ../artifacts/stats_df.csv \
  --output synthetic_trajectories.csv \
  --n_samples 1000
```

### Evaluate Generated Data

```bash
python3 evaluate_distribution.py \
  --real ../artifacts/stats_df.csv \
  --synthetic synthetic_trajectories.csv
```

## Model Download

Pre-trained weights (167MB): [Dropbox Link - TBD]

Place in `v1.5/model/diffusion_final.pt`

## Key Results

- **Distributional Fidelity**: Good matching (WD Speed 0.56)
- **2D Correlation**: Excellent SAFD matching (0.0005) 
- **Kinematic Validity**: 0% boundary violations
- **Smoothness**: Slight over-smoothing (LDLJ -13.29)

See paper Table 3 for full comparison with other models.
