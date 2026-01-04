# CSDI v1.3 - Physics-Informed Transformer Diffusion

Conditional Score Diffusion Imputation (CSDI) with transformer backbone and physics-informed training.

## Model Architecture

- **Type**: Transformer-based diffusion with self-attention
- **Physics Constraints**: Temporal smoothness, jerk penalties, energy consistency
- **Input**: Boundary conditions + cluster conditioning
- **Output**: Speed trajectory sequence
- **Performance**: WD Speed=0.30, WD Accel=0.026 (BEST)

## Files

- `csdi_trajectory.py` - Main training script with physics losses
- `evaluate_csdi.py` - Evaluation metrics
- `v1.3/` - Final model checkpoints and results
- `README.md` - Original detailed documentation
- `*.sbatch` - SLURM batch scripts for HPC training

## Requirements

```bash
pip install torch numpy pandas matplotlib scipy einops
```

## Usage

### Generate Synthetic Trajectories

```bash
python3 csdi_trajectory.py \
  --model_path v1.3/model/csdi_final.pt \
  --data_path ../artifacts/stats_df.csv \
  --cluster_path ../artifacts/trip_clusters.csv \
  --output synthetic_trajectories.csv \
  --n_samples 1000
```

### Evaluate Generated Data

```bash
python3 evaluate_csdi.py \
  --real ../artifacts/stats_df.csv \
  --synthetic synthetic_trajectories.csv
```

## Model Download

**Pre-trained CSDI v1.3 weights** (21MB):

```bash
curl -L -o v1.3/model/csdi_best.pt \
  "https://www.dropbox.com/scl/fi/igeuh5j6i2dhb4dc90kek/csdi_best.pt?rlkey=878ys1xw4u3mvb4nxbqmyjmwe&dl=1"
```

Or download manually: [csdi_best.pt](https://www.dropbox.com/scl/fi/igeuh5j6i2dhb4dc90kek/csdi_best.pt?rlkey=878ys1xw4u3mvb4nxbqmyjmwe&dl=1)

Place in `v1.3/model/csdi_best.pt`

## Key Results

- **Distributional Fidelity**: State-of-the-art (WD Speed 0.30 - 6× better than Markov, 2× better than Diffusion)
- **Acceleration Matching**: Exceptional (WD Accel 0.026)
- **Kinematic Validity**: 0% boundary violations
- **Smoothness**: Closest to real data (LDLJ -11.45 vs -11.08 real)
- **Discriminative Score**: 0.49 (indistinguishable from real)

## Physics-Informed Training

The model incorporates:
1. **Temporal smoothness penalty**: Reduces high-frequency artifacts
2. **Jerk minimization**: Ensures realistic acceleration changes
3. **Energy consistency**: VSP-based physical plausibility
4. **Hard boundary constraints**: Exact start/end speed matching via inpainting

See `csdi_trajectory.py` for loss function details.

## Citation

This is the primary model recommended in the paper due to superior distribution matching and physical realism.
