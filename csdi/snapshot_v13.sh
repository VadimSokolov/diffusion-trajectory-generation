#!/bin/bash
# =============================================================================
# CSDI v1.3 Snapshot Script
# Creates a complete snapshot of the production-ready CSDI model
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="v1.3"
SNAPSHOT_DIR="${SCRIPT_DIR}/${VERSION}"

# Hopper connection (uses SSH ControlMaster from ~/.ssh/config)
HOPPER_USER="vsokolov"
REMOTE_DIR="/home/${HOPPER_USER}/projects/svtrip/paper3/code/csdi"

echo "=============================================="
echo "Creating CSDI ${VERSION} Snapshot"
echo "=============================================="
echo "Destination: ${SNAPSHOT_DIR}"
echo ""

# Create snapshot directory structure
mkdir -p "${SNAPSHOT_DIR}/code"
mkdir -p "${SNAPSHOT_DIR}/results"
mkdir -p "${SNAPSHOT_DIR}/model"

# -----------------------------------------------------------------------------
# 1. Copy local code files
# -----------------------------------------------------------------------------
echo "Copying code files..."
cp "${SCRIPT_DIR}/csdi_trajectory.py" "${SNAPSHOT_DIR}/code/"
cp "${SCRIPT_DIR}/evaluate_csdi.py" "${SNAPSHOT_DIR}/code/"
cp "${SCRIPT_DIR}/submit.sbatch" "${SNAPSHOT_DIR}/code/"
cp "${SCRIPT_DIR}/generate.sbatch" "${SNAPSHOT_DIR}/code/"
cp "${SCRIPT_DIR}/requirements.txt" "${SNAPSHOT_DIR}/code/" 2>/dev/null || echo "  (no requirements.txt)"
cp "${SCRIPT_DIR}/sync_to_hopper.sh" "${SNAPSHOT_DIR}/code/"
cp "${SCRIPT_DIR}/fetch_results.sh" "${SNAPSHOT_DIR}/code/"

# -----------------------------------------------------------------------------
# 2. Fetch model and results from Hopper
# -----------------------------------------------------------------------------
echo ""
echo "Fetching model checkpoint from Hopper..."
rsync -avz --progress \
    "${HOPPER_USER}@hopper:${REMOTE_DIR}/csdi_best.pt" \
    "${SNAPSHOT_DIR}/model/" 2>/dev/null || echo "  (model not found on Hopper)"

echo ""
echo "Fetching latest results from Hopper..."
# Get the most recent evaluation files
ssh hopper "ls -t ${REMOTE_DIR}/csdi_evaluation_*.png 2>/dev/null | head -1" | while read f; do
    rsync -avz "${HOPPER_USER}@hopper:$f" "${SNAPSHOT_DIR}/results/"
done

ssh hopper "ls -t ${REMOTE_DIR}/csdi_metrics_*.json 2>/dev/null | head -1" | while read f; do
    rsync -avz "${HOPPER_USER}@hopper:$f" "${SNAPSHOT_DIR}/results/"
done

ssh hopper "ls -t ${REMOTE_DIR}/csdi_samples_*.png 2>/dev/null | head -1" | while read f; do
    rsync -avz "${HOPPER_USER}@hopper:$f" "${SNAPSHOT_DIR}/results/"
done

ssh hopper "ls -t ${REMOTE_DIR}/csdi_report_*.csv 2>/dev/null | head -1" | while read f; do
    rsync -avz "${HOPPER_USER}@hopper:$f" "${SNAPSHOT_DIR}/results/"
done

# Also copy local results if they exist
cp "${SCRIPT_DIR}/results/csdi_evaluation_20260101_200324.png" "${SNAPSHOT_DIR}/results/" 2>/dev/null || true
cp "${SCRIPT_DIR}/results/csdi_metrics_20260101_200324.json" "${SNAPSHOT_DIR}/results/" 2>/dev/null || true
cp "${SCRIPT_DIR}/results/csdi_samples_20260101_200318.png" "${SNAPSHOT_DIR}/results/" 2>/dev/null || true

# -----------------------------------------------------------------------------
# 3. Create VERSION.txt with full configuration
# -----------------------------------------------------------------------------
echo ""
echo "Creating VERSION.txt..."
cat > "${SNAPSHOT_DIR}/VERSION.txt" << 'EOF'
================================================================================
CSDI v1.3 - Physics-Informed Trajectory Generation
================================================================================

Release Date: January 2026
Status: Production-Ready

================================================================================
MODEL ARCHITECTURE
================================================================================

Component               Value           Description
------------------------------------------------------------------------
d_model                 256             Transformer hidden dimension
n_heads                 8               Multi-head attention heads
n_layers                6               Transformer encoder layers
d_ff                    1024            Feed-forward hidden dimension
dropout                 0.1             Regularization
max_length              512             Maximum trajectory length (seconds)
cond_dim                4               Condition vector size

Conditioning Vector: [avg_speed/30, duration/1000, max_speed/40, vehicle_dynamics]

Total Parameters: ~5.5M

================================================================================
DIFFUSION SETTINGS
================================================================================

Parameter               Value           Description
------------------------------------------------------------------------
diffusion_steps         200             Reverse diffusion iterations
beta_start              0.0001          Noise schedule start
beta_end                0.02            Noise schedule end
schedule                cosine          Beta schedule type

================================================================================
TRAINING SETTINGS
================================================================================

Parameter               Value           Description
------------------------------------------------------------------------
epochs                  200             Training epochs
batch_size              32              Samples per batch
learning_rate           1e-4            Initial learning rate
lr_schedule             cosine          Learning rate decay
weight_decay            0               No L2 regularization

Data Augmentation:
- Original passenger car data: ~6,000 trajectories
- Augmented heavy truck data: ~6,000 trajectories
- Augmented bus data: ~6,000 trajectories
- Total training data: ~18,000 trajectories

================================================================================
PHYSICS-BASED LOSS WEIGHTS (v1.3 New)
================================================================================

Loss                    Weight          Threshold       Purpose
------------------------------------------------------------------------
MSE (noise)             1.0             -               Primary denoising loss
Temporal smoothness     0.1             -               Reduce high-freq patterns
Acceleration penalty    0.03            4.0 m/s²        Limit max acceleration
Deceleration penalty    0.03            5.0 m/s²        Limit max braking
Jerk penalty            0.02            2.0 m/s³        Smooth accel changes
Accel distribution      0.05            data-driven     Match real accel dist

================================================================================
GENERATION SETTINGS (Recommended)
================================================================================

Parameter               Value           Description
------------------------------------------------------------------------
smooth_kernel           7               Gaussian smoothing kernel size
boundary_ramp           3               Boundary ramp length (seconds)
speed_boost             1.0             Condition sampling weight (uniform)
temperature             1.0             Sampling temperature (not used)

================================================================================
VEHICLE TYPE CONDITIONING
================================================================================

Vehicle Type            Dynamics Value  Typical Accel   Typical Decel
------------------------------------------------------------------------
heavy_truck             0.15            0.5-1.0 m/s²    2-3 m/s²
truck                   0.25            1.0-1.5 m/s²    2-4 m/s²
bus                     0.35            1.0-1.5 m/s²    3-4 m/s²
suv                     0.45            2.0-3.0 m/s²    5-7 m/s²
car / passenger_car     0.55            2.5-4.0 m/s²    7-9 m/s²
sedan                   0.60            3.0-4.0 m/s²    8-10 m/s²
sports_car              0.85            4.0-6.0 m/s²    10-12 m/s²

================================================================================
FINAL METRICS (v1.3 with speed_boost=1.0)
================================================================================

Metric                          Real            CSDI v1.3       Status
------------------------------------------------------------------------
Mean Speed (m/s)                16.96           17.84           ✅ Close
Std Speed (m/s)                 5.92            5.92            ✅ Exact
Max Speed (m/s)                 39.56           39.22           ✅ Close
Mean Duration (s)               307.3           263.8           ⚠️ Shorter

Speed Wasserstein               -               0.2957          ✅ Excellent
Accel Wasserstein               -               0.0261          ✅ Excellent
SAFD Wasserstein                -               0.0001          ✅ Perfect
MMD                             -               0.0101          ✅ Low
Discriminative Score            -               0.4913          ✅ Near 0.5
Boundary Violation Rate         -               0.0000          ✅ Perfect
TSTR MAE                        -               0.3880          ✅ Low

================================================================================
FILES IN THIS SNAPSHOT
================================================================================

v1.3/
├── VERSION.txt              # This file
├── README.md                # Comprehensive usage guide
├── code/
│   ├── csdi_trajectory.py   # Main training/generation script
│   ├── evaluate_csdi.py     # Evaluation metrics script
│   ├── submit.sbatch        # Full training + generation job
│   ├── generate.sbatch      # Generation-only job
│   ├── sync_to_hopper.sh    # Sync to HPC cluster
│   └── fetch_results.sh     # Fetch results from HPC
├── model/
│   └── csdi_best.pt         # Trained model checkpoint (~22MB)
└── results/
    ├── csdi_evaluation_*.png    # Distribution comparison plots
    ├── csdi_samples_*.png       # Sample trajectory plots
    ├── csdi_metrics_*.json      # Full metrics JSON
    └── csdi_report_*.csv        # Summary report

================================================================================
EOF

# -----------------------------------------------------------------------------
# 4. Create comprehensive README for the snapshot
# -----------------------------------------------------------------------------
echo "Creating README.md..."
cat > "${SNAPSHOT_DIR}/README.md" << 'EOF'
# CSDI v1.3: Physics-Informed Vehicle Trajectory Generation

## Overview

This is the production-ready release of CSDI (Conditional Score-based Diffusion with Transformers) for synthetic vehicle speed trajectory generation. Version 1.3 introduces physics-based training constraints and vehicle type conditioning.

**Key Achievements:**
- ✅ **Speed Wasserstein: 0.30** (excellent distribution match)
- ✅ **Accel Wasserstein: 0.026** (accurate acceleration dynamics)
- ✅ **Discriminative Score: 0.49** (indistinguishable from real)
- ✅ **Zero boundary violations** (perfect start/end conditions)
- ✅ **Vehicle type conditioning** (cars, trucks, buses)

---

## Quick Start

### Generate Trajectories (Recommended Settings)

```bash
python code/csdi_trajectory.py \
    --generate \
    --model_path model/csdi_best.pt \
    --data_path /path/to/Microtrips \
    --n_samples 1000 \
    --smooth_kernel 7 \
    --speed_boost 1.0
```

### Generate for Specific Vehicle Types

```bash
# Heavy truck (slow acceleration, smooth)
python code/csdi_trajectory.py --generate --model_path model/csdi_best.pt \
    --vehicle_type heavy_truck

# Bus (moderate dynamics)
python code/csdi_trajectory.py --generate --model_path model/csdi_best.pt \
    --vehicle_type bus

# Sports car (aggressive acceleration)
python code/csdi_trajectory.py --generate --model_path model/csdi_best.pt \
    --vehicle_type sports_car
```

---

## Available Parameters (Knobs)

### Generation Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--n_samples` | 1000 | 1-10000 | Number of trajectories to generate |
| `--smooth_kernel` | 7 | 0-15 | Gaussian smoothing (0=disable, higher=smoother) |
| `--speed_boost` | 1.0 | 0.5-2.0 | Condition sampling weight (1.0=uniform) |
| `--diffusion_steps` | 200 | 100-500 | Reverse diffusion steps (more=smoother) |
| `--vehicle_type` | None | see below | Vehicle type for generation |

### Vehicle Type Options

| Option | Dynamics | Use Case |
|--------|----------|----------|
| `heavy_truck` | 0.15 | Semi-trucks, 18-wheelers |
| `truck` | 0.25 | Delivery trucks, box trucks |
| `bus` | 0.35 | City buses, coach buses |
| `suv` | 0.45 | SUVs, crossovers |
| `car` | 0.55 | Average passenger cars |
| `sedan` | 0.60 | Sedans, compact cars |
| `sports_car` | 0.85 | Performance vehicles |
| `0.0-1.0` | custom | Direct numeric input |

### Fine-Tuning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 200 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--limit_files` | None | Limit training data (for testing) |

---

## Fine-Tuning the Model

If you have new trajectory data or want to adapt the model:

```bash
# Fine-tune on new data (start from pretrained)
python code/csdi_trajectory.py \
    --train \
    --data_path /path/to/new_data \
    --epochs 50 \
    --lr 5e-5 \
    --batch_size 32

# Full retraining from scratch
python code/csdi_trajectory.py \
    --train \
    --data_path /path/to/data \
    --epochs 200 \
    --lr 1e-4
```

### Fine-Tuning Tips

1. **Lower learning rate** (5e-5) when fine-tuning from checkpoint
2. **Fewer epochs** (50-100) for domain adaptation
3. **Include diverse vehicle types** in training data for better generalization
4. **Monitor acceleration distribution** to ensure physics constraints are satisfied

---

## Results

### Distribution Comparison

![Distribution Comparison](results/csdi_evaluation_20260101_200324.png)

### Sample Trajectories

![Sample Trajectories](results/csdi_samples_20260101_200318.png)

### Metrics Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Speed Wasserstein** | 0.2957 | Low = good distribution match |
| **Accel Wasserstein** | 0.0261 | Low = accurate dynamics |
| **SAFD Wasserstein** | 0.0001 | Low = joint distribution match |
| **MMD** | 0.0101 | Low = similar feature space |
| **Discriminative Score** | 0.4913 | 0.5 = indistinguishable |
| **Boundary Violations** | 0.0% | Perfect boundary conditions |
| **TSTR MAE** | 0.388 | Low = good for prediction tasks |

### Real vs Synthetic Comparison

| Statistic | Real Data | CSDI v1.3 | Match |
|-----------|-----------|-----------|-------|
| Mean Speed | 16.96 m/s | 17.84 m/s | ✅ |
| Std Speed | 5.92 m/s | 5.92 m/s | ✅ Exact |
| Max Speed | 39.56 m/s | 39.22 m/s | ✅ |
| Mean Duration | 307.3 s | 263.8 s | ⚠️ |

---

## Model Architecture

```
CSDITransformer (5.5M parameters)
├── Input Embedding: Linear(1 → 256)
├── Positional Encoding: Sinusoidal (max 512)
├── Time Embedding: MLP(1 → 256 → 256)
├── Condition Embedding: Linear(4 → 256)
├── Transformer Encoder: 6 layers
│   ├── Multi-Head Attention (8 heads)
│   ├── Feed-Forward (256 → 1024 → 256)
│   └── LayerNorm + Dropout(0.1)
└── Output: Linear(256 → 1)
```

### Conditioning Vector

```
[avg_speed/30, duration/1000, max_speed/40, vehicle_dynamics]
     ↓              ↓              ↓              ↓
  0-1 range     0-0.5 range     0-1 range      0-1 range
```

---

## Files

```
v1.3/
├── VERSION.txt          # Full configuration details
├── README.md            # This file
├── code/
│   ├── csdi_trajectory.py   # Main script (1100 lines)
│   ├── evaluate_csdi.py     # Evaluation metrics
│   ├── submit.sbatch        # HPC training job
│   ├── generate.sbatch      # HPC generation job
│   └── *.sh                 # Utility scripts
├── model/
│   └── csdi_best.pt         # Trained weights (~22MB)
└── results/
    └── *.png, *.json        # Evaluation outputs
```

---

## Citation

If you use this model, please cite:

```bibtex
@software{csdi_trajectory_v13,
  title={CSDI v1.3: Physics-Informed Vehicle Trajectory Generation},
  author={GMU Transportation Lab},
  year={2026},
  version={1.3}
}
```

---

## Changelog

- **v1.3**: Physics-based losses, vehicle type conditioning, data augmentation
- **v1.2**: Optimized smoothing (kernel=7), conditional boundary ramps
- **v1.1**: Post-processing smoothing, temporal smoothness loss
- **v1.0**: Initial CSDI implementation

EOF

echo ""
echo "=============================================="
echo "Snapshot Complete!"
echo "=============================================="
echo ""
echo "Created: ${SNAPSHOT_DIR}"
echo ""
echo "Contents:"
ls -la "${SNAPSHOT_DIR}"
echo ""
echo "To use:"
echo "  cd ${SNAPSHOT_DIR}"
echo "  python code/csdi_trajectory.py --generate --model_path model/csdi_best.pt"
echo ""

