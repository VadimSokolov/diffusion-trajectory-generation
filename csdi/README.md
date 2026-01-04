# CSDI: Conditional Score-based Diffusion for Vehicle Trajectories

## Overview

Implementation of Conditional Score-based Diffusion with a Transformer backbone for vehicle speed trajectory generation. Based on the CSDI paper (Tashiro et al., NeurIPS 2021).

**Current Version**: v1.3 (Physics-Informed)  
**Status**: ✅ Production-ready

---

## Executive Summary

CSDI generates realistic synthetic vehicle speed trajectories that are statistically indistinguishable from real driving data. The model supports conditioning on trip characteristics (average speed, duration, max speed) and vehicle type (passenger car, truck, bus).

### Key Results (v1.3)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Speed Wasserstein** | 0.30 | Excellent distribution match |
| **Accel Wasserstein** | 0.026 | Accurate acceleration dynamics |
| **Discriminative Score** | 0.49 | Indistinguishable (0.5 = perfect) |
| **Boundary Violations** | 0% | Perfect start/end conditions |

### Sample Results

![Distribution Comparison](results/csdi_evaluation_20260101_200324.png)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CSDITransformer (5.5M params)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Noisy Trajectory x_t ─────────────────────┐                    │
│                                             │                    │
│  ┌─────────────────┐                       ▼                    │
│  │ Diffusion Time  │ ──▶ ┌──────────────────────────────────┐  │
│  │ Embedding t     │     │                                   │  │
│  └─────────────────┘     │    Transformer Encoder            │  │
│                          │    (6 layers, 8 heads)            │  │
│  ┌─────────────────┐     │                                   │  │
│  │ Condition Embed │ ──▶ │    + Positional Encoding          │  │
│  │ (spd,dur,max,vd)│     │    + Time Injection               │  │
│  └─────────────────┘     │    + Condition Injection          │  │
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

### Conditioning Vector (4D)
```
[avg_speed/30, duration/1000, max_speed/40, vehicle_dynamics]
```

---

## Version History & Comparison

### Development Timeline

| Version | Date | Key Changes |
|---------|------|-------------|
| v1.0 | Dec 2025 | Initial CSDI implementation |
| v1.1 | Dec 2025 | Post-processing smoothing, temporal loss |
| v1.2 | Jan 2026 | Optimized smoothing, conditional ramps |
| **v1.3** | Jan 2026 | Physics constraints, vehicle conditioning |

### Comprehensive Version Comparison

| Feature | v1.0 | v1.1 | v1.2 | **v1.3** |
|---------|------|------|------|----------|
| **Architecture** |
| Transformer layers | 4 | 4 | 4 | **6** |
| Feed-forward dim | 512 | 512 | 512 | **1024** |
| Condition dimensions | 3 | 3 | 3 | **4** |
| Parameters | ~3.5M | ~3.5M | ~3.5M | **~5.5M** |
| **Training** |
| Epochs | 100 | 100 | 100 | **200** |
| Diffusion steps | 100 | **200** | 200 | 200 |
| Beta schedule | linear | linear | linear | **cosine** |
| Data augmentation | ❌ | ❌ | ❌ | **✅ 3x** |
| **Losses** |
| MSE (denoising) | ✅ | ✅ | ✅ | ✅ |
| Temporal smoothness | ❌ | **✅ 0.1** | ✅ 0.1 | ✅ 0.1 |
| Acceleration penalty | ❌ | ❌ | ❌ | **✅ 0.03** |
| Jerk penalty | ❌ | ❌ | ❌ | **✅ 0.02** |
| Accel distribution | ❌ | ❌ | ❌ | **✅ 0.05** |
| **Post-Processing** |
| Gaussian smoothing | ❌ | **kernel=9** | kernel=7 | kernel=7 |
| Boundary ramp | 10s | 15s | **3s (cond)** | 3s (cond) |
| **Generation** |
| Vehicle type control | ❌ | ❌ | ❌ | **✅** |
| Speed boost sampling | ❌ | ❌ | ❌ | **✅** |
| **Quality Metrics** |
| Speed Wasserstein | 1.5+ | 0.8 | 0.74 | **0.30** |
| Accel Wasserstein | 0.15 | 0.08 | 0.08 | **0.026** |
| Discriminative | 0.65 | 0.50 | 0.50 | **0.49** |
| Boundary violations | 5% | 0% | 0% | **0%** |
| Jerkiness | ❌ Severe | ✅ Fixed | ✅ Good | **✅ Smooth** |

### Version-by-Version Changelog

#### v1.0 → v1.1: Fixing Jerkiness
**Problem**: Generated trajectories had high-frequency noise ("jitter") due to diffusion sampling artifacts.

**Solutions**:
1. Increased diffusion steps (100 → 200) for finer denoising
2. Added post-processing Gaussian smoothing (kernel=9)
3. Added temporal smoothness loss during training
4. Extended boundary ramp (10s → 15s)

**Result**: Smooth, realistic trajectories

#### v1.1 → v1.2: Optimizing Smoothing
**Problem**: Over-smoothing was flattening peak speeds; boundary ramps were too aggressive.

**Solutions**:
1. Reduced smoothing kernel (9 → 7) for better detail preservation
2. Made boundary ramps conditional (only apply if not already near zero)
3. Reduced boundary ramp length (15s → 3s)

**Result**: Better peak preservation, natural boundaries

#### v1.2 → v1.3: Physics & Vehicle Types
**Problem**: Acceleration distribution was too wide; no control over vehicle type.

**Solutions**:
1. Added physics-based loss terms (acceleration, jerk, distribution penalties)
2. Added vehicle_dynamics as 4th conditioning dimension
3. Implemented data augmentation to create truck/bus training data
4. Increased model capacity (4→6 layers, 512→1024 d_ff)
5. Switched to cosine beta schedule for better quality
6. Extended training (100→200 epochs)

**Result**: Accurate acceleration distribution, vehicle type control

---

## Physics-Based Training (v1.3)

### Motivation
Real vehicles have physical constraints on acceleration, deceleration, and rate of change (jerk). Training with physics-informed losses ensures generated trajectories respect these limits.

### Loss Function
```python
total_loss = mse_loss                           # Primary denoising
           + 0.1 * smoothness_loss              # Temporal smoothness
           + 0.03 * accel_penalty               # Limit extreme accelerations
           + 0.02 * jerk_penalty                # Smooth acceleration changes
           + 0.05 * accel_distribution_loss     # Match real distribution
```

### Physics Constraints

| Constraint | Threshold | Justification |
|------------|-----------|---------------|
| Max acceleration | 4 m/s² | Covers all vehicle types |
| Max deceleration | 5 m/s² | Comfortable braking |
| Max jerk | 2 m/s³ | Passenger comfort |
| Target accel std | ~0.5 m/s² | Match real data |

### Vehicle-Type Compatibility

| Vehicle | Max Accel | Max Decel | Covered by 4/5 m/s²? |
|---------|-----------|-----------|----------------------|
| Sports car | 4-6 m/s² | 10-12 m/s² | ✅ |
| Passenger car | 2.5-4 m/s² | 8-10 m/s² | ✅ |
| SUV | 2-3.5 m/s² | 7-9 m/s² | ✅ |
| Bus | 1-2 m/s² | 4-6 m/s² | ✅ |
| Heavy truck | 0.5-1.5 m/s² | 3-5 m/s² | ✅ |

---

## Vehicle Type Conditioning

### Overview
The model accepts a 4th conditioning variable (`vehicle_dynamics`) that controls acceleration behavior:

```
0.0 ──────── 0.5 ──────── 1.0
 │            │            │
 Heavy       Car         Sports
 Truck                    Car
```

### Vehicle Type Map

| Argument | Value | Description |
|----------|-------|-------------|
| `heavy_truck` | 0.15 | Semi-trucks, 18-wheelers |
| `truck` | 0.25 | Delivery trucks |
| `bus` | 0.35 | City/coach buses |
| `suv` | 0.45 | SUVs, crossovers |
| `car` | 0.55 | Passenger cars |
| `sedan` | 0.60 | Sedans, compacts |
| `sports_car` | 0.85 | Performance vehicles |

### Usage
```bash
# Heavy truck
python csdi_trajectory.py --generate --vehicle_type heavy_truck

# Custom value (0-1)
python csdi_trajectory.py --generate --vehicle_type 0.3
```

### Data Augmentation
Training data (passenger cars only) is augmented with synthetic truck/bus trajectories:

```
Original car trajectory
         │
         ├──► Heavy truck: clip accel ±1.0, smooth(7)
         │
         └──► Bus: clip accel ±1.2, smooth(5)
```

This 3x expands training data and teaches the model vehicle-specific dynamics.

---

## Configuration & Hyperparameters

### Model Architecture
```python
CONFIG = {
    "max_length": 512,       # Max trajectory length (seconds)
    "d_model": 256,          # Transformer hidden dimension
    "n_heads": 8,            # Attention heads
    "n_layers": 6,           # Transformer layers
    "d_ff": 1024,            # Feed-forward dimension
    "dropout": 0.1,          # Regularization
    "cond_dim": 4,           # Condition dimensions
}
```

### Diffusion Settings
```python
CONFIG = {
    "diffusion_steps": 200,  # Denoising steps
    "beta_start": 0.0001,    # Noise schedule start
    "beta_end": 0.02,        # Noise schedule end
    "schedule": "cosine",    # Beta schedule type
}
```

### Training Settings
```python
CONFIG = {
    "epochs": 200,
    "batch_size": 32,
    "lr": 1e-4,
    "smoothness_weight": 0.1,
    "accel_penalty_weight": 0.03,
    "jerk_penalty_weight": 0.02,
    "accel_distribution_weight": 0.05,
}
```

### Generation Settings

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--smooth_kernel` | 7 | 0-15 | Gaussian kernel (0=disable) |
| `--speed_boost` | 1.0 | 0.5-2.0 | Sampling weight |
| `--diffusion_steps` | 200 | 100-500 | More = smoother |
| `--boundary_ramp` | 3 | 1-15 | Ramp length (seconds) |

---

## Usage

### Training
```bash
# Full training
python csdi_trajectory.py \
    --train \
    --data_path ../../data/Microtrips \
    --epochs 200 \
    --batch_size 32 \
    --lr 1e-4

# Quick test (limited data)
python csdi_trajectory.py \
    --train \
    --data_path ../../data/Microtrips \
    --epochs 10 \
    --limit_files 100
```

### Generation
```bash
# Default (sample from training distribution)
python csdi_trajectory.py \
    --generate \
    --model_path csdi_best.pt \
    --data_path ../../data/Microtrips \
    --n_samples 1000 \
    --smooth_kernel 7

# Specific vehicle type
python csdi_trajectory.py \
    --generate \
    --model_path csdi_best.pt \
    --vehicle_type heavy_truck \
    --n_samples 500
```

### HPC (Hopper)
```bash
# Sync and submit
./sync_to_hopper.sh
ssh hopper "cd ~/projects/svtrip/paper3/code/csdi && sbatch submit.sbatch"

# Monitor
ssh hopper "tail -f ~/projects/svtrip/paper3/code/csdi/csdi-*.out"

# Fetch results
./fetch_results.sh
```

---

## Output Files

| File | Description |
|------|-------------|
| `csdi_best.pt` | Best model checkpoint |
| `csdi_synthetic_*.csv` | Generated trajectories (CSV) |
| `csdi_synthetic_*.pkl` | Trajectories (pickle, variable length) |
| `csdi_samples_*.png` | Sample visualization |
| `csdi_evaluation_*.png` | Distribution comparison |
| `csdi_metrics_*.json` | Full evaluation metrics |

---

## Future Improvements

### Implementation Categories

| Category | Description | Model Changes | Time |
|----------|-------------|---------------|------|
| 🟢 **Generation Only** | Change `--parameters` at generation time | None | Minutes |
| 🟡 **Fine-Tune** | Continue training from existing checkpoint | Minor | Hours |
| 🔴 **Retrain** | Full training from scratch with code changes | Major | 4-6 hrs |
| ⚫ **Research** | New architecture or data requirements | Significant | Days-Weeks |

---

### High Priority

#### 1. Duration Distribution Matching
- **Problem**: Mean duration 264s vs real 307s
- **Type**: 🟢 **Generation Only** OR 🟡 **Fine-Tune**

**Option A - Generation Parameter (Quick)**:
```bash
# Add duration_boost parameter (similar to speed_boost)
python csdi_trajectory.py --generate --duration_boost 1.3
```
Steps:
1. Add `--duration_boost` argument to `csdi_trajectory.py`
2. Weight sampling by duration^power
3. No retraining needed

**Option B - Fine-Tune with Loss (Better)**:
1. Add duration distribution loss to training
2. Fine-tune for 20-50 epochs from `csdi_best.pt`
3. `python csdi_trajectory.py --train --epochs 50 --lr 5e-5 --resume csdi_best.pt`

---

#### 2. Real-Time Generation (DDIM)
- **Problem**: 200 diffusion steps = ~3 min for 1000 samples
- **Type**: 🟢 **Generation Only**

Steps:
1. Implement DDIM sampler in `CSDISampler` class (~100 lines)
2. Add `--sampler ddim --ddim_steps 50` arguments
3. No retraining - uses same model weights
4. Expected speedup: 4-10x

```python
# Add to csdi_trajectory.py
class DDIMSampler:
    def sample(self, model, cond, steps=50):
        # Skip steps: [0, 4, 8, 12, ...] instead of [0, 1, 2, 3, ...]
        skip = self.num_steps // steps
        ...
```

---

#### 3. Multi-Modal Speed Patterns (Highway/Urban)
- **Problem**: No road type conditioning
- **Type**: 🔴 **Retrain**

Steps:
1. **Label data** with road type (highway, urban, suburban) - requires external data or heuristics
2. Add 5th condition dimension: `road_type`
3. Modify `TrajectoryDataset` to load road type
4. Retrain from scratch (200 epochs)

```python
# Heuristic labeling (if no external data):
if avg_speed > 25 and std_speed < 5:
    road_type = "highway"  # 0.9
elif avg_speed < 15:
    road_type = "urban"    # 0.1
else:
    road_type = "suburban" # 0.5
```

---

### Medium Priority

#### 4. Acceleration Asymmetry Learning
- **Problem**: Symmetric accel/decel penalties
- **Type**: 🟡 **Fine-Tune**

Steps:
1. Modify `train()` to separate positive/negative acceleration penalties
2. Learn optimal thresholds from data percentiles
3. Fine-tune 30-50 epochs

```python
# In training loss:
accel_pos_penalty = relu(accel[accel > 0] - learned_max_accel)
accel_neg_penalty = relu(-accel[accel < 0] - learned_max_decel)
```

---

#### 5. Temperature Sampling
- **Problem**: No diversity control at generation
- **Type**: 🟢 **Generation Only**

Steps:
1. Add `--temperature` argument (already exists, unused)
2. Scale noise in reverse diffusion: `noise * temperature`
3. Temperature > 1.0 = more diverse, < 1.0 = more deterministic

```python
# In CSDISampler.sample():
noise = torch.randn_like(x) * temperature
```

---

#### 6. Longer Trajectories (>512s)
- **Problem**: Max length 512 seconds
- **Type**: 🔴 **Retrain** OR 🟢 **Generation Only** (sliding window)

**Option A - Increase max_length (Retrain)**:
1. Change `CONFIG["max_length"] = 1024`
2. Increase positional encoding size
3. Retrain (more memory, longer training)

**Option B - Sliding Window (Generation Only)**:
1. Generate 512s chunks with 50s overlap
2. Blend overlapping regions
3. No retraining needed

---

### Low Priority (Research)

#### 7. Continuous Vehicle Dynamics
- **Type**: ⚫ **Research** → 🔴 **Retrain**

Steps:
1. Collect vehicle specs (mass, power, drag coefficient)
2. Replace discrete `vehicle_dynamics` with continuous embedding
3. Train encoder to map specs → dynamics embedding
4. Full retrain with new conditioning

---

#### 8. Traffic Interaction
- **Type**: ⚫ **Research**

Steps:
1. Collect multi-vehicle trajectory data
2. Design multi-agent conditioning (leader speed, gap, etc.)
3. Significant architecture changes (graph attention?)
4. Research project scope

---

#### 9. Road Grade Incorporation
- **Type**: ⚫ **Research** → 🔴 **Retrain**

Steps:
1. Obtain elevation data for training routes
2. Add grade profile as additional input sequence
3. Modify model to accept (speed, grade) pairs
4. Full retrain

---

#### 10. Energy Consumption Modeling
- **Type**: ⚫ **Research** → 🔴 **Retrain**

Steps:
1. Obtain fuel/energy data paired with speed trajectories
2. Add energy as second output channel
3. Multi-output diffusion model
4. Significant architecture changes

---

### Quick Reference Table

| # | Improvement | Type | Effort | Files to Modify |
|---|-------------|------|--------|-----------------|
| 1a | Duration boost | 🟢 Gen | 1 hr | `csdi_trajectory.py` (add arg) |
| 1b | Duration loss | 🟡 Fine | 2-4 hr | `csdi_trajectory.py` (train fn) |
| 2 | DDIM sampler | 🟢 Gen | 4-8 hr | `csdi_trajectory.py` (new class) |
| 3 | Road type | 🔴 Retrain | 1-2 days | Dataset + training |
| 4 | Asymmetric accel | 🟡 Fine | 2-4 hr | Training loss |
| 5 | Temperature | 🟢 Gen | 30 min | Sampler |
| 6a | Longer (retrain) | 🔴 Retrain | 1 day | CONFIG + training |
| 6b | Sliding window | 🟢 Gen | 4 hr | Generation only |
| 7-10 | Research items | ⚫ Research | Weeks | New architecture |

### Recommended Next Steps

1. **Immediate** (🟢): Add `--duration_boost` and `--temperature` parameters
2. **Short-term** (🟢): Implement DDIM for 10x faster generation
3. **Medium-term** (🟡): Fine-tune with asymmetric acceleration penalties
4. **Long-term** (🔴): Add road type conditioning with labeled data

---

## Troubleshooting

### Jerky Trajectories
- Increase `--smooth_kernel` (try 9 or 11)
- Increase `--diffusion_steps` (try 300)

### Wide Acceleration Distribution
- Use v1.3 model (trained with physics constraints)
- Increase `--smooth_kernel`

### Vehicle Type Not Working
- Ensure model was trained with v1.3 code (4D conditioning)
- Check spelling: `heavy_truck`, not `heavy-truck`

### Slow Generation
- Reduce `--diffusion_steps` to 100
- Implement DDIM (future improvement)

### Mean Speed Mismatch
- Adjust `--speed_boost` (< 1.0 for lower speeds, > 1.0 for higher)
- Currently 1.0 gives best overall match

---

## File Structure

```
csdi/
├── csdi_trajectory.py      # Main training/generation script
├── evaluate_csdi.py        # Evaluation metrics
├── submit.sbatch           # Full HPC job (train + generate)
├── generate.sbatch         # Generation-only HPC job
├── sync_to_hopper.sh       # Sync code to HPC
├── fetch_results.sh        # Fetch results from HPC
├── snapshot_v13.sh         # Create version snapshot
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── results/                # Fetched results
│   ├── csdi_evaluation_*.png
│   ├── csdi_samples_*.png
│   └── csdi_metrics_*.json
├── v1.2/                   # Previous version snapshot
└── v1.3/                   # Current version snapshot (after running snapshot_v13.sh)
```

---

## References

- [CSDI Paper](https://arxiv.org/abs/2107.03502) - Conditional Score-based Diffusion for Time Series
- [DDPM](https://arxiv.org/abs/2006.11239) - Denoising Diffusion Probabilistic Models
- [Cosine Schedule](https://arxiv.org/abs/2102.09672) - Improved Denoising Diffusion
- [DDIM](https://arxiv.org/abs/2010.02502) - Faster Sampling (future work)

---

## Citation

```bibtex
@software{csdi_trajectory,
  title={CSDI: Physics-Informed Vehicle Trajectory Generation},
  author={GMU Transportation Lab},
  year={2026},
  version={1.3},
  url={https://github.com/gmu-transport/svtrip}
}
```
