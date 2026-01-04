# Project Organization & Logistics

This document covers the operational aspects of the Diffusion Trajectory project, including environment setup, cluster usage, and standard workflows.

---

## 1. Environment Setup

### Local Machine (macOS)
```bash
conda activate base  # or your preferred env with PyTorch
pip install torch numpy pandas matplotlib scipy tqdm
```

### HPC Cluster (Hopper)
```bash
# SSH to cluster
ssh hopper

# Navigate to project
cd projects/svtrip/diffusion

# Activate environment
source /opt/sw/spack/apps/linux-rhel8-x86_64_v2/gcc-10.3.0/miniconda3-22.11.1-gy/etc/profile.d/conda.sh
conda activate base
```

### Data Location
- **Local:** `/Users/vsokolov/Dropbox/prj/svtrip/paper3/artifacts/` (or `data/` in project root)
- **Hopper:** `/projects/vsokolov/svtrip/data/Microtrips/` (6,367 CSV files)

---

## 2. Project Structure

```
code/diffusion/
├── bin/                       # Simplified SLURM and utility scripts
│   ├── train.sh               # Unified Training (Standard/PID)
│   ├── eval.sh                # Unified Evaluation (flexible)
│   ├── sweep.sh               # Run CFG scale sweep
│   ├── cache.sh               # Rebuild dataset cache
│   ├── clean.sh               # Cleanup artifacts
│   ├── sync.sh                # Sync code to Hopper
│   └── fetch.sh               # Fetch results from Hopper
├── data/                      # Model checkpoints and generated CSVs
├── report/                    # Evaluation plots and reports
├── v1.0/                      # Stable verified snapshot
├── diffusion_trajectory.py    # Main pipeline logic
├── evaluate_distribution.py   # Metrics and Plotting
├── projectorg.md              # Logistics & Ops (This file)
└── model.md                   # Modeling & Results
```

---

## 3. SLURM Usage Policy (Hopper)

> **⚠️ CRITICAL: Never run compute-intensive tasks on login nodes!**

All training, generation, and evaluation must be submitted as jobs.

### Batch Jobs (Preferred)
```bash
sbatch bin/train.sh        # Standard Training
sbatch bin/train.sh --mode pid  # PID Training
sbatch bin/eval.sh         # Generation + Evaluation
```

### Interactive Session (Debugging)
```bash
# Request GPU session for 2 hours
salloc --partition=gpuq --gres=gpu:3g.40gb:1 --time=02:00:00 --mem=16G
```

---

## 4. Operational Workflows

### A. Syncing Code
Keep local and remote environments aligned.
```bash
# Push local code to Hopper
bash bin/sync.sh
```

### B. Fetching Results
Retrieve plots, reports, and CSVs from Hopper for analysis.
```bash
# Pull results to local machine
bash bin/fetch.sh
```

### C. Troubleshooting
- **"FileNotFoundError: diffusion_final.pt"**: Ensure you have synced the model or trained it. Check `data/`.
- **High Loss / Exploding Gradients**: Verify loss weights in `train_pid`. Use normalized reconstruction loss.
- **Job Pending (PD)**: Resources exhausted. Try a different partition in the SLURM script or wait.
