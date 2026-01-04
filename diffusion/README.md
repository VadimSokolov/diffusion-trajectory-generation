# Physics-Informed Diffusion for Vehicle Speed Trajectory Synthesis

This project implements a **conditional 1D Diffusion Model** to generate realistic vehicle speed trajectories from the CMAP 2007 Household Travel Survey.

---

## 📚 Documentation

The documentation has been split into two detailed guides:

### 1. [Modeing & Results (model.md)](model.md)
**Read this for:**
*   **Model Architecture:** 1D U-Net, FiLM Conditioning.
*   **Training & Physics:** Loss functions (Standard vs PID).
*   **Generation:** Inpainting logic and post-processing.
*   **Results:** Verified metrics and side-by-side comparison of **v1.0 (Baseline)** vs **v1.1 (PID)**.

### 2. [Project Logistics (projectorg.md)](projectorg.md)
**Read this for:**
*   **Environment Setup:** Installation (Local/HPC).
*   **Cluster Usage:** SLURM policy and commands.
*   **Operations:** Syncing code, fetching results, and troubleshooting.

---

## 🚀 Quick Start

### Generate trajectories (v1.0 Best Model)
```bash
python diffusion_trajectory.py --generate --model_path data/diffusion_final.pt --n_samples 500
```

### Run Evaluation
```bash
python evaluate_distribution.py
```
