# v1.1 (PID Experimental - Failed)

**Status:** Unsuccessful / Regression
**Date:** Jan 3, 2026

This directory contains the snapshot of the code used for the first Physics-Informed Diffusion (PID) experiment (`diffusion_final_pid.pt`, Ep 4700).

## Issue
The model failed to converge to a realistic distribution.
- **Boundary Violations:** 100% (Model ignores v=0 constraint)
- **High Speed Artifacts:** Generated speeds up to 60+ m/s.
- **Discriminative Score:** ~0.48 (vs 0.10 for v1.0).

## Hypothesis
The loss weights for the physics terms were set too high, overpowering the diffusion reconstruction loss.

- **Bad Weights:**
    - `L_boundary`: 0.1 (Training conflict)
    - `L_dist`: 0.1 (Training conflict)
    - `L_asym`: 0.5 (Too aggressive)
    - `L_jerk`: 0.01

**Next Step:** v1.2 will adopt lower weights inspired by CSDI (0.03 for acceleration, 0.0 for boundary/distance).
