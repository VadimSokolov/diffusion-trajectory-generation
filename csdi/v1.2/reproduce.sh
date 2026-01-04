#!/bin/bash
# Reproduce v1.2 results using the saved model
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "Reproducing CSDI v1.2 results..."
echo ""
echo "Settings:"
echo "  - Smoothing kernel: 7"
echo "  - Boundary ramp: 3 (conditional)"
echo "  - Diffusion steps: 200"
echo ""

# Copy model to parent directory
cp "${SCRIPT_DIR}/model/csdi_best.pt" "${PARENT_DIR}/"

# Run generation with v1.2 settings
cd "${PARENT_DIR}"
python csdi_trajectory.py \
    --generate \
    --model_path csdi_best.pt \
    --n_samples 1000 \
    --smooth_kernel 7 \
    --diffusion_steps 200 \
    --data_path ../../data/Microtrips

echo ""
echo "Done! Check for new csdi_samples_*.png and csdi_synthetic_*.csv files."
