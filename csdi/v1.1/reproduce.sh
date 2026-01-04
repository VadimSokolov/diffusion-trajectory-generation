#!/bin/bash
# Reproduce v1.1 results using the saved model
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "Reproducing CSDI v1.1 results..."
echo ""

# Copy model to parent directory
cp "${SCRIPT_DIR}/model/csdi_best.pt" "${PARENT_DIR}/"

# Run generation with v1.1 settings
cd "${PARENT_DIR}"
python csdi_trajectory.py \
    --generate \
    --model_path csdi_best.pt \
    --n_samples 1000 \
    --smooth_kernel 9 \
    --diffusion_steps 200

echo ""
echo "Done! Check for new csdi_samples_*.png and csdi_synthetic_*.csv files."
