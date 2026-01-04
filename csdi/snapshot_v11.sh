#!/bin/bash
# Snapshot CSDI v1.1 - Creates a reproducible snapshot of the current working version
# This fetches all necessary files from Hopper and saves locally

set -e

REMOTE_DIR="/home/vsokolov/projects/svtrip/paper3/code/csdi"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT_DIR="${SCRIPT_DIR}/v1.1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "CSDI v1.1 Snapshot Creator"
echo "=============================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Destination: ${SNAPSHOT_DIR}"
echo ""

# Create snapshot directory structure
mkdir -p "${SNAPSHOT_DIR}/model"
mkdir -p "${SNAPSHOT_DIR}/results"
mkdir -p "${SNAPSHOT_DIR}/code"
mkdir -p "${SNAPSHOT_DIR}/logs"

# --- 1. Fetch model files from Hopper ---
echo "[1/4] Fetching model files from Hopper..."
rsync -avz --progress \
    --include="csdi_best.pt" \
    --include="csdi_final.pt" \
    --exclude="*" \
    "hopper:${REMOTE_DIR}/" \
    "${SNAPSHOT_DIR}/model/"

# --- 2. Fetch results (plots, CSVs, pickles) ---
echo ""
echo "[2/4] Fetching results from Hopper..."
rsync -avz --progress \
    --include="*.csv" \
    --include="*.pkl" \
    --include="*.png" \
    --include="*.json" \
    --exclude="*" \
    "hopper:${REMOTE_DIR}/" \
    "${SNAPSHOT_DIR}/results/"

# --- 3. Fetch logs ---
echo ""
echo "[3/4] Fetching logs from Hopper..."
rsync -avz --progress \
    --include="*.out" \
    --include="*.err" \
    --exclude="*" \
    "hopper:${REMOTE_DIR}/" \
    "${SNAPSHOT_DIR}/logs/"

# --- 4. Copy local code files ---
echo ""
echo "[4/4] Copying local code files..."
cp "${SCRIPT_DIR}/csdi_trajectory.py" "${SNAPSHOT_DIR}/code/"
cp "${SCRIPT_DIR}/submit.sbatch" "${SNAPSHOT_DIR}/code/"
cp "${SCRIPT_DIR}/README.md" "${SNAPSHOT_DIR}/code/"
cp "${SCRIPT_DIR}/requirements.txt" "${SNAPSHOT_DIR}/code/" 2>/dev/null || true

# --- Create version info file ---
cat > "${SNAPSHOT_DIR}/VERSION.txt" << EOF
CSDI v1.1 Snapshot
==================
Created: $(date)
Git commit: $(cd "${SCRIPT_DIR}" && git rev-parse HEAD 2>/dev/null || echo "N/A")

Key Settings:
- Diffusion steps: 200
- Smoothing kernel: 9
- Boundary ramp: 15 seconds
- Temporal smoothness loss: 0.1 weight (code ready, model not retrained with it)

Files Included:
- model/csdi_best.pt - Best trained model checkpoint
- model/csdi_final.pt - Final model after training
- results/*.csv - Generated trajectories
- results/*.png - Sample visualizations
- results/*.pkl - Trajectories as numpy arrays
- logs/*.out, *.err - Training/generation logs
- code/ - Source code snapshot

To Reproduce:
1. Copy model file: cp v1.1/model/csdi_best.pt .
2. Generate: python csdi_trajectory.py --generate --model_path csdi_best.pt --smooth_kernel 9 --diffusion_steps 200
EOF

# --- Create reproduction script ---
cat > "${SNAPSHOT_DIR}/reproduce.sh" << 'EOF'
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
EOF
chmod +x "${SNAPSHOT_DIR}/reproduce.sh"

# --- Summary ---
echo ""
echo "=============================================="
echo "Snapshot Complete!"
echo "=============================================="
echo ""
echo "Directory structure:"
find "${SNAPSHOT_DIR}" -type f | head -20
echo ""
echo "Total size: $(du -sh "${SNAPSHOT_DIR}" | cut -f1)"
echo ""
echo "To reproduce results later:"
echo "  cd ${SNAPSHOT_DIR} && ./reproduce.sh"
echo ""

