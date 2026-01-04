#!/bin/bash
# Snapshot CSDI v1.2 - Creates a reproducible snapshot of kernel=7 results
# Best configuration before attempting correlated noise approach

set -e

REMOTE_DIR="/home/vsokolov/projects/svtrip/paper3/code/csdi"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT_DIR="${SCRIPT_DIR}/v1.2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "CSDI v1.2 Snapshot Creator"
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

# --- 2. Fetch results (plots, CSVs, metrics) ---
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
cp "${SCRIPT_DIR}/generate.sbatch" "${SNAPSHOT_DIR}/code/"
cp "${SCRIPT_DIR}/evaluate_csdi.py" "${SNAPSHOT_DIR}/code/" 2>/dev/null || true
cp "${SCRIPT_DIR}/README.md" "${SNAPSHOT_DIR}/code/"
cp "${SCRIPT_DIR}/requirements.txt" "${SNAPSHOT_DIR}/code/" 2>/dev/null || true

# --- Create version info file ---
cat > "${SNAPSHOT_DIR}/VERSION.txt" << EOF
CSDI v1.2 Snapshot - Best Configuration
========================================
Created: $(date)
Git commit: $(cd "${SCRIPT_DIR}" && git rev-parse HEAD 2>/dev/null || echo "N/A")

Model Architecture:
- Type: Conditional Score-based Diffusion with Transformer backbone
- d_model: 256 (embedding dimension)
- n_heads: 8 (attention heads)
- n_layers: 6 (transformer layers)
- d_ff: 1024 (feedforward dimension)
- dropout: 0.1
- max_length: 512 (max trajectory length in seconds)
- Conditioning: avg_speed, duration, max_speed (3 features)

Diffusion Parameters:
- diffusion_steps: 200
- beta_start: 0.0001
- beta_end: 0.02
- schedule: cosine

Training Settings:
- epochs: 200
- batch_size: 32
- learning_rate: 1e-4
- smoothness_weight: 0.1 (temporal smoothness loss)
- speed_limit: 40.0 m/s (normalization ceiling)

Generation Settings (OPTIMAL):
- Smoothing kernel: 7 (sweet spot - balances smoothness vs distribution matching)
- Boundary ramp: 3 seconds (conditional - only applies if not already near zero)
- Condition sampling: from real data distribution

Results Summary:
- Speed Wasserstein: 0.736 (excellent)
- Accel Wasserstein: 0.080 (good)
- Discriminative Score: 0.499 (near-perfect, ~0.5 = indistinguishable)
- Boundary Violations: 0%
- LDLJ (jerk smoothness): -1.69 (synthetic smoother than real -0.47)

Known Limitations:
- Acceleration distribution has slightly lighter tails than real data
- LDLJ indicates synthetic trajectories are smoother than real

Future Improvements to Try:
1. Correlated noise post-processing (add medium-freq variation)
2. Retraining with acceleration distribution matching loss

Files Included:
- model/csdi_best.pt - Best trained model checkpoint
- results/*.csv - Generated trajectories
- results/*.png - Evaluation plots and samples
- results/*.json - Metrics
- logs/*.out, *.err - Training/generation logs
- code/ - Source code snapshot

To Reproduce:
1. Copy model file: cp v1.2/model/csdi_best.pt .
2. Generate: python csdi_trajectory.py --generate --model_path csdi_best.pt --smooth_kernel 7 --data_path ../../data/Microtrips
EOF

# --- Create reproduction script ---
cat > "${SNAPSHOT_DIR}/reproduce.sh" << 'EOF'
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
EOF
chmod +x "${SNAPSHOT_DIR}/reproduce.sh"

# --- Summary ---
echo ""
echo "=============================================="
echo "Snapshot Complete!"
echo "=============================================="
echo ""
echo "Directory structure:"
find "${SNAPSHOT_DIR}" -type f | head -25
echo ""
echo "Total size: $(du -sh "${SNAPSHOT_DIR}" | cut -f1)"
echo ""
echo "To reproduce results later:"
echo "  cd ${SNAPSHOT_DIR} && ./reproduce.sh"
echo ""

