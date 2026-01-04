#!/bin/bash -l
#SBATCH --job-name=csdi_gen
#SBATCH --output=csdi-gen-%j.out
#SBATCH --error=csdi-gen-%j.err
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=0:30:00

echo "=============================================="
echo "CSDI Trajectory Generator - Generation Only"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo ""

# --- Module Setup ---
source /etc/profile
for lmod_init in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash /opt/ohpc/admin/lmod/lmod/init/bash; do
    [ -f "$lmod_init" ] && source "$lmod_init" && break
done

module load gnu10/10.3.0-ya
module load miniconda3/22.11.1-gy

# Initialize conda for script usage
eval "$(conda shell.bash hook)"
conda activate base

# --- Environment ---
PROJECT_DIR="/home/vsokolov/projects/svtrip"
CODE_DIR="${PROJECT_DIR}/paper3/code/csdi"
DATA_DIR="${PROJECT_DIR}/data/Microtrips"

export SVTRIP_DATA_DIR="${PROJECT_DIR}/data"
export PYTHONUNBUFFERED=1

cd "${CODE_DIR}"

echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Data path: ${DATA_DIR}"
echo "Data exists: $(python -c \"import os; print(os.path.exists('${DATA_DIR}'))\")"
echo ""

# --- Generation ---
echo "Starting generation..."
python csdi_trajectory.py \
    --generate \
    --model_path csdi_best.pt \
    --data_path "${DATA_DIR}" \
    --n_samples 1000 \
    --smooth_kernel 7 \
    --speed_boost 1.0 \
    --correlated_noise 0.0

# --- Evaluation ---
echo ""
echo "Starting evaluation..."

# Install sklearn if needed
pip install --user scikit-learn --quiet 2>/dev/null || true

# Find the most recent synthetic CSV
LATEST_SYNTH_CSV=$(ls -t "${CODE_DIR}"/csdi_synthetic_*.csv 2>/dev/null | head -n 1)

if [ -n "$LATEST_SYNTH_CSV" ]; then
    echo "Evaluating: $LATEST_SYNTH_CSV"
    
    python evaluate_csdi.py \
        --real_data "${DATA_DIR}" \
        --synth_csv "${LATEST_SYNTH_CSV}" \
        --output_dir "${CODE_DIR}"
else
    echo "WARNING: No synthetic CSV found for evaluation"
fi

echo ""
echo "=============================================="
echo "Completed: $(date)"
echo "=============================================="

