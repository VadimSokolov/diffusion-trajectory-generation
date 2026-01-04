#!/bin/bash -l
#SBATCH --job-name=csdi_eval
#SBATCH --output=csdi_eval-%j.out
#SBATCH --error=csdi_eval-%j.err
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00

echo "=============================================="
echo "CSDI Evaluation"
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

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate base

# --- Environment ---
PROJECT_DIR="/home/vsokolov/projects/svtrip"
CODE_DIR="${PROJECT_DIR}/paper3/code/csdi"
DATA_DIR="${PROJECT_DIR}/data/Microtrips"

export PYTHONUNBUFFERED=1

cd "${CODE_DIR}"

echo "Python: $(which python)"
echo ""

# --- Install dependencies ---
pip install --user scikit-learn fastdtw --quiet 2>/dev/null || true

# --- Run Evaluation ---
python evaluate_csdi.py \
    --real_data "${DATA_DIR}" \
    --output_dir "${CODE_DIR}"

echo ""
echo "=============================================="
echo "Completed: $(date)"
echo "=============================================="

