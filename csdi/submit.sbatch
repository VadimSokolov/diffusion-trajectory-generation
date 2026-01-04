#!/bin/bash -l
#SBATCH --job-name=csdi_traj
#SBATCH --output=csdi-%j.out
#SBATCH --error=csdi-%j.err
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=6:00:00

echo "=============================================="
echo "CSDI Trajectory Generator"
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

# Initialize conda for this shell
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
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# --- Install dependencies if needed ---
pip install --user tqdm matplotlib pandas numpy --quiet

# --- Training ---
echo "Starting training..."
echo "Config: 6 layers, d_ff=1024, cosine schedule"
echo "Physics losses: accel_penalty=0.05, jerk_penalty=0.02, accel_dist=0.1"
python csdi_trajectory.py \
    --train \
    --data_path "${DATA_DIR}" \
    --epochs 200 \
    --batch_size 32 \
    --lr 1e-4

# --- Generation ---
echo ""
echo "Starting generation..."
python csdi_trajectory.py \
    --generate \
    --model_path csdi_best.pt \
    --data_path "${DATA_DIR}" \
    --n_samples 1000 \
    --smooth_kernel 7

# --- Evaluation ---
echo ""
echo "Starting evaluation..."
pip install --user scikit-learn --quiet 2>/dev/null || true
python evaluate_csdi.py \
    --real_data "${DATA_DIR}" \
    --output_dir "${CODE_DIR}"

echo ""
echo "=============================================="
echo "Completed: $(date)"
echo "=============================================="

