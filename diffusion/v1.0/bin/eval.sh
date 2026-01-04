#!/bin/bash
#SBATCH --job-name=diff_eval
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --output=report/eval_%j.out
#SBATCH --error=report/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -e
# Default values
MODEL_PATH="data/diffusion_final.pt"
CFG_SCALE=1.0
SUFFIX=""
N_SAMPLES=500

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --cfg_scale) CFG_SCALE="$2"; shift ;;
        --suffix) SUFFIX="$2"; shift ;;
        --n_samples) N_SAMPLES="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Environment Setup
if [ -f /etc/profile ]; then source /etc/profile; fi
for init_script in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /opt/ohpc/admin/lmod/lmod/init/bash; do
    if [ -f "$init_script" ]; then source "$init_script"; break; fi
done
module load gnu10/10.3.0-ya miniconda3/22.11.1-gy 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate base

echo "Running Evaluation..."
echo "Model: $MODEL_PATH"
echo "CFG Scale: $CFG_SCALE"
echo "Suffix: $SUFFIX"

python diffusion_trajectory.py --generate --model_path "$MODEL_PATH" --cfg_scale "$CFG_SCALE" --n_samples "$N_SAMPLES"
python evaluate_distribution.py --suffix "$SUFFIX"

echo "Done!"
