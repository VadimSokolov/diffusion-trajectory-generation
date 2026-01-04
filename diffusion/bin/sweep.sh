#!/bin/bash
#SBATCH --job-name=cfg_sweep
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --output=report/sweep_%j.out
#SBATCH --error=report/sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -e
# Environment Setup
if [ -f /etc/profile ]; then source /etc/profile; fi
for init_script in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /opt/ohpc/admin/lmod/lmod/init/bash; do
    if [ -f "$init_script" ]; then source "$init_script"; break; fi
done
module load gnu10/10.3.0-ya miniconda3/22.11.1-gy 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate base

MODEL_PATH="data/diffusion_final.pt"

for CFG in 1.0 3.0 5.0; do
    echo "--- Running CFG $CFG ---"
    python diffusion_trajectory.py --generate --model_path "$MODEL_PATH" --cfg_scale "$CFG" --n_samples 500
    python evaluate_distribution.py --suffix "_cfg$CFG"
done

echo "Sweep Complete"
