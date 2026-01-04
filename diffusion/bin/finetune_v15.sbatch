#!/bin/bash
#SBATCH --job-name=diff_ft15
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --output=report/ft15_%j.out
#SBATCH --error=report/ft15_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -e

# Environment Setup
if [ -f /etc/profile ]; then source /etc/profile; fi
for init_script in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /opt/ohpc/admin/lmod/lmod/init/bash; do
    if [ -f "$init_script" ]; then source "$init_script"; break; fi
done
module load gnu10/10.3.0-ya miniconda3/22.11.1-gy 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate base

echo "Starting v1.5 Fine-Tuning..."
echo "Base Model: data/diffusion_final_pid.pt"
echo "Physics: Accel=0.05, Jerk=0.02"

# Fine-tune for 200 epochs (should be enough to tighten accel)
python diffusion_trajectory.py --pid \
    --model_path data/diffusion_final_pid.pt \
    --epochs 200 \
    --lr 2e-5 \
    --save_dir data/v1.5_checkpoints \
    --accel_weight 0.05 \
    --jerk_weight 0.02

echo "Fine-tuning Complete!"
