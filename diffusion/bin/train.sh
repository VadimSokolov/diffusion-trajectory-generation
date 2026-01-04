#!/bin/bash
#SBATCH --job-name=diff_train
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --output=report/train_%j.out
#SBATCH --error=report/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e
# Default values
MODE="standard"
EPOCHS=5000
DATA_PATH="/projects/vsokolov/svtrip/data/Microtrips"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --data_path) DATA_PATH="$2"; shift ;;
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
pip install --user torch pandas numpy matplotlib scipy tqdm 2>/dev/null

echo "Starting Training Job: $MODE mode, $EPOCHS epochs"
echo "Data Path: $DATA_PATH"

if [ "$MODE" == "pid" ]; then
    python diffusion_trajectory.py --pid --epochs "$EPOCHS" --save_dir data --data_path "$DATA_PATH"
else
    python diffusion_trajectory.py --train --epochs "$EPOCHS" --save_dir data --data_path "$DATA_PATH"
fi

echo "Training complete. Generating baseline samples..."
MODEL_PATH="data/diffusion_final.pt"
if [ "$MODE" == "pid" ]; then MODEL_PATH="data/diffusion_final_pid.pt"; fi

python diffusion_trajectory.py --generate --model_path "$MODEL_PATH" --n_samples 500
python evaluate_distribution.py --suffix "_${MODE}_final"

echo "Job Complete"
