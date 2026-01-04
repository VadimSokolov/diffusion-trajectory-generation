#!/bin/bash
#SBATCH --job-name=rebuild_cache
#SBATCH --output=report/rebuild_%j.out
#SBATCH --error=report/rebuild_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --qos=standard
#SBATCH --partition=standard

set -e
source /opt/sw/spack/apps/linux-rhel8-x86_64_v2/gcc-10.3.0/miniconda3-22.11.1-gy/etc/profile.d/conda.sh
conda activate /home/vsokolov/.local

echo "Rebuilding dataset cache with full metadata..."
rm -f data/dataset_cache.pt data/real_data_stats.pkl
python -c "
import sys
sys.path.insert(0, '.')
from diffusion_trajectory import TrajectoryDataset, CONFIG
dataset = TrajectoryDataset('/projects/vsokolov/svtrip/data/Microtrips', save_stats=True, cache_file='data/dataset_cache.pt')
print(f'Rebuilt cache with {len(dataset)} trips')
"

echo "Cache rebuild complete!"
