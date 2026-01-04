#!/bin/bash
# Reproduce v1.0 results on Hopper Cluster
# Run this from the v1.0 directory

# Configuration
REMOTE_HOST="hopper"
REMOTE_DIR="~/projects/svtrip/diffusion/v1.0"

echo "--- v1.0 Reproduction Pipeline ---"

# Step 1: Sync v1.0 snapshot to Hopper
echo "1. Syncing v1.0 to Hopper..."
ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR"
rsync -avz --exclude 'report/*' ./ $REMOTE_HOST:$REMOTE_DIR

# Step 2: Submit SLURM job
echo "2. Submitting SLURM job..."
# Runs from within the v1.0 remote directory to use v1.0 code/model
ssh $REMOTE_HOST "cd $REMOTE_DIR && sbatch bin/eval.sh --model_path data/diffusion_final.pt --suffix _v1.0_repro"

# Step 3: Wait and monitor
echo "3. Monitor job progress with:"
echo "ssh $REMOTE_HOST 'squeue -u \$USER'"

# Step 4: Fetch results back (manually once job is complete)
# echo "4. Fetch results back:"
# rsync -avz $REMOTE_HOST:$REMOTE_DIR/report/*.png ./report/
# rsync -avz $REMOTE_HOST:$REMOTE_DIR/report/*.md ./report/
# rsync -avz $REMOTE_HOST:$REMOTE_DIR/report/*.out ./report/
# rsync -avz $REMOTE_HOST:$REMOTE_DIR/data/synthetic_trajectories_*.csv ./data/
