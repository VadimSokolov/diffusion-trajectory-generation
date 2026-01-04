#!/bin/bash

# Configuration
# Replace 'hopper' with your actual SSH host alias or user@hostname (e.g., vsokolov@hopper.orc.gmu.edu)
REMOTE_HOST="hopper" 
REMOTE_TARGET_DIR="/home/vsokolov/projects/svtrip/diffusion"

echo "Syncing code to $REMOTE_HOST:$REMOTE_TARGET_DIR ..."

# Create remote directory first to be safe
ssh $REMOTE_HOST "mkdir -p $REMOTE_TARGET_DIR"

# Rsync
# -a: archive mode (preserves permissions, times, etc.)
# -v: verbose
# -z: compress
# --exclude: skip venv, caches, and local output files/checkpoints
rsync -avz \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '.DS_Store' \
    --exclude 'data/*.pt' \
    --exclude 'data/*.csv' \
    --exclude 'report/*.png' \
    --exclude '*.out' \
    --exclude '*.err' \
    ./ \
    "$REMOTE_HOST:$REMOTE_TARGET_DIR"

echo "Done."
