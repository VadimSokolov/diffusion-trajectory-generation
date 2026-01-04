#!/bin/bash

# Configuration
REMOTE_HOST="hopper" 
REMOTE_DIR="/home/vsokolov/projects/svtrip/diffusion"

echo "Fetching Diffusion results from $REMOTE_HOST:$REMOTE_DIR ..."

# Fetch Report, Plots, and Logs
rsync -avz "$REMOTE_HOST:$REMOTE_DIR/report/*.md" ./report/
rsync -avz "$REMOTE_HOST:$REMOTE_DIR/report/*.png" ./report/
rsync -avz "$REMOTE_HOST:$REMOTE_DIR/report/*.out" ./report/
rsync -avz "$REMOTE_HOST:$REMOTE_DIR/report/*.err" ./report/
rsync -avz "$REMOTE_HOST:$REMOTE_DIR/*.out" ./
rsync -avz "$REMOTE_HOST:$REMOTE_DIR/*.err" ./

# Optional: Fetch synthetic CSV (might be large)
# rsync -avz "$REMOTE_HOST:$REMOTE_DIR/data/synthetic_trajectories_*.csv" ./data/

echo "Done. Check report/ directory for results."
