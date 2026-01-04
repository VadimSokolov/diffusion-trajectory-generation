#!/bin/bash
# Sync CSDI code to Hopper cluster
# Uses 'hopper' alias from ~/.ssh/config with 24-hour ControlMaster

set -e

REMOTE_DIR="/home/vsokolov/projects/svtrip"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Syncing CSDI code to Hopper..."

# Create remote directory (uses ControlMaster from ~/.ssh/config)
ssh hopper "mkdir -p ${REMOTE_DIR}/paper3/code/csdi"

# Sync code files
rsync -avz --progress \
    -e "ssh" \
    --exclude="*.pt" \
    --exclude="*.pkl" \
    --exclude="*.csv" \
    --exclude="*.png" \
    --exclude="*.json" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude="v1.1" \
    --exclude="results" \
    "${SCRIPT_DIR}/" \
    "hopper:${REMOTE_DIR}/paper3/code/csdi/"

echo ""
echo "Done! To run on Hopper:"
echo "  ssh hopper"
echo "  cd ${REMOTE_DIR}/paper3/code/csdi"
echo "  sbatch submit.sbatch"

