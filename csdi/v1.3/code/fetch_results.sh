#!/bin/bash
# Fetch CSDI results from Hopper (plots and metrics only, no models)
# Uses 'hopper' alias from ~/.ssh/config with ControlMaster

set -e

REMOTE_DIR="/home/vsokolov/projects/svtrip/paper3/code/csdi"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Create results directory
mkdir -p "${RESULTS_DIR}"

echo "=============================================="
echo "Fetching CSDI results from Hopper"
echo "=============================================="
echo "Destination: ${RESULTS_DIR}"
echo ""

rsync -avz --progress \
    --include="*.csv" \
    --include="*.png" \
    --include="*.json" \
    --include="*.out" \
    --include="*.err" \
    --include="evaluation_report.md" \
    --exclude="*" \
    "hopper:${REMOTE_DIR}/" \
    "${RESULTS_DIR}/"

echo ""
echo "=============================================="
echo "Done! Results in: ${RESULTS_DIR}"
echo "=============================================="
ls -la "${RESULTS_DIR}"/ 2>/dev/null || echo "(no results yet)"
