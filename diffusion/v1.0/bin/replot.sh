#!/bin/bash
# Re-generate grid plots from synthetic trajectory CSV
# Usage: ./bin/replot.sh data/synthetic_trajectories_XXX.csv

CSV_PATH=$1

if [ -z "$CSV_PATH" ]; then
    echo "Usage: ./bin/replot.sh <path_to_csv>"
    exit 1
fi

python diffusion_trajectory.py --plot_csv "$CSV_PATH"
