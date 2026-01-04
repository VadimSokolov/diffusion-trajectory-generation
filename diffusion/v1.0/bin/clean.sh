#!/bin/bash

echo "Cleaning up artifacts..."

# Remove logs
rm -f *.err *.out

# Remove checkpoints and final model
rm -f data/diffusion_ckpt*.pt data/diffusion_final*.pt

# Remove plots and CSVs
rm -f report/*.png data/*.csv report/*.md

# Clear cache (necessary when switching from subset to full data)
rm -f data/dataset_cache.pt data/real_data_stats.pkl

echo "Clean complete. (Cache cleared for full data reload)"
