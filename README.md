# Reproduction Package for "Diffusion Models for Conditional Vehicle Speed Trajectory Generation"

This repository contains all scripts, data, and documentation needed to reproduce the figures and tables in the IEEE ITS journal paper.

## Contents

- `generate_figures.py` - Main script to generate all figures
- `artifacts/` - Preprocessed data files
  - `cluster_to_trip_ids.json` - Cluster assignments for 6,367 microtrips
  - `microtrips-summary.md` - Dataset statistics
- `fig/` - Output directory for generated figures
- `README.md` - This file

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

**Python Version**: 3.8+

## Quick Start

Generate all figures for the paper:

```bash
python3 generate_figures.py --artifacts-path artifacts --output-dir fig
```

This will create:
- `fig1_data_distributions.pdf` - Trip duration, distance, speed distributions
- `fig2and3_cluster_projections.pdf` - PCA and t-SNE side-by-side
- `fig4-7_cluster*_trajectories.pdf` - Representative trajectories with trip IDs
- `fig_main_results.pdf` - **Main results comparison (CSDI vs Diffusion)**
- `fig11_model_comparison.pdf` - 6-model comparison grid
- `fig12_distribution_comparisons.pdf` - Distribution overlays
- `fig13_tsne_real_synthetic.pdf` - Real vs synthetic visualization

## Dataset

**Source**: Chicago Metropolitan Agency for Planning (CMAP) 2007-2008 Regional Household Travel Survey

**Processing**: 
- Extracted 6,367 microtrips (individual vehicle trips)
- Clustered into 4 driving regimes using K-Means (k=4)
- Features: avg speed, max speed, std speed, idle time ratio, stops/km, accel noise

**Clusters**:
1. **Cluster 0** (n=2,224): Arterial/Suburban - moderate speeds,periodic patterns
2. **Cluster 1** (n=1,020): Highway/Interstate - high sustained speeds
3. **Cluster 2** (n=636): Congested/City - low speeds, frequent stops
4. **Cluster 3** (n=2,487): Free-flow Arterial - smooth moderate speeds

## Figure Descriptions

### Figure 1: Data Distributions
Histograms showing trip duration (34-12,841s), distance (0.26-393km), and average speed (5.18-31.57 m/s).

### Figures 2-3: Cluster Projections (Side-by-Side)
- **(a) PCA**: First 2 components explain ~65% variance
- **(b) t-SNE**: Nonlinear projection with perplexity=30

### Figures 4-7: Representative Trajectories
Two example speed profiles per cluster, annotated with actual CMAP trip IDs for traceability.

### Main Results Figure (CSDI vs Diffusion)
**NEW: Side-by-side comparison of the two best-performing models**

**Left Column - CSDI v1.3:**
- (a) Speed/Acceleration distributions (WD Speed=0.30, WD Accel=0.026)
- (c) Sample highway trajectory
- (e) SAFD heatmap (WD=0.0008)

**Right Column - Diffusion v1.5:**
- (b) Speed/Acceleration distributions (WD Speed=0.56, WD Accel=0.080)  
- (d) Sample highway trajectory
- (f) SAFD heatmap (WD=0.0005)

**Key Insight**: Visual proof that CSDI achieves 2× better distribution matching than standard U-Net diffusion (0.30 vs 0.56), and 6× better than Markov chains (0.30 vs 1.82).

### Figure 11: Model Comparison Grid
Sample trajectories from 6 models: Real, Diffusion v1.5, CSDI v1.3, Markov, DoppelGANger, SDV

### Figure 12: Distribution Comparisons
Overlaid histograms for speed and acceleration distributions across models.

### Figure 13: t-SNE Real vs Synthetic
Scatter plot showing CSDI and Diffusion samples cluster near real data (discriminative score ≈ 0.49).

## Tables

All tables are embedded in `paper.qmd`. Key results summary:

**Table 3: Model Performance**

| Model | WD Speed | WD Accel | Discriminative | Boundary Violations |
|:------|:---------|:---------|:---------------|:--------------------|
| CSDI v1.3 | **0.30** | **0.026** | **0.49** | **0%** |
| Diffusion v1.5 | 0.56 | 0.080 | 0.38 | 0% |
| Markov | 1.82 | 0.145 | 0.31 | 0% |
| Chronos | 2.15 | 0.198 | 0.24 | 18.3% |
| DoppelGANger | 3.42 | 0.312 | 0.08 | 12.1% |
| SDV | 2.87 | 0.421 | 0.15 | 23.4% |

## Citation

If you use this code or data, please cite:

```bibtex
@article{sokolov2026diffusion,
  title={Diffusion Models for Conditional Vehicle Speed Trajectory Generation: A Comparative Study},
  author={Sokolov, Vadim and Behnia, Farnaz and Karbowski, Dominik},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026},
  note={Under review}
}
```

## License

MIT License - See LICENSE file for details

## Contact

- Vadim Sokolov - George Mason University
- Farnaz Behnia - Argonne National Laboratory  
- Dominik Karbowski - Argonne National Laboratory

## Acknowledgments

This work was supported by the U.S. Department of Energy's Vehicle Technologies Office. Data from CMAP 2007-2008 Regional Household Travel Survey.
