# Diffusion Models for Vehicle Speed Trajectory Generation

This repository contains the reproducibility package for the IEEE ITS paper:

**"Diffusion Models for Conditional Vehicle Speed Trajectory Generation: A Comparative Study"**

## Repository Structure

- **`diffusion/`** - U-Net diffusion model (v1.5) implementation
- **`csdi/`** - CSDI transformer diffusion model (v1.3) implementation  
- **`generate_figures.py`** - Script to generate all paper figures
- **`artifacts/`** - Preprocessed CMAP 2007 travel survey data
- **`LICENSE`** - MIT License

Each model subfolder contains complete training code, evaluation scripts, and documentation.

## Quick Start

### 1. Generate Paper Figures

```bash
python3 generate_figures.py --artifacts-path artifacts --output-dir fig
```

This creates all 13 figures from the paper, including the comprehensive main results figure comparing CSDI and Diffusion across Highway, Arterial, and Congested driving regimes.

### 2. Model Training & Generation

See subfolder READMEs for detailed instructions:
- **Diffusion v1.5**: `diffusion/README.md`
- **CSDI v1.3**: `csdi/README.md`

## Dataset

**Source**: Chicago Metropolitan Agency for Planning (CMAP) 2007-2008 Regional Household Travel Survey

**Processed Data** (in `artifacts/`):
- 6,367 microtrips clustered into 4 driving regimes
- Cluster 0 (n=2,224): Arterial/Suburban
- Cluster 1 (n=1,020): Highway/Interstate  
- Cluster 2 (n=636): Congested/City
- Cluster 3 (n=2,487): Free-flow Arterial

## Model Weights

Pretrained model weights are available via Dropbox (too large for GitHub):

- **Diffusion v1.5** (167MB): [Dropbox Link - TBD by authors]
- **CSDI v1.3** (21MB): [Dropbox Link - TBD by authors]

Place downloaded weights in respective `diffusion/v1.5/model/` and `csdi/v1.3/model/` directories.

## Citation

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

MIT License - See LICENSE file

## Contact

- **Vadim Sokolov** - George Mason University
- **Farnaz Behnia** - Argonne National Laboratory
- **Dominik Karbowski** - Argonne National Laboratory

## Acknowledgments

Supported by U.S. Department of Energy's Vehicle Technologies Office.
