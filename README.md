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

**Download Full Dataset**:
- **Microtrips Data** (30MB): [Download from Dropbox](https://www.dropbox.com/scl/fi/56b1usdppj0dv8id3zqvw/Microtrips.zip?rlkey=kxs96s40mtepjtwr6fs46yk60&dl=1)
- Extract to `data/Microtrips/` for full trip CSV files

**Processed Data** (in `artifacts/`):
- 6,367 microtrips clustered into 4 driving regimes
- Cluster 0 (n=2,224): Arterial/Suburban
- Cluster 1 (n=1,020): Highway/Interstate  
- Cluster 2 (n=636): Congested/City
- Cluster 3 (n=2,487): Free-flow Arterial

## Model Weights

Pretrained model weights are available via Dropbox:

**Diffusion v1.5** (167MB):
```bash
curl -L -o diffusion/v1.5/model/diffusion_final.pt \
  "https://www.dropbox.com/scl/fi/cug0sm4t7ck347obt8x5r/diffusion_final.pt?rlkey=b6bo8otagxmlsybyabis2g7sd&dl=1"
```

**CSDI v1.3** (21MB):
```bash
curl -L -o csdi/v1.3/model/csdi_best.pt \
  "https://www.dropbox.com/scl/fi/igeuh5j6i2dhb4dc90kek/csdi_best.pt?rlkey=878ys1xw4u3mvb4nxbqmyjmwe&dl=1"
```

Or download manually from links above and place in respective model directories.

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
- **Farnaz Behnia** - Rocket Close
- **Dominik Karbowski** - Argonne National Laboratory

## Acknowledgments

Supported by U.S. Department of Energy's Vehicle Technologies Office.
