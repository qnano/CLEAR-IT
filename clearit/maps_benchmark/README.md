# CLEAR-IT MAPS Benchmark Runtime

This directory contains CLEAR-IT's public MAPS-compatible benchmark runtime for the low-label cell phenotyping experiments used in the paper's benchmarking figures.

The benchmark logic in this directory is based on the original MAPS project by Mahmood Lab:

- Upstream repository: https://github.com/mahmoodlab/MAPS
- Original paper: Shaban, M., Bai, Y., Qiu, H. et al. *MAPS: pathologist-level
  cell type annotation from tissue images through machine learning*. Nature
  Communications 15, 28 (2024).
  https://doi.org/10.1038/s41467-023-44188-w

## Why this code is vendored here

The original MAPS repository expects CSV-based inputs. For the public CLEAR-IT
reproduction pipeline, we adapted the relevant benchmark components so they can:

- read canonical CLEAR-IT embedding HDF5 files
- benchmark `expressions`, `features`, or their concatenation from one HDF5
- run from CLEAR-IT YAML recipes and `config.yaml`
- write directly into the published `outputs/maps_benchmark/...` directory tree

This avoids requiring users to install or patch a separate private MAPS checkout
just to reproduce the CLEAR-IT benchmark stage.

## Provenance

The files most directly adapted from MAPS are:

- `datasets.py`
- `networks.py`
- `trainer.py`

The file `pipeline.py` is CLEAR-IT-specific orchestration around that runtime.

## License and attribution

The vendored MAPS-derived files in this directory remain subject to the upstream
MAPS license terms. See `THIRD_PARTY_NOTICES.md` in this directory for the
required attribution and license notice.
