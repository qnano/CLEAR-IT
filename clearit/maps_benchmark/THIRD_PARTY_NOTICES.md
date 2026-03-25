# Third-Party Notices for `clearit.maps_benchmark`

This directory contains code derived in part from the MAPS project.

## Upstream project

- Project: MAPS: Machine learning for Analysis of Proteomics in Spatial biology
- Repository: https://github.com/mahmoodlab/MAPS
- Paper: Shaban, M., Bai, Y., Qiu, H. et al. *MAPS: pathologist-level cell type
  annotation from tissue images through machine learning*. Nature
  Communications 15, 28 (2024).
  https://doi.org/10.1038/s41467-023-44188-w

## Vendored/adapted files

The following CLEAR-IT files are adapted from MAPS code:

- `datasets.py`
- `networks.py`
- `trainer.py`

The file `pipeline.py` is CLEAR-IT-specific orchestration built around those
adapted components.

## Upstream license notice

The local MAPS repository bundled during development includes the following
license notice:

> “Commons Clause” License Condition v1.0
>
> The Software is provided to you by the Licensor under the License, as defined
> below, subject to the following condition.
>
> Without limiting other conditions in the License, the grant of rights under
> the License will not include, and the License does not grant to you, the right
> to Sell the Software.
>
> For purposes of the foregoing, “Sell” means practicing any or all of the
> rights granted to you under the License to provide to third parties, for a fee
> or other consideration (including without limitation fees for hosting or
> consulting/support services related to the Software), a product or service
> whose value derives, entirely or substantially, from the functionality of the
> Software. Any license notice or attribution required by the License must also
> include this Commons Clause License Condition notice.
>
> Software: MAPS
>
> License: Apache 2.0 with Commons Clause
>
> Licensor: Mahmood Lab

## CLEAR-IT-specific changes

For the public CLEAR-IT reproduction pipeline, the MAPS-derived components were
adapted to:

- read canonical CLEAR-IT embedding HDF5 files instead of MAPS CSV inputs
- benchmark `expressions`, `features`, or their concatenation from one HDF5
- integrate with CLEAR-IT YAML recipes and `config.yaml`
- write directly into the published CLEAR-IT benchmark output layout
