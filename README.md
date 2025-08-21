# CLEAR-IT

This is the repository for [CLEAR-IT: Contrastive Learning to Capture the Immune Composition of Tumor Microenvironments](https://doi.org/10.1101/2024.08.20.608738).

For pre-trained models, embeddings, and model predictions, see our [data repository](https://data.4tu.nl/my/datasets/ebc792ad-4767-4aef-b8ff-ae653e901e3f/10.4121/126d8103-6de5-4493-a48e-5d529fef471e) (not published yet as of August 21st 2025)

## Repository structure and `config.yaml`

This repository's structure is as follows:
```
.
├── clearit                # CLEAR-IT Python library
├── clearit.Dockerfile     # Dockerfile for running CLEAR-IT in a Docker container
├── config_template.yaml   # Template config file. Modify and rename this to config.yaml
├── experiments            # YAML recipe files for training all models and performing linear evaluation
├── notebooks              # Jupyter Notebooks for plotting
├── requirements.txt       # requirements.txt for custom environments
├── scripts                # Scripts for converting external datasets used in the study to a unified format
└── setup.py               # setup.py for a local install of clearit
```
We recommend placing the contents of the [data repository](https://data.4tu.nl/my/datasets/ebc792ad-4767-4aef-b8ff-ae653e901e3f/10.4121/126d8103-6de5-4493-a48e-5d529fef471e) in this directory (or somewhere else on fast storage), extending the structure as follows:
```
├── datasets               # Location of the converted datasets, ready to be used by CLEAR-IT
├── embeddings             # Pre-computed embeddings for benchmarking purposes
├── models                 # Pre-trained CLEAR-IT encoders and linear classifiers
├── outputs                # Predictions made via linear evaluation or benchmarking, survival classifiers
├── raw_datasets           # Location of the unconverted datasets - the conversion scripts in the scripts directory will move these to the datasets directory
```
The `config_template.yaml` file contains a template for a `config.yaml` file, which scripts and notebooks will look for:
```yaml
# config_template.yaml
# Create a copy of this file and name it `config.yaml` to point to custom paths
paths:
  # Absolute or relative path to the unpacked CLEAR-IT-Data directory
  data_root: /path/to/data/repository/CLEAR-IT                     # Corresponds to the GitHub repository's root directory
  datasets_dir: /path/to/data/repository/CLEAR-IT/datasets         # The datasets directory from the data repository
  raw_datasets_dir: /path/to/data/repository/CLEAR-IT/raw_datasets # The raw_datasets directory from the data repository
  models_dir: /path/to/data/repository/CLEAR-IT/models             # The models directory from the data repository
  outputs_dir: /path/to/data/repository/CLEAR-IT/outputs           # The outputs directory from the data repository
  experiments_dir: /path/to/data/repository/CLEAR-IT/experiments   # The experiments directory from the GitHub repository
```
By modifying the `config.yaml`, you are free to choose where you place individual directories (if space is a concern). If you want to train models, we recommend putting the `datasets` directory on fast storage (for example an SSD).
