# CLEAR-IT

This is the repository for [CLEAR-IT: Contrastive Learning to Capture the Immune Composition of Tumor Microenvironments](https://doi.org/10.1101/2024.08.20.608738).

For pre-trained models, embeddings, and model predictions, see our [data repository](https://data.4tu.nl/my/datasets/ebc792ad-4767-4aef-b8ff-ae653e901e3f/10.4121/126d8103-6de5-4493-a48e-5d529fef471e) (not published yet)

## Installation

### Option A — Local install (recommended for development)

1. Clone the repository and (optionally) create a fresh Python environment.

   ```bash
   git clone https://github.com/qnano/CLEAR-IT.git
   cd CLEAR-IT
   # (optional) create & activate a virtual environment
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. Install dependencies and the package.

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

### Option B — Docker

A ready-to-build Dockerfile is provided as `clearit.Dockerfile`.

Build the image:

```bash
docker build -f clearit.Dockerfile -t clearit:latest .
```

Run a container (mount your repo and data; add `--gpus all` if you have NVIDIA GPUs configured):

```bash
docker run -it --rm \
  -v $(pwd):/workspace/CLEAR-IT \
  -v /path/to/data:/data \
  clearit:latest bash
```

Inside the container, install in editable mode if needed:

```bash
cd /workspace/CLEAR-IT && pip install -e .
```

> **Tip:** Whether you install locally or via Docker, make sure to create a `config.yaml` (see below). The scripts look for this file to resolve all paths.

## Usage

The CLEAR-IT library exposes three driver scripts to (1) pre-train encoders, (2) train classification heads, and (3) perform linear evaluation. Each script is pointed to a YAML **recipe** describing one or more experiments. Recipe files live under the `experiments/` folder in this repository.

### 1) Configure paths

Copy the template and edit paths to where your data lives (see the [Repository structure and `config.yaml`](#repository-structure-and-configyaml) section for details on the directory layout):

```bash
cp config_template.yaml config.yaml
# then open config.yaml and update the paths under `paths:`
```

The scripts and notebooks will look for a `config.yaml` in your working directory (typically the repo root). The locations set here determine where datasets are read from and where models, embeddings, and outputs are written.

### 2) Pick a recipe and run

Each recipe can contain multiple encoders/heads to be trained or parameters to use for linear evaluation. Example commands:

```bash
python -m clearit.scripts.run_pretrain --recipe ./experiments/01_hyperopt/tnbc1-mxif8/round01/01_pretrain/01_batch-tau.yaml

python -m clearit.scripts.run_train_heads --recipe ./experiments/01_hyperopt/tnbc1-mxif8/round01/02_classifier/01_batch-tau.yaml

python -m clearit.scripts.run_inference_pipeline --recipe ./experiments/01_hyperopt/tnbc1-mxif8/round01/03_linear-eval/01_batch-tau.yaml
```

**Where do results go?**

* Trained encoders and heads are saved under `models_dir`.
* Predictions are written under `outputs_dir`.
* All of these locations are defined in your `config.yaml` (see below).

### 3) (Optional) Convert raw datasets

If you want to train the models yourself and are starting from raw sources, use the scripts in `scripts/` to convert external datasets into the unified format expected by CLEAR-IT. These scripts read from `raw_datasets` and write to `datasets` as configured in `config.yaml`.

We recommend downloading the prepared data from our [data repository](https://data.4tu.nl/my/datasets/ebc792ad-4767-4aef-b8ff-ae653e901e3f/10.4121/126d8103-6de5-4493-a48e-5d529fef471e), which contains the folder structure and instructions on how to obtain the raw datasets for conversion..

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

## Troubleshooting

* **`FileNotFoundError: config.yaml`** — ensure you copied `config_template.yaml` to `config.yaml` and that you run commands from the repository root (or point your working directory accordingly).
* **Docker can’t see your data** — double-check your `-v /host/path:/container/path` volume mounts and that `config.yaml` uses the *container* paths when running inside Docker.
