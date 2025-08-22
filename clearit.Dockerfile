# Base NVIDIA PyTorch image (CUDA/cuDNN preinstalled)
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Keep Python snappy and logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Workdir inside the container
WORKDIR /workspace

# Copy dependencies first to leverage build cache
COPY requirements.txt /tmp/requirements.txt

# Upgrade packaging tooling (compat constraints), then install deps
RUN pip install --no-cache-dir -U "pip<24.3" "setuptools<70" "wheel<0.45" \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# --- Install CLEAR-IT into the image ---
COPY setup.py /opt/clearit/src/setup.py
COPY clearit /opt/clearit/src/clearit
COPY scripts /opt/clearit/src/scripts
COPY notebooks /opt/clearit/src/notebooks

# Install the CLEAR-IT library (non-editable install)
RUN pip install --no-cache-dir /opt/clearit/src

# Bundle experiments and the config template for convenience
COPY experiments /opt/clearit/experiments
COPY config_template.yaml /opt/clearit/config_template.yaml

# --- Lightweight entrypoint to auto-create config.yaml when /data is mounted ---
# This script writes /workspace/config.yaml pointing to /data/* and bundled experiments,
# only if the file doesn't already exist and /data is present.
RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -euo pipefail' \
'CFG="/workspace/config.yaml"' \
'if [ ! -f "$CFG" ] && [ -d "/data" ]; then' \
'  cat > "$CFG" <<'"'"'YAML'"'"'' \
'paths:' \
'  data_root: /data' \
'  datasets_dir: /data/datasets' \
'  raw_datasets_dir: /data/raw_datasets' \
'  models_dir: /data/models' \
'  outputs_dir: /data/outputs' \
'  experiments_dir: /opt/clearit/experiments' \
'YAML' \
'  echo "[CLEAR-IT] Wrote default config to $CFG"' \
'fi' \
'exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root "$@"' \
> /usr/local/bin/clearit-entrypoint.sh \
 && chmod +x /usr/local/bin/clearit-entrypoint.sh

# JupyterLab port
EXPOSE 8888

# Default: run entrypoint that ensures config.yaml exists, then starts JupyterLab
ENTRYPOINT ["clearit-entrypoint.sh"]
