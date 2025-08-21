# Base NVIDIA PyTorch image (CUDA/cuDNN preinstalled)
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Keep Python snappy and logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Workdir inside the container
WORKDIR /workspace

# Copy deps first to leverage build cache
COPY requirements.txt /tmp/requirements.txt

# Upgrade packaging tooling (compat constraints), then install deps
RUN pip install --no-cache-dir -U "pip<24.3" "setuptools<70" "wheel<0.45" \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# JupyterLab port
EXPOSE 8888

# Default: run JupyterLab accessible from host
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
