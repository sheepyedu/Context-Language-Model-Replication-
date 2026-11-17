#!/bin/bash
set -euo pipefail
ROOT=/mount-data/CLM-Replication
VENV="$ROOT/env/rl"
export CUDA_VISIBLE_DEVICES=GPU-101b4bf3-3f4e-5706-eef9-7e3d38ba4ea3,GPU-c64fa322-ea6c-ed73-b81b-67bddfd2ad9e
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$VENV/bin:/mount-data/yangyang-h20-2gpu-train-20261107/env/skill100-20261110/runtime/usr/bin:$PATH"
export DOCKER_HOST=unix:///var/docker/proxy/docker-proxy.sock
export LD_LIBRARY_PATH="$VENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/mount-data/yangyang-h20-2gpu-train-20261107/env/deepseek41f-vllm031-20261113/cuda-compat/usr/local/cuda-13.0/compat:$VENV/lib/python3.11/site-packages/nvidia/cudnn/lib:$VENV/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$ROOT/src/ProRL/src:$ROOT/src/slime:$ROOT/src/Megatron-LM"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HUB_OFFLINE=1 HF_HOME="$ROOT/env/hf-cache"
export LITELLM_LOCAL_MODEL_COST_MAP=true
export XDG_CACHE_HOME="$ROOT/env/runtime-cache"
export TRITON_CACHE_DIR="$ROOT/env/triton-cache"
export TORCH_EXTENSIONS_DIR="$ROOT/env/torch-extensions"
export POLAR_DUAL_CHANNEL_W_EFF=0.25
export PYTHONNOUSERSITE=1
exec "$@"
