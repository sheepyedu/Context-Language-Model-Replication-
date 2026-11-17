#!/bin/bash
set -euo pipefail
ROOT=/mount-data/CLM-Replication
exec 9>"$ROOT/management/megatron-bootstrap.lock"
flock -n 9 || exit 73
trap 'code=$?; printf "%s\n" "$code" > "$ROOT/management/megatron-bootstrap.exit"' EXIT
export CUDA_VISIBLE_DEVICES=4,5
export CUDA_HOME=/usr/local/cuda-12.4
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
export UV_CACHE_DIR="$ROOT/env/uv-cache"
export MAX_JOBS=8
export NVTE_BUILD_THREADS_PER_JOB=2
export TORCH_CUDA_ARCH_LIST=9.0
export NVTE_FRAMEWORK=pytorch
PY="$ROOT/env/rl/bin/python"
UV="$ROOT/env/rl/bin/uv"
while [ ! -f "$ROOT/management/bootstrap.exit" ]; do sleep 10; done
[ "$(cat "$ROOT/management/bootstrap.exit")" = 0 ] || exit 74
mkdir -p "$ROOT/src/Megatron-LM"
[ -f "$ROOT/src/Megatron-LM/pyproject.toml" ] || tar -xzf "$ROOT/management/Megatron-LM.tar.gz" --strip-components=1 -C "$ROOT/src/Megatron-LM"
"$UV" pip install --python "$PY" -e "$ROOT/src/Megatron-LM" --no-deps --no-build-isolation
CUDNN_PATH=$("$PY" -c 'import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0])')
export CPATH="$CUDNN_PATH/include:$CUDA_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$CUDNN_PATH/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
"$UV" pip install --python "$PY" --no-build-isolation 'transformer-engine[pytorch]==2.10.0'
"$UV" pip install --python "$PY" ray tensorboard wandb omegaconf numba pylatexenc qwen_vl_utils nvtx tensorstore flask-restful sentencepiece
"$UV" pip install --python "$PY" 'numpy==1.26.4' 'pyarrow==21.0.0' 'pandas==2.2.3' 'datasets==4.3.0' 'ml-dtypes==0.5.4'
"$UV" pip freeze --python "$PY" > "$ROOT/management/pip-freeze.txt"
