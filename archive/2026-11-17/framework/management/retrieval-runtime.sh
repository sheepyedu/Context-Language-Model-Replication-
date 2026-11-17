#!/bin/bash
set -euo pipefail
export JAVA_HOME="/mount-data/CLM-Replication/env/java21/jdk-21.0.13+9-jre"
export PATH="$JAVA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="" USE_TORCH=0 USE_TF=0 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="/mount-data/CLM-Replication/management/retrieval-v1:/mount-data/CLM-Replication/env/rl/lib/python3.11/site-packages"
exec "/mount-data/CLM-Replication/env/retrieval-v1/bin/python" "$@"
