#!/bin/bash
set -euo pipefail
ROOT=/mount-data/CLM-Replication
exec "$ROOT/management/runtime.sh" bash -c '
  export CUDA_VISIBLE_DEVICES=4,5,6,7
  export CLM_RAY_ADDRESS=127.0.0.1:29477
  export PYTHONPATH="/mount-data/CLM-Replication/env/attention-fa283:$PYTHONPATH"
  exec "$@"
' clm-four-card-runtime "$@"
