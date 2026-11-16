#!/bin/bash
set -euo pipefail
ROOT=/mount-data/CLM-Replication
exec "$ROOT/management/runtime.sh" bash -c '
 export PYTHONPATH="/mount-data/CLM-Replication/env/attention-fa283:$PYTHONPATH"
 exec "$@"
' clm-attention-runtime "$@"
