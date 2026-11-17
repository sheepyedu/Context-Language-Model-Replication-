#!/bin/bash
set -euo pipefail
ROOT=/mount-data/CLM-Replication
exec 9>"$ROOT/management/agent-bootstrap.lock"
flock -n 9 || exit 73
trap 'code=$?; printf "%s\n" "$code" > "$ROOT/management/agent-bootstrap.exit"' EXIT
UV="$ROOT/env/rl/bin/uv"
export UV_CACHE_DIR="$ROOT/env/uv-cache"
PY_SRC=$("$UV" python find 3.12)
if [ ! -d "$ROOT/env/cpython312" ]; then cp -a "$(dirname "$(dirname "$PY_SRC")")" "$ROOT/env/cpython312"; fi
"$UV" venv --python "$ROOT/env/cpython312/bin/python3.12" "$ROOT/env/agent"
"$UV" pip install --python "$ROOT/env/agent/bin/python" 'harbor==0.16.1' 'litellm>=1.70,<2' tiktoken pyyaml requests beautifulsoup4 --index-url https://mirrors.aliyun.com/pypi/simple
"$ROOT/env/agent/bin/python" -m pip freeze > "$ROOT/management/agent-freeze.txt" || "$UV" pip freeze --python "$ROOT/env/agent/bin/python" > "$ROOT/management/agent-freeze.txt"
