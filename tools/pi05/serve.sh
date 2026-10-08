#!/usr/bin/env bash
set -euo pipefail
OPENPI_DIR="${OPENPI_DIR:-$HOME/projects/openpi}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"
exec "$OPENPI_DIR/.venv/bin/python" "$script_dir/serve.py" "$@"
