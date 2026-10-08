#!/usr/bin/env bash
set -euo pipefail
OPENPI_DIR="${OPENPI_DIR:-$HOME/projects/openpi}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MUJOCO_GL=glx
export LIBGL_ALWAYS_SOFTWARE=1
export PYTHONPATH="$OPENPI_DIR/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
exec xvfb-run -a "$OPENPI_DIR/examples/libero/.venv/bin/python" \
  "$script_dir/demo.py" --openpi-dir "$OPENPI_DIR" "$@"
