#!/usr/bin/env bash
set -euo pipefail
export GIT_LFS_SKIP_SMUDGE=1
OPENPI_DIR="${OPENPI_DIR:-$HOME/projects/openpi}"
revision=215abfb217dbac7d5f1273282331b9b1866c0479
python3 -m pip install --user uv
uv_bin="$HOME/.local/bin/uv"
if [ ! -d "$OPENPI_DIR/.git" ]; then
  git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR"
  git -C "$OPENPI_DIR" checkout "$revision"
fi
if [ "$(git -C "$OPENPI_DIR" rev-parse HEAD)" != "$revision" ]; then
  echo "Expected OpenPI $revision; use a separate OPENPI_DIR for this demo." >&2
  exit 1
fi
cd "$OPENPI_DIR"
git submodule update --init --recursive
"$uv_bin" sync --python 3.11 --no-dev
"$uv_bin" venv --python 3.8 --allow-existing examples/libero/.venv
"$uv_bin" pip sync --python examples/libero/.venv/bin/python \
  examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
"$uv_bin" pip install --python examples/libero/.venv/bin/python \
  -e packages/openpi-client -e third_party/libero
