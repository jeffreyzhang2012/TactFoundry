#!/usr/bin/env bash
set -eo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$script_dir/../../.." && pwd)"
export UF850_RUN_DIR="${UF850_RUN_DIR:-$repo/data/uf850_tuning_diverse_v6}"
export UF850_DATA_DIR="${UF850_DATA_DIR:-$repo/data/uf850_diverse_sim_v6}"
export UF850_REPO_ID="${UF850_REPO_ID:-tactfoundry/uf850_diverse_sim_v6}"
export UF850_BASE_WEIGHTS="${UF850_BASE_WEIGHTS:-$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-.70}"
exec bash "$script_dir/run.sh" "$@"
