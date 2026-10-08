#!/usr/bin/env bash
set -eo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$script_dir/../../.." && pwd)"
export UF850_RUN_DIR="${UF850_RUN_DIR:-$repo/data/uf850_tuning_multi_v3}"
export UF850_DATA_DIR="${UF850_DATA_DIR:-$repo/data/uf850_multi_sim_v3_wide}"
export UF850_REPO_ID="${UF850_REPO_ID:-tactfoundry/uf850_multi_sim_v3}"
export UF850_BASE_WEIGHTS="${UF850_BASE_WEIGHTS:-$repo/data/uf850_tuning_v2/checkpoints/pi05_uf850_lora/refine500/499}"
exec bash "$script_dir/run.sh" "$@"
