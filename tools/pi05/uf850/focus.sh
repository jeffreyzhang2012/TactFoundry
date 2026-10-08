#!/usr/bin/env bash
set -eo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$script_dir/../../.." && pwd)"
previous="$repo/data/uf850_tuning_multi_v3"
export UF850_RUN_DIR="${UF850_RUN_DIR:-$repo/data/uf850_tuning_multi_v4}"
export UF850_DATA_DIR="${UF850_DATA_DIR:-$repo/data/uf850_multi_sim_v3_wide}"
export UF850_REPO_ID="${UF850_REPO_ID:-tactfoundry/uf850_multi_sim_v3}"
export UF850_BASE_WEIGHTS="${UF850_BASE_WEIGHTS:-$previous/checkpoints/pi05_uf850_lora/multi2000/1999}"
checkpoint="$UF850_RUN_DIR/checkpoints/pi05_uf850_lora/grounding1000/999"
stage="${1:-}"; if [ "$#" -gt 0 ]; then shift; fi
case "$stage" in
  train)
    # Keep the exact normalization used by the source checkpoint.
    if [ ! -d "$UF850_RUN_DIR/assets" ]; then
      mkdir -p "$UF850_RUN_DIR"
      cp -a "$previous/assets" "$UF850_RUN_DIR/assets"
    fi
    exec bash "$script_dir/run.sh" train --steps 1000 --exp-name grounding1000 \
      --selection-manifest "$UF850_DATA_DIR/split.json" "$@" ;;
  evaluate)
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-.75}"
    exec bash "$script_dir/run.sh" evaluate --checkpoint "$checkpoint" --samples 96 \
      --language-probe-source "$UF850_DATA_DIR" "$@" ;;
  serve)
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-.75}"
    exec bash "$script_dir/run.sh" serve --checkpoint "$checkpoint" "$@" ;;
  rollout|language-probe)
    exec bash "$script_dir/run.sh" "$stage" "$@" ;;
  *)
    echo 'Usage: focus.sh {train|evaluate|serve|rollout|language-probe} [options]' >&2
    exit 2 ;;
esac
