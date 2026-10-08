#!/usr/bin/env bash
set -eo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$script_dir/../../.." && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$HOME/projects/openpi}"
root="${UF850_RUN_DIR:-$repo/data/uf850_tuning_v1}"
data_dir="${UF850_DATA_DIR:-$repo/data/uf850_sim_v1}"
repo_id="${UF850_REPO_ID:-tactfoundry/uf850_bowl_sim_v1}"
weights="${UF850_BASE_WEIGHTS:-$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base}"
python="$OPENPI_DIR/.venv/bin/python"
source /opt/ros/humble/setup.bash
source "$repo/install/setup.bash"
case "${1:-}" in
  generate)
    shift
    exec python3 "$script_dir/generate.py" --output "$data_dir" "$@" ;;
  convert)
    shift
    exec "$python" "$script_dir/convert.py" --source "$data_dir" --repo-id "$repo_id" "$@" ;;
  download)
    exec "$python" -c 'from openpi.shared import download; print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_base", token="anon"))' ;;
  stats|train|evaluate|serve)
    stage="$1"; shift
    export WANDB_MODE=disabled
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-.85}"
    if [ "$stage" = stats ]; then export JAX_PLATFORMS=cpu; fi
    if [ "$stage" = stats ] || [ "$stage" = train ]; then
      exec "$python" "$script_dir/train.py" "$stage" --openpi "$OPENPI_DIR" --root "$root" --weights "$weights" --repo-id "$repo_id" "$@"
    else
      exec "$python" "$script_dir/$stage.py" --root "$root" --weights "$weights" --repo-id "$repo_id" "$@"
    fi ;;
  rollout)
    shift
    exec python3 "$script_dir/rollout.py" --output "$root/rollouts" "$@" ;;
  *)
    echo 'Usage: run.sh {generate|convert|download|stats|train|evaluate|serve|rollout} [options]' >&2
    exit 2 ;;
esac
