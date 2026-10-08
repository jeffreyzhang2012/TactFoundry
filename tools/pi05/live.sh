#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set +u
source /opt/ros/humble/setup.bash
set -u
export ROS_DOMAIN_ID=43
export FASTRTPS_DEFAULT_PROFILES_FILE="$script_dir/../../src/tactile_simulation/config/fastdds_udp.xml"
python3 "$script_dir/rviz_bridge.py" &
bridge_pid=$!
rviz2 -d "$script_dir/libero.rviz" &
rviz_pid=$!
trap 'kill "$bridge_pid" "$rviz_pid" 2>/dev/null || true' EXIT
sleep 2
while true; do
  bash "$script_dir/demo.sh" --live-port 8766 --step-seconds .07 --hold-seconds 5 "$@"
done
