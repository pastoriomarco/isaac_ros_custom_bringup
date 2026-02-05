#!/usr/bin/env bash
set -euo pipefail

# Opt-in automation: set `ISAAC_ROS_MANIPULATION_AUTO_BUILD=1` to run at container start.
if [[ "${ISAAC_ROS_MANIPULATION_AUTO_BUILD:-1}" != "1" ]]; then
  exit 0
fi

exec /usr/local/bin/isaac-manipulation-build.sh
