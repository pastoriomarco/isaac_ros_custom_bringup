#!/usr/bin/env bash
set -euo pipefail

# Opt-in automation: set `ISAAC_ROS_MANIPULATION_AUTO_SETUP=1` to run at container start.
if [[ "${ISAAC_ROS_MANIPULATION_AUTO_SETUP:-0}" != "1" ]]; then
  exit 0
fi

# Avoid blocking container startup with interactive prompts.
if [[ -z "${ISAAC_ROS_ACCEPT_EULA:-}" ]]; then
  echo "isaac-manipulation: auto-setup skipped (set ISAAC_ROS_ACCEPT_EULA=1 or run /usr/local/bin/isaac-manipulation-setup.sh --eula)"
  exit 0
fi

exec /usr/local/bin/isaac-manipulation-setup.sh --accept-eula
