#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bootstrap_isaac_ros_cli_files.sh
  [--force]
  [--user-scope]
  [--source] [--rl]
  [--realsense]
  [--minor <N> | --latest]
  [--pull] [--force-build] [--skip-asset-install]
  [--disable-auto-build] [--disable-auto-setup] [--force-asset-setup]
  [--disable-accept-eula]

Creates Isaac ROS CLI helper files.

Default (recommended): WORKSPACE-scoped
  - ${ISAAC_ROS_WS}/.isaac-ros-cli/config.yaml
  - ${ISAAC_ROS_WS}/../scripts/.isaac_ros_dev-dockerargs
  - ${ISAAC_ROS_WS}/../scripts/.isaac_ros_common-config

With --user-scope:
  - ~/.config/isaac-ros-cli/config.yaml
  - ~/.isaac_ros_dev-dockerargs
  - ${ISAAC_ROS_WS}/../scripts/.isaac_ros_common-config

Image key selection:
  --source     Use `isaac_manipulation_source` instead of `isaac_manipulation`.
  --rl         Append `isaac_manipulation_rsl_rl`.
  --realsense  Include the `realsense` layer (optional).

Minor selection for source checkouts:
  --minor N    Pin NVIDIA source repos to `release-4.N` (default: 1).
  --latest     Resolve latest Isaac ROS 4.x minor at container start.

Runtime behavior (written to dockerargs):
  --pull               Set ISAAC_ROS_MANIPULATION_PULL_REPOS=1
  --force-build        Set ISAAC_ROS_MANIPULATION_FORCE_BUILD=1
  --skip-asset-install Set ISAAC_MANIPULATION_SKIP_ASSET_INSTALL=1
  --disable-auto-build Set ISAAC_ROS_MANIPULATION_AUTO_BUILD=0
  --disable-auto-setup Set ISAAC_ROS_MANIPULATION_AUTO_SETUP=0
  --force-asset-setup  Set ISAAC_ROS_MANIPULATION_FORCE_ASSET_SETUP=1
  --disable-accept-eula Set ISAAC_ROS_ACCEPT_EULA=0
EOF
}

infer_ws_from_script() {
  # Expected script location:
  #   <ISAAC_ROS_WS>/src/isaac_ros_custom_bringup/isaac_ros_4/4.1/scripts/this_script.sh
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  (cd "${script_dir}/../../../../.." && pwd)
}

FORCE=0
USER_SCOPE=0
USE_SOURCE=0
USE_RL=0
USE_REALSENSE=0

AUTO_BUILD=1
FORCE_BUILD=0
PULL_REPOS=0
SKIP_ASSET_INSTALL=0
AUTO_SETUP=1
FORCE_ASSET_SETUP=0
ACCEPT_EULA=1

TARGET_MINOR="1"
USE_LATEST_MINOR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --user-scope)
      USER_SCOPE=1
      shift
      ;;
    --source)
      USE_SOURCE=1
      shift
      ;;
    --rl)
      USE_RL=1
      shift
      ;;
    --realsense)
      USE_REALSENSE=1
      shift
      ;;
    --minor)
      TARGET_MINOR="${2:-}"
      USE_LATEST_MINOR=0
      shift 2
      ;;
    --latest)
      USE_LATEST_MINOR=1
      shift
      ;;
    --disable-auto-build)
      AUTO_BUILD=0
      shift
      ;;
    --force-build)
      FORCE_BUILD=1
      shift
      ;;
    --pull)
      PULL_REPOS=1
      shift
      ;;
    --skip-asset-install)
      SKIP_ASSET_INSTALL=1
      shift
      ;;
    --disable-auto-setup)
      AUTO_SETUP=0
      shift
      ;;
    --force-asset-setup)
      FORCE_ASSET_SETUP=1
      shift
      ;;
    --disable-accept-eula)
      ACCEPT_EULA=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "${TARGET_MINOR}" && ! "${TARGET_MINOR}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: invalid --minor: ${TARGET_MINOR} (expected integer)" >&2
  exit 2
fi

WS="${ISAAC_ROS_WS:-}"
if [[ -z "${WS}" ]]; then
  WS="$(infer_ws_from_script)"
fi

TARGET_COMMON_DIR="${WS}/../scripts"
TARGET_COMMON="${TARGET_COMMON_DIR}/.isaac_ros_common-config"

if [[ "${USER_SCOPE}" == "1" ]]; then
  TARGET_DOCKERARGS="${HOME}/.isaac_ros_dev-dockerargs"
  TARGET_CLI_CFG_DIR="${HOME}/.config/isaac-ros-cli"
  TARGET_CLI_CFG="${TARGET_CLI_CFG_DIR}/config.yaml"
else
  TARGET_DOCKERARGS="${WS}/../scripts/.isaac_ros_dev-dockerargs"
  TARGET_CLI_CFG_DIR="${WS}/.isaac-ros-cli"
  TARGET_CLI_CFG="${TARGET_CLI_CFG_DIR}/config.yaml"
fi

should_write() {
  local dst="$1"

  if [[ "${FORCE}" == "1" ]]; then
    return 0
  fi

  if [[ ! -e "${dst}" ]]; then
    return 0
  fi

  echo "==> exists: ${dst}"
  if [[ -t 0 ]]; then
    local reply
    read -r -p "Overwrite ${dst}? [y/N] " reply
    if [[ "${reply}" =~ ^[Yy]$ ]]; then
      return 0
    fi
  else
    echo "==> no TTY; skipping ${dst} (use --force to overwrite)"
  fi

  return 1
}

write_common_config() {
  local dst="$1"
  local tmp

  if ! should_write "${dst}"; then
    echo "==> kept: ${dst}"
    return 0
  fi

  tmp="$(mktemp)"
  cat <<'EOF' > "${tmp}"
# Isaac ROS CLI common config (sourced by isaac-ros-cli).
#
# This adds this repo's `isaac_ros_4/4.1/` directory to the Dockerfile search path so that the CLI can find:
#   - Dockerfile.isaac_manipulation
#   - Dockerfile.isaac_manipulation_source
#
# Note: this file is intended to live at:
#   ${ISAAC_ROS_WS}/../scripts/.isaac_ros_common-config

CONFIG_DOCKER_SEARCH_DIRS=(/etc/isaac-ros-cli/docker ${ISAAC_ROS_WS}/docker ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4/4.1)
EOF

  mkdir -p "$(dirname "${dst}")"
  install -m 0644 "${tmp}" "${dst}"
  rm -f "${tmp}"
  echo "==> wrote: ${dst}"
}

write_cli_config() {
  local dst="$1"
  local tmp
  local layers=()

  if [[ "${USE_REALSENSE}" == "1" ]]; then
    layers+=(realsense)
  fi

  if [[ "${USE_SOURCE}" == "1" ]]; then
    layers+=(isaac_manipulation_source)
  else
    layers+=(isaac_manipulation)
  fi

  if [[ "${USE_RL}" == "1" ]]; then
    layers+=(isaac_manipulation_rsl_rl)
  fi

  if ! should_write "${dst}"; then
    echo "==> kept: ${dst}"
    return 0
  fi

  tmp="$(mktemp)"
  {
    cat <<'EOF'
# Isaac ROS CLI config (user or workspace scope).
#
# This file is generated by bootstrap_isaac_ros_cli_files.sh.

docker:
  image:
    additional_image_keys:
EOF
    for layer in "${layers[@]}"; do
      echo "      - ${layer}"
    done
  } > "${tmp}"

  mkdir -p "$(dirname "${dst}")"
  install -m 0644 "${tmp}" "${dst}"
  rm -f "${tmp}"
  echo "==> wrote: ${dst}"
}

write_dockerargs() {
  local dst="$1"
  local tmp

  if ! should_write "${dst}"; then
    echo "==> kept: ${dst}"
    return 0
  fi

  tmp="$(mktemp)"
  {
    echo "-e ISAAC_ROS_MANIPULATION_AUTO_BUILD=${AUTO_BUILD}"
    echo "-e ISAAC_ROS_MANIPULATION_FORCE_BUILD=${FORCE_BUILD}"
    echo "-e ISAAC_ROS_MANIPULATION_PULL_REPOS=${PULL_REPOS}"
    echo "-e ISAAC_MANIPULATION_SKIP_ASSET_INSTALL=${SKIP_ASSET_INSTALL}"
    echo "-e ISAAC_ROS_MANIPULATION_AUTO_SETUP=${AUTO_SETUP}"
    echo "-e ISAAC_ROS_MANIPULATION_FORCE_ASSET_SETUP=${FORCE_ASSET_SETUP}"
    echo "-e ISAAC_ROS_ACCEPT_EULA=${ACCEPT_EULA}"
    echo "-e ISAAC_ROS_MANIPULATION_TARGET_MINOR=${TARGET_MINOR}"
    echo "-e ISAAC_ROS_MANIPULATION_USE_LATEST_MINOR=${USE_LATEST_MINOR}"
    echo "-w /workspaces/isaac_ros-dev"
  } > "${tmp}"

  mkdir -p "$(dirname "${dst}")"
  install -m 0644 "${tmp}" "${dst}"
  rm -f "${tmp}"
  echo "==> wrote: ${dst}"
}

mkdir -p "${TARGET_COMMON_DIR}"
mkdir -p "${TARGET_CLI_CFG_DIR}"

write_common_config "${TARGET_COMMON}"
write_dockerargs "${TARGET_DOCKERARGS}"
write_cli_config "${TARGET_CLI_CFG}"

echo
echo "Done."
echo "  - Wrote: ${TARGET_COMMON}"
echo "  - Wrote: ${TARGET_CLI_CFG}"
echo "  - Wrote: ${TARGET_DOCKERARGS}"
echo "Next:"
echo "  - Run: isaac-ros activate --build-local"

