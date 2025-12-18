#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bootstrap_isaac_ros_cli_files.sh [--force]

Creates/copies Isaac ROS CLI helper files with default values:
  - ~/.config/isaac-ros-cli/config.yaml
  - ~/.isaac_ros_dev-dockerargs
  - ${ISAAC_ROS_WS}/../scripts/.isaac_ros_common-config

By default, existing files are not overwritten. Use --force to overwrite.
EOF
}

infer_ws_from_script() {
  # Expected script location:
  #   <ISAAC_ROS_WS>/src/isaac_ros_custom_bringup/isaac_ros_4/scripts/this_script.sh
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  (cd "${script_dir}/../../../.." && pwd)
}

FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
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

WS="${ISAAC_ROS_WS:-}"
if [[ -z "${WS}" ]]; then
  WS="$(infer_ws_from_script)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES_DIR="${SCRIPT_DIR}/../setup_files"

SRC_COMMON="${TEMPLATES_DIR}/.isaac_ros_common-config"
SRC_DOCKERARGS="${TEMPLATES_DIR}/.isaac_ros_dev-dockerargs"
SRC_CLI_CFG="${TEMPLATES_DIR}/isaac-ros-cli.config.yaml"

if [[ ! -f "${SRC_COMMON}" ]] || [[ ! -f "${SRC_DOCKERARGS}" ]] || [[ ! -f "${SRC_CLI_CFG}" ]]; then
  echo "ERROR: expected setup_files templates not found under: ${TEMPLATES_DIR}" >&2
  exit 1
fi

TARGET_COMMON_DIR="${WS}/../scripts"
TARGET_COMMON="${TARGET_COMMON_DIR}/.isaac_ros_common-config"
TARGET_DOCKERARGS="${HOME}/.isaac_ros_dev-dockerargs"
TARGET_CLI_CFG_DIR="${HOME}/.config/isaac-ros-cli"
TARGET_CLI_CFG="${TARGET_CLI_CFG_DIR}/config.yaml"

maybe_install() {
  local src="$1"
  local dst="$2"

  if [[ -e "${dst}" ]] && [[ "${FORCE}" != "1" ]]; then
    echo "==> exists: ${dst} (skipping; pass --force to overwrite)"
    return 0
  fi

  mkdir -p "$(dirname "${dst}")"
  install -m 0644 "${src}" "${dst}"
  echo "==> wrote: ${dst}"
}

mkdir -p "${TARGET_COMMON_DIR}"
mkdir -p "${TARGET_CLI_CFG_DIR}"

maybe_install "${SRC_COMMON}" "${TARGET_COMMON}"
maybe_install "${SRC_DOCKERARGS}" "${TARGET_DOCKERARGS}"
maybe_install "${SRC_CLI_CFG}" "${TARGET_CLI_CFG}"

echo
echo "Done."
echo "  - ISAAC_ROS_MANIPULATION_AUTO_SETUP and ISAAC_ROS_ACCEPT_EULA are set to 1 by default"
echo "Next:"
echo "  - Edit ${TARGET_COMMON} to modify CONFIG_DOCKER_SEARCH_DIRS if needed for other custom dockerfiles"
echo "  - Edit ${TARGET_CLI_CFG} to add or remove custom layers"
echo "  - Edit ${TARGET_DOCKERARGS} to modify ISAAC_ROS_MANIPULATION_AUTO_SETUP and ISAAC_ROS_ACCEPT_EULA if needed"
echo "  - Run: isaac-ros activate --build-local"
