#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bootstrap_isaac_ros_cli_files.sh [--force] [--source] [--rl] [--pull] [--auto-build <0|1>] [--force-build <0|1>] [--auto-setup <0|1>] [--force-asset-setup <0|1>] [--accept-eula <0|1>]

Creates/copies Isaac ROS CLI helper files with default values:
  - ~/.config/isaac-ros-cli/config.yaml
  - ~/.isaac_ros_dev-dockerargs
  - ${ISAAC_ROS_WS}/../scripts/.isaac_ros_common-config

By default, existing files are not overwritten; you will be prompted per-file.
Use --force to overwrite without prompts.

Config options:
  --source   Use isaac_manipulation_source instead of isaac_manipulation.
  --rl       Append isaac_manipulation_rsl_rl to additional_image_keys.
  --auto-build <0|1>   Set ISAAC_ROS_MANIPULATION_AUTO_BUILD in ~/.isaac_ros_dev-dockerargs.
  --force-build <0|1>  Set ISAAC_ROS_MANIPULATION_FORCE_BUILD in ~/.isaac_ros_dev-dockerargs.
  --pull              Set ISAAC_ROS_MANIPULATION_PULL_REPOS=1 in ~/.isaac_ros_dev-dockerargs.
  --auto-setup <0|1>   Set ISAAC_ROS_MANIPULATION_AUTO_SETUP in ~/.isaac_ros_dev-dockerargs.
  --force-asset-setup <0|1> Set ISAAC_ROS_MANIPULATION_FORCE_ASSET_SETUP in ~/.isaac_ros_dev-dockerargs.
  --accept-eula <0|1>  Set ISAAC_ROS_ACCEPT_EULA in ~/.isaac_ros_dev-dockerargs.
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
USE_SOURCE=0
USE_RL=0
AUTO_BUILD=1
FORCE_BUILD=0
PULL_REPOS=0
AUTO_SETUP=1
FORCE_ASSET_SETUP=0
ACCEPT_EULA=1

parse_bool() {
  case "$1" in
    1|true|True|yes|on)
      echo "1"
      ;;
    0|false|False|no|off)
      echo "0"
      ;;
    *)
      return 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
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
    --auto-build)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --auto-build requires a value (0 or 1)" >&2
        usage >&2
        exit 2
      fi
      if ! AUTO_BUILD="$(parse_bool "${2}")"; then
        echo "ERROR: invalid value for --auto-build: ${2} (use 0 or 1)" >&2
        usage >&2
        exit 2
      fi
      shift 2
      ;;
    --force-build)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --force-build requires a value (0 or 1)" >&2
        usage >&2
        exit 2
      fi
      if ! FORCE_BUILD="$(parse_bool "${2}")"; then
        echo "ERROR: invalid value for --force-build: ${2} (use 0 or 1)" >&2
        usage >&2
        exit 2
      fi
      shift 2
      ;;
    --pull)
      PULL_REPOS=1
      shift
      ;;
    --auto-setup)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --auto-setup requires a value (0 or 1)" >&2
        usage >&2
        exit 2
      fi
      if ! AUTO_SETUP="$(parse_bool "${2}")"; then
        echo "ERROR: invalid value for --auto-setup: ${2} (use 0 or 1)" >&2
        usage >&2
        exit 2
      fi
      shift 2
      ;;
    --force-asset-setup)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --force-asset-setup requires a value (0 or 1)" >&2
        usage >&2
        exit 2
      fi
      if ! FORCE_ASSET_SETUP="$(parse_bool "${2}")"; then
        echo "ERROR: invalid value for --force-asset-setup: ${2} (use 0 or 1)" >&2
        usage >&2
        exit 2
      fi
      shift 2
      ;;
    --accept-eula)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --accept-eula requires a value (0 or 1)" >&2
        usage >&2
        exit 2
      fi
      if ! ACCEPT_EULA="$(parse_bool "${2}")"; then
        echo "ERROR: invalid value for --accept-eula: ${2} (use 0 or 1)" >&2
        usage >&2
        exit 2
      fi
      shift 2
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

if [[ ! -f "${SRC_COMMON}" ]]; then
  echo "ERROR: expected setup_files templates not found under: ${TEMPLATES_DIR}" >&2
  exit 1
fi

TARGET_COMMON_DIR="${WS}/../scripts"
TARGET_COMMON="${TARGET_COMMON_DIR}/.isaac_ros_common-config"
TARGET_DOCKERARGS="${HOME}/.isaac_ros_dev-dockerargs"
TARGET_CLI_CFG_DIR="${HOME}/.config/isaac-ros-cli"
TARGET_CLI_CFG="${TARGET_CLI_CFG_DIR}/config.yaml"

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

maybe_install() {
  local src="$1"
  local dst="$2"

  if ! should_write "${dst}"; then
    echo "==> kept: ${dst}"
    return 0
  fi

  mkdir -p "$(dirname "${dst}")"
  install -m 0644 "${src}" "${dst}"
  echo "==> wrote: ${dst}"
}

write_cli_config() {
  local dst="$1"
  local tmp
  local layers=(
    realsense
  )

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
# Isaac ROS CLI user config.
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
    echo "-e ISAAC_ROS_MANIPULATION_AUTO_SETUP=${AUTO_SETUP}"
    echo "-e ISAAC_ROS_MANIPULATION_FORCE_ASSET_SETUP=${FORCE_ASSET_SETUP}"
    echo "-e ISAAC_ROS_ACCEPT_EULA=${ACCEPT_EULA}"
    echo "-w /workspaces/isaac_ros-dev"
  } > "${tmp}"

  mkdir -p "$(dirname "${dst}")"
  install -m 0644 "${tmp}" "${dst}"
  rm -f "${tmp}"
  echo "==> wrote: ${dst}"
}

mkdir -p "${TARGET_COMMON_DIR}"
mkdir -p "${TARGET_CLI_CFG_DIR}"

maybe_install "${SRC_COMMON}" "${TARGET_COMMON}"
write_dockerargs "${TARGET_DOCKERARGS}"
write_cli_config "${TARGET_CLI_CFG}"

echo
echo "Done."
echo "  - ISAAC_ROS_MANIPULATION_AUTO_BUILD=${AUTO_BUILD}, ISAAC_ROS_MANIPULATION_FORCE_BUILD=${FORCE_BUILD}"
echo "  - ISAAC_ROS_MANIPULATION_AUTO_SETUP=${AUTO_SETUP}, ISAAC_ROS_MANIPULATION_FORCE_ASSET_SETUP=${FORCE_ASSET_SETUP}"
echo "  - ISAAC_ROS_ACCEPT_EULA=${ACCEPT_EULA} (see ${TARGET_DOCKERARGS})"
echo "Next:"
echo "  - Edit ${TARGET_COMMON} to modify CONFIG_DOCKER_SEARCH_DIRS if needed for other custom dockerfiles"
echo "  - Edit ${TARGET_CLI_CFG} to add or remove custom layers"
echo "  - Edit ${TARGET_DOCKERARGS} to modify ISAAC_ROS_MANIPULATION_AUTO_SETUP and ISAAC_ROS_ACCEPT_EULA if needed"
echo "  - Run: isaac-ros activate --build-local"
