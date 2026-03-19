#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[isaac-manipulation-bootstrap] $*"
}

ISAAC_ROS_WS="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}"
FORCE="${ISAAC_ROS_4_2_BOOTSTRAP_FORCE:-0}"
COLCON_BUILD_BASE="${ISAAC_ROS_COLCON_BUILD_BASE:-${ISAAC_ROS_WS}/build}"
COLCON_LOG_BASE="${ISAAC_ROS_COLCON_LOG_BASE:-${ISAAC_ROS_WS}/log}"
COLCON_INSTALL_BASE="${ISAAC_ROS_COLCON_INSTALL_BASE:-${ISAAC_ROS_WS}/install}"

if [[ ! -d "${ISAAC_ROS_WS}" ]]; then
  echo "ERROR: ISAAC_ROS_WS does not exist: ${ISAAC_ROS_WS}" >&2
  exit 1
fi

mkdir -p "${ISAAC_ROS_WS}/src" "${COLCON_BUILD_BASE}" "${COLCON_LOG_BASE}" "${COLCON_INSTALL_BASE}"

source_if_exists() {
  local setup_file="$1"
  if [[ -f "${setup_file}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${setup_file}"
    set -u
    return 0
  fi
  return 1
}

apt_pkg_installed() {
  dpkg -s "$1" >/dev/null 2>&1
}

colcon_build() {
  colcon \
    --log-base "${COLCON_LOG_BASE}" \
    build \
    --build-base "${COLCON_BUILD_BASE}" \
    --install-base "${COLCON_INSTALL_BASE}" \
    "$@"
}

clone_if_missing() {
  local target_path="$1"
  shift

  if [[ -e "${target_path}" ]]; then
    log "Skipping clone (already exists): ${target_path}"
    return 0
  fi

  git clone "$@"
}

sync_topic_based_ros2_control_headers() {
  local source_include_dir="${ISAAC_ROS_WS}/src/topic_based_ros2_control/include"
  local install_include_dir="${COLCON_INSTALL_BASE}/topic_based_ros2_control/include"

  if [[ ! -d "${source_include_dir}" ]]; then
    log "topic_based_ros2_control headers not found at ${source_include_dir}; skipping header sync."
    return 0
  fi

  log "Syncing topic_based_ros2_control headers into ${COLCON_INSTALL_BASE}."
  mkdir -p "${install_include_dir}"
  cp -a "${source_include_dir}/." "${install_include_dir}/"
}

is_known_broken_overlay_prefix() {
  case "$1" in
    isaac_ros_nitros_camera_info_type|isaac_ros_nitros_detection3_d_array_type|isaac_ros_nitros_disparity_image_type)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

prune_incomplete_overlay_prefixes() {
  local prefix_dir
  local package_name
  local package_share_dir

  for prefix_dir in "${COLCON_INSTALL_BASE}"/*; do
    [[ -d "${prefix_dir}" ]] || continue
    package_name="$(basename "${prefix_dir}")"
    package_share_dir="${prefix_dir}/share/${package_name}"

    if is_known_broken_overlay_prefix "${package_name}" && [[ -f "${package_share_dir}/package.dsv" && ! -f "${package_share_dir}/local_setup.bash" ]]; then
      log "Removing incomplete overlay prefix ${package_name}."
      rm -rf "${prefix_dir}"
    fi
  done
}

if ! source_if_exists "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"; then
  echo "ERROR: ROS setup file not found: /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" >&2
  exit 1
fi

if apt_pkg_installed ros-jazzy-rmw-cyclonedds-cpp; then
  log "CycloneDDS RMW package already installed; skipping"
else
  log "Installing CycloneDDS RMW package"
  log "CycloneDDS RMW package not installed yet; refreshing apt package lists first"
  sudo apt-get update
  sudo apt-get install -y ros-jazzy-rmw-cyclonedds-cpp
fi
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

if apt_pkg_installed python3-pip; then
  log "python3-pip already installed; skipping"
else
  log "Installing python3-pip"
  sudo apt update
  sudo apt install -y python3-pip
fi

if apt_pkg_installed ros-jazzy-isaac-manipulator-bringup; then
  log "ros-jazzy-isaac-manipulator-bringup already installed; skipping"
else
  log "Refreshing apt package lists"
  sudo apt-get update
  log "Installing ros-jazzy-isaac-manipulator-bringup"
  sudo apt-get install -y ros-jazzy-isaac-manipulator-bringup
fi

log "Cloning Robotiq + serial (if missing)"
cd "${ISAAC_ROS_WS}/src"
clone_if_missing "${ISAAC_ROS_WS}/src/ros2_robotiq_gripper" --recursive https://github.com/NVIDIA-ISAAC-ROS/ros2_robotiq_gripper
clone_if_missing "${ISAAC_ROS_WS}/src/serial" -b ros2 https://github.com/tylerjw/serial
if [[ -e "${ISAAC_ROS_WS}/src/ros2_robotiq_gripper/.git" ]]; then
  log "Ensuring ros2_robotiq_gripper submodules are initialized"
  git -C "${ISAAC_ROS_WS}/src/ros2_robotiq_gripper" submodule update --init --recursive
fi

log "Building Robotiq + serial"
cd "${ISAAC_ROS_WS}"
colcon_build --symlink-install --packages-select-regex 'robotiq.*' serial --cmake-args -DBUILD_TESTING=OFF

prune_incomplete_overlay_prefixes
source_if_exists "${COLCON_INSTALL_BASE}/setup.bash" || true

log "Cloning topic_based_ros2_control (if missing)"
cd "${ISAAC_ROS_WS}/src"
clone_if_missing "${ISAAC_ROS_WS}/src/topic_based_ros2_control" https://github.com/karanchahal-nv/topic_based_ros2_control

if [[ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ]]; then
  log "Initializing rosdep (sudo rosdep init)"
  sudo rosdep init
fi

log "Installing topic_based_ros2_control deps (rosdep)"
sudo apt-get update
rosdep update
rosdep install --from-paths "${ISAAC_ROS_WS}/src/topic_based_ros2_control" --ignore-src -y

log "Building topic_based_ros2_control"
cd "${ISAAC_ROS_WS}"
colcon_build --symlink-install --packages-up-to topic_based_ros2_control
sync_topic_based_ros2_control_headers

prune_incomplete_overlay_prefixes
source_if_exists "${COLCON_INSTALL_BASE}/setup.bash" || true

#touch "${MARKER_FILE}"
log "Done."
