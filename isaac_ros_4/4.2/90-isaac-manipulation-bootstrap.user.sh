#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[isaac-manipulation-bootstrap] $*"
}

ISAAC_ROS_WS="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}"
FORCE="${ISAAC_ROS_4_2_BOOTSTRAP_FORCE:-0}"
#MARKER_FILE="${ISAAC_ROS_WS}/.isaac_ros_4_2_bootstrap_done"

#if [[ "${FORCE}" != "1" && -f "${MARKER_FILE}" ]]; then
#  log "Already bootstrapped (${MARKER_FILE}); skipping."
#  exit 0
#fi

if [[ ! -d "${ISAAC_ROS_WS}" ]]; then
  echo "ERROR: ISAAC_ROS_WS does not exist: ${ISAAC_ROS_WS}" >&2
  exit 1
fi

mkdir -p "${ISAAC_ROS_WS}/src"

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

clone_if_missing() {
  local target_path="$1"
  shift

  if [[ -e "${target_path}" ]]; then
    log "Skipping clone (already exists): ${target_path}"
    return 0
  fi

  git clone "$@"
}

patch_pip_shim_constraints() {
  local shim_file="/usr/share/isaac-ros-cli/pip_shim_constraints.txt"
  local shim_backup="${shim_file}.bak"

  if [[ ! -f "${shim_file}" ]]; then
    log "Skipping pip shim patch; file not found: ${shim_file}"
    return 0
  fi

  if [[ ! -f "${shim_backup}" ]]; then
    sudo cp "${shim_file}" "${shim_backup}"
  fi

  sudo sed -i \
    -e 's/^tensordict==.*$/tensordict==0.10.0/' \
    -e 's/^warp-lang==.*$/warp-lang==1.9.0/' \
    "${shim_file}"
}

if ! source_if_exists "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"; then
  echo "ERROR: ROS setup file not found: /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" >&2
  exit 1
fi

log "Patching isaac-ros-cli pip shim constraints (tensordict, warp-lang)"
patch_pip_shim_constraints

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
colcon build --symlink-install --packages-select-regex 'robotiq.*' serial --cmake-args -DBUILD_TESTING=OFF

source_if_exists "${ISAAC_ROS_WS}/install/setup.bash" || true

log "Cloning topic_based_ros2_control (if missing)"
cd "${ISAAC_ROS_WS}/src"
clone_if_missing "${ISAAC_ROS_WS}/src/topic_based_ros2_control" https://github.com/karanchahal-nv/topic_based_ros2_control

if [[ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ]]; then
  log "Initializing rosdep (sudo rosdep init)"
  sudo rosdep init
fi

log "Installing topic_based_ros2_control deps (rosdep)"
rosdep update
rosdep install --from-paths "${ISAAC_ROS_WS}/src/topic_based_ros2_control" --ignore-src -y

log "Building topic_based_ros2_control"
cd "${ISAAC_ROS_WS}"
colcon build --symlink-install --packages-up-to topic_based_ros2_control

source_if_exists "${ISAAC_ROS_WS}/install/setup.bash" || true

#touch "${MARKER_FILE}"
log "Done."
