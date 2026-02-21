#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[isaac-manipulation-bootstrap] $*"
}

ISAAC_ROS_WS="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}"
FORCE="${ISAAC_ROS_4_1_BOOTSTRAP_FORCE:-0}"
#MARKER_FILE="${ISAAC_ROS_WS}/.isaac_ros_4_1_bootstrap_done"

#if [[ "${FORCE}" != "1" && -f "${MARKER_FILE}" ]]; then
#  log "Already bootstrapped (${MARKER_FILE}); skipping."
#  exit 0
#fi

if [[ ! -d "${ISAAC_ROS_WS}" ]]; then
  echo "ERROR: ISAAC_ROS_WS does not exist: ${ISAAC_ROS_WS}" >&2
  exit 1
fi

mkdir -p "${ISAAC_ROS_WS}/src"

if [[ -f "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
  set -u
else
  echo "ERROR: ROS setup file not found: /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" >&2
  exit 1
fi

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

log "Cloning Robotiq + serial (if missing)"
cd "${ISAAC_ROS_WS}/src"
if [[ ! -d "${ISAAC_ROS_WS}/src/ros2_robotiq_gripper" ]]; then
  git clone --recursive https://github.com/NVIDIA-ISAAC-ROS/ros2_robotiq_gripper
fi
if [[ ! -d "${ISAAC_ROS_WS}/src/serial" ]]; then
  git clone -b ros2 https://github.com/tylerjw/serial
fi

log "Building Robotiq + serial"
cd "${ISAAC_ROS_WS}"
colcon build --symlink-install --packages-select-regex 'robotiq.*' serial --cmake-args -DBUILD_TESTING=OFF

if [[ -f "${ISAAC_ROS_WS}/install/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${ISAAC_ROS_WS}/install/setup.bash"
  set -u
fi

log "Cloning topic_based_ros2_control (if missing)"
cd "${ISAAC_ROS_WS}/src"
if [[ ! -d "${ISAAC_ROS_WS}/src/topic_based_ros2_control" ]]; then
  git clone https://github.com/karanchahal-nv/topic_based_ros2_control
fi

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
colcon build --symlink-install --packages-up-to topic_based_ros2_control

if [[ -f "${ISAAC_ROS_WS}/install/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${ISAAC_ROS_WS}/install/setup.bash"
  set -u
fi

#touch "${MARKER_FILE}"
log "Done."

