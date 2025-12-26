#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[apply-isaac-manipulation-fixes] $*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXES_DIR="${SCRIPT_DIR}/fixes"
WORKSPACE_ROOT="${ISAAC_ROS_WS:-$(cd "${SCRIPT_DIR}/../../../../" && pwd)}"

if [[ ! -d "${WORKSPACE_ROOT}/src" ]]; then
  echo "ERROR: Workspace src directory not found at ${WORKSPACE_ROOT}/src" >&2
  exit 1
fi

if [[ ! -d "${FIXES_DIR}" ]]; then
  echo "ERROR: Fixes directory not found at ${FIXES_DIR}" >&2
  exit 1
fi

files=(
  "src/isaac_manipulator/isaac_manipulator_ros_python_utils/isaac_manipulator_ros_python_utils/robot_description_utils.py"
  "src/isaac_manipulator/isaac_manipulator_robot_description/urdf/ur10e_robotiq_base_sim.ros2_control.xacro"
  "src/isaac_manipulator/isaac_manipulator_robot_description/urdf/ur_sim.urdf.xacro"
  "src/isaac_manipulator/isaac_manipulator_robot_description/urdf/ur_sim_macro.xacro"
  "src/isaac_ros_cumotion/isaac_ros_cumotion_examples/ur_config/ur.ros2_control.xacro"
)

log "Applying fixes into ${WORKSPACE_ROOT}."
for rel_path in "${files[@]}"; do
  src_path="${FIXES_DIR}/${rel_path}"
  dst_path="${WORKSPACE_ROOT}/${rel_path}"
  if [[ ! -f "${src_path}" ]]; then
    echo "ERROR: Missing fixes file ${src_path}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${dst_path}")"
  cp -a "${src_path}" "${dst_path}"
done

if ! command -v colcon >/dev/null 2>&1; then
  echo "ERROR: colcon not found in PATH; cannot build packages." >&2
  exit 1
fi

log "Building affected packages."
(
  cd "${WORKSPACE_ROOT}"
  colcon build --packages-select \
    isaac_manipulator_ros_python_utils \
    isaac_manipulator_robot_description \
    isaac_ros_cumotion_examples
)

log "Done."
