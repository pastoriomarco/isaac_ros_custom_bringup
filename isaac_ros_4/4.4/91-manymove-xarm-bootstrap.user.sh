#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[manymove-xarm-bootstrap] $*"
}

if [[ "${ISAAC_ROS_4_4_BOOTSTRAP:-1}" != "1" ]]; then
  log "Skipping ManyMove/xArm bootstrap because ISAAC_ROS_4_4_BOOTSTRAP=${ISAAC_ROS_4_4_BOOTSTRAP}."
  exit 0
fi

colcon_build() {
  colcon \
    --log-base "${COLCON_LOG_BASE}" \
    build \
    --build-base "${COLCON_BUILD_BASE}" \
    --install-base "${COLCON_INSTALL_BASE}" \
    "$@"
}

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

configure_parallelism() {
  local requested="${MANYMOVE_COLCON_WORKERS:-}"
  if [[ -z "${requested}" ]]; then
    return
  fi
  if ! [[ "${requested}" =~ ^[0-9]+$ ]] || [[ "${requested}" -le 0 ]]; then
    log "Ignoring MANYMOVE_COLCON_WORKERS='${requested}' because it is not a positive integer."
    return
  fi
  export MANYMOVE_PARALLEL_LEVEL="${requested}"
  export CMAKE_BUILD_PARALLEL_LEVEL="${requested}"
  log "Limiting build parallelism to ${requested} worker(s)."
}

write_head_marker() {
  local repo_dir="$1"
  local marker_name="$2"
  local head=""

  head="$(git -C "${repo_dir}" rev-parse HEAD 2>/dev/null || true)"
  if [[ -n "${head}" ]]; then
    printf '%s\n' "${head}" > "${ISAAC_ROS_WS}/${marker_name}"
  fi
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

ensure_xarm_repo() {
  local just_cloned=false
  local repo_name
  local current_remote=""
  local current_branch=""

  repo_name="$(basename "${XARM_SOURCE_DIR}")"

  if [[ ! -d "${XARM_SOURCE_DIR}" ]]; then
    log "Cloning xarm_ros2 from ${XARM_REPO_URL} (branch ${XARM_BRANCH_NAME})."
    mkdir -p "$(dirname "${XARM_SOURCE_DIR}")"
    git -C "$(dirname "${XARM_SOURCE_DIR}")" clone --recursive --branch "${XARM_BRANCH_NAME}" "${XARM_REPO_URL}" "${repo_name}"
    just_cloned=true
  else
    log "xarm_ros2 already present at ${XARM_SOURCE_DIR}; skipping clone."
  fi

  if [[ ! -e "${XARM_SOURCE_DIR}/.git" ]]; then
    echo "ERROR: ${XARM_SOURCE_DIR} exists but is not a git repository." >&2
    exit 1
  fi

  current_remote="$(git -C "${XARM_SOURCE_DIR}" remote get-url origin 2>/dev/null || true)"
  current_branch="$(git -C "${XARM_SOURCE_DIR}" branch --show-current 2>/dev/null || true)"
  if [[ -n "${current_remote}" && "${current_remote}" != "${XARM_REPO_URL}" ]]; then
    log "xarm_ros2 remote is ${current_remote}; expected ${XARM_REPO_URL}. Leaving the existing checkout unchanged."
  fi
  if [[ -n "${current_branch}" && "${current_branch}" != "${XARM_BRANCH_NAME}" ]]; then
    log "xarm_ros2 branch is ${current_branch}; expected ${XARM_BRANCH_NAME}. Leaving the existing checkout unchanged."
  fi

  if [[ "${just_cloned}" == true && -n "${XARM_PINNED_COMMIT}" ]]; then
    log "Checking out xarm_ros2 commit ${XARM_PINNED_COMMIT}."
    git -C "${XARM_SOURCE_DIR}" fetch --depth 1 origin "${XARM_PINNED_COMMIT}"
    git -C "${XARM_SOURCE_DIR}" checkout --detach "${XARM_PINNED_COMMIT}"
  elif [[ -n "${XARM_PINNED_COMMIT}" ]]; then
    log "xarm_ros2 commit override requested (${XARM_PINNED_COMMIT}) but repository already exists; leaving checkout unchanged."
  fi

  log "Updating xarm_ros2 submodules."
  git -C "${XARM_SOURCE_DIR}" submodule sync --recursive
  git -C "${XARM_SOURCE_DIR}" submodule update --init --recursive
  write_head_marker "${XARM_SOURCE_DIR}" ".xarm_ros2_commit"
}

ensure_groot() {
  local just_cloned=false
  local repo_name

  repo_name="$(basename "${GROOT_SOURCE_DIR}")"

  if [[ ! -d "${GROOT_SOURCE_DIR}" ]]; then
    log "Cloning Groot from ${GROOT_REPO_URL} (branch ${GROOT_BRANCH_NAME})."
    mkdir -p "$(dirname "${GROOT_SOURCE_DIR}")"
    git -C "$(dirname "${GROOT_SOURCE_DIR}")" clone --recurse-submodules --branch "${GROOT_BRANCH_NAME}" "${GROOT_REPO_URL}" "${repo_name}"
    just_cloned=true
  else
    log "Groot already present at ${GROOT_SOURCE_DIR}; skipping clone."
  fi

  if [[ ! -e "${GROOT_SOURCE_DIR}/.git" ]]; then
    echo "ERROR: ${GROOT_SOURCE_DIR} exists but is not a git repository." >&2
    exit 1
  fi

  if [[ "${just_cloned}" == true && -n "${GROOT_PINNED_COMMIT}" ]]; then
    log "Checking out Groot commit ${GROOT_PINNED_COMMIT}."
    git -C "${GROOT_SOURCE_DIR}" fetch --depth 1 origin "${GROOT_PINNED_COMMIT}"
    git -C "${GROOT_SOURCE_DIR}" checkout --detach "${GROOT_PINNED_COMMIT}"
  elif [[ -n "${GROOT_PINNED_COMMIT}" ]]; then
    log "Groot commit override requested (${GROOT_PINNED_COMMIT}) but repository already exists; leaving checkout unchanged."
  fi

  log "Updating Groot submodules."
  git -C "${GROOT_SOURCE_DIR}" submodule update --init --recursive

  log "Configuring Groot."
  cmake -S "${GROOT_SOURCE_DIR}" -B "${GROOT_SOURCE_DIR}/build"

  log "Building Groot."
  cmake --build "${GROOT_SOURCE_DIR}/build" --parallel
  write_head_marker "${GROOT_SOURCE_DIR}" ".groot_commit"
}

ISAAC_ROS_WS="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
SRC_DIR="${ISAAC_ROS_WS}/src"
MANYMOVE_DIR="${SRC_DIR}/manymove"
ISAAC_ROS_CUSTOM_BRINGUP_DIR="${SRC_DIR}/isaac_ros_custom_bringup"
COLCON_BUILD_BASE="${ISAAC_ROS_COLCON_BUILD_BASE:-${ISAAC_ROS_WS}/build}"
COLCON_LOG_BASE="${ISAAC_ROS_COLCON_LOG_BASE:-${ISAAC_ROS_WS}/log}"
COLCON_INSTALL_BASE="${ISAAC_ROS_COLCON_INSTALL_BASE:-${ISAAC_ROS_WS}/install}"
XARM_SOURCE_DIR="${XARM_SOURCE_DIR:-${SRC_DIR}/xarm_ros2}"
XARM_REPO_URL="${XARM_REPO:-https://github.com/pastoriomarco/xarm_ros2.git}"
XARM_BRANCH_NAME="${XARM_BRANCH:-${ROS_DISTRO}_no_gazebo}"
XARM_PINNED_COMMIT="${XARM_COMMIT:-}"
GROOT_SOURCE_DIR="${GROOT_SOURCE_DIR:-${SRC_DIR}/Groot}"
GROOT_REPO_URL="${GROOT_REPO:-https://github.com/pastoriomarco/Groot.git}"
GROOT_BRANCH_NAME="${GROOT_BRANCH:-master}"
GROOT_PINNED_COMMIT="${GROOT_COMMIT:-}"
export MANYMOVE_PARALLEL_LEVEL="${MANYMOVE_PARALLEL_LEVEL:-}"
ROSDEP_SKIP_KEYS="${ROSDEP_SKIP_KEYS:-sdformat14 gz-sim8 gz_ros2_control ros_gz_bridge ros_gz_sim}"

if [[ ! -d "${ISAAC_ROS_WS}" ]]; then
  echo "ERROR: ISAAC_ROS_WS does not exist: ${ISAAC_ROS_WS}" >&2
  exit 1
fi

mkdir -p "${SRC_DIR}" "${COLCON_BUILD_BASE}" "${COLCON_LOG_BASE}" "${COLCON_INSTALL_BASE}"

if [[ ! -d "${MANYMOVE_DIR}" ]]; then
  echo "ERROR: ManyMove workspace sources not found at ${MANYMOVE_DIR}" >&2
  exit 1
fi

if [[ ! -d "${ISAAC_ROS_CUSTOM_BRINGUP_DIR}" ]]; then
  echo "ERROR: isaac_ros_custom_bringup not found at ${ISAAC_ROS_CUSTOM_BRINGUP_DIR}" >&2
  exit 1
fi

if ! source_if_exists "/opt/ros/${ROS_DISTRO}/setup.bash"; then
  echo "ERROR: ROS setup file not found: /opt/ros/${ROS_DISTRO}/setup.bash" >&2
  exit 1
fi

configure_parallelism
ensure_xarm_repo
ensure_groot
prune_incomplete_overlay_prefixes
source_if_exists "${COLCON_INSTALL_BASE}/setup.bash" || true

if [[ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ]]; then
  log "Initializing rosdep."
  sudo rosdep init
fi

log "Refreshing apt package lists."
sudo apt-get update

log "Updating rosdep indexes."
if ! rosdep update; then
  log "rosdep update failed; continuing with the existing rosdep cache."
fi

log "Installing ManyMove/xArm system dependencies with rosdep."
rosdep install \
  --from-paths \
    "${MANYMOVE_DIR}" \
    "${XARM_SOURCE_DIR}" \
    "${ISAAC_ROS_CUSTOM_BRINGUP_DIR}" \
  --ignore-src \
  --rosdistro "${ROS_DISTRO}" \
  --skip-keys "${ROSDEP_SKIP_KEYS}" \
  -y \
  -r

BUILD_ARGS=(
  --symlink-install
  --packages-select
  isaac_ros_custom_bringup
  manymove_msgs
  manymove_object_manager
  manymove_planner
  manymove_cpp_trees
  manymove_hmi
  manymove_bringup
  uf_ros_lib
  xarm_msgs
  xarm_sdk
  xarm_description
  xarm_api
  xarm_controller
  xarm_moveit_config
)

if [[ -n "${MANYMOVE_PARALLEL_LEVEL}" ]]; then
  BUILD_ARGS+=(--parallel-workers "${MANYMOVE_PARALLEL_LEVEL}")
fi

log "Building ManyMove/xArm overlay packages."
cd "${ISAAC_ROS_WS}"
colcon_build "${BUILD_ARGS[@]}"

prune_incomplete_overlay_prefixes
source_if_exists "${COLCON_INSTALL_BASE}/setup.bash" || true
log "Done."
