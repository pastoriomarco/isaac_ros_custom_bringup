#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[isaac-manipulation-build] $*"
}

ISAAC_ROS_WS="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}"

SRC_DIR=""
if [[ -d "${ISAAC_ROS_WS}/src" ]]; then
  SRC_DIR="${ISAAC_ROS_WS}/src"
elif [[ -d "${ISAAC_ROS_WS}/ros_ws/src" ]]; then
  SRC_DIR="${ISAAC_ROS_WS}/ros_ws/src"
else
  SRC_DIR="${ISAAC_ROS_WS}/src"
fi

INSTALL_SETUP="${ISAAC_ROS_WS}/install/setup.bash"
FORCE_BUILD="${ISAAC_ROS_MANIPULATION_FORCE_BUILD:-0}"

REPOS_FILE="/usr/local/share/isaac-manipulation/isaac_ros_manipulation.repos"

if [[ -d "${HOME}/.gitconfig" ]]; then
  fallback_gitconfig="${HOME}/.config/git/config"
  if [[ -f "${fallback_gitconfig}" ]]; then
    export GIT_CONFIG_GLOBAL="${fallback_gitconfig}"
    log "WARNING: ${HOME}/.gitconfig is a directory; using ${fallback_gitconfig} via GIT_CONFIG_GLOBAL."
  else
    export GIT_CONFIG_GLOBAL="/dev/null"
    log "WARNING: ${HOME}/.gitconfig is a directory; disabling global git config with GIT_CONFIG_GLOBAL=/dev/null."
  fi
fi

if ! command -v vcs >/dev/null 2>&1; then
  echo "ERROR: vcstool not found. Install python3-vcstool in the image." >&2
  exit 1
fi

resolve_latest_minor_from_ngc() {
  python3 - <<'PY'
import json
import re
import urllib.request

major = 4
org = "nvidia"
team = "isaac"
resource = "isaac_ros_ess_assets"
url = (
    "https://catalog.ngc.nvidia.com/api/resources/versions"
    f"?orgName={org}&teamName={team}&name={resource}"
    "&isPublic=true&pageNumber=0&pageSize=100&sortOrder=CREATED_DATE_DESC"
)

with urllib.request.urlopen(url, timeout=30) as resp:
    data = json.loads(resp.read().decode("utf-8"))

best = None  # (major, minor, patch)
for entry in data.get("recipeVersions", []):
    version = entry.get("versionId", "")
    m = re.match(r"^(\\d+)\\.(\\d+)\\.(\\d+)$", version)
    if not m:
        continue
    mj, mn, pt = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if mj != major:
        continue
    tup = (mj, mn, pt)
    if best is None or tup > best:
        best = tup

if best is None:
    raise SystemExit(2)

print(best[1])
PY
}

resolve_target_minor() {
  local use_latest="${ISAAC_ROS_MANIPULATION_USE_LATEST_MINOR:-0}"
  local minor="${ISAAC_ROS_MANIPULATION_TARGET_MINOR:-1}"

  if [[ "${use_latest}" == "1" ]]; then
    log "Resolving latest Isaac ROS 4.x minor from NGC."
    minor="$(resolve_latest_minor_from_ngc)"
  fi

  if [[ -z "${minor}" || ! "${minor}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid minor version: ${minor} (expected integer)" >&2
    exit 2
  fi

  echo "${minor}"
}

render_repos_for_minor() {
  local minor="$1"
  local target_version="release-4.${minor}"

  if [[ ! -f "${REPOS_FILE}" ]]; then
    echo "ERROR: repos file not found at ${REPOS_FILE}" >&2
    exit 1
  fi

  python3 - "${REPOS_FILE}" "${target_version}" <<'PY'
import sys

repos_path = sys.argv[1]
target_version = sys.argv[2]

current = None
out_lines: list[str] = []

with open(repos_path, "r", encoding="utf-8") as f:
    for line in f:
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if indent == 2 and stripped.rstrip().endswith(":") and not stripped.strip().startswith("#"):
            current = stripped.rstrip()[:-1].strip()
        if current == "isaac_manipulator" and stripped.strip().startswith("version:"):
            prefix = line.split("version:", 1)[0] + "version: "
            line = prefix + target_version + "\n"
        out_lines.append(line)

sys.stdout.write("".join(out_lines))
PY
}

import_missing_repos() {
  local repos_yaml="$1"

  local tmp_missing
  tmp_missing="$(mktemp)"
  python3 - "${repos_yaml}" "${SRC_DIR}" > "${tmp_missing}" <<'PY'
import os
import sys

repos_path = sys.argv[1]
src_dir = sys.argv[2]

def parse_scalar(raw: str):
    raw = raw.strip()
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw

repos = {}
current = None

with open(repos_path, "r", encoding="utf-8") as f:
    for original in f:
        line = original.rstrip("\n")
        stripped = line.lstrip(" ")
        if not stripped or stripped.startswith("#"):
            continue
        if "#" in stripped:
            stripped = stripped.split("#", 1)[0].rstrip()
            if not stripped:
                continue
        indent = len(line) - len(stripped)
        if indent == 0 and stripped.startswith("repositories:"):
            continue
        if indent == 2 and stripped.endswith(":"):
            current = stripped[:-1].strip()
            repos[current] = {}
            continue
        if indent >= 4 and current and ":" in stripped:
            key, rest = stripped.split(":", 1)
            repos[current][key.strip()] = parse_scalar(rest.strip())

missing = {}
for name, meta in repos.items():
    repo_path = os.path.join(src_dir, name)
    if not os.path.isdir(repo_path):
        missing[name] = meta

if not missing:
    raise SystemExit(0)

print("repositories:")
for name, meta in missing.items():
    print(f"  {name}:")
    for key in ("type", "url", "version"):
        if key in meta:
            print(f"    {key}: {meta[key]}")
PY

  if [[ -s "${tmp_missing}" ]]; then
    log "Cloning missing repositories."
    vcs import "${SRC_DIR}" < "${tmp_missing}"
  fi
  rm -f "${tmp_missing}"
}

mkdir -p "${SRC_DIR}"

minor="$(resolve_target_minor)"
log "Target Isaac ROS minor: 4.${minor}"

tmp_repos="$(mktemp)"
render_repos_for_minor "${minor}" > "${tmp_repos}"

if [[ -z "$(ls -A "${SRC_DIR}" 2>/dev/null || true)" ]]; then
  log "Workspace source tree is empty; importing repos."
  vcs import "${SRC_DIR}" < "${tmp_repos}"
else
  import_missing_repos "${tmp_repos}"
fi

rm -f "${tmp_repos}"

if [[ "${ISAAC_ROS_MANIPULATION_PULL_REPOS:-0}" == "1" ]]; then
  log "Updating repositories (vcs pull)."
  vcs pull "${SRC_DIR}"
fi

if command -v git-lfs >/dev/null 2>&1; then
  git lfs install --local >/dev/null 2>&1 || true
  while IFS= read -r -d '' attrs; do
    if grep -q 'filter=lfs' "${attrs}"; then
      repo_root="$(git -C "$(dirname "${attrs}")" rev-parse --show-toplevel 2>/dev/null || true)"
      if [[ -n "${repo_root}" ]]; then
        git -C "${repo_root}" lfs pull || true
      fi
    fi
  done < <(find "${SRC_DIR}" -name .gitattributes -print0)
fi

if find "${SRC_DIR}" -name .gitmodules -print -quit | grep -q .; then
  while IFS= read -r -d '' gm; do
    repo_dir="$(dirname "${gm}")"
    git -C "${repo_dir}" submodule update --init --recursive
  done < <(find "${SRC_DIR}" -name .gitmodules -print0)
fi

if [[ "${FORCE_BUILD}" != "1" && -f "${INSTALL_SETUP}" ]]; then
  log "install/setup.bash already exists; skipping colcon build."
  set +u
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
  # shellcheck disable=SC1090
  source "${INSTALL_SETUP}"
  set -u
  exit 0
fi

set +u
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
set -u

if [[ "${ISAAC_MANIPULATION_SKIP_ASSET_INSTALL:-0}" == "1" ]]; then
  export MANIPULATOR_INSTALL_ASSETS=0
else
  export MANIPULATOR_INSTALL_ASSETS=1
fi

cd "${ISAAC_ROS_WS}"

log "Building isaac_manipulator_bringup (MANIPULATOR_INSTALL_ASSETS=${MANIPULATOR_INSTALL_ASSETS})"
colcon build --symlink-install --packages-up-to isaac_manipulator_bringup --cmake-args -DBUILD_TESTING=OFF

if colcon list --names-only 2>/dev/null | grep -Eq '^(robotiq_driver|robotiq_controllers|robotiq_description|robotiq_hardware_tests|serial)$'; then
  log "Building Robotiq + serial packages"
  colcon build --symlink-install --packages-select-regex 'robotiq.*|serial' --cmake-args -DBUILD_TESTING=OFF
fi

if colcon list --names-only 2>/dev/null | grep -qx 'topic_based_ros2_control'; then
  log "Building topic_based_ros2_control"
  colcon build --symlink-install --packages-up-to topic_based_ros2_control --cmake-args -DBUILD_TESTING=OFF
fi

log "Build complete."

