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

mkdir -p "${SRC_DIR}"
clone_missing_repos() {
  local repos_file="/usr/local/share/isaac-manipulation/isaac_ros_manipulation.repos"
  if [[ ! -f "${repos_file}" ]]; then
    echo "WARNING: repos file not found at ${repos_file}; skipping clone." >&2
    return 0
  fi

  local tmp_repos
  tmp_repos="$(mktemp)"
  python3 - "${repos_file}" "${SRC_DIR}" > "${tmp_repos}" <<'PY'
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
    sys.exit(0)

print("repositories:")
for name, meta in missing.items():
    print(f"  {name}:")
    for key in ("type", "url", "version"):
        if key in meta:
            print(f"    {key}: {meta[key]}")
PY

  if [[ -s "${tmp_repos}" ]]; then
    log "Cloning missing repositories."
    vcs import "${SRC_DIR}" < "${tmp_repos}"
  fi
  rm -f "${tmp_repos}"
}

if [[ -z "$(ls -A "${SRC_DIR}" 2>/dev/null || true)" ]]; then
  log "Workspace source tree is empty; importing repos."
  vcs import "${SRC_DIR}" < /usr/local/share/isaac-manipulation/isaac_ros_manipulation.repos
else
  clone_missing_repos
fi

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

if [[ "${ISAAC_MANIPULATION_SKIP_ASSET_INSTALL:-0}" == "1" ]]; then
  ASSET_CMAKE="${SRC_DIR}/isaac_ros_common/isaac_ros_common/cmake/isaac_ros_common-extras-assets.cmake"
  if [[ -f "${ASSET_CMAKE}" ]]; then
    if grep -q 'add_custom_target("${TARGET_NAME}" ALL DEPENDS ${OUTPUT_PATHS})' "${ASSET_CMAKE}"; then
      sed -i 's/add_custom_target("${TARGET_NAME}" ALL DEPENDS ${OUTPUT_PATHS})/add_custom_target("${TARGET_NAME}" DEPENDS ${OUTPUT_PATHS})/' "${ASSET_CMAKE}"
    else
      echo "WARNING: unexpected assets CMake format; asset installs may still run: ${ASSET_CMAKE}" >&2
    fi
  else
    echo "WARNING: assets CMake file not found; asset installs may still run: ${ASSET_CMAKE}" >&2
  fi
else
  ASSET_CMAKE="${SRC_DIR}/isaac_ros_common/isaac_ros_common/cmake/isaac_ros_common-extras-assets.cmake"
  if [[ -f "${ASSET_CMAKE}" ]]; then
    if grep -q 'add_custom_target("${TARGET_NAME}" DEPENDS ${OUTPUT_PATHS})' "${ASSET_CMAKE}"; then
      sed -i 's/add_custom_target("${TARGET_NAME}" DEPENDS ${OUTPUT_PATHS})/add_custom_target("${TARGET_NAME}" ALL DEPENDS ${OUTPUT_PATHS})/' "${ASSET_CMAKE}"
    fi
  fi
fi

if [[ "${FORCE_BUILD}" != "1" && -f "${INSTALL_SETUP}" ]]; then
  log "install/setup.bash already exists; skipping colcon build."
  exit 0
fi

detect_arch_list() {
  local caps
  if command -v nvidia-smi >/dev/null 2>&1; then
    caps="$(
      nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | \
        tr -d ' ' | sort -V | uniq
    )"
  elif python3 - <<'PY' >/dev/null 2>&1
import torch
print(".".join(str(x) for x in torch.cuda.get_device_capability(0)))
PY
  then
    caps="$(python3 - <<'PY'
import torch
cc = torch.cuda.get_device_capability(0)
print(f"{cc[0]}.{cc[1]}")
PY
)"
  fi

  if [[ -z "${caps:-}" ]]; then
    echo "8.6+PTX"
    return 0
  fi

  mapfile -t cap_list < <(printf '%s\n' "${caps}")
  local last_index=$(( ${#cap_list[@]} - 1 ))
  if [[ "${cap_list[${last_index}]}" != *+PTX ]]; then
    cap_list[${last_index}]="${cap_list[${last_index}]}+PTX"
  fi
  local joined
  joined="$(IFS=';'; echo "${cap_list[*]}")"
  echo "${joined}"
}

ARCH_LIST="${ISAAC_MANIPULATION_TORCH_CUDA_ARCH_LIST:-}"
if [[ -z "${ARCH_LIST}" ]]; then
  ARCH_LIST="$(detect_arch_list)"
fi
export TORCH_CUDA_ARCH_LIST="${ARCH_LIST}"

set +u
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
set -u

CFG_PATH="/usr/local/share/isaac-manipulation/isaac_manipulation_assets.yaml"
read_config_flags() {
  python3 - "${CFG_PATH}" <<'PY'
import sys

cfg_path = sys.argv[1]

def parse_scalar(raw: str):
    raw = raw.strip()
    if raw in ("true", "True"):
        return True
    if raw in ("false", "False"):
        return False
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    try:
        if raw.startswith("0") and raw != "0" and raw.isdigit():
            return raw
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    root = {}
    stack = [(0, root)]
    for original in text.splitlines():
        line = original.rstrip("\n")
        stripped = line.lstrip(" ")
        if not stripped or stripped.startswith("#"):
            continue
        if "#" in stripped:
            stripped = stripped.split("#", 1)[0].rstrip()
            if not stripped:
                continue
        indent = len(line) - len(stripped)
        if indent % 2 != 0:
            raise ValueError(f"unsupported indentation: {original!r}")
        if ":" not in stripped:
            raise ValueError(f"expected mapping entry: {original!r}")
        key, rest = stripped.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        cur = stack[-1][1]
        if rest == "":
            nxt = {}
            cur[key] = nxt
            stack.append((indent + 2, nxt))
        else:
            cur[key] = parse_scalar(rest)
    return root

try:
    cfg = load_cfg(cfg_path)
except FileNotFoundError:
    cfg = {}

components = cfg.get("components", {})

def enabled(component: str) -> bool:
    value = components.get(component, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value.get("enabled", True))
    return True

keys = [
    "ess",
    "foundationstereo",
    "foundationpose",
    "rtdetr",
    "grounding_dino",
    "dope",
    "segment_anything",
    "segment_anything2",
    "gear_assembly",
]

for key in keys:
    print(f"{key}={'1' if enabled(key) else '0'}")
PY
}

declare -A COMPONENTS=()
while IFS='=' read -r key value; do
  COMPONENTS["${key}"]="${value}"
done < <(read_config_flags)

targets=(
  isaac_manipulator_bringup
  isaac_manipulator_asset_bringup
  isaac_manipulator_pick_and_place
  isaac_ros_cumotion
  isaac_ros_nvblox
  serial
  robotiq_driver
  robotiq_controllers
  robotiq_description
  robotiq_hardware_tests
)

if [[ "${COMPONENTS[ess]:-0}" == "1" ]]; then
  targets+=(isaac_ros_ess)
fi
if [[ "${COMPONENTS[foundationstereo]:-0}" == "1" ]]; then
  targets+=(isaac_ros_foundationstereo)
fi
if [[ "${COMPONENTS[foundationpose]:-0}" == "1" ]]; then
  targets+=(isaac_ros_foundationpose)
fi
if [[ "${COMPONENTS[rtdetr]:-0}" == "1" ]]; then
  targets+=(isaac_ros_rtdetr)
fi
if [[ "${COMPONENTS[grounding_dino]:-0}" == "1" ]]; then
  targets+=(isaac_ros_grounding_dino)
fi
if [[ "${COMPONENTS[dope]:-0}" == "1" ]]; then
  targets+=(isaac_ros_dope)
fi
if [[ "${COMPONENTS[segment_anything]:-0}" == "1" ]]; then
  targets+=(isaac_ros_segment_anything)
fi
if [[ "${COMPONENTS[segment_anything2]:-0}" == "1" ]]; then
  targets+=(isaac_ros_segment_anything2)
fi

if [[ "${COMPONENTS[gear_assembly]:-0}" == "1" ]]; then
  if python3 - <<'PY' >/dev/null 2>&1
import importlib
import sys
for mod in ("tensordict", "rsl_rl"):
    try:
        importlib.import_module(mod)
    except Exception:
        sys.exit(1)
sys.exit(0)
PY
  then
    targets+=(isaac_manipulator_gear_assembly isaac_manipulator_ur_dnn_policy)
  else
    echo "WARNING: gear_assembly enabled but RSL-RL deps not installed; skipping gear assembly packages." >&2
  fi
fi

dedupe_targets=()
declare -A seen=()
for t in "${targets[@]}"; do
  if [[ -z "${seen[${t}]:-}" ]]; then
    dedupe_targets+=("${t}")
    seen["${t}"]=1
  fi
done
targets=("${dedupe_targets[@]}")

base_paths=()
while IFS= read -r path; do
  base_paths+=("${path}")
done < <(
  python3 - "${SRC_DIR}" /usr/local/share/isaac-manipulation/isaac_ros_manipulation.repos <<'PY'
import os
import sys

src_dir = sys.argv[1]
repos_path = sys.argv[2]

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

for name in repos:
    path = os.path.join(src_dir, name)
    if os.path.isdir(path):
        print(path)
PY
)
if [[ ${#base_paths[@]} -eq 0 ]]; then
  base_paths=("${SRC_DIR}")
fi

colcon_args=()
if [[ "${ISAAC_ROS_MANIPULATION_SYMLINK_INSTALL:-1}" == "1" ]]; then
  colcon_args+=(--symlink-install)
fi
colcon_args+=(--cmake-args -DBUILD_TESTING=OFF)

log "Building targets: ${targets[*]}"
log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
cd "${ISAAC_ROS_WS}"
colcon build \
  "${colcon_args[@]}" \
  --packages-up-to "${targets[@]}" \
  --base-paths "${base_paths[@]}"

log "Build complete."
