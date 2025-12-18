#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: prefetch_quickstart_assets_host.sh [--ws <path>] [--major <N>] [--minor <N>|--latest-minor] [--force]

Downloads Isaac ROS "quickstart.tar.gz" asset bundles from NGC and extracts them into:
  ${ISAAC_ROS_WS}/isaac_ros_assets

This is a host-side helper intended to be run *before* entering the dev container / running `isaac-ros activate`.

Options:
  --ws <path>     Workspace root (defaults to $ISAAC_ROS_WS if set; otherwise inferred from script location)
  --major <N>     Isaac ROS major version to match (default: 4)
  --minor <N>     Isaac ROS minor version upper-bound (default: 0). Selects latest X.Y.Z with Y<=minor.
  --latest-minor  Select latest X.Y.Z for the given major (no minor upper-bound).
  --force         Re-download even if the asset directory already exists
  -h, --help      Show this help

Requires: curl, jq, tar, python3
EOF
}

need_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: missing required command: ${cmd}" >&2
    exit 127
  fi
}

infer_ws_from_script() {
  # Expected script location:
  #   <ISAAC_ROS_WS>/src/isaac_ros_custom_bringup/isaac_ros_4/scripts/this_script.sh
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  (cd "${script_dir}/../../../.." && pwd)
}

default_config_from_script() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  echo "${script_dir}/../config/isaac_manipulation_assets.yaml"
}

component_enabled() {
  local cfg_path="$1"
  local component="$2"
  python3 - "${cfg_path}" "${component}" <<'PY'
import sys

cfg_path = sys.argv[1]
component = sys.argv[2]

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
    # Minimal YAML subset parser: supports nested mappings with 2-space indentation and scalar values.
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
            raise ValueError(f"unsupported indentation (expected multiples of 2): {original!r}")
        if ":" not in stripped:
            raise ValueError(f"expected 'key: value' mapping entry: {original!r}")
        key, rest = stripped.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"invalid indentation: {original!r}")
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
    print("1")
    raise SystemExit(0)
except Exception as e:
    print(f"ERROR: failed to parse YAML config: {cfg_path}: {e}", file=sys.stderr)
    raise SystemExit(2)

components = cfg.get("components", {})
value = components.get(component, False)

if isinstance(value, bool):
    enabled = value
elif isinstance(value, dict):
    enabled = bool(value.get("enabled", True))
else:
    enabled = True

print("1" if enabled else "0")
PY
}

config_get_str() {
  local cfg_path="$1"
  local dotted_key="$2"
  local default_value="$3"
  python3 - "${cfg_path}" "${dotted_key}" "${default_value}" <<'PY'
import sys

cfg_path, dotted_key, default_value = sys.argv[1:4]

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
    # Minimal YAML subset parser: supports nested mappings with 2-space indentation and scalar values.
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
            raise ValueError(f"unsupported indentation (expected multiples of 2): {original!r}")
        if ":" not in stripped:
            raise ValueError(f"expected 'key: value' mapping entry: {original!r}")
        key, rest = stripped.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"invalid indentation: {original!r}")
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
    print(default_value)
    raise SystemExit(0)
except Exception as e:
    print(f"ERROR: failed to parse YAML config: {cfg_path}: {e}", file=sys.stderr)
    raise SystemExit(2)

cur = cfg
for part in dotted_key.split("."):
    if isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        print(default_value)
        raise SystemExit(0)

if cur is None:
    print(default_value)
elif isinstance(cur, bool):
    print("1" if cur else "0")
else:
    print(str(cur))
PY
}

WS="${ISAAC_ROS_WS:-}"
MAJOR_VERSION="4"
MINOR_VERSION="0"
FORCE="0"
MAJOR_SET="0"
MINOR_MODE_SET="0" # minor|latest-minor

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ws)
      WS="${2:-}"
      shift 2
      ;;
    --major)
      MAJOR_VERSION="${2:-}"
      MAJOR_SET="1"
      shift 2
      ;;
    --minor)
      MINOR_VERSION="${2:-}"
      MINOR_MODE_SET="1"
      shift 2
      ;;
    --latest-minor)
      MINOR_VERSION=""
      MINOR_MODE_SET="1"
      shift
      ;;
    --force)
      FORCE="1"
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

need_cmd curl
need_cmd jq
need_cmd tar
need_cmd python3

if [[ -z "${WS}" ]]; then
  WS="$(infer_ws_from_script)"
fi

if [[ ! -d "${WS}" ]]; then
  echo "ERROR: workspace directory not found: ${WS}" >&2
  exit 1
fi

ASSETS_DIR="${WS}/isaac_ros_assets"
mkdir -p "${ASSETS_DIR}"

NGC_ORG="nvidia"
NGC_TEAM="isaac"
NGC_FILENAME="quickstart.tar.gz"

select_latest_version_id() {
  local available_versions_json="$1"
  if [[ -z "${MINOR_VERSION}" ]]; then
    echo "${available_versions_json}" | jq -r \
      --argjson major "${MAJOR_VERSION}" \
      '
        .recipeVersions[]
        | .versionId as $v
        | ($v | capture("^(?<major>[0-9]+)\\.(?<minor>[0-9]+)\\.(?<patch>[0-9]+)$")?) as $ver
        | select($ver != null)
        | select(($ver.major | tonumber) == $major)
        | $v
      ' | sort -V | tail -n 1
    return 0
  fi

  echo "${available_versions_json}" | jq -r \
    --argjson major "${MAJOR_VERSION}" \
    --argjson minor "${MINOR_VERSION}" \
    '
      .recipeVersions[]
      | .versionId as $v
      | ($v | capture("^(?<major>[0-9]+)\\.(?<minor>[0-9]+)\\.(?<patch>[0-9]+)$")?) as $ver
      | select($ver != null)
      | select(($ver.major | tonumber) == $major and ($ver.minor | tonumber) <= $minor)
      | $v
    ' | sort -V | tail -n 1
}

prefetch_bundle() {
  local package_name="$1"
  local ngc_resource="$2"

  local sentinel_dir="${ASSETS_DIR}/${package_name}"
  if [[ "${FORCE}" != "1" ]] && [[ -d "${sentinel_dir}" ]]; then
    echo "==> ${package_name}: already present at ${sentinel_dir} (skipping)"
    return 0
  fi

  echo "==> ${package_name}: resolving NGC version for ${ngc_resource}"
  local version_req_url="https://catalog.ngc.nvidia.com/api/resources/versions?orgName=${NGC_ORG}&teamName=${NGC_TEAM}&name=${ngc_resource}&isPublic=true&pageNumber=0&pageSize=100&sortOrder=CREATED_DATE_DESC"
  local available_versions
  available_versions="$(curl -fsSL -H "Accept: application/json" "${version_req_url}")"

  local latest_version_id
  latest_version_id="$(select_latest_version_id "${available_versions}")"

  if [[ -z "${latest_version_id}" ]] || [[ "${latest_version_id}" == "null" ]]; then
    if [[ -z "${MINOR_VERSION}" ]]; then
      echo "ERROR: ${package_name}: no corresponding version found for Isaac ROS ${MAJOR_VERSION}.*" >&2
    else
      echo "ERROR: ${package_name}: no corresponding version found for Isaac ROS ${MAJOR_VERSION}.${MINOR_VERSION}" >&2
    fi
    echo "Found versions:" >&2
    echo "${available_versions}" | jq -r '.recipeVersions[].versionId' >&2 || true
    return 1
  fi

  local file_req_url="https://api.ngc.nvidia.com/v2/resources/${NGC_ORG}/${NGC_TEAM}/${ngc_resource}/versions/${latest_version_id}/files/${NGC_FILENAME}"
  echo "==> ${package_name}: downloading ${NGC_FILENAME} (${latest_version_id})"

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  (
    trap 'rm -rf "${tmp_dir}"' EXIT

    curl -fL --retry 3 --retry-delay 2 -o "${tmp_dir}/${NGC_FILENAME}" "${file_req_url}"

    echo "==> ${package_name}: extracting into ${ASSETS_DIR}"
    tar -xf "${tmp_dir}/${NGC_FILENAME}" -C "${ASSETS_DIR}"
  )

  if [[ ! -d "${sentinel_dir}" ]]; then
    echo "WARNING: ${package_name}: extracted but expected directory not found: ${sentinel_dir}" >&2
    echo "         (Tarball layout may have changed; verify extracted contents under: ${ASSETS_DIR})" >&2
  fi
}

CFG_PATH="$(default_config_from_script)"
if [[ ! -f "${CFG_PATH}" ]]; then
  echo "ERROR: expected config file not found: ${CFG_PATH}" >&2
  exit 1
fi
echo "Using config: ${CFG_PATH}"

# Apply versioning from config unless explicitly overridden by CLI args.
if [[ "${MAJOR_SET}" != "1" ]]; then
  cfg_major="$(config_get_str "${CFG_PATH}" "versioning.quickstart_assets.major" "")"
  if [[ -n "${cfg_major}" ]]; then
    MAJOR_VERSION="${cfg_major}"
  fi
fi

if [[ "${MINOR_MODE_SET}" != "1" ]]; then
  cfg_mode="$(config_get_str "${CFG_PATH}" "versioning.quickstart_assets.mode" "")"
  case "${cfg_mode}" in
    latest)
      MINOR_VERSION=""
      ;;
    pinned_minor)
      cfg_minor="$(config_get_str "${CFG_PATH}" "versioning.quickstart_assets.minor" "")"
      if [[ -n "${cfg_minor}" ]]; then
        MINOR_VERSION="${cfg_minor}"
      fi
      ;;
    "")
      ;;
    *)
      echo "ERROR: unsupported versioning.quickstart_assets.mode in config: ${cfg_mode} (expected: latest|pinned_minor)" >&2
      exit 2
      ;;
  esac
fi

maybe_prefetch() {
  local component="$1"
  local package_name="$2"
  local ngc_resource="$3"

  if [[ "$(component_enabled "${CFG_PATH}" "${component}")" == "1" ]]; then
    prefetch_bundle "${package_name}" "${ngc_resource}"
  else
    echo "==> ${package_name}: disabled by config (${component}) (skipping)"
  fi
}

maybe_prefetch "foundationpose" "isaac_ros_foundationpose" "isaac_ros_foundationpose_assets"
maybe_prefetch "ess" "isaac_ros_ess" "isaac_ros_ess_assets"
maybe_prefetch "rtdetr" "isaac_ros_rtdetr" "isaac_ros_rtdetr_assets"
maybe_prefetch "foundationstereo" "isaac_ros_foundationstereo" "isaac_ros_foundationstereo_assets"
maybe_prefetch "grounding_dino" "isaac_ros_grounding_dino" "isaac_ros_grounding_dino_assets"

echo
echo "Done. Quickstart assets live under: ${ASSETS_DIR}"
