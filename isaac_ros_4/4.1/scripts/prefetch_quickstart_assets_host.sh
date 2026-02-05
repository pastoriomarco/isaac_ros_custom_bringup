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
  --minor <N>     Isaac ROS minor version upper-bound (default: 1). Selects latest X.Y.Z with Y<=minor.
  --latest-minor  Select latest X.Y.Z for the given major (no minor upper-bound).
  --latest        Alias for --latest-minor.
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
  #   <ISAAC_ROS_WS>/src/isaac_ros_custom_bringup/isaac_ros_4/4.1/scripts/this_script.sh
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  (cd "${script_dir}/../../../../.." && pwd)
}

WS="${ISAAC_ROS_WS:-}"
MAJOR_VERSION="4"
MINOR_VERSION="1"
FORCE="0"
MAJOR_SET="0"
MINOR_MODE_SET="0" # minor|latest

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
    --latest)
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

prefetch_bundle "isaac_ros_foundationpose" "isaac_ros_foundationpose_assets"
prefetch_bundle "isaac_ros_ess" "isaac_ros_ess_assets"
prefetch_bundle "isaac_ros_rtdetr" "isaac_ros_rtdetr_assets"
prefetch_bundle "isaac_ros_foundationstereo" "isaac_ros_foundationstereo_assets"
prefetch_bundle "isaac_ros_grounding_dino" "isaac_ros_grounding_dino_assets"

echo
echo "Done. Quickstart assets live under: ${ASSETS_DIR}"
