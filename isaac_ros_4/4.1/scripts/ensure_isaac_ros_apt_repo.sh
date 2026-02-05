#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ensure_isaac_ros_apt_repo.sh [--suite auto|noble|noble-jetpack] [--list-file <path>] [--check-package <apt_pkg>]

Ensures the NVIDIA Isaac ROS apt source list is usable on the current platform.

Motivation:
  On some Jetson/JetPack-based setups the Isaac ROS apt repo needs the `noble-jetpack` suite
  (instead of `noble`) for packages like `ros-jazzy-isaac-ros-common` to be discoverable.

Behavior:
  - In `--suite auto` mode (default), runs `apt-get update` and checks availability of the given package.
    If unavailable, it will try switching `noble` <-> `noble-jetpack` and re-check.
  - In explicit suite mode, it forces that suite and verifies the package is available.

Defaults:
  --list-file      /etc/apt/sources.list.d/nvidia-isaac-ros.list
  --check-package  ros-${ROS_DISTRO:-jazzy}-isaac-ros-common
EOF
}

SUITE="auto"
LIST_FILE="/etc/apt/sources.list.d/nvidia-isaac-ros.list"
ROS_PKG="ros-${ROS_DISTRO:-jazzy}-isaac-ros-common"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      SUITE="${2:-}"
      shift 2
      ;;
    --list-file)
      LIST_FILE="${2:-}"
      shift 2
      ;;
    --check-package)
      ROS_PKG="${2:-}"
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

if [[ ! -f "${LIST_FILE}" ]]; then
  echo "==> Isaac ROS apt list not found at ${LIST_FILE} (skipping)"
  exit 0
fi

apt_update() {
  echo "==> apt-get update"
  apt-get update
}

candidate_version() {
  apt-cache policy "${ROS_PKG}" 2>/dev/null | awk -F': ' '/Candidate:/ {print $2; exit 0}'
}

has_candidate() {
  local cand
  cand="$(candidate_version || true)"
  [[ -n "${cand}" && "${cand}" != "(none)" && "${cand}" != "none" ]]
}

current_suite() {
  # Parse the suite token from the first non-comment "deb ..." line.
  # Format: deb [opts] URL SUITE COMPONENTS...
  python3 - "${LIST_FILE}" <<'PY'
import re
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    for raw in f:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^deb\\s+(?:\\[[^\\]]*\\]\\s+)?\\S+\\s+(\\S+)\\s+.*$", line)
        if m:
            print(m.group(1))
            raise SystemExit(0)
print("")
PY
}

set_suite() {
  local desired="$1"
  if [[ -z "${desired}" ]]; then
    return 0
  fi

  local before
  before="$(current_suite)"
  if [[ "${before}" == "${desired}" ]]; then
    echo "==> apt suite already ${desired}"
    return 0
  fi

  echo "==> Updating Isaac ROS apt suite: ${before:-<unknown>} -> ${desired}"
  # Only rewrite the suite token after the repo URL. Limit replacement to noble variants.
  sed -i -E "s|(^(\\s*deb\\s+(\\[[^\\]]+\\]\\s+)?\\S+\\s+))(noble-jetpack|noble)(\\s+)|\\1${desired}\\6|g" "${LIST_FILE}"
}

verify_or_die() {
  if has_candidate; then
    echo "==> OK: ${ROS_PKG} is available (Candidate: $(candidate_version))"
    return 0
  fi

  echo "ERROR: ${ROS_PKG} not available after configuring ${LIST_FILE}" >&2
  echo "  - Suite: $(current_suite)" >&2
  echo "  - apt-cache policy output:" >&2
  apt-cache policy "${ROS_PKG}" >&2 || true
  return 1
}

case "${SUITE}" in
  auto)
    apt_update
    if has_candidate; then
      echo "==> OK: ${ROS_PKG} is available (Candidate: $(candidate_version))"
      exit 0
    fi

    # Best-effort auto-fix: try the opposite suite.
    cur="$(current_suite)"
    if [[ "${cur}" == "noble" ]]; then
      set_suite "noble-jetpack"
      apt_update
    elif [[ "${cur}" == "noble-jetpack" ]]; then
      set_suite "noble"
      apt_update
    else
      echo "==> Unknown suite in ${LIST_FILE} (${cur}); leaving as-is." >&2
    fi

    verify_or_die
    ;;
  noble|noble-jetpack)
    set_suite "${SUITE}"
    apt_update
    verify_or_die
    ;;
  *)
    echo "ERROR: unsupported --suite: ${SUITE} (expected: auto|noble|noble-jetpack)" >&2
    exit 2
    ;;
esac

