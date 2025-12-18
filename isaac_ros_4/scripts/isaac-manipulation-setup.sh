#!/usr/bin/env bash
set -euo pipefail

show_usage() {
  cat <<'EOF'
Usage: isaac-manipulation-setup.sh [--accept-eula|--eula|--show-eula] [--model-res <low_res|high_res|both>]

Downloads/converts the models + assets required by the Isaac Manipulation tutorials.

Notes:
  - Assets/models are installed into: $ISAAC_ROS_WS/isaac_ros_assets
  - Idempotent: installer scripts skip if outputs already exist.
  - For non-interactive use, pass --accept-eula (or set ISAAC_ROS_ACCEPT_EULA=1).
  - Uses the baked-in config at: /usr/local/share/isaac-manipulation/isaac_manipulation_assets.yaml
EOF
}

MODEL_RES="${FOUNDATIONSTEREO_MODEL_RES:-}"
EULA_MODE="none" # none|accept|show

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_usage
      exit 0
      ;;
    --model-res)
      MODEL_RES="${2:-}"
      shift 2
      ;;
    --accept-eula)
      EULA_MODE="accept"
      shift
      ;;
    --eula|--show-eula)
      EULA_MODE="show"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      show_usage >&2
      exit 2
      ;;
  esac
done

resolve_config_path() {
  echo "/usr/local/share/isaac-manipulation/isaac_manipulation_assets.yaml"
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
value = components.get(component, True)

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
  local component="$2"
  local key="$3"
  local default_value="$4"
  python3 - "${cfg_path}" "${component}" "${key}" "${default_value}" <<'PY'
import sys

cfg_path, component, key, default_value = sys.argv[1:5]

try:
    with open(cfg_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Minimal YAML subset parser: supports nested mappings with 2-space indentation and scalar values.
    root = {}
    stack = [(0, root)]

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
        k, rest = stripped.split(":", 1)
        k = k.strip()
        rest = rest.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"invalid indentation: {original!r}")
        cur = stack[-1][1]
        if rest == "":
            nxt = {}
            cur[k] = nxt
            stack.append((indent + 2, nxt))
        else:
            cur[k] = parse_scalar(rest)

    cfg = root
except FileNotFoundError:
    print(default_value)
    raise SystemExit(0)
except Exception as e:
    print(f"ERROR: failed to parse YAML config: {cfg_path}: {e}", file=sys.stderr)
    raise SystemExit(2)

components = cfg.get("components", {})
value = components.get(component, {})

if isinstance(value, dict) and key in value and value[key] is not None:
    print(str(value[key]))
else:
    print(default_value)
PY
}

if [[ -z "${ISAAC_ROS_WS:-}" ]]; then
  echo "ERROR: ISAAC_ROS_WS is not set (isaac-ros sets this inside the container)." >&2
  exit 1
fi

if [[ -f "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" ]]; then
  # Ensure `ros2` and installed packages are on PATH for non-interactive shells.
  #
  # NOTE: This script runs with `set -u`; some ROS setup files reference optional env vars
  # (e.g., AMENT_TRACE_SETUP_FILES) that may be unset. Temporarily disable nounset.
  set +u
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
  set -u
fi

case "${EULA_MODE}" in
  accept)
    export ISAAC_ROS_ACCEPT_EULA=1
    ;;
  show)
    ;;
  none)
    ;;
esac

CFG_PATH="$(resolve_config_path)"
if [[ ! -f "${CFG_PATH}" ]]; then
  echo "ERROR: expected config file not found: ${CFG_PATH}" >&2
  exit 1
fi
echo "Using config: ${CFG_PATH}"

if [[ -z "${MODEL_RES}" ]]; then
  MODEL_RES="$(config_get_str "${CFG_PATH}" "foundationstereo" "model_res" "low_res")"
fi

eula_args=()
if [[ "${EULA_MODE}" == "show" ]]; then
  # Installer scripts documented in Isaac ROS use `--eula` (some also accept `--show-eula`).
  # Pass the more widely supported flag downstream.
  eula_args+=(--eula)
fi

# Support both workspace layouts:
# - $ISAAC_ROS_WS/src/...
# - $ISAAC_ROS_WS/ros_ws/src/...
src_root=""
if [[ -d "${ISAAC_ROS_WS}/src" ]]; then
  src_root="${ISAAC_ROS_WS}/src"
elif [[ -d "${ISAAC_ROS_WS}/ros_ws/src" ]]; then
  src_root="${ISAAC_ROS_WS}/ros_ws/src"
fi

if [[ -n "${src_root}" ]] && [[ -f "${src_root}/isaac_ros_common/isaac_ros_common/scripts/isaac_ros_asset_eula.sh" ]]; then
  export ISAAC_ROS_ASSET_EULA_SH="${src_root}/isaac_ros_common/isaac_ros_common/scripts/isaac_ros_asset_eula.sh"
  export PATH="$(dirname "${ISAAC_ROS_ASSET_EULA_SH}"):${PATH}"
elif command -v isaac_ros_asset_eula.sh >/dev/null 2>&1; then
  # Binary installs typically place this in /opt/ros/${ROS_DISTRO}/bin (in PATH).
  export ISAAC_ROS_ASSET_EULA_SH
  ISAAC_ROS_ASSET_EULA_SH="$(command -v isaac_ros_asset_eula.sh)"
fi

MODEL_RES_MODE="${MODEL_RES}"
if [[ "${MODEL_RES_MODE}" == "both" ]]; then
  # Keep env var compatible with upstream defaults; we pass explicit --model_res for installs below.
  export FOUNDATIONSTEREO_MODEL_RES="low_res"
else
  export FOUNDATIONSTEREO_MODEL_RES="${MODEL_RES_MODE}"
fi
export MANIPULATOR_INSTALL_ASSETS=1

echo "ISAAC_ROS_WS=${ISAAC_ROS_WS}"
echo "FOUNDATIONSTEREO_MODEL_RES=${FOUNDATIONSTEREO_MODEL_RES}"

prefer_source_or_ros2_run() {
  local description="$1"; shift
  local source_script="$1"; shift
  local ros2_pkg="$1"; shift
  local ros2_exe="$1"; shift

  if [[ -n "${source_script}" ]] && [[ -f "${source_script}" ]]; then
    run_or_die "${description} (source)" bash "${source_script}" "$@"
    return 0
  fi

  run_or_die "${description} (ros2 run)" ros2 run "${ros2_pkg}" "${ros2_exe}" "$@"
}

run_or_die() {
  local desc="$1"; shift
  echo
  echo "==> ${desc}"
  "$@"
}

ess_install_src=""
foundationstereo_install_src=""
foundationpose_install_src=""
rtdetr_install_src=""
grounding_dino_install_src=""
setup_perception_models_src=""

if [[ -n "${src_root}" ]]; then
  ess_install_src="${src_root}/isaac_ros_dnn_stereo_depth/isaac_ros_ess_models_install/asset_scripts/install_ess_models.sh"
  foundationstereo_install_src="${src_root}/isaac_ros_dnn_stereo_depth/isaac_ros_foundationstereo_models_install/asset_scripts/install_foundationstereo_models.sh"
  foundationpose_install_src="${src_root}/isaac_ros_pose_estimation/isaac_ros_foundationpose_models_install/asset_scripts/install_foundationpose_models.sh"
  rtdetr_install_src="${src_root}/isaac_ros_object_detection/isaac_ros_rtdetr_models_install/asset_scripts/install_rtdetr_models.sh"
  grounding_dino_install_src="${src_root}/isaac_ros_object_detection/isaac_ros_grounding_dino_models_install/asset_scripts/install_grounding_dino_models.sh"
  setup_perception_models_src="${src_root}/isaac_manipulator/isaac_manipulator_asset_bringup/scripts/setup_perception_models.py"
fi

if [[ "$(component_enabled "${CFG_PATH}" "ess")" == "1" ]]; then
  prefer_source_or_ros2_run "ESS (stereo depth) models" \
    "${ess_install_src}" isaac_ros_ess_models_install install_ess_models.sh "${eula_args[@]}"
else
  echo
  echo "==> ESS models: disabled by config (ess) (skipping)"
fi

if [[ "$(component_enabled "${CFG_PATH}" "foundationstereo")" == "1" ]]; then
  if [[ "${MODEL_RES_MODE}" == "both" ]]; then
    for res in low_res high_res; do
      prefer_source_or_ros2_run "FoundationStereo models (${res})" \
        "${foundationstereo_install_src}" isaac_ros_foundationstereo_models_install install_foundationstereo_models.sh \
        --model_res "${res}" "${eula_args[@]}"
    done
  else
    prefer_source_or_ros2_run "FoundationStereo models (${FOUNDATIONSTEREO_MODEL_RES})" \
      "${foundationstereo_install_src}" isaac_ros_foundationstereo_models_install install_foundationstereo_models.sh \
      --model_res "${FOUNDATIONSTEREO_MODEL_RES}" "${eula_args[@]}"
  fi
else
  echo
  echo "==> FoundationStereo models: disabled by config (foundationstereo) (skipping)"
fi

if [[ "$(component_enabled "${CFG_PATH}" "foundationpose")" == "1" ]]; then
  prefer_source_or_ros2_run "FoundationPose models" \
    "${foundationpose_install_src}" isaac_ros_foundationpose_models_install install_foundationpose_models.sh "${eula_args[@]}"
else
  echo
  echo "==> FoundationPose models: disabled by config (foundationpose) (skipping)"
fi

if [[ "$(component_enabled "${CFG_PATH}" "rtdetr")" == "1" ]]; then
  prefer_source_or_ros2_run "SyntheticaDETR (RT-DETR) models" \
    "${rtdetr_install_src}" isaac_ros_rtdetr_models_install install_rtdetr_models.sh "${eula_args[@]}"
else
  echo
  echo "==> RT-DETR models: disabled by config (rtdetr) (skipping)"
fi

if [[ "$(component_enabled "${CFG_PATH}" "grounding_dino")" == "1" ]]; then
  prefer_source_or_ros2_run "Grounding DINO models" \
    "${grounding_dino_install_src}" isaac_ros_grounding_dino_models_install install_grounding_dino_models.sh "${eula_args[@]}"
else
  echo
  echo "==> Grounding DINO models: disabled by config (grounding_dino) (skipping)"
fi

# Downloads sample object assets + DOPE weights + Segment Anything assets and performs SAM PTH->ONNX conversion (x86).
if [[ "$(component_enabled "${CFG_PATH}" "manipulator_assets")" == "1" ]]; then
  if ros2 pkg prefix isaac_manipulator_asset_bringup >/dev/null 2>&1; then
    run_or_die "Manipulation perception assets (downloads + verification) (ros2 run)" \
      ros2 run isaac_manipulator_asset_bringup setup_perception_models.py --models all
  elif [[ -n "${setup_perception_models_src}" ]] && [[ -f "${setup_perception_models_src}" ]]; then
    # Best-effort: run from source if the python deps are present in the workspace.
    if [[ -n "${src_root}" ]] && [[ -d "${src_root}/isaac_ros_common/isaac_common_py" ]]; then
      export PYTHONPATH="${src_root}/isaac_ros_common/isaac_common_py:${PYTHONPATH:-}"
    fi
    run_or_die "Manipulation perception assets (downloads + verification) (python)" \
      python3 "${setup_perception_models_src}" --models all
  else
    echo "WARNING: Couldn't find isaac_manipulator_asset_bringup. Skipping setup_perception_models.py." >&2
    echo "         Install the package (binary) or build your workspace, then re-run this script." >&2
  fi
else
  echo
  echo "==> Manipulation perception assets: disabled by config (manipulator_assets) (skipping)"
fi

echo
echo "Done. Assets/models live under: ${ISAAC_ROS_WS}/isaac_ros_assets"

maybe_link_tmp_model() {
  local src="$1"
  local dst="$2"

  if [[ ! -e "${src}" ]]; then
    echo "WARNING: Expected model artifact not found: ${src} (won't link ${dst})" >&2
    return 0
  fi

  mkdir -p "$(dirname "${dst}")"
  ln -sfn "${src}" "${dst}"
}

# Many Isaac ROS launch files default to reading models from /tmp/*.onnx/.plan.
# Provide convenient symlinks so examples work even when you only pass engine paths.
if [[ "$(component_enabled "${CFG_PATH}" "foundationpose")" == "1" ]]; then
  maybe_link_tmp_model "${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_model.onnx" /tmp/refine_model.onnx
  maybe_link_tmp_model "${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_model.onnx" /tmp/score_model.onnx
  maybe_link_tmp_model "${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan" /tmp/refine_trt_engine.plan
  maybe_link_tmp_model "${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_trt_engine.plan" /tmp/score_trt_engine.plan
fi

# Manipulation bringup defaults to /tmp/rtdetr.plan; the installer produces sdetr_grasp.plan.
if [[ "$(component_enabled "${CFG_PATH}" "rtdetr")" == "1" ]]; then
  maybe_link_tmp_model "${ISAAC_ROS_WS}/isaac_ros_assets/models/synthetica_detr/sdetr_grasp.plan" /tmp/rtdetr.plan
fi
