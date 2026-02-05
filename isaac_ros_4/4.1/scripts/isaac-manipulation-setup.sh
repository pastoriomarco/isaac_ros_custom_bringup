#!/usr/bin/env bash
set -euo pipefail

show_usage() {
  cat <<'EOF'
Usage: isaac-manipulation-setup.sh [--accept-eula|--eula|--show-eula] [--model-res <low_res|high_res|both>] [--skip-perception]

Downloads/converts the models + assets required by the Isaac Manipulation tutorials.

Notes:
  - Assets/models are installed into: $ISAAC_ROS_WS/isaac_ros_assets
  - Idempotent: installer scripts skip if outputs already exist.
  - For non-interactive use, pass --accept-eula (or set ISAAC_ROS_ACCEPT_EULA=1).
  - By default, also runs: `setup_perception_models.py --models all`
  - To enable SAM2 export, set: ISAAC_ROS_MANIPULATION_SETUP_SAM2=1 (x86_64 only)
EOF
}

MODEL_RES="${FOUNDATIONSTEREO_MODEL_RES:-}"
EULA_MODE="none" # none|accept|show
SETUP_PERCEPTION_MODELS="${ISAAC_ROS_MANIPULATION_SETUP_PERCEPTION_MODELS:-1}"

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
    --skip-perception)
      SETUP_PERCEPTION_MODELS="0"
      shift
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

if [[ -z "${ISAAC_ROS_WS:-}" ]]; then
  echo "ERROR: ISAAC_ROS_WS is not set (isaac-ros sets this inside the container)." >&2
  exit 1
fi

ASSET_MARKER="${ISAAC_ROS_WS}/isaac_ros_assets/.isaac_manipulation_assets_ready"
if [[ "${ISAAC_ROS_MANIPULATION_FORCE_ASSET_SETUP:-0}" != "1" && -f "${ASSET_MARKER}" ]]; then
  echo "Assets already prepared (${ASSET_MARKER}); skipping."
  exit 0
fi

# Ensure `ros2` and installed packages are on PATH for non-interactive shells.
#
# NOTE: This script runs with `set -u`; some ROS setup files reference optional env vars
# (e.g., AMENT_TRACE_SETUP_FILES) that may be unset. Temporarily disable nounset.
if [[ -f "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
  set -u
fi
if [[ -f "${ISAAC_ROS_WS}/install/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${ISAAC_ROS_WS}/install/setup.bash"
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

if [[ -z "${MODEL_RES}" ]]; then
  MODEL_RES="low_res"
fi

case "${MODEL_RES}" in
  low_res|high_res|both)
    ;;
  *)
    echo "ERROR: invalid --model-res: ${MODEL_RES} (expected: low_res|high_res|both)" >&2
    exit 2
    ;;
esac

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

  if ros2 pkg prefix "${ros2_pkg}" >/dev/null 2>&1; then
    run_or_die "${description} (ros2 run)" ros2 run "${ros2_pkg}" "${ros2_exe}" "$@"
    return 0
  fi

  echo
  echo "==> ${description}: not available (${ros2_pkg}); skipping"
  return 0
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

ensure_python_module() {
  local module="$1"
  local install_cmd="$2"

  if python3 - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("${module}")
PY
  then
    return 0
  fi

  echo "==> Installing Python dependency for ${module}"
  eval "${install_cmd}"
}

prefer_source_or_ros2_run "ESS (stereo depth) models" \
  "${ess_install_src}" isaac_ros_ess_models_install install_ess_models.sh "${eula_args[@]}"

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

prefer_source_or_ros2_run "FoundationPose models" \
  "${foundationpose_install_src}" isaac_ros_foundationpose_models_install install_foundationpose_models.sh "${eula_args[@]}"

prefer_source_or_ros2_run "SyntheticaDETR (RT-DETR) models" \
  "${rtdetr_install_src}" isaac_ros_rtdetr_models_install install_rtdetr_models.sh "${eula_args[@]}"

prefer_source_or_ros2_run "Grounding DINO models" \
  "${grounding_dino_install_src}" isaac_ros_grounding_dino_models_install install_grounding_dino_models.sh "${eula_args[@]}"

if [[ "${SETUP_PERCEPTION_MODELS}" == "1" ]]; then
  if ros2 pkg prefix isaac_manipulator_asset_bringup >/dev/null 2>&1; then
    run_or_die "Manipulation tutorial assets (downloads + verification) (ros2 run)" \
      ros2 run isaac_manipulator_asset_bringup setup_perception_models.py --models all
  elif [[ -n "${setup_perception_models_src}" ]] && [[ -f "${setup_perception_models_src}" ]]; then
    # Best-effort: run from source if the python deps are present in the workspace.
    if [[ -n "${src_root}" ]] && [[ -d "${src_root}/isaac_ros_common/isaac_common_py" ]]; then
      export PYTHONPATH="${src_root}/isaac_ros_common/isaac_common_py:${PYTHONPATH:-}"
    fi
    run_or_die "Manipulation tutorial assets (downloads + verification) (python)" \
      python3 "${setup_perception_models_src}" --models all
  else
    echo "WARNING: Couldn't find isaac_manipulator_asset_bringup. Skipping setup_perception_models.py." >&2
    echo "         Install the package (binary) or build your workspace, then re-run this script." >&2
  fi
else
  echo
  echo "==> Manipulation tutorial assets: disabled (--skip-perception) (skipping)"
fi

if [[ "${ISAAC_ROS_MANIPULATION_SETUP_SAM2:-0}" == "1" ]]; then
  echo
  echo "==> Segment Anything2 models"
  if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "WARNING: SAM2 ONNX export is only supported on x86_64. Copy the ONNX to this device." >&2
  else
    SAM2_ASSETS_DIR="${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_segment_anything2"
    SAM2_CHECKPOINT="${ISAAC_ROS_SEGMENT_ANYTHING2_CHECKPOINT:-${SAM2_ASSETS_DIR}/sam2.1_hiera_tiny.pt}"
    SAM2_MODEL_DIR="${ISAAC_ROS_WS}/isaac_ros_assets/models/segment_anything2/1"
    SAM2_MODEL_PATH="${SAM2_MODEL_DIR}/model.onnx"
    SAM2_FP16="${ISAAC_ROS_SEGMENT_ANYTHING2_FP16:-1}"

    if [[ -f "${SAM2_MODEL_PATH}" && "${ISAAC_ROS_MANIPULATION_FORCE_ASSET_SETUP:-0}" != "1" ]]; then
      echo "SAM2 ONNX already exists; skipping export."
    elif [[ ! -f "${SAM2_CHECKPOINT}" ]]; then
      echo "WARNING: SAM2 checkpoint not found at ${SAM2_CHECKPOINT}; skipping export." >&2
    else
      ensure_python_module "sam2" "python3 -m pip install --break-system-packages git+https://github.com/facebookresearch/sam2.git -v"
      if [[ "${SAM2_FP16}" == "1" ]]; then
        ensure_python_module "onnxconverter_common" "python3 -m pip install --break-system-packages onnxconverter-common==1.14.0"
      fi
      mkdir -p "${SAM2_MODEL_DIR}"
      export_cmd=(ros2 run isaac_ros_segment_anything2 sam2_onnx_exporter.py \
        --checkpoint "${SAM2_CHECKPOINT}" \
        --output "${SAM2_MODEL_PATH}")
      if [[ "${SAM2_FP16}" == "1" ]]; then
        export_cmd+=(--fp16)
      fi
      run_or_die "Export SAM2 ONNX" "${export_cmd[@]}"
    fi

    if [[ -f "${SAM2_ASSETS_DIR}/sam2_config.pbtxt" ]]; then
      mkdir -p "${ISAAC_ROS_WS}/isaac_ros_assets/models/segment_anything2"
      cp "${SAM2_ASSETS_DIR}/sam2_config.pbtxt" "${ISAAC_ROS_WS}/isaac_ros_assets/models/segment_anything2/config.pbtxt"
    else
      echo "WARNING: SAM2 config not found at ${SAM2_ASSETS_DIR}/sam2_config.pbtxt" >&2
    fi
    if [[ -d "${SAM2_ASSETS_DIR}/warmup" ]]; then
      cp -r "${SAM2_ASSETS_DIR}/warmup" "${ISAAC_ROS_WS}/isaac_ros_assets/models/segment_anything2/"
    else
      echo "WARNING: SAM2 warmup data not found at ${SAM2_ASSETS_DIR}/warmup" >&2
    fi
  fi
else
  echo
  echo "==> Segment Anything2 models: disabled (set ISAAC_ROS_MANIPULATION_SETUP_SAM2=1 to enable) (skipping)"
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
maybe_link_tmp_model "${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_model.onnx" /tmp/refine_model.onnx
maybe_link_tmp_model "${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_model.onnx" /tmp/score_model.onnx
maybe_link_tmp_model "${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan" /tmp/refine_trt_engine.plan
maybe_link_tmp_model "${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_trt_engine.plan" /tmp/score_trt_engine.plan

# Manipulation bringup defaults to /tmp/rtdetr.plan; the installer produces sdetr_grasp.plan.
maybe_link_tmp_model "${ISAAC_ROS_WS}/isaac_ros_assets/models/synthetica_detr/sdetr_grasp.plan" /tmp/rtdetr.plan

mkdir -p "$(dirname "${ASSET_MARKER}")"
touch "${ASSET_MARKER}"
