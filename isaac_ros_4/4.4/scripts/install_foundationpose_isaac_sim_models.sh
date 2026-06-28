#!/usr/bin/env bash
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# One-time model setup for the "FoundationPose with Isaac Sim" tutorial on
# Isaac ROS 4.4. Run this INSIDE the activated container, once:
#
#   isaac-ros activate
#   src/isaac_ros_custom_bringup/isaac_ros_4/4.4/scripts/install_foundationpose_isaac_sim_models.sh
#
# It fetches + converts the three models the launch graph needs and is
# idempotent: each step is skipped when its output already exists.
#   - FoundationPose : refine/score ONNX (1.0.1_onnx) -> *_trt_engine.plan
#   - RT-DETR        : sdetr_grasp.plan  (via install_rtdetr_models.sh --eula)
#   - ESS / light_ess: light_ess.engine (via install_ess_models.sh --eula)
set -euo pipefail

log() { echo "[fp-isaac-sim-models] $*"; }

ISAAC_ROS_WS="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}"
ASSETS="${ISAAC_ROS_WS}/isaac_ros_assets"
TRTEXEC="${TENSORRT_COMMAND:-/usr/src/tensorrt/bin/trtexec}"
# Set FP_MODELS_FORCE=1 to delete existing TensorRT engines first and rebuild them.
# Needed when engines were compiled by a different TensorRT (e.g. carried over from
# an older Isaac ROS release) — TensorRT refuses version-mismatched .plan/.engine files.
FORCE="${FP_MODELS_FORCE:-0}"

# ROS setup.bash references unset vars; relax nounset only while sourcing it.
set +u
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
set -u

if [[ "${FORCE}" == "1" ]]; then
  log "FP_MODELS_FORCE=1 — removing existing TensorRT engines so they rebuild for the current TensorRT."
  rm -f "${ASSETS}/models/foundationpose/refine_trt_engine.plan" \
        "${ASSETS}/models/foundationpose/score_trt_engine.plan" \
        "${ASSETS}/models/synthetica_detr/sdetr_grasp.plan"
  find "${ASSETS}/models/dnn_stereo_disparity" -name 'light_ess.engine' -delete 2>/dev/null || true
fi

# --- FoundationPose ONNX -> TensorRT engines ---------------------------------
FP_DIR="${ASSETS}/models/foundationpose"
mkdir -p "${FP_DIR}"

if [[ ! -f "${FP_DIR}/refine_model.onnx" ]]; then
  log "Downloading FoundationPose refine_model.onnx"
  wget -nv -O "${FP_DIR}/refine_model.onnx" \
    'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.1_onnx/files/refine_model.onnx'
fi
if [[ ! -f "${FP_DIR}/score_model.onnx" ]]; then
  log "Downloading FoundationPose score_model.onnx"
  wget -nv -O "${FP_DIR}/score_model.onnx" \
    'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.1_onnx/files/score_model.onnx'
fi

if [[ ! -f "${FP_DIR}/refine_trt_engine.plan" ]]; then
  log "Building FoundationPose refine_trt_engine.plan (trtexec)"
  "${TRTEXEC}" \
    --onnx="${FP_DIR}/refine_model.onnx" \
    --saveEngine="${FP_DIR}/refine_trt_engine.plan" \
    --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
    --optShapes=input1:1x160x160x6,input2:1x160x160x6 \
    --maxShapes=input1:42x160x160x6,input2:42x160x160x6
fi
if [[ ! -f "${FP_DIR}/score_trt_engine.plan" ]]; then
  log "Building FoundationPose score_trt_engine.plan (trtexec)"
  "${TRTEXEC}" \
    --onnx="${FP_DIR}/score_model.onnx" \
    --saveEngine="${FP_DIR}/score_trt_engine.plan" \
    --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
    --optShapes=input1:1x160x160x6,input2:1x160x160x6 \
    --maxShapes=input1:252x160x160x6,input2:252x160x160x6
fi

# --- RT-DETR (SyntheticaDETR) sdetr_grasp.plan -------------------------------
if [[ ! -f "${ASSETS}/models/synthetica_detr/sdetr_grasp.plan" ]]; then
  log "Installing RT-DETR models (install_rtdetr_models.sh --eula)"
  ros2 run isaac_ros_rtdetr_models_install install_rtdetr_models.sh --eula
else
  log "RT-DETR sdetr_grasp.plan already present; skipping."
fi

# --- ESS light_ess engine ----------------------------------------------------
if ! ls "${ASSETS}"/models/dnn_stereo_disparity/*/light_ess.engine >/dev/null 2>&1; then
  log "Installing ESS light_ess model (install_ess_models.sh --eula)"
  ros2 run isaac_ros_ess_models_install install_ess_models.sh --eula
else
  log "ESS light_ess.engine already present; skipping."
fi

log "Done. You can now run:"
log "  ros2 launch isaac_ros_foundationpose isaac_ros_foundationpose_isaac_sim.launch.py"
