# Isaac ROS 4.x (Jazzy) — custom bringup helpers

This directory is versioned by **Isaac ROS minor release**:

- `4.4/`: **recommended** — Isaac ROS 4.4 rename (`isaac_manipulator` → `isaac_ros_manipulation`)
  + FoundationPose / RT-DETR / ESS baked in for the "FoundationPose with Isaac Sim" example
- `4.3/`: previous (workspace-root `build/`, `log/`, `install/` + idempotent clones)
- `4.2/`: previous recommended layout
- `4.1/`: workspace-scoped `isaac-ros-cli` config + lighter bootstrap
- `4.0/`: legacy bootstrap (writes user-global `~/.config/isaac-ros-cli/config.yaml` and `~/.isaac_ros_dev-dockerargs`)

## Quick start (4.4)

```bash
# After the host is on release-4.4 and CONFIG_DOCKER_SEARCH_DIRS points at 4.4/ (see 4.4/README.md):
isaac-ros activate --build-local
```

Notes:
- **4.4 is a breaking rename.** Configure the Dockerfile search path + the renamed
  `additional_image_keys: [isaac_ros_manipulation]`, and move the host to `release-4.4` first —
  full steps in [4.4/README.md](4.4/README.md).
- FoundationPose + Isaac Sim model setup is a one-time in-container step
  (`4.4/scripts/install_foundationpose_isaac_sim_models.sh`).
- RealSense support notes for the older helper flow remain in `4.1/README.md`.

## Perception example — YOLOv8 → FoundationPose (Isaac Sim), ported from 3.x

`launch/yolov8_foundationpose_isaac_sim.launch.py` is the Isaac ROS 4.4 (Jazzy) port of the
isaac_ros_3 `yolov8_foundationpose_realsense_remote.launch.py` trocar example: a **custom YOLOv8
detector → FoundationPose**, driven by a (simulated) RealSense stream published by Isaac Sim.
Depth comes straight from the sim RealSense aligned-depth topic, so there's **no ESS** (unlike the
stock `isaac_ros_foundationpose_isaac_sim.launch.py`, which uses RT-DETR + ESS).

### 1) Build the launch package (one-time, inside the 4.4 container)
The `isaac_ros_manipulation` bootstrap does not build `isaac_ros_custom_bringup`, so build it once:
```bash
cd ${ISAAC_ROS_WS}
colcon build --packages-select isaac_ros_custom_bringup
source install/setup.bash
```

If you want the ManyMove/xArm pick-and-place step in this same Isaac ROS container, enable the
`manymove_xarm` image key before rebuilding the image:

```yaml
docker:
  image:
    additional_image_keys:
      - isaac_ros_manipulation
      - manymove_xarm
```

The `manymove_xarm` startup hook builds `isaac_ros_custom_bringup`, ManyMove, and xArm packages.

### 2) Assets (same as the 3.x example — already staged)
- YOLOv8 trocar engine: `isaac_ros_assets/models/yolov8/trocar_short.{onnx,plan}`
- FoundationPose refine/score engines: `isaac_ros_assets/models/foundationpose/` — **rebuild for this
  TensorRT** if stale (`FP_MODELS_FORCE=1 4.4/scripts/install_foundationpose_isaac_sim_models.sh`).
- Trocar mesh + texture: `src/isaac_sim_custom_examples/trocar_short.obj` + `grey.png`

### 3) Run
Start Isaac Sim, open `src/isaac_sim_custom_examples/test_scene_realsense_foundationpose_trocar.usd`,
press **Play** (publishes `/image_rect`, `/camera_info`, `/depth` as 32FC1 meters). For
multi-machine runs, apply the DDS/static-peer and Isaac Sim camera QoS notes in
[`dds/README.md`](dds/README.md) before playback. Then:
```bash
export YOLO_MODEL_NAME=trocar_short
export MESH_FILE_PATH=${ISAAC_ROS_WS}/src/isaac_sim_custom_examples/trocar_short.obj
export TEXTURE_PATH=${ISAAC_ROS_WS}/src/isaac_sim_custom_examples/grey.png
ros2 launch isaac_ros_custom_bringup yolov8_foundationpose_isaac_sim.launch.py \
  yolov8_model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/${YOLO_MODEL_NAME}.onnx \
  yolov8_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/${YOLO_MODEL_NAME}.plan \
  force_engine_update:=True \
  input_binding_names:='["images"]' output_binding_names:='["output0"]' \
  num_classes:=1 confidence_threshold:=0.25 nms_threshold:=0.45 \
  mesh_file_path:=${MESH_FILE_PATH} texture_path:=${TEXTURE_PATH} \
  refine_model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_model.onnx \
  refine_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan \
  score_model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_model.onnx \
  score_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_trt_engine.plan \
  depth_is_float:=True launch_rviz:=False
```
The sim-RealSense topics and `depth_is_float:=True` are already the defaults, so those args are
optional. Output pose is on `/output`; YOLOv8 detections on `/detections_output`.
Use `force_engine_update:=True` only for the first run after a TensorRT/GPU image change; remove it
after `trocar_short.plan` has been rebuilt successfully.

### 4) Run the ManyMove consumer (optional)
Once `/output` is publishing FoundationPose detections and the ManyMove packages are built/sourced:

```bash
ros2 run tf2_ros tf2_echo world camera_color_optical_frame
ros2 launch manymove_bringup lite_foundationpose_movegroup_fake_cpp_trees.launch.py
```

The TF check matters because the ManyMove FoundationPose behavior transforms detections from the
camera frame into `world` before applying pick/approach offsets and workspace bounds.

### What changed vs the 3.x launch (verified against `release-4.4` source)
- **CropNode** dropped its `encoding_desired` parameter → removed here (the mono8 mask is preserved).
- **Encoder → TensorRT → decoder** now use 4.4 default topics
  (`/tensor_pub` → `tensor_pub`/`tensor_sub` → `detections_output`); the 3.x `tensor_input` remap is gone.
- Defaults switched to the Isaac Sim sim-RealSense topics (`/image_rect`, `/camera_info`, `/depth`) + float depth.
- `NitrosCameraDropNode` now explicitly uses `input_qos: SENSOR_DATA` and a larger sync queue, matching
  4.4 FoundationPose launch patterns and common Isaac Sim / RealSense camera QoS.
- `ConvertMetricNode`, `dnn_image_encoder.launch.py`, and all `FoundationPose*` / `YoloV8DecoderNode`
  plugins remain the same package/plugin surfaces, including `symmetry_axes` and the rviz config.

### Verify / caveats
- **Scene resolution must be 1280×720** (RealSense default). If your scene differs, edit
  `REALSENSE_IMAGE_WIDTH/HEIGHT` at the top of the launch — the letterbox/crop math derives from them.
- The encoder letterboxes with **CENTER** padding (pad top/bottom 140 for 1280×720→640×640); the
  unletterbox crop assumes this. If the mask looks misaligned with detections, check there.
- rviz needs an X display (`launch_rviz:=False` + Foxglove is the headless option).
- Isaac ROS 4.4 perception has produced `/output` as `vision_msgs/msg/Detection3DArray` in the
  live trocar scene, with poses in `camera_color_optical_frame`. Full ManyForge Lite6 pick/drop
  remains a separate qualification step.
- Current observed `/output` carries empty detection id/class, score `0.0`, and zero covariance.
  ManyForge's bootstrap scenario uses `selection: first` and explicit zero-field allowances only
  for the controlled single-trocar scene; investigate score/covariance propagation before treating
  this as a production perception gate.
