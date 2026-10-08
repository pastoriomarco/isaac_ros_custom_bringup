# Custom trocar YOLOv8 → FoundationPose on Isaac ROS 5.0 (Thor)

**Status, 2026-10-07: validated on Thor against the laptop Isaac Sim scene,
for one object (the highest-confidence detection), with pose estimation only
(no tracking).** The trocar pose agrees with the simulator's ground truth
within 0.8 mm and 0.9°. The 4.x launch was restored on 5.0 interfaces; no
detector, model or multi-object change.

## Revisions and identities

| Item | Identity |
| --- | --- |
| Launch | `isaac_ros_5/launch/yolov8_foundationpose_isaac_sim.launch.py`, sha256 `62390cd1…`, on `isaac_ros_custom_bringup@a844844` plus uncommitted changes (`CMakeLists.txt` `057d8883…`, `package.xml` `dfd9b214…`) |
| Perception image | `nvcr.io/nvidia/isaac/ros:isaac_ros-isaac_ros_perception_cdd1894bfc775397b805f2e3c8e60213-arm64-jetpack` (local id `sha256:75145137…`), container `isaac_ros5_perception_dev`, ROS 2 Lyrical, TensorRT 10.16.2 |
| Simulator | Isaac Sim 5.1.0-rc.19 on the laptop, internal ROS 2 Jazzy bridge, `rmw_fastrtps_cpp`, domain 0 |
| Scene | `isaac_sim_custom_examples@11c3901` `test_scene_realsense_foundationpose_trocar.usda` (sha256 `47574d43…`): RealSense D455 (`/World/rsd455`), six `/World/trocar_short*` |
| Mesh | `isaac_sim_custom_examples@11c3901`: `trocar_short.obj` `ac8254fc…`, `trocar_short.mtl` `7219d44a…`, `grey.png` `4646b999…` |
| YOLO model | `isaac_ros_assets/models/trocar_yolov8/best.onnx` `99753676…` (`sdg_training_custom@f90b6cf`) |
| YOLO engine | `isaac_ros_assets/models/trocar_yolov8/engines/trocar_yolov8_thor_trt10.16.2_fp16.plan` `4aed44d1…` (built on Thor with `trtexec --fp16`; build log next to it) |
| FoundationPose engines (default) | `/reference_assets/models/foundationpose/{refine,score}_trt_engine.plan` (existing, built with the Isaac ROS documented command) |
| FoundationPose engines (optional) | `isaac_ros_assets/models/foundationpose_thor/{refine,score}_thor_trt10.16.2.plan` (`3ef78750…`, `69668039…`): same documented trtexec command, built on this Thor |

### Model facts (from `best.onnx`)

- Ultralytics 8.3.20, task `detect`, **one class** (`{0: 'custom'}`).
- Input `images`: 1×3×640×640 FP32, static batch.
- Output `output0`: 1×5×8400 (four box values plus one class score).
- Preprocessing: RGB, values scaled to [0, 1] (mean 0, std 1), letterboxed to 640.
- The decoder therefore needs `num_classes: 1`. The 4.x launch's default of 80 was wrong for this model.

### Mesh and scene frames

The scene loads `trocar_short.usdz`, authored in millimetres and converted with a 0.001 unit scale. Its geometry matches the OBJ:
- long axis along X, −44.50 to +44.54 mm;
- other two axes ±14.00 mm;
- centred to within 0.02 mm.

So the trocar prim's simulator pose (`/get_entity_state`) is directly comparable with the FoundationPose pose of the OBJ frame. The OBJ is in metres. Symmetry is `['x_full']`: rotation about the long axis is arbitrary.

## Run

The full startup procedure, one-time preparation, and why each step is needed are in
[TROCAR_STACK_STARTUP.md](TROCAR_STACK_STARTUP.md).

Inside `isaac_ros5_perception_dev`; the targeted build is already in `worktrees/isaac_ros5_trocar-build`:

```bash
source /opt/ros/lyrical/setup.bash
source /workspaces/isaac_ros-dev/worktrees/isaac_ros5_trocar-build/install/setup.bash
export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp   # simulator topology, discovery range SUBNET
ros2 launch isaac_ros_custom_bringup yolov8_foundationpose_isaac_sim.launch.py \
  yolov8_engine_file_path:=/workspaces/isaac_ros-dev/isaac_ros_assets/models/trocar_yolov8/engines/trocar_yolov8_thor_trt10.16.2_fp16.plan \
  camera_drop_drop_count:=0 camera_drop_window:=1
```

To rebuild after editing the launch, build only this package:

```bash
colcon build --packages-select isaac_ros_custom_bringup \
  --base-paths worktrees/isaac_ros5_trocar \
  --build-base worktrees/isaac_ros5_trocar-build/build \
  --install-base worktrees/isaac_ros5_trocar-build/install
```

- **Camera drop:** `CameraDropNode` drops X of every Y frames (X = `camera_drop_drop_count`, Y = `camera_drop_window`). The default 28/30 (from 4.x) keeps 1/15 of the frames. That suits a 30 Hz camera but leaves about 0.35 Hz with the simulator's current output. The simulator already skips 9 of every 10 rendered frames (`frameSkipCount` 9), giving about 5 Hz, so use `0`/`1` for it.
- **Offline replay:** large best-effort images over localhost lose most frames, so make the player reliable and set `camera_drop_input_qos:=DEFAULT`. Use `ros2 bag play … --qos-profile-overrides-path validation/replay_qos.yaml` (see `validation/bag_run.sh`).
- **Shutdown:** stop with SIGINT and allow about 15 s. In 3 of 11 runs the 5.0 FoundationPose node waited 5 s for a TensorRT component that had already stopped. One of those then crashed (SIGSEGV), and one needed SIGKILL. The stock FoundationPose launch logs the same timeout. No process was left behind.

## Interface for consumers (ManyForge)

**Inputs.** All are subscribed with SENSOR_DATA (best-effort) QoS:
- `/image_rect`: `rgb8`, 1280×720.
- `/depth`: `32FC1`, metres, aligned to the colour image.
- `/camera_info`: `plumb_bob`, all-zero distortion, fx = fy = 634.09, cx = 640, cy = 360.

All three use frame `camera_color_optical_frame` and simulation-time stamps. The simulator publishes them best-effort with keep-last depth 1, at about 5.3 Hz.

**Outputs.** All are reliable, keep-last 10, volatile, with the same stamp as the source image:

| Topic | Type | Content |
| --- | --- | --- |
| `/output` | `vision_msgs/msg/Detection3DArray` | One detection: `results[0].pose.pose` is the trocar OBJ frame in `camera_color_optical_frame`. `hypothesis.class_id` is `''` and `score` is `0.0`; FoundationPose fills neither, so don't use them as confidence. |
| `/detections_output` | `vision_msgs/msg/Detection2DArray` | All YOLO detections in 640×640 letterbox coordinates. Full image: x·2, (y − 140)·2. Each has a score. |
| `/segmentation` | `sensor_msgs/msg/Image` `mono8` 1280×720 | The rectangle mask given to FoundationPose. |

**Timing** (one camera stream at about 5.3 Hz, one object, no tracking):
- Every input frame produces a pose.
- From input arrival on Thor to pose: 164 ms median, 184 ms max.
- To the mask: 26 ms median.

**Clock.** Stamps are simulation time. The pipeline nodes don't use `/clock`, so `use_sim_time` doesn't matter to them; a consumer that combines `/output` with `/tf` must use simulation time.
- **Pause:** sim time stops. Inputs and outputs stop, and processing resumes on Play.
- **Stop:** the scene resets, and stamps restart from about 0 on the next Play (*Reset Simulation Time On Stop* is enabled). The pipeline continued with valid poses without a restart, but consumers must accept stamps that go backwards.

**Missing or stale input.** Nothing is ever republished:
- With no detection, `/detections_output` publishes an empty array, and `/segmentation` and `/output` publish nothing.
- With paused or missing input, nothing is published.

A consumer must therefore treat a pose as stale based on its stamp's age and the arrival of newer images, not wait for an explicit "lost" message.

**Topology used.**
- Thor container on host network, domain 0, Fast DDS, discovery range SUBNET (multicast).
- No TF and no commands are published.
- The output names are generic (`/output`, `/segmentation`); give them a namespace before sharing a domain with other pipelines.

## Evidence

Stored on Thor in `validation/trocar_yolo_foundationpose_5_0/`; the recording and ground truth are in `isaac_ros_assets/recordings/`.

| Run | What | Result |
| --- | --- | --- |
| `live1` | Live simulator, existing FoundationPose engines, 45 s | 230/230 frames posed. 6 trocars detected per frame (score 0.92–0.95). Target `trocar_short_03`. Error 0.7 mm median / 0.8 mm max; long axis 0.8° / 0.9°; jitter < 0.1 mm |
| `live2` | As `live1` with the Thor-built FoundationPose engines | Same accuracy, 164 ms latency. The "plan file across different models of devices" warning disappears; no measurable performance or accuracy change |
| `stale1` | Pause, play, stop, play during a run | See Clock above; `sim_commands.txt` has the timeline |
| `bag2` | Offline replay of `recordings/trocar_sim_20261007` (21.5 s MCAP: `/image_rect` `/camera_info` `/depth` `/tf` `/clock`) | 106/108 frames posed; 0.7 mm / 0.7° against `trocar_sim_20261007.ground_truth.json` |

`overlay.png` shows the input mask, the unletterboxed YOLO boxes, the true mesh positions of all six trocars, and the estimated pose's mesh. The true positions are drawn through the camera intrinsics and TF, and sit on the trocars in the image.

**Resources** (`live2`, 1280×720 RGB plus depth at 5.3 Hz, one object, compared with idle):
- Pipeline process: about one CPU core (top 104% of one core; Thor has 14).
- GPU rail: 39.9 W (idle 2.9 W).
- CPU/SoC rail: 11.7 W (idle 6.3 W).
- System RAM: +0.6 GB.

Engine-only YOLO inference: 0.92 ms mean (`trtexec`).

## Limitations

Multi-object follow-up, with recorded decisions: `manyforge_specs/docs/reference/PERCEPTION_MULTI_INSTANCE_POSE_PROPOSAL.md`.
ManyForge-side evidence summary: `manyforge_specs/docs/implementation/perception-trocar-isaac-ros5-restoration-evidence-2026-10-07.md`.

- **One object per frame:** the highest-confidence detection. Which trocar is chosen is not controlled.
- **Rectangle masks:** they include neighbouring objects when boxes overlap. Accuracy was unaffected here.
- **Tracking:** tested only with the stock bag during the port, not live.
- **Unreliable shutdown:** the upstream 5.0 FoundationPose node, as described under Run.
- **No confidence on the pose:** the pose output has no confidence value, and the 2D detection score is published separately.
- **`symmetry_axes`:** can't be set to an empty list from the command line.
- **Sim only:** validated only with the simulator. No real camera, Orin or AMD64.
