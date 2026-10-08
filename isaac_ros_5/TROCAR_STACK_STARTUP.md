# Trocar perception stack: startup procedure and required changes

Status, 2026-10-07: the working configuration validated in
[TROCAR_YOLO_FOUNDATIONPOSE.md](TROCAR_YOLO_FOUNDATIONPOSE.md). The procedure
below is what was actually done; each step says why it was needed. It is the
basis for packaging the stack later, not a packaged installer.

## Topology

| Host | Role | Software |
| --- | --- | --- |
| Laptop `tndlux-G16` (RTX 4070 Laptop) | Simulator and camera source | Isaac Sim 5.1.0-rc.19, internal ROS 2 Jazzy bridge |
| Jetson AGX Thor `192.168.1.136` | Perception | JetPack 7.2 / L4T R39.2, Docker container `isaac_ros5_perception_dev` (Isaac ROS 5.0, ROS 2 Lyrical, TensorRT 10.16.2) |

- **Network:** both hosts are on the same LAN, using ROS domain 0, Fast DDS (`rmw_fastrtps_cpp`) and multicast discovery (`ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`).
- **Cross-distribution traffic:** Jazzy (simulator) and Lyrical (Thor) exchange the standard sensor, TF and clock messages directly. No bridge is needed.
- **Thor container:** it uses the host network and host IPC, so it takes part in DDS discovery like a host process.

## One-time preparation

### Thor

1. **SSH:** use `ssh -o StrictHostKeyChecking=yes tndlux@192.168.1.136`.
   - *Why:* the `tndlux-thor` alias has a stale host key from before the JetPack 7 reinstall. Don't bypass the check.
2. **Image and CLI configuration:** follow the [5.0 README](README.md#build-configuration-and-normal-startup). It copies three workspace-level files and runs the build once:
   - `.isaac-ros-cli/config.yaml` selects the `isaac_ros_perception` image layer and names the container.
   - `scripts/.isaac_ros_common-config` selects the pinned ARM64 base before NVIDIA's own Dockerfile.
   - `scripts/.isaac_ros_dev-dockerargs` mounts `~/workspaces/isaac_ros-dev/isaac_ros_assets` read-only at `/reference_assets`.
   - *Why:* this reuses the downloaded NVIDIA base instead of rebuilding it, and keeps the original tutorial assets read-only.
3. **YOLO model:** place it in `~/workspaces/dev_ws/isaac_ros_assets/models/trocar_yolov8/`: `best.onnx`, `provenance.json`, `SHA256SUMS`. Check it with `sha256sum -c SHA256SUMS`.
4. **Trocar mesh:** clone it:
   `git clone https://github.com/pastoriomarco/isaac_sim_custom_examples ~/workspaces/dev_ws/src/isaac_sim_custom_examples`.
   The tested revision is `11c3901`.
   - *Why:* FoundationPose needs the OBJ, MTL and texture of the same object the simulator renders. The scene's `.usdz` has the same geometry and frame as `trocar_short.obj` (metres, centred, long axis +X). A different mesh, such as the tutorial mustard bottle, gives wrong poses.
5. **TensorRT engines:** build them on this device, inside the container. Engines depend on the device and the TensorRT version; none is shipped.

   ```bash
   cd /workspaces/isaac_ros-dev/isaac_ros_assets/models/trocar_yolov8 && mkdir -p engines
   /usr/src/tensorrt/bin/trtexec --onnx=best.onnx --fp16 --skipInference \
     --saveEngine=engines/trocar_yolov8_thor_trt10.16.2_fp16.plan
   ```

   - *Why:* no engine was shipped with the model. The engine name records the device and TensorRT version, so a stale engine can't be picked up by mistake.
   - The FoundationPose refine/score engines in `/reference_assets` load and work. TensorRT warns that they were built on a different device model.
   - A same-command Thor build is in `isaac_ros_assets/models/foundationpose_thor/` (refine max batch 42, score 252). It removes the warning, with no measurable change in accuracy or speed. Commands are in that folder's `*.build.log`.
6. **Build the bringup package only**, into its own build tree, inside the container:

   ```bash
   cd /workspaces/isaac_ros-dev
   colcon build --packages-select isaac_ros_custom_bringup \
     --base-paths worktrees/isaac_ros5_trocar \
     --build-base worktrees/isaac_ros5_trocar-build/build \
     --install-base worktrees/isaac_ros5_trocar-build/install
   ```

   - *Why:* on Lyrical, the package installs only the 5.0 launch file. The 3.x/4.x launches need NITROS packages that 5.0 removed, and the 4.x launch has the same file name.
   - A targeted build avoids a whole-workspace rosdep and build.
   - For the shared checkout, use `src/isaac_ros_custom_bringup` and the default `build`/`install` instead.

### Laptop (Isaac Sim)

1. **Isaac Sim 5.1:** installed in `~/isaac-sim`. It was started through the App Selector (`isaac-sim.selector.sh`), with ROS bridge `isaacsim.ros2.bridge` and **internal ROS libraries**. That gives the running process:
   - `ROS_DISTRO=jazzy`
   - `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
   - `LD_LIBRARY_PATH=~/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib`
   - no `ROS_DOMAIN_ID`, so domain 0.

   *Why:* the bundled Jazzy bridge needs no system ROS installation on the simulator host.
2. **Simulation-control extension:** enable `isaacsim.ros2.sim_control`. Today it is enabled only in the user settings of the "Isaac-Sim Full" app (`~/.local/share/ov/data/Kit/Isaac-Sim Full/5.1/user.config.json`), not by the scene.
   - *Why:* it provides the `simulation_interfaces` services (`/set_simulation_state`, `/reset_simulation`, `/step_simulation`, `/get_entity_state`, ...). They let a remote host play, pause and stop the simulator and read true object poses, which the accuracy checks use.
3. **Scene:** `~/workspaces/isaac_ros-dev/src/isaac_sim_custom_examples/test_scene_realsense_foundationpose_trocar.usda` at `11c3901`. Its action graph publishes `/image_rect`, `/depth`, `/camera_info`, `/tf` and `/clock` with frame `camera_color_optical_frame`, and the camera topics are best-effort with depth 1.
4. **Camera frame skip and physics rate:** all three camera publishers need `frameSkipCount` = **9**, and the physics rate is **30 steps/s** (the scene default is 60). The saved scene file has frame skip **5**; both values were set in the open session. Save the scene while stopped and check its diff.
   - *Why:* it keeps the 1280×720 RGB and depth stream to about 5.3 Hz (about 35 MB/s), so the LAN isn't saturated.
   - *Keep best-effort QoS:* reliable delivery of 2.7–3.7 MB frames over the LAN would retransmit and could stall publishing.

## Startup sequence

1. **Laptop:** start Isaac Sim as above and open the scene. Check that the frame skip is 9. Leave the simulation stopped.
2. **Thor, start the container:**

   ```bash
   export ISAAC_ROS_WS="$HOME/workspaces/dev_ws"; cd "$ISAAC_ROS_WS"
   isaac-ros activate --start-only
   ```

   - *Why `--start-only`:* it uses the local image and never rebuilds.
   - **Without a terminal it exits immediately and removes the container.** In an interactive SSH session that's fine. Unattended, it needs a held pseudo-terminal. This is what keeps the current container alive:

     ```bash
     nohup setsid bash -c 'sleep infinity | script -qfec "isaac-ros activate --start-only" /dev/null' > activate.log 2>&1 &
     ```

     Other shells then use `docker exec -u admin isaac_ros5_perception_dev bash`.
3. **Thor, launch the pipeline** inside the container:

   ```bash
   source /opt/ros/lyrical/setup.bash
   source /workspaces/isaac_ros-dev/worktrees/isaac_ros5_trocar-build/install/setup.bash
   export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
   ros2 launch isaac_ros_custom_bringup yolov8_foundationpose_isaac_sim.launch.py \
     yolov8_engine_file_path:=/workspaces/isaac_ros-dev/isaac_ros_assets/models/trocar_yolov8/engines/trocar_yolov8_thor_trt10.16.2_fp16.plan \
     camera_drop_drop_count:=0 camera_drop_window:=1
   ```

   - It is ready when the log shows 18 "Loaded node" lines, about 3 s.
   - *Why drop 0/1:* the simulator already sends only one frame in ten. The launch default of dropping 28 of 30 (meant for a 30 Hz camera) would leave about 0.35 Hz.
   - To use the Thor-built FoundationPose engines, add:
     `refine_engine_file_path:=/workspaces/isaac_ros-dev/isaac_ros_assets/models/foundationpose_thor/refine_thor_trt10.16.2.plan score_engine_file_path:=/workspaces/isaac_ros-dev/isaac_ros_assets/models/foundationpose_thor/score_thor_trt10.16.2.plan`
4. **Play the simulator:** use the GUI, or from any ROS host on domain 0:
   `ros2 service call /set_simulation_state simulation_interfaces/srv/SetSimulationState "{state: {state: 1}}"`.
5. **Check:**
   - `ros2 topic hz /output` gives about 5 Hz.
   - `/detections_output` shows the visible trocars.
   - `/output` gives a pose in `camera_color_optical_frame`.

   The [validation scripts](validation/README.md) measure accuracy against the simulator's true poses.

## Shutdown

1. **Stop the pipeline:** Ctrl-C, or SIGINT to the launch, then allow about **15 s** before forcing.
   - *Why:* NVIDIA's 5.0 FoundationPose node sometimes waits 5 s for a TensorRT component that has already stopped, then crashes or hangs (3 of 11 runs).
   - No process was left behind. Check with `pgrep -a component_container`.
2. **Simulator:** stop it (state 0) when done; stopping resets the scene and its time.
3. **Container:** stop it only when no one else uses it. Kill the keep-alive process group, or exit the interactive activation shell.

## Offline replay (no simulator)

Recording: `isaac_ros_assets/recordings/trocar_sim_20261007/` (MCAP, 21.5 s), with its ground-truth JSON next to it.
- **Isolate it:** replay on a separate domain (78) with `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`.
- **Make it reliable:** use `--qos-profile-overrides-path validation/replay_qos.yaml` on the player and `camera_drop_input_qos:=DEFAULT` on the launch.

*Why:* best-effort 1280×720 frames over localhost lose most fragments (13 of 113 images arrived), while reliable delivery passes all of them. See `validation/bag_run.sh`.

## Port changes and why they were needed

| Change | Why |
| --- | --- |
| `NitrosCameraDropNode` → `isaac_ros_topic_tools` `CameraDropNode` (mode `mono+depth`, X/Y = drop X of Y); `depth_format_string` removed | 5.0 removed the NITROS topic tools; the new node has different topics and semantics. A measured 30 Hz input with 28/30 gave 2.0 Hz out. |
| FoundationPose refine/score (and tracking refine) moved to separate `TensorRTNode` components, batch 42/252/1 | In 5.0, FoundationPose no longer runs TensorRT itself; this follows NVIDIA's `isaac_ros_foundationpose_core` launch. |
| `texture_path` removed; mesh path loads OBJ, MTL and texture together | The parameter no longer exists in the 5.0 node. |
| Encoder `tensor_name:=input_tensor` | The 5.0 encoder launch's default output name changed to `output_tensor`. |
| `num_classes` default 80 → 1 | The model has one class (output 1×5×8400). 80 makes the decoder read past the tensor. |
| Letterbox crop/resize computed from `image_width`/`image_height` | The 5.0 `ResizeNode` lost its input-size parameters. The values must match its centred padding: 140 rows for 1280×720 into 640. |
| Crop node publishes its own camera-info topic | The 4.x launch republished onto the encoder's topic, a loop. |
| Tracking uses the Selector's default topics | The 4.x launch bypassed the Selector. |
| `CMakeLists.txt` / `package.xml`: on Lyrical, install and depend on 5.0 parts only | See one-time step 6. Other distributions are unchanged. |

## What packaging has to capture

These steps are manual or session-specific today, and an application package must make them explicit:

- **Simulator settings:** the frame skip of 9 and physics at 30 steps/s (until the scene is saved), the simulation-control extension (user settings), and the bridge and internal-library choice (App Selector). These belong in the scene file or a start script, e.g. `isaac-sim.sh --enable isaacsim.ros2.sim_control` with the bridge environment. The script form isn't tested yet.
- **Container keep-alive:** unattended activation needs a terminal (see startup step 2), or a different supervisor.
- **Engine generation:** engines are generated per device and TensorRT version at install time, with the version in the name. Source models and the original engines stay read-only.
- **Asset paths:** the `/reference_assets` mount points at the old `~/workspaces/isaac_ros-dev/isaac_ros_assets`. The custom assets live in `~/workspaces/dev_ws/isaac_ros_assets`.
- **Worktree build:** the launch currently runs from a worktree build; packaging should install from the reviewed checkout.
- **Topology:** domain, discovery range, RMW and generic output names (`/output`, `/segmentation`). Give outputs a namespace before the domain is shared.
- **Shutdown grace:** at least 15 s, and the shutdown fault is upstream.
- **Mesh source:** `isaac_sim_custom_examples` must be checked out on the perception host.
