# Isaac ROS 5.0 — focused perception environment

**Status, 2026-10-07: image build, CLI activation and stock FoundationPose
reference-bag smoke check passed on Thor. The custom YOLO/trocar launch is
ported to 5.0 and validated on Thor against the laptop Isaac Sim scene for one
object: see [TROCAR_YOLO_FOUNDATIONPOSE.md](TROCAR_YOLO_FOUNDATIONPOSE.md).**
Orin and AMD64 execution remain unverified.

## Start with the existing working environment

Keep Thor's original tutorial assets as the integration reference. The candidate
image now passes the recorded-data check described below. Next prove that the
laptop simulator's camera stream reaches it and its pose messages reach the
consumer. Base image, package versions and model identities are recorded.
There is no need to install the complete suite on Orin to test a ROS arm there.

The intended first custom pipeline is:

```text
Isaac Sim RGB + aligned depth + camera calibration
  -> bounded camera sampling
  -> image encoder -> custom YOLO TensorRT -> detection decoder
  -> selected detection -> mask, mapped back to camera coordinates
  -> FoundationPose + refine/score TensorRT nodes
  -> Detection3DArray + required camera/frame transforms
  -> application pose consumer
```

This restores the existing single-object workflow first. A bounding-box mask
is not instance segmentation or all-object bin picking. Tracking, multi-object
requests, mapping and alternate detectors remain separate follow-ups.

## What to install and what to run

The extension is [Dockerfile.isaac_ros_perception](5.0/Dockerfile.isaac_ros_perception).
It retains official binary packages, without the full manipulation metapackage.

| Capability | Package or dependency | Initial treatment |
| --- | --- | --- |
| Pose estimation | `isaac_ros_foundationpose` | Required; brings its supporting dependencies. |
| Custom workpiece detector | `isaac_ros_yolov8` | Required for the existing custom YOLO workflow. |
| Camera sampling | `isaac_ros_topic_tools` | Required for the selected camera-drop graph; replaces the old NITROS package. |
| Image/tensor processing and inference | `isaac_ros_dnn_image_encoder`, `isaac_ros_image_proc`, `isaac_ros_tensor_proc`, `isaac_ros_tensor_rt` | Installed transitively; launch only the components used by the graph. |
| Depth conversion | `isaac_ros_depth_image_proc` | Included for the existing 16UC1-to-meters option; no conversion node for already aligned 32FC1 meter depth. |
| Stock launch fragments | `isaac_ros_examples` | Included for baseline tutorial launches; not necessary for an explicit custom graph. |
| Pose visualization | `rviz2`, `vision_msgs_rviz_plugins` | RViz is already a FoundationPose dependency; add its detection plugin. Start visualization only when needed. |
| RT-DETR | `isaac_ros_rtdetr` | Upstream FoundationPose depends on it, so its package is installed. The custom YOLO graph need not run it or prepare its engines. |
| Learned stereo | ESS / FoundationStereo | Not directly requested: the custom simulator path already supplies depth. A stock stereo tutorial needs its own extra packages and models. |
| Physical camera driver | RealSense or another selected driver | Not included; add only for the actual camera, in the chosen acquisition provider. |
| Mapping / GPU motion planning | Nvblox / cuMotion | Not included initially; independent feature and integration decisions. |
| Complete reference manipulation | `isaac_ros_manipulation_bringup` and its workflow packages | Not required by this direct perception graph. |
| Robot drivers and application runtimes | xArm, Robotiq, MoveIt, ManyMove, ManyForge | Not requested by this image extension; belong to their selected control/application environments. |

The inspected NVIDIA ARM64 package index **still contains**
`ros-lyrical-isaac-ros-manipulation-bringup`. Its absence from a documentation
menu is not proof that all its packages disappeared. It pulls a broad set of
robot, orchestration, perception, planning and mapping dependencies that the
first workflow does not require.

Likewise, `isaac_manipulator_custom_server` is not needed for the existing
direct-topic trocar path. It provides a separate action-oriented workflow;
port it only if a selected application needs that interface. Do not install
the whole manipulation stack merely to obtain FoundationPose.

This is a **focused dependency selection on NVIDIA's development image**, not a
claim of a minimal deployment image. The base already contains development
tools, and upstream dependencies include RT-DETR, RViz and CUDA tooling.
`--no-install-recommends` does not remove required dependencies. Avoid forking
upstream packages just to save disk space. Idle installed packages do not
consume the runtime memory/compute of launched pipelines; measure the actual
running graph rather than estimating its load from image size.

## Changes needed before the custom launch works on 5.0

Use [the 4.x custom launch](../isaac_ros_4/launch/yolov8_foundationpose_isaac_sim.launch.py)
as a behavioral reference, not a launchable 5.0 file. Review against the pinned
5.0 sources and installed plugins:

1. Replace `isaac_ros_nitros_topic_tools` / `NitrosCameraDropNode` with
   `isaac_ros_topic_tools` / `nvidia::isaac_ros::topic_tools::CameraDropNode`.
   Remove the old `depth_format_string`. In the new node, X/Y means **drop X
   of Y**; 28/30 keeps roughly two frames per thirty inputs. Name the settings
   clearly and verify the emitted rate instead of copying the old comment.
2. FoundationPose's refine and score inference now uses **separate TensorRT
   components**. Follow the upstream core launch's topic, tensor-binding and
   batch configuration. Engine paths belong to those components; copying the
   old FoundationPose node parameters is insufficient. Tracking has its own
   refine inference component and must route its output to the consumer too.
3. Review crop/resize/encoding parameters against the 5.0 implementations.
   Preserve the custom YOLO preprocessing, model class count, input/output
   bindings and correct reversal of letterboxing when producing the mask.
4. Mount the OBJ together with its MTL and referenced textures. 5.0 loads mesh
   textures through the asset; the old standalone `texture_path` parameter is
   not the same interface. Prepare compatible engines explicitly on the target
   GPU/runtime; do not assume older engines or Thor engines work on Orin.
5. Preserve aligned depth units, image/calibration timestamps, optical frame
   identity and the camera-to-cell transform required by the pose consumer.
   Qualify normal ROS message delivery across the selected Jazzy/Lyrical
   endpoints before investigating special transports or bridges.

The 5.0 `rosidl::Buffer` implementation changes internal buffer handling;
upstream documents unchanged message definitions and a CPU transport fallback
where CUDA sharing cannot be used. That does not establish arbitrary
cross-distro compatibility. Check the exact Image, CameraInfo, Detection3DArray
and TF interfaces, QoS, RMW and clocks at our actual boundary.

The repository's root `package.xml` still aggregates older launch dependencies,
including RealSense and `isaac_ros_nitros_topic_tools`. Do not run a blind
whole-workspace `rosdep install` for this profile. The launch port must update
the package/install arrangement while keeping the existing 3.x/4.x launch
paths usable. Creating this folder does not declare that packaging problem solved.

## Build configuration and normal startup

Workspace configuration templates are in [5.0/config](5.0/config). For the Thor
candidate, [base/Dockerfile.isaac_ros](5.0/base/Dockerfile.isaac_ros) deliberately
selects the already downloaded NVIDIA ARM64 JetPack image by digest. The search
order selects that file before the system Dockerfile. This avoids rebuilding
NVIDIA's entire base, which CLI 2.6.0 attempted when its intermediate registry
cache lookup failed. No NVIDIA CLI source is patched. This pin is an ARM64
profile, not an AMD64 image or an Orin qualification result.

On Thor, use the usual source workspace, `~/workspaces/dev_ws`. With the repository present at
`src/isaac_ros_custom_bringup`, install the templates there:

```bash
export ISAAC_ROS_WS="$HOME/workspaces/dev_ws"
cd "$ISAAC_ROS_WS"
mkdir -p .isaac-ros-cli scripts validation
cp src/isaac_ros_custom_bringup/isaac_ros_5/5.0/config/config.yaml .isaac-ros-cli/config.yaml
cp src/isaac_ros_custom_bringup/isaac_ros_5/5.0/config/isaac_ros_common-config scripts/.isaac_ros_common-config
cp src/isaac_ros_custom_bringup/isaac_ros_5/5.0/config/isaac_ros_dev-dockerargs scripts/.isaac_ros_dev-dockerargs
set -o pipefail
isaac-ros activate --build-local --build-only --no-push 2>&1 | tee validation/image-build-pinned.log
```

Run that build once, not from multiple terminals. Follow it from another terminal
on Thor with `tail -n 80 -F ~/workspaces/dev_ws/validation/image-build-pinned.log`.
The candidate container name is `isaac_ros5_perception_dev`. The original demo
assets are mounted read-only at `/reference_assets`; create new outputs in the
candidate workspace. Check that the original asset path exists before activation.
The config templates are workspace-level settings, not global host settings.
The Docker-arguments file must contain arguments only: CLI 2.6.0 passes comment
lines into its shell command rather than ignoring them.

### Custom workpiece model for the integration handoff

The maintainer selected the YOLO export in
[`sdg_training_custom/yolov8/example_output`](https://github.com/pastoriomarco/sdg_training_custom/tree/f90b6cf0c8ac1dab8e5e9f8888085c28f2702c41/yolov8/example_output).
The source commit is `f90b6cf0c8ac1dab8e5e9f8888085c28f2702c41`.
Keep these assets outside the bringup Git repository:

```text
~/workspaces/dev_ws/isaac_ros_assets/models/trocar_yolov8/
  best.onnx
  best.pt
  provenance.json
  SHA256SUMS
```

The same workspace is mounted inside the candidate container at
`/workspaces/isaac_ros-dev`, so its ONNX path is
`/workspaces/isaac_ros-dev/isaac_ros_assets/models/trocar_yolov8/best.onnx`.
`provenance.json` records source URLs, Git blob identities and SHA-256 hashes;
verify with `sha256sum -c SHA256SUMS` from the asset directory. Source license
and training README copies accompany the models. Verify the ONNX bindings,
input dimensions, class metadata and preprocessing during the custom launch
port. The downloaded export is not yet qualified with the 5.0 decoder.
Generate any Thor TensorRT engine separately; do not replace the source ONNX
or infer engine portability to Orin from the shared ARM64 architecture.

### Normal startup: use the existing image

For every normal startup after the first build, activate it **without rebuilding**:

```bash
export ISAAC_ROS_WS="$HOME/workspaces/dev_ws"
cd "$ISAAC_ROS_WS"
isaac-ros activate --start-only
```

`--start-only` starts or attaches to the configured container using the local
image; if that image is missing, it refuses instead of building. Do not combine
it with `--build` or `--build-local`. Reserve build commands for intentional
image dependency changes. Moving the workspace or editing these run arguments
does not change the Dockerfile content hash; activation after the move to
`dev_ws` was verified with the same image. Keep validation results in this
README rather than changing Dockerfile labels merely to record a passed test.

CLI 2.6.0's outer activation wrapper does not propagate every child failure as a
nonzero exit status. Check the build log and final local image identity and
actually test activation; a zero shell exit status alone is not build evidence.
Its post-build availability check also tries `docker pull` before checking the
local image. A custom tag built with `--no-push` is absent from NVIDIA's registry,
so that pull prints `not found` even after a successful build. The local image
inspection and actual activation distinguish this expected registry miss from
a build failure. `--start-only` uses the local image check directly.

### Selecting another base or platform

The general customization rules below still apply when selecting a different
base/platform. They are an alternative to the pinned ARM64 search configuration,
not additional instructions to overwrite it.

Use a matching **Isaac ROS release-5.0 / Lyrical** NVIDIA development base for
the target architecture. The layer requires `BASE_IMAGE`; plain Ubuntu is not
a substitute. Keep both the release-5.0 repository and NVIDIA's Noble/Lyrical
buildfarm configured in the base. Do not copy the Jazzy example in the CLI's
generic customization documentation literally.

Direct Isaac packages are pinned to the versions inspected in NVIDIA's
`noble-jetpack` ARM64 repository. This is not a complete dependency lock: record
the base digest, resolved package manifest, CUDA/TensorRT and assets at the
first successful build. AMD64 availability/build and Orin execution remain
unverified. Do not select `latest` or rewrite Thor's working apt sources.

In the chosen workspace's `scripts/.isaac_ros_common-config`, retain the default
Dockerfile search directory and add this layer's directory:

```bash
CONFIG_DOCKER_SEARCH_DIRS=(/etc/isaac-ros-cli/docker ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_5/5.0)
```

Select the new key in that workspace's `.isaac-ros-cli/config.yaml`:

```yaml
docker:
  image:
    additional_image_keys:
      - isaac_ros_perception
```

Check all CLI configuration scopes first: sequence values are **appended**,
so a workspace file does not remove inherited `isaac_ros_manipulation` or
`manymove_xarm` keys. The unique new key avoids Dockerfile-name shadowing but
does not select the 5.0 base on its own. Use an isolated configuration/container
for validation rather than recreating the working demo container.

Then, from the host with that workspace selected:

```bash
isaac-ros activate --build-local
```

There are no custom startup hooks, source clones, colcon builds, model downloads
or engine conversions in this layer. Build dependencies into the image;
prepare mounted model assets once; launch the chosen pipeline explicitly.
Run RViz separately or remotely when useful. Moving visualization off Thor
reduces its rendering work but still consumes transport resources; verify the
actual combination before choosing a deployment default.

## Small validation sequence

1. Record the existing Thor image/package/model identities without replacing it.
2. Build the candidate layer separately and check the resolved dependencies,
   component registrations and reference-bag inference. Existing standalone
   demos remain comparison inputs; the Thor result is recorded below.
3. With the laptop's existing Isaac Sim 5/5.1 internal Jazzy bridge, prove real
   camera messages and pose results cross the selected common DDS domain.
   Start with one CameraInfo sample and a synthetic, namespaced
   Detection3DArray received in the control environment; then test image
   delivery and real inference. These first checks need no robot motion.
   Check payloads, timestamps and frames, not just discovery. Do not source
   Lyrical into the simulator process. NVIDIA documents a separate 5/5.1 path.
4. Port the custom launch and prove one custom workpiece pose, with correct
   geometry, mask alignment and optional RViz display. Keep pose inference
   bounded/on demand as appropriate; validate tracking separately if selected.
5. Connect the application pose consumer, then qualify the complete pick cycle
   with its control, grasp and contact-policy prerequisites. Perception output
   alone does not prove a complete pick workflow.

### Thor image verification, 2026-10-07

- Base digest: `sha256:4b8dd4ed835c06a191775d0fef9d9dc9ac942d37a7cf5c042c9bd206afed877e`.
- Local image tag: `nvcr.io/nvidia/isaac/ros:isaac_ros-isaac_ros_perception_cdd1894bfc775397b805f2e3c8e60213-arm64-jetpack`.
- Inspected image ID: `sha256:751451374506d62d22b04e40b6fdcc07d567aa55f6c9bf01b6bebebfea29a26c`, `linux/arm64`.
- Docker-reported size: 29,940,934,639 bytes (29.94 GB). This is a development
  image, not a minimal runtime artifact. The upstream TensorRT ROS package
  requires the CUDA toolkit and TensorRT development packages.
- FoundationPose, YOLOv8 and topic-tools: `5.0.0-0noble.20260918011409000`;
  TensorRT: `10.16.2.10-1+cuda13.2`; CUDA toolkit: `13.2.2-1`;
  Isaac ROS CLI: `2.6.0-1.20260919010821`.
- CLI activation from `~/workspaces/dev_ws` passed. Original tutorial assets
  are mounted read-only at `/reference_assets`; custom YOLO files are in the
  writable workspace asset directory above.
- Required component registrations passed, including FoundationPose, tracking,
  detection filtering/mask conversion, TensorRT, YOLOv8 and camera-drop.
- Existing refine, score and RT-DETR engines deserialized successfully.
  The stock FoundationPose fragment, fed the original reference bag, emitted
  a nonempty finite pose in `tf_camera`. No ONNX export or engine rebuild ran.
- The check used local-only DDS domain 77; it did not test simulator/network
  interoperability or the custom YOLO graph. Test processes were stopped after
  the result; the development container remains available.

Evidence on Thor is under `~/workspaces/dev_ws/validation/`: build and smoke
logs, `image-validation.json`, `reference-environment.json`, `packages.tsv`,
component registrations, engine-load logs, launch/playback logs and the captured
`foundationpose-observation.json`. The one-off bounded check and its invocation
are preserved there as `perception_smoke.py` and `run-smoke.sh`.

This is functional image evidence, not a latency, throughput or custom-workflow
qualification. The Dockerfile's original `not-validated` label is unchanged;
this dated record describes the narrower checks actually performed.
The existing 4.6 adaptation stays separate and NOT VALIDATED; it is not a
prerequisite for the 5.0 path.

## Reviewed sources

- [NVIDIA 5.0 getting started and simulator versions](https://nvidia-isaac-ros.github.io/v/release-5.0/getting_started/index.html#integrate-external-data-sources-optional)
- [Development image layers and configuration merging](https://nvidia-isaac-ros.github.io/v/release-5.0/concepts/dev_env/index.html)
- [Noble/Lyrical buildfarm](https://nvidia-isaac-ros.github.io/v/release-5.0/getting_started/isaac_ros_buildfarm_cdn.html)
- [FoundationPose package and dependency declarations](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/blob/release-5.0/isaac_ros_foundationpose/package.xml)
- [FoundationPose core inference graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/blob/release-5.0/isaac_ros_foundationpose/launch/isaac_ros_foundationpose_core.launch.py)
- [FoundationPose tracking graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/blob/release-5.0/isaac_ros_foundationpose/launch/isaac_ros_foundationpose_tracking_core.launch.py)
- [Camera-drop implementation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/release-5.0/isaac_ros_topic_tools/src/isaac_ros_camera_drop_node.cpp)
- [Buffer message contracts](https://nvidia-isaac-ros.github.io/v/release-5.0/concepts/rosidl_buffer/index.html)
- [CUDA buffer backend](https://nvidia-isaac-ros.github.io/v/release-5.0/concepts/rosidl_buffer/cuda_buffer_backend.html)
- [NVIDIA release-5.0 ARM64 package metadata](https://isaac.download.nvidia.com/isaac-ros/release-5.0/dists/noble-jetpack/main/binary-arm64/Packages.gz)
