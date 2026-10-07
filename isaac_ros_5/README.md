# Isaac ROS 5.0 — focused perception environment

**Status, 2026-10-07: dependency/source review only; custom image NOT VALIDATED.**
The maintainer confirmed NVIDIA's RT-DETR and FoundationPose examples working
on Thor. That evidence belongs to the existing Thor environment; it does not
validate this Dockerfile, the custom YOLO/trocar launch or an Orin installation.
No 5.0 custom launch is installed by this repository yet.

## Start with the existing working environment

Keep Thor's working Isaac ROS 5.0 container and assets as the first integration
reference. First prove the laptop simulator's camera stream reaches it and that
its pose messages reach the consumer. Build this image extension separately
after recording the working base image, package versions and model identities.
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

The proposed extension is [Dockerfile.isaac_ros_perception](5.0/Dockerfile.isaac_ros_perception).
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

This is a **smaller dependency addition to NVIDIA's development image**, not a
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

## Image configuration when a build is scheduled

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
2. With the laptop's existing Isaac Sim 5/5.1 internal Jazzy bridge, prove real
   camera messages and pose results cross the selected common DDS domain.
   Start with one CameraInfo sample and a synthetic, namespaced
   Detection3DArray received in the control environment; then test image
   delivery and real inference. These first checks need no robot motion.
   Check payloads, timestamps and frames, not just discovery. Do not source
   Lyrical into the simulator process. NVIDIA documents a separate 5/5.1 path.
3. Build the candidate layer separately and check the resolved dependencies and
   component registrations. Existing standalone demos remain comparison inputs.
4. Port the custom launch and prove one custom workpiece pose, with correct
   geometry, mask alignment and optional RViz display. Keep pose inference
   bounded/on demand as appropriate; validate tracking separately if selected.
5. Connect the application pose consumer, then qualify the complete pick cycle
   with its control, grasp and contact-policy prerequisites. Perception output
   alone does not prove a complete pick workflow.

No runtime, image-size or performance result is claimed for the new layer yet.
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
