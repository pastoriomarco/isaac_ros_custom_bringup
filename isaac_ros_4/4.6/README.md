# Isaac ROS 4.6 (Jazzy) — NOT VALIDATED

**Status: not validated.** Adapted from the 4.4 helpers on 2026-10-05. No Docker
image build, dependency installation, model conversion, simulator connection, or
robot/perception example has been tested with this adaptation. Static syntax checks
are not runtime validation. The successful Isaac ROS 5.0 demos on Thor do not
validate this 4.6 setup.

## Scope and differences from 4.4

This keeps the existing two-layer installation:

- `Dockerfile.isaac_ros_manipulation` installs the complete NVIDIA manipulation
  bringup plus FoundationPose, RT-DETR, YOLOv8, ESS and their supporting packages.
- `Dockerfile.manymove_xarm` adds the ManyMove/xArm development dependencies.
- `90-isaac-ros-manipulation-bootstrap.user.sh` and
  `91-manymove-xarm-bootstrap.user.sh` retain the 4.4 startup behavior: dependency
  installation, source checkout and colcon builds during activation.
- `scripts/install_foundationpose_isaac_sim_models.sh` retains the one-time model
  download/conversion flow. It skips existing outputs; that is not a compatibility
  check.

The adaptation remains on **Ubuntu 24.04 / ROS 2 Jazzy**, expects a matching
**Isaac ROS release-4.6 base**, and changes the helper environment prefix to
`ISAAC_ROS_4_6_*`. The image keys remain `isaac_ros_manipulation` and `manymove_xarm`.
The manipulation package rename happened in 4.4, not 4.6.

The perception layer explicitly adds `ros-jazzy-isaac-ros-dnn-stereo-decoder`, used
by the 4.6 stock FoundationPose/Isaac Sim launch, and `wget`, used by the model
helper. The custom YOLOv8/trocar launch under `../launch/` still uses camera depth
directly, without ESS; its 4.6 compatibility remains untested.

Installing the complete manipulation bringup preserves the previous environment;
it does not mean ManyMove or ManyForge uses NVIDIA's complete orchestration.
The separate `isaac_manipulator_custom_server` repository is not built by these
hooks. Its older package names/interfaces require their own migration before use.

References:

- [Isaac ROS 4.6 release notes](https://nvidia-isaac-ros.github.io/v/release-4.6/releases/index.html)
- [Isaac ROS 4.6 getting started](https://nvidia-isaac-ros.github.io/v/release-4.6/getting_started/index.html)
- [4.6 FoundationPose/Isaac Sim launch source](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/blob/release-4.6/isaac_ros_foundationpose/launch/isaac_ros_foundationpose_isaac_sim.launch.py)

## Prepare an isolated 4.6 environment when validation is scheduled

Do not downgrade the running Thor 5.0 environment or rewrite its host apt sources
for this draft. Use a separate workspace and container configuration. Keep Jazzy
build/install outputs and TensorRT engines separate from the Lyrical/5.0 workspace.

The Dockerfiles are extension layers, not standalone Ubuntu installers. Their
`BASE_IMAGE` must be supplied by a matching 4.6 NVIDIA image chain with ROS Jazzy
and release-4.6 apt sources already configured. The `ubuntu:24.04` fallback does
not supply those dependencies. Selecting the 4.6 custom directory alone does not
change a 5.0 CLI's base into a 4.6 image. Establish and pin that base using the
versioned NVIDIA setup instructions before building these layers; record its
image digest and installed CUDA/TensorRT versions during validation.

In the isolated environment's effective `.isaac_ros_common-config`, select:

```bash
CONFIG_DOCKER_SEARCH_DIRS=(/etc/isaac-ros-cli/docker ${ISAAC_ROS_WS}/docker ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4/4.6)
```

The default Docker search directory in that environment must also contain the
matching 4.6 base definitions. Remove earlier custom directories with the same
Dockerfile suffixes from this configuration so they cannot shadow this draft.

In the isolated workspace's `.isaac-ros-cli/config.yaml`, use:

```yaml
docker:
  image:
    additional_image_keys:
      - isaac_ros_manipulation
      - manymove_xarm  # omit if ManyMove/xArm is not needed
```

Once the matching base/configuration has been prepared:

```bash
isaac-ros activate --build-local
```

The optional ManyMove hook expects `src/manymove` and `src/isaac_ros_custom_bringup`
to exist. It defaults to `pastoriomarco/xarm_ros2`, branch `jazzy_no_gazebo`, and
clones Groot when absent. Existing checkouts are preserved. These inherited hooks
do not provide a fully pinned source closure: record and qualify the exact source
commits, including Robotiq, serial and topic-based control, when testing the image.

## Runtime options

- `ISAAC_ROS_4_6_BOOTSTRAP=0` skips both startup hooks.
- `ISAAC_ROS_4_6_USE_CYCLONEDDS=1` opts into CycloneDDS; otherwise an unset RMW
  defaults to Fast DDS, as in 4.4. Actual simulator DDS interoperability is untested.
- `ISAAC_ROS_COLCON_BUILD_BASE`, `ISAAC_ROS_COLCON_LOG_BASE` and
  `ISAAC_ROS_COLCON_INSTALL_BASE` override workspace-local build outputs.
- `MANYMOVE_COLCON_WORKERS` controls the optional ManyMove build parallelism.

## Model setup and examples — pending validation

Run model setup **inside the isolated 4.6 container**:

```bash
src/isaac_ros_custom_bringup/isaac_ros_4/4.6/scripts/install_foundationpose_isaac_sim_models.sh
```

The helper downloads FoundationPose ONNX files and builds refine/score engines,
then invokes NVIDIA's RT-DETR and ESS installers with `--eula` (license acceptance).
Verify their installed command/options and output paths before using the helper.
TensorRT engine compatibility depends on GPU and runtime versions, not on filenames.
`FP_MODELS_FORCE=1` deletes the helper's existing engines and regenerates them;
use it only in the isolated 4.6 assets directory, never against the running 5.0 assets.

The intended stock example is:

```bash
ros2 launch isaac_ros_foundationpose isaac_ros_foundationpose_isaac_sim.launch.py
```

For the custom trocar pipeline and ManyMove consumer, the commands in
[the shared 4.x guide](../README.md#perception-example--yolov8--foundationpose-isaac-sim-ported-from-3x)
are the starting reference. Their reported 4.4 observations must not be attributed
to 4.6. Check launch arguments/plugins, model outputs, image/depth alignment, TF,
DDS/QoS and controller/gripper behavior before claiming the examples work.

## Deferred validation

All runtime validation is pending:

- Build the image and verify the selected ROS distro, apt release and package closure.
- Run both startup hooks with the chosen source revisions; verify a second activation.
- Generate/load compatible models and run the stock perception example.
- Run the custom trocar pipeline and verify fresh poses and the camera-to-robot TF chain.
- Exercise the intended ManyMove simulator example, including arm and gripper control;
  exercise cuMotion separately if that example selects it.

Keep the **NOT VALIDATED** status until the relevant results and exact environment
are recorded. No physical robot operation is implied by these preparation steps.
