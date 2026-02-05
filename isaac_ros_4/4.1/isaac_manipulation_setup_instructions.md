# Isaac Manipulation (Isaac ROS 4.1) setup — annotated command transcript

This document transcribes **every command** found in `src/isaac_manipulation_install_log.md` in the **same order**, including repeated commands, and annotates each one with what it does and whether it succeeded.

References:

- Setup guide (web): https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/tutorials/setup/setup_guide_isaac_sim.html
- Setup guide (local): `src/NVIDIA-ISAAC-ROS.github.io/public/reference_workflows/isaac_for_manipulation/tutorials/setup/setup_guide_isaac_sim.html`

Notes from the log:

- Commands were run in a container at `/workspaces/isaac_ros-dev`.
- The platform appears to be `arm64`/SBSA (e.g., `CMAKE_DEVICE: sbsa`).
- `ISAAC_ROS_WS` is assumed to already be set (it is used by multiple commands).
- Some commands appear in the log **without a prompt** (e.g., `export ...`). They were likely pasted right after the previous prompt line.

## Command sequence (chronological)

### 1) Clone the `isaac_manipulator` repository

```bash
cd ${ISAAC_ROS_WS}/src && git clone --recursive -b release-4.1 https://github.com/NVIDIA-ISAAC-ROS/isaac_manipulator.git isaac_manipulator
```

- Purpose: Fetch the Isaac Manipulator sources for Isaac ROS `release-4.1`.
- Result: Succeeded.

### 2) Install CycloneDDS RMW for ROS 2 Jazzy

```bash
sudo apt-get install -y ros-jazzy-rmw-cyclonedds-cpp
```

- Purpose: Install the CycloneDDS middleware implementation package for ROS 2 Jazzy.
- Result: Succeeded (packages were installed and configured).

### 3) Select CycloneDDS as the active RMW implementation

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

- Purpose: Configure ROS 2 in the current shell to use CycloneDDS.
- Result: Succeeded (no output expected for `export`).
- Notes: This only affects the current shell session unless persisted (e.g., in `.bashrc`).

### 4) Update apt, install `pip`, then install Python deps (`cupy`, `hdbscan`)

```bash
sudo apt update && sudo apt install -y python3-pip && \
python3 -m pip install --break-system-packages cupy-cuda13x && \
python3 -m pip install --break-system-packages hdbscan
```

- Purpose: Refresh apt indices, ensure `python3-pip` exists, then install Python packages needed by parts of the manipulation workflow.
- Result: Succeeded (`cupy-cuda13x` and `hdbscan` installed).
- Notes: `pip` defaulted to a user install because system site-packages was not writeable.

### 5) Update apt indices (again)

```bash
sudo apt-get update
```

- Purpose: Refresh apt indices (same goal as `apt update`; often done after changing repos or as a troubleshooting step).
- Result: Succeeded (with a warning about a legacy keyring for one repo).

### 6) Install ROS/apt dependencies via `rosdep` (attempt 1/4)

```bash
rosdep update && rosdep install --from-paths ${ISAAC_ROS_WS}/src/isaac_manipulator/isaac_manipulator_bringup --ignore-src -y
```

- Purpose: Update rosdep rules and install system dependencies needed by `isaac_manipulator_bringup`.
- Result: Failed.
- Failure reason (from log): `E: Unable to locate package ros-jazzy-isaac-ros-common`.
- Why it was repeated: The apt repository configuration did not provide the `ros-jazzy-isaac-ros-common` package, so the repo list was inspected/edited and `rosdep` was re-run.

### 7) Update apt indices (after `rosdep` failure)

```bash
sudo apt update
```

- Purpose: Refresh apt indices to see if the missing package becomes available.
- Result: Succeeded, but did not resolve the missing `ros-jazzy-isaac-ros-common` package by itself.

### 8) Inspect the NVIDIA Isaac ROS apt source list

```bash
cat /etc/apt/sources.list.d/nvidia-isaac-ros.list
```

- Purpose: Confirm which Isaac ROS apt repository is configured.
- Result: Succeeded (printed the configured `release-4.1` apt line).

### 9) Edit the Isaac ROS apt source list (attempt 1/2)

```bash
sudo nano /etc/apt/sources.list.d/nvidia-isaac-ros.list
```

- Purpose: Edit the apt source list to fix the missing package issue.
- Result: Failed.
- Failure reason (from log): `sudo: nano: command not found` (editor not installed).
- Why it was repeated: `nano` was installed and the edit was attempted again.

### 10) Install `nano`

```bash
sudo apt install nano
```

- Purpose: Install the `nano` editor so the apt source list can be edited.
- Result: Succeeded.

### 11) Edit the Isaac ROS apt source list (attempt 2/2)

```bash
sudo nano /etc/apt/sources.list.d/nvidia-isaac-ros.list
```

- Purpose: Update the Isaac ROS apt source configuration so `ros-jazzy-isaac-ros-common` can be found.
- Result: Succeeded (no terminal output; `nano` opened/closed).
- Inferred change (based on subsequent `apt update` output): the repo suite was updated to a `noble-jetpack` variant so JetPack/arm64 packages became available.

### 12) Update apt indices after editing the repo list

```bash
sudo apt update
```

- Purpose: Refresh apt indices so the newly configured repository is used.
- Result: Succeeded (the `noble-jetpack` repo started being queried).

### 13) Check if the missing package is now available

```bash
apt-cache policy ros-jazzy-isaac-ros-common
```

- Purpose: Confirm apt can see a candidate version for `ros-jazzy-isaac-ros-common`.
- Result: Succeeded (a candidate version was shown from the `noble-jetpack` repo).

### 14) Search apt for the missing package (ran immediately after the policy check)

```bash
apt-cache search isaac-ros-common
```

- Purpose: Double-check the package name/availability in apt.
- Result: Succeeded (package description was returned).

### 15) Install ROS/apt dependencies via `rosdep` (attempt 2/4)

```bash
rosdep update && rosdep install --from-paths ${ISAAC_ROS_WS}/src/isaac_manipulator/isaac_manipulator_bringup --ignore-src -y
```

- Purpose: Retry dependency installation now that the apt repo is fixed.
- Result: Failed.
- Failure reason (from log): a `pip` download timed out while configuring `python3-warp-lang-pip-shim` (`TimeoutError: The read operation timed out`), which caused `dpkg` to leave packages unconfigured and return `error code (1)`.
- Why it was repeated: Re-running `rosdep install` is a common way to recover after transient network issues or partially configured packages.

### 16) Install ROS/apt dependencies via `rosdep` (attempt 3/4)

```bash
rosdep update && rosdep install --from-paths ${ISAAC_ROS_WS}/src/isaac_manipulator/isaac_manipulator_bringup --ignore-src -y
```

- Purpose: Retry dependency installation after the previous `pip` timeout.
- Result: Failed.
- Failure reason (from log): the apt operation was interrupted and `rosdep` was aborted (`^C`), while installing `ros-jazzy-rqt-image-view`.
- Why it was repeated: A subsequent run was needed to complete the interrupted install.

### 17) Install ROS/apt dependencies via `rosdep` (attempt 4/4)

```bash
rosdep update && rosdep install --from-paths ${ISAAC_ROS_WS}/src/isaac_manipulator/isaac_manipulator_bringup --ignore-src -y
```

- Purpose: Finish dependency installation after the interruption.
- Result: Succeeded (`#All required rosdeps installed successfully`).
- Notes: The log contains non-fatal warnings during this run (e.g., `pip` dependency conflict warnings), but `rosdep` completed successfully.

### 18) Install Segment Anything (Python) from GitHub

```bash
pip install --no-deps --break-system-packages git+https://github.com/facebookresearch/segment-anything.git
```

- Purpose: Install Meta's Segment Anything Python package from source (without pulling dependencies).
- Result: Succeeded (`Successfully installed segment-anything-1.0`).

### 19) Change directory to the workspace root

```bash
cd ${ISAAC_ROS_WS}
```

- Purpose: Ensure subsequent build commands run from the workspace root.
- Result: Succeeded (no output expected for `cd`).

### 20) Enable automatic asset/model downloads during the build

```bash
export MANIPULATOR_INSTALL_ASSETS=1
```

- Purpose: Tell the build/bringup tooling to download required models/assets into `isaac_ros_assets`.
- Result: Succeeded (no output expected for `export`).

### 21) Build up to `isaac_manipulator_bringup`

```bash
colcon build --symlink-install --packages-up-to isaac_manipulator_bringup
```

- Purpose: Build the Isaac Manipulator packages needed for the workflow.
- Result: Succeeded (`Summary: 12 packages finished`).
- Notes: The build printed warnings/stderr (e.g., `listing git files failed - pretending there aren't any`) and asset setup messages; some "ERROR:" lines appeared before assets were downloaded, but the overall build and asset setup completed.

## Continuation (from `isaac_manipulation_install_log_2.md`)
Commands also from https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/tutorials/setup/setup_with_gripper.html, but only the section `Set Up Perception Deep Learning Models`

The following steps continue the transcript using `src/isaac_manipulation_install_log_2.md`.

### 22) Source the workspace overlay

```bash
source install/setup.bash
```

- Purpose: Load the workspace’s `install/` overlay into the current shell so `ros2` can find built packages.
- Result: Succeeded (no output in log).

### 23) Clone Robotiq gripper dependencies (Robotiq + `serial`)

```bash
cd ${ISAAC_ROS_WS}/src && \
  git clone --recursive https://github.com/NVIDIA-ISAAC-ROS/ros2_robotiq_gripper && \
  git clone -b ros2 https://github.com/tylerjw/serial
```

- Purpose: Fetch the Robotiq gripper ROS 2 stack (Isaac ROS fork) and a `serial` package version compatible with ROS 2.
- Result: Succeeded (both repos cloned successfully).

### 24) Return to the workspace root

```bash
cd ${ISAAC_ROS_WS}
```

- Purpose: Run subsequent `colcon` builds from the workspace root.
- Result: Succeeded.

### 25) Build Robotiq gripper dependencies, then re-source the overlay

```bash
colcon build --symlink-install --packages-select-regex robotiq* serial --cmake-args "-DBUILD_TESTING=OFF" && \
source install/setup.bash  # Source the workspace after building gripper dependencies
```

- Purpose: Build only the `robotiq*` packages plus `serial` (disabling tests), then refresh the environment so the new packages are discoverable.
- Result: Succeeded (`Summary: 5 packages finished`).
- Notes: The log shows `colcon` override warnings (packages already present in `/opt/ros/jazzy`) and compiler warnings, but the build completed.

### 26) Clone `topic_based_ros2_control`

```bash
cd ${ISAAC_ROS_WS}/src && git clone https://github.com/karanchahal-nv/topic_based_ros2_control
```

- Purpose: Fetch the `topic_based_ros2_control` package required by the workflow.
- Result: Succeeded.

### 27) Install `topic_based_ros2_control` dependencies via `rosdep`

```bash
rosdep update && rosdep install --from-paths ${ISAAC_ROS_WS}/src/topic_based_ros2_control --ignore-src -y
```

- Purpose: Install missing system/ROS dependencies for `topic_based_ros2_control`.
- Result: Succeeded (`#All required rosdeps installed successfully`).
- Notes: `rosdep` installed additional packages via apt (e.g., `ros-jazzy-ros-testing`).

### 28) Build `topic_based_ros2_control`, then re-source the overlay

```bash
cd ${ISAAC_ROS_WS} && colcon build --symlink-install --packages-up-to topic_based_ros2_control && source install/setup.bash
```

- Purpose: Build `topic_based_ros2_control` and its dependencies, then refresh the environment so `ros2` can find it.
- Result: Succeeded (`Summary: 1 package finished`).
- Notes: The log shows a `colcon` override warning (already in `/opt/ros/jazzy`) and compiler warnings, but the build completed.

### 29) Point the workflow at the bringup config directory

```bash
export ISAAC_MANIPULATOR_WORKFLOW_CONFIG_DIR="${ISAAC_ROS_WS}/src/isaac_manipulator/isaac_manipulator_bringup/params"
```

- Purpose: Tell the workflow where to find the default configuration/params YAMLs.
- Result: Succeeded (no output expected for `export`).

### 30) Install ESS (stereo disparity) models/assets (EULA-gated)

```bash
ros2 run isaac_ros_ess_models_install install_ess_models.sh --eula
```

- Purpose: Download and set up ESS model assets (and build TensorRT engines) into `isaac_ros_assets`.
- Result: Succeeded (the log shows `&&&& PASSED TensorRT.trtexec ...`).
- Notes: Interactive — the prompt `Do you accept? [y/n]` was answered with `y`.

### 31) Select FoundationStereo model resolution (first time)

```bash
export FOUNDATIONSTEREO_MODEL_RES=low_res
```

- Purpose: Configure subsequent FoundationStereo model installation to use the `low_res` variant.
- Result: Succeeded.
- Why it was repeated later: The variable was set again immediately before running the perception model setup script, likely to ensure the correct resolution is used in that invocation (and/or because this was a fresh shell context).

### 32) Install FoundationStereo models/assets (EULA-gated)

```bash
ros2 run isaac_ros_foundationstereo_models_install install_foundationstereo_models.sh --eula \
--model_res low_res
```

- Purpose: Download FoundationStereo ONNX and convert it to a TensorRT engine for the selected resolution.
- Result: Succeeded (the log shows `&&&& PASSED TensorRT.trtexec ...`).
- Notes: Interactive — the prompt `Do you accept? [y/n]` was answered with `y`.

### 33) Install FoundationPose models/assets (EULA-gated)

```bash
ros2 run isaac_ros_foundationpose_models_install install_foundationpose_models.sh --eula
```

- Purpose: Download FoundationPose model files into `isaac_ros_assets`.
- Result: Succeeded (downloads completed; no errors shown).
- Notes: Interactive — the prompt `Do you accept? [y/n]` was answered with `y`.

### 34) Install RT-DETR models/assets (EULA-gated)

```bash
ros2 run isaac_ros_rtdetr_models_install install_rtdetr_models.sh --eula
```

- Purpose: Download RT-DETR model assets and convert to TensorRT engines.
- Result: Succeeded (the log shows `&&&& PASSED TensorRT.trtexec ...`).
- Notes: Interactive — the prompt `Do you accept? [y/n]` was answered with `y`.

### 35) Install GroundingDINO models/assets (EULA-gated)

```bash
ros2 run isaac_ros_grounding_dino_models_install install_grounding_dino_models.sh --eula
```

- Purpose: Download the GroundingDINO ONNX model and convert it to a TensorRT engine.
- Result: Succeeded (the log shows `&&&& PASSED TensorRT.trtexec ...`).
- Notes: Interactive — the prompt `Do you accept? [y/n]` was answered with `y`.

### 36) Re-run Segment Anything install from GitHub (repeat)

```bash
pip install --no-deps --break-system-packages git+https://github.com/facebookresearch/segment-anything.git
```

- Purpose: Install (or re-install) the Segment Anything Python package from source.
- Result: Indeterminate from the captured output (the log ends after `Preparing metadata (setup.py) ... done` with no explicit success/error line).
- Why it was repeated: It was already installed earlier in `isaac_manipulation_install_log.md`; this re-run was likely to ensure it exists in the current environment or to follow the guide steps verbatim.

### 37) Enable manipulator asset installation (repeat)

```bash
export MANIPULATOR_INSTALL_ASSETS=1
```

- Purpose: Enable automatic asset/model setup for manipulator perception/bringup tooling.
- Result: Succeeded.
- Why it was repeated: It was also set during the earlier `colcon build` step; this re-export ensures it is set for the upcoming model setup command.

### 38) Select FoundationStereo model resolution (second time; repeat)

```bash
export FOUNDATIONSTEREO_MODEL_RES=low_res
```

- Purpose: Ensure the perception model setup uses the `low_res` FoundationStereo variant.
- Result: Succeeded.
- Second-time outcome: No errors were reported; the following model setup completed successfully.

### 39) Set up all perception models for the manipulator workflow

```bash
ros2 run isaac_manipulator_asset_bringup setup_perception_models.py --models all
```

- Purpose: Download/prepare all required perception assets (FoundationPose, DOPE, SAM/SAM2, UR DNN Policy, etc.) into `isaac_ros_assets`.
- Result: Succeeded (`INFO: All requested models were set up successfully!`).
- Notes: Many assets were already present, so the script skipped downloads/copies where possible.

## Continuation (from `isaac_manipulation_install_log_3.md` — FoundationPose tutorial)

The following steps continue the transcript using `src/isaac_manipulation_install_log_3.md`.

### 40) Update apt indices

```bash
sudo apt-get update
```

- Purpose: Refresh apt package indices before installing tutorial dependencies.
- Result: Succeeded (with a warning about a legacy keyring for one repo).

### 41) Ensure the FoundationPose package is installed

```bash
sudo apt-get install -y ros-jazzy-isaac-ros-foundationpose
```

- Purpose: Install the `isaac_ros_foundationpose` ROS package for Jazzy from apt.
- Result: Succeeded; package was already installed (`already the newest version`).
- Why it was run: Likely to confirm the tutorial dependency is present even if it was installed earlier via `rosdep`.

### 42) Convert FoundationPose refine model ONNX → TensorRT engine

```bash
/usr/src/tensorrt/bin/trtexec --onnx=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_model.onnx --saveEngine=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan --minShapes=input1:1x160x160x6,input2:1x160x160x6 --optShapes=input1:1x160x160x6,input2:1x160x160x6 --maxShapes=input1:42x160x160x6,input2:42x160x160x6
```

- Purpose: Build a TensorRT plan (`refine_trt_engine.plan`) from the FoundationPose refine ONNX model using the specified min/opt/max dynamic input shapes.
- Result: Succeeded (`&&&& PASSED TensorRT.trtexec ...`).

### 43) Convert FoundationPose score model ONNX → TensorRT engine

```bash
/usr/src/tensorrt/bin/trtexec --onnx=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_model.onnx --saveEngine=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_trt_engine.plan --minShapes=input1:1x160x160x6,input2:1x160x160x6 --optShapes=input1:1x160x160x6,input2:1x160x160x6 --maxShapes=input1:252x160x160x6,input2:252x160x160x6
```

- Purpose: Build a TensorRT plan (`score_trt_engine.plan`) from the FoundationPose score ONNX model using the specified min/opt/max dynamic input shapes.
- Result: Succeeded (`&&&& PASSED TensorRT.trtexec ...`).

### 44) Install Isaac ROS examples package

```bash
sudo apt-get install -y ros-jazzy-isaac-ros-examples
```

- Purpose: Install the Isaac ROS examples bundle (often used by tutorials for sample launch files, assets, or helper scripts).
- Result: Succeeded (package was newly installed).

### 45) Install RViz2 and VisionMsgs RViz plugins

```bash
sudo apt-get install -y ros-jazzy-rviz2 ros-jazzy-vision-msgs-rviz-plugins
```

- Purpose: Install visualization tooling needed for FoundationPose tutorial visualization in RViz.
- Result: Succeeded (`ros-jazzy-rviz2` was already installed; `ros-jazzy-vision-msgs-rviz-plugins` was installed).

### 46) Source the system ROS 2 Jazzy environment

```bash
source /opt/ros/jazzy/setup.bash
```

- Purpose: Ensure the current shell has the baseline ROS 2 Jazzy environment (useful before launching RViz/ROS nodes if the shell was fresh).
- Result: Succeeded (no output in log).
