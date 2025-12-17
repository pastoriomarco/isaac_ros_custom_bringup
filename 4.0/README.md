# Isaac Manipulation dev-image customization

This folder contains a custom Isaac ROS CLI image layer (`Dockerfile.isaac_manipulation`) and helper scripts to
install the packages and perception models needed by the Isaac Manipulation tutorials (Isaac Sim).

## What this layer does

- Installs `ros-jazzy-isaac-manipulator-bringup` and the core Isaac ROS packages used by the manipulation
  tutorials (e.g., `isaac_ros_examples`, FoundationPose/RT-DETR/GroundingDINO, ESS/FoundationStereo, SegmentAnything, DOPE).
- Installs Python deps needed by model setup utilities (e.g., `segment-anything`, `gdown`).
- Adds an optional entrypoint hook that can auto-run the model setup on container start.

## How to use with `isaac-ros` CLI

1. Make sure this directory is in the Isaac ROS CLI Dockerfile search path (`CONFIG_DOCKER_SEARCH_DIRS`).
   The CLI reads this from the first `.isaac_ros_common-config` it finds. A convenient override is:

   - `${ISAAC_ROS_WS}/../scripts/.isaac_ros_common-config`

   Example contents:

   ```bash
   CONFIG_DOCKER_SEARCH_DIRS=(/etc/isaac-ros-cli/docker ${ISAAC_ROS_WS}/docker)
   ```

2. Add the image key to your Isaac ROS CLI config (e.g., `~/.config/isaac-ros-cli/config.yaml`):

   ```yaml
   docker:
     image:
       additional_image_keys:
         - realsense
         - isaac_manipulation
   ```

3. Build locally:

   ```bash
   isaac-ros activate --build-local
   ```

4. In the container, run the setup script once:

   ```bash
   /usr/local/bin/isaac-manipulation-setup.sh --show-eula
   ```

   Assets/models are installed under `${ISAAC_ROS_WS}/isaac_ros_assets` and installer scripts skip work if files
   already exist.

## Optional: run setup automatically at container start

Set both environment variables when launching the dev container:

```bash
export ISAAC_ROS_MANIPULATION_AUTO_SETUP=1
export ISAAC_ROS_ACCEPT_EULA=1
```

You can inject env vars permanently via `~/.isaac_ros_dev-dockerargs` by adding:

```bash
-e ISAAC_ROS_MANIPULATION_AUTO_SETUP=1
-e ISAAC_ROS_ACCEPT_EULA=1
```
