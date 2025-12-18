# Isaac Manipulation dev-image customization

This folder contains a custom Isaac ROS CLI image layer (`Dockerfile.isaac_manipulation`) and helper scripts to
install the packages and perception models needed by the Isaac Manipulation tutorials (Isaac Sim).

## What this layer does

- Installs `ros-jazzy-isaac-manipulator-bringup` and the core Isaac ROS packages used by the manipulation
  tutorials (e.g., `isaac_ros_examples`, FoundationPose/RT-DETR/GroundingDINO, ESS/FoundationStereo, SegmentAnything, DOPE).
- Installs Python deps needed by model setup utilities (e.g., `segment-anything`, `gdown`).
- Adds an optional entrypoint hook that can auto-run the model setup on container start.
- Provides a host-side helper script to prefetch NGC quickstart assets into `${ISAAC_ROS_WS}/isaac_ros_assets`.
- Versioning behavior:
  - Host prefetch (`scripts/prefetch_quickstart_assets_host.sh`) is driven by `versioning.quickstart_assets` in the config:
    - Default config uses `mode: latest`, so it floats to the newest `4.Y.Z` available on NGC for major `4`.
    - To *pin* to a specific minor (e.g., `4.0.x`), set `mode: pinned_minor` and `minor: 0` (it will still pick the latest
      patch `x` within that minor).
    - CLI flags override the config (`--minor N` or `--latest-minor`).
  - In-container model install uses the upstream installer scripts shipped in the installed Isaac ROS packages; any NGC/model
    version pinning is controlled by those scripts (this layer just invokes them).

## Config-driven components

To avoid downloading/installing models and assets you don't need, both the host prefetch script and the in-container
setup script can be driven by a YAML config:

- Configure components and versioning in `src/isaac_ros_custom_bringup/isaac_ros_4/config/isaac_manipulation_assets.yaml`.
  - Host prefetch reads this file.
  - The Docker layer copies it into the image at `/usr/local/share/isaac-manipulation/isaac_manipulation_assets.yaml` for in-container setup.

Keys under `components` gate both the corresponding quickstart asset bundle (host) and the corresponding model install
(container):

- `ess`
- `foundationstereo` (also supports `model_res`: `low_res`/`high_res`/`both`)
- `foundationpose`
- `rtdetr`
- `grounding_dino`
- `manipulator_assets` (DOPE weights, Segment Anything conversion, sample objects via `setup_perception_models.py`)

## How to use with `isaac-ros` CLI

1. Make sure this directory is in the Isaac ROS CLI Dockerfile search path (`CONFIG_DOCKER_SEARCH_DIRS`).
   The CLI reads this from the first `.isaac_ros_common-config` it finds. A convenient override is:

   - `${ISAAC_ROS_WS}/../scripts/.isaac_ros_common-config`

   Example contents:

   ```bash
   CONFIG_DOCKER_SEARCH_DIRS=(/etc/isaac-ros-cli/docker ${ISAAC_ROS_WS}/docker ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4)
   ```

2. Add the image key to your Isaac ROS CLI config (e.g., `~/.config/isaac-ros-cli/config.yaml`):

   ```yaml
   docker:
     image:
       additional_image_keys:
         - realsense
         - isaac_manipulation
   ```

3. Optional (host): prefetch NGC quickstart assets **before** starting a dev container:

   ```bash
   ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4/scripts/prefetch_quickstart_assets_host.sh
   # or (when running from the workspace root):
   ./src/isaac_ros_custom_bringup/isaac_ros_4/scripts/prefetch_quickstart_assets_host.sh
   ```

   This downloads and extracts `quickstart.tar.gz` bundles for enabled components in the config (defaults to all):
   `isaac_ros_foundationpose`, `isaac_ros_ess`, `isaac_ros_rtdetr`, `isaac_ros_foundationstereo`, `isaac_ros_grounding_dino`.

   Notes:
   - Requires `curl`, `jq`, `tar`, and `python3` on the host.
   - Idempotent by default; pass `--force` to re-download.
   - If `ISAAC_ROS_WS` is not set, the script infers it from its own location; or pass `--ws /path/to/isaac_ros_ws`.

4. Build locally:

   ```bash
   isaac-ros activate --build-local
   ```

5. In the container, run the setup script once:

   ```bash
   /usr/local/bin/isaac-manipulation-setup.sh --eula
   ```

   Assets/models are installed under `${ISAAC_ROS_WS}/isaac_ros_assets` and installer scripts skip work if files
   already exist.

## Optional: slim image builds

`Dockerfile.isaac_manipulation` supports build args to skip installing packages for disabled components:

- `ISAAC_MANIPULATION_ENABLE_ESS`
- `ISAAC_MANIPULATION_ENABLE_FOUNDATIONSTEREO`
- `ISAAC_MANIPULATION_ENABLE_FOUNDATIONPOSE`
- `ISAAC_MANIPULATION_ENABLE_RTDETR`
- `ISAAC_MANIPULATION_ENABLE_GROUNDING_DINO`
- `ISAAC_MANIPULATION_ENABLE_MANIPULATOR_ASSETS`

To derive these from the config file:

```bash
./src/isaac_ros_custom_bringup/isaac_ros_4/scripts/print_docker_build_args_from_config.sh
```

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
