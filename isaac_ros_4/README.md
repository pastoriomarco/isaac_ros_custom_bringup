# Isaac Manipulation dev-image customization

This folder contains a custom Isaac ROS CLI image layer (`Dockerfile.isaac_manipulation`) and helper scripts to
install the packages and perception models needed by the Isaac Manipulation tutorials (Isaac Sim).

## Prerequisites

This README assumes your host is already set up with Isaac ROS CLI and GPU-enabled Docker as described in the Isaac ROS
*Developer Environment Setup* docs, including:

- `pip install termcolor --break-system-packages`
- `sudo apt-get install isaac-ros-cli` and `sudo isaac-ros init docker`
- Docker installed and NVIDIA Container Toolkit configured for Docker (for GPU access)

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

Keys under `components` gate the relevant downloads/installs during host prefetch and in-container setup:

- `ess`
- `foundationstereo` (also supports `model_res`: `low_res`/`high_res`/`both`)
- `foundationpose` (also downloads sample object mesh/texture assets via `setup_perception_models.py`)
- `rtdetr`
- `grounding_dino`
- `dope` (downloads DOPE weights via `setup_perception_models.py`)
- `segment_anything` (downloads SAM checkpoint/assets + performs PTH->ONNX conversion via `setup_perception_models.py` on x86)
- `gear_assembly` (downloads UR DNN Policy assets for gear assembly via `setup_perception_models.py`)

## How to use with `isaac-ros` CLI

1. Clone this repo into your workspace:

   ```bash
   cd ${ISAAC_ROS_WS}/src
   git clone https://github.com/pastoriomarco/isaac_ros_custom_bringup
   ```

2. Create the Isaac ROS CLI helper files (recommended: run the bootstrap script):

   **WARNING**: update the helper files in this section according to your needs!

   ```bash
   bash ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4/scripts/bootstrap_isaac_ros_cli_files.sh
   ```

   This script writes:
   - `${ISAAC_ROS_WS}/../scripts/.isaac_ros_common-config` (from `src/isaac_ros_custom_bringup/isaac_ros_4/setup_files/.isaac_ros_common-config`)
   - `~/.config/isaac-ros-cli/config.yaml` (from `src/isaac_ros_custom_bringup/isaac_ros_4/setup_files/isaac-ros-cli.config.yaml`)
   - `~/.isaac_ros_dev-dockerargs` (from `src/isaac_ros_custom_bringup/isaac_ros_4/setup_files/.isaac_ros_dev-dockerargs`)

   Manual copy (equivalent):

   ```bash
   mkdir -p ${ISAAC_ROS_WS}/../scripts
   cp ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4/setup_files/.isaac_ros_common-config ${ISAAC_ROS_WS}/../scripts/.isaac_ros_common-config

   mkdir -p ~/.config/isaac-ros-cli
   cp ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4/setup_files/isaac-ros-cli.config.yaml ~/.config/isaac-ros-cli/config.yaml

   cp ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4/setup_files/.isaac_ros_dev-dockerargs ~/.isaac_ros_dev-dockerargs
   ```

3. Prefetch NGC quickstart assets **before** starting a dev container:

   ```bash
   bash ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4/scripts/prefetch_quickstart_assets_host.sh
   ```

   This downloads and extracts `quickstart.tar.gz` bundles for enabled components in the config (defaults to all):
   `isaac_ros_foundationpose`, `isaac_ros_ess`, `isaac_ros_rtdetr`, `isaac_ros_foundationstereo`, `isaac_ros_grounding_dino`.

   Notes:
   - Requires `curl`, `jq`, `tar`, and `python3` on the host.
   - Idempotent by default; pass `--force` to re-download.
   - If `ISAAC_ROS_WS` is not set, the script infers it from its own location; or pass `--ws /path/to/isaac_ros_ws`.

5. Build locally:

   ```bash
   isaac-ros activate --build-local
   ```

## Optional: 

### Disable running setup automatically at container start

   At container start you will have to download and convert the models: the helper files are set to it's done automatically.

   To disable, edit `~/.isaac_ros_dev-dockerargs` and set:

   ```bash
   -e ISAAC_ROS_MANIPULATION_AUTO_SETUP=0
   -e ISAAC_ROS_ACCEPT_EULA=0
   ```
   
   If you disable auto-setup, you can run it from inside the container with:

   ```bash
   /usr/local/bin/isaac-manipulation-setup.sh --eula
   ```

   Assets/models are installed under `${ISAAC_ROS_WS}/isaac_ros_assets` and installer scripts skip work if files
   already exist.

### Slim image builds

`Dockerfile.isaac_manipulation` supports build args to skip installing packages for disabled components:

- `ISAAC_MANIPULATION_ENABLE_ESS`
- `ISAAC_MANIPULATION_ENABLE_FOUNDATIONSTEREO`
- `ISAAC_MANIPULATION_ENABLE_FOUNDATIONPOSE`
- `ISAAC_MANIPULATION_ENABLE_RTDETR`
- `ISAAC_MANIPULATION_ENABLE_GROUNDING_DINO`
- `ISAAC_MANIPULATION_ENABLE_SEGMENT_ANYTHING`
- `ISAAC_MANIPULATION_ENABLE_DOPE`

To derive these from the config file:

```bash
bash ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4/scripts/print_docker_build_args_from_config.sh
```
