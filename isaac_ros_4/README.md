# Isaac Manipulation dev-image customization

This folder contains custom Isaac ROS CLI image layers (`Dockerfile.isaac_manipulation`,
`Dockerfile.isaac_manipulation_source`, `Dockerfile.isaac_manipulation_rsl_rl`) and helper scripts to install the
packages and perception models needed by the Isaac Manipulation tutorials (Isaac Sim).

## Prerequisites

This README assumes your host is already set up with Isaac ROS CLI and GPU-enabled Docker as described in the Isaac ROS
*Developer Environment Setup* docs, including:

- [Getting Started](https://nvidia-isaac-ros.github.io/getting_started/index.html)
- [Isaac ROS Development Environment](https://nvidia-isaac-ros.github.io/concepts/dev_env/index.html)
- [Compute Setup](https://nvidia-isaac-ros.github.io/concepts/dev_env/index.html)
- [Sensors Setup](https://nvidia-isaac-ros.github.io/getting_started/sensors/index.html)

For Thor also check:
- [Quick Start Guide](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html)
- [Docker Setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_docker.html)

## What this layer does

- **Binary layer (`Dockerfile.isaac_manipulation`)**: installs the Isaac Manipulation Debian packages
  (cuMotion, nvblox, NITROS, perception stacks, etc.) and wires in the asset setup hook.
- **Source layer (`Dockerfile.isaac_manipulation_source`)**: installs build tooling + all `rosdep` dependencies
  for the manipulation source repos (using a temporary workspace at image build time), but **does not** build the
  packages inside the image.
- Adds entrypoint hooks that:
  - auto-build the required source packages on first container start (or when forced), and
  - auto-setup models/assets (idempotent; skips if already prepared).
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
- `segment_anything2` (optional; ONNX export is x86-only, copy to Jetson/Thor)
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
   - `~/.isaac_ros_dev-dockerargs` (generated)
   - Clones missing repositories from `source/isaac_ros_manipulation.repos` into `${ISAAC_ROS_WS}/src`
     (add `--pull` to update all repos)

   Manual copy (equivalent):

   ```bash
   mkdir -p ${ISAAC_ROS_WS}/../scripts
   cp ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4/setup_files/.isaac_ros_common-config ${ISAAC_ROS_WS}/../scripts/.isaac_ros_common-config

   mkdir -p ~/.config/isaac-ros-cli
   cp ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4/setup_files/isaac-ros-cli.config.yaml ~/.config/isaac-ros-cli/config.yaml

   cat <<'EOF' > ~/.isaac_ros_dev-dockerargs
    -e ISAAC_ROS_MANIPULATION_AUTO_BUILD=1
    -e ISAAC_ROS_MANIPULATION_FORCE_BUILD=0
    -e ISAAC_ROS_MANIPULATION_AUTO_SETUP=1
    -e ISAAC_ROS_MANIPULATION_FORCE_ASSET_SETUP=0
    -e ISAAC_ROS_ACCEPT_EULA=1
    EOF
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

4. Build locally:

   ```bash
   isaac-ros activate --build-local
   ```

## Source build layer (build Isaac ROS from source)

To build Isaac ROS packages from source at container start, use `Dockerfile.isaac_manipulation_source`
and the repo list in `src/isaac_ros_custom_bringup/isaac_ros_4/source/isaac_ros_manipulation.repos`:

1. Edit `~/.config/isaac-ros-cli/config.yaml` and replace `isaac_manipulation` with
   `isaac_manipulation_source` under `additional_image_keys`.
2. (Optional) Pin branches/commits by editing
   `src/isaac_ros_custom_bringup/isaac_ros_4/source/isaac_ros_manipulation.repos`.
3. Rebuild the image layers:

   ```bash
   isaac-ros activate --build-local
   ```

Notes:
- The source layer preserves the default `ISAAC_ROS_WS` behavior (typically `/workspaces/isaac_ros-dev`). When no host mount is present, `/workspaces/isaac_ros-dev` is symlinked to `/opt/isaac_ros_ws`.
- The repo list covers the packages pulled in by `ros-jazzy-isaac-manipulator-bringup` and Isaac Sim setup dependencies
  (cuMotion, nvblox, NITROS, perception stacks, etc.).
- **Packages are built at container start**, not during the image build. The auto-build hook runs if
  `${ISAAC_ROS_WS}/install/setup.bash` is missing, or if forced.

### Auto-build behavior

At container start, `/usr/local/bin/isaac-manipulation-build.sh` runs automatically when
`ISAAC_ROS_MANIPULATION_AUTO_BUILD=1` (default). It builds a **specific list** of packages using
`colcon build --packages-up-to ...` (not the entire workspace). The target list is derived from
`config/isaac_manipulation_assets.yaml`, plus `isaac_ros_nvblox` (always included).

To force a rebuild, set:

```bash
-e ISAAC_ROS_MANIPULATION_FORCE_BUILD=1
```

To disable auto-build:

```bash
-e ISAAC_ROS_MANIPULATION_AUTO_BUILD=0
```

To override the CUDA arch list used by cuRobo:

```bash
-e ISAAC_MANIPULATION_TORCH_CUDA_ARCH_LIST=8.9+PTX
```

## Optional: 

### Enable RSL-RL dependencies (optional)

RSL-RL is off by default. To install the RL dependencies (tensordict + rsl-rl-lib) during image build, add
`isaac_manipulation_rsl_rl` to `additional_image_keys` in `~/.config/isaac-ros-cli/config.yaml`, then rebuild:

```bash
isaac-ros activate --build-local
```

### Disable running setup automatically at container start

   At container start the helper files are set to do this automatically.

   To disable, edit `~/.isaac_ros_dev-dockerargs` and set:

   ```bash
   -e ISAAC_ROS_MANIPULATION_AUTO_BUILD=0
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

Note: `Dockerfile.isaac_manipulation_source` always builds the repositories listed in
`src/isaac_ros_custom_bringup/isaac_ros_4/source/isaac_ros_manipulation.repos`. The **auto-build target list**
is derived from `config/isaac_manipulation_assets.yaml` (plus `isaac_ros_nvblox` which is always included). To slim
a source build further, edit the config and/or the repo list (or add `COLCON_IGNORE` files in the workspace).

---

## DISCLAIMER

This package builds on and integrates software components from [NVIDIA’s Isaac ROS](https://nvidia-isaac-ros.github.io/index.html) and [Isaac SIM](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) platforms. All copyrights, trademarks, and ownership of the original software remain with NVIDIA Corporation.

This tutorial and the associated launch files are **community-created** and are **not officially maintained, endorsed, or supported by NVIDIA**.

It is intended to serve as a **reference and example** for combining Isaac ROS packages (e.g., YOLOv8, FoundationPose) and Isaac SIM in a practical perception pipeline. While care has been taken to test the setup, **there are no guarantees of correctness, completeness, or compatibility** with future Isaac ROS or Isaac SIM releases.

Use this material **at your own discretion and risk**. For official documentation, support, and best practices, refer to the official NVIDIA documentation.

---
