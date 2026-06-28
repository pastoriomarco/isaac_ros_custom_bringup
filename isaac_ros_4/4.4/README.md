# Isaac ROS manipulation + FoundationPose layer (Isaac ROS 4.4) — `isaac-ros` CLI integration

This directory contains custom Isaac ROS CLI image layers for **Isaac ROS 4.4** (Ubuntu 24.04
Noble / ROS 2 Jazzy / CUDA 13 / TensorRT 10.13.3.9, pairs with Isaac Sim 5.0–5.1):

- `Dockerfile.isaac_ros_manipulation` — manipulation stack **+ FoundationPose / RT-DETR / ESS**
  for the "FoundationPose with Isaac Sim" example
- `90-isaac-ros-manipulation-bootstrap.user.sh` (copied into the image as an entrypoint hook)
- `Dockerfile.manymove_xarm` + `91-manymove-xarm-bootstrap.user.sh` (optional layer that extends
  `isaac_ros_manipulation`)
- `scripts/install_foundationpose_isaac_sim_models.sh` — one-time, idempotent model setup

## What changed from 4.3

Isaac ROS 4.4 **renamed `isaac_manipulator` → `isaac_ros_manipulation`** and moved the planner
backend to cuMotion. This layer follows that rename end-to-end:

| 4.3 | 4.4 |
| --- | --- |
| apt `ros-jazzy-isaac-manipulator-bringup` | apt `ros-jazzy-isaac-ros-manipulation-bringup` |
| image key `isaac_manipulation` | image key `isaac_ros_manipulation` |
| `Dockerfile.isaac_manipulation` | `Dockerfile.isaac_ros_manipulation` |
| `90-isaac-manipulation-bootstrap.user.sh` | `90-isaac-ros-manipulation-bootstrap.user.sh` |
| env `ISAAC_ROS_4_3_*` | env `ISAAC_ROS_4_4_*` |
| apt source `release-4.3` | apt source `release-4.4` |

`ros2_robotiq_gripper`, `serial`, and `topic_based_ros2_control` are still cloned (not apt-provided),
same upstream URLs/branches as 4.3.

## 0) Host prerequisites — move the base image to 4.4 first

> Your custom Dockerfile is layered **on top of NVIDIA's base image**, and it runs `apt install`
> against the **base image's** apt config. NVIDIA's `Dockerfile.isaac_ros` pins the in-image apt
> source to a specific `release-X.Y`. So updating *this* repo is **not enough** — the base image
> (i.e. the installed `isaac-ros-cli` and its Dockerfiles) must be 4.4, or
> `ros-jazzy-isaac-ros-manipulation-bringup` won't resolve against a 4.3 repo.

On this aarch64 Jetson host (current source is `release-4.3 noble-jetpack`):

```bash
# 1. Point the host apt source at release-4.4 (keep the noble-jetpack suffix for Jetson Thor)
sudo sed -i 's#/isaac-ros/release-4\.3 #/isaac-ros/release-4.4 #' \
  /etc/apt/sources.list.d/nvidia-isaac-ros.list
sudo apt-get update

# 2. Upgrade the CLI (this ships /etc/isaac-ros-cli/docker/* pinned to release-4.4)
sudo apt-get install --only-upgrade -y isaac-ros-cli

# 3. Refresh the system Docker assets for the new CLI
sudo isaac-ros init docker
```

Notes:
- `sudo isaac-ros init docker` can overwrite files under `/etc/isaac-ros-cli/` — check it didn't
  clobber a customization you care about (e.g. `.build_image_layers.yaml`, `.isaac_ros_dev-dockerargs`).
- Verify `grep release /etc/isaac-ros-cli/docker/Dockerfile.isaac_ros` now shows `release-4.4`.
- The base image has **no `:4.4` tag** — the CLI pulls it by content hash automatically.

## 1) Point the Dockerfile search path at `4.4/`

`isaac-ros` reads `CONFIG_DOCKER_SEARCH_DIRS` from the first existing `.isaac_ros_common-config`
(precedence: `${ISAAC_ROS_WS}/../scripts/...`, then `${ISAAC_ROS_WS}/scripts/...` (new in 4.4), …,
then `/etc/isaac-ros-cli/.isaac_ros_common-config`). On this host it is
`${ISAAC_ROS_WS}/../scripts/.isaac_ros_common-config`. Point it at `4.4`:

```bash
cat > "${ISAAC_ROS_WS}/../scripts/.isaac_ros_common-config" <<'EOF'
CONFIG_DOCKER_SEARCH_DIRS=(/etc/isaac-ros-cli/docker ${ISAAC_ROS_WS}/docker ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4/4.4)
EOF
```

Keep `/etc/isaac-ros-cli/docker` in the list so the default CLI Dockerfiles stay available, and put
`4.4/` **before** any older `4.3/`/`4.1/` entry (first `Dockerfile.<key>` match wins).

## 2) Use the renamed image key in the CLI build sequence

The build keys come from `docker.image.base_image_keys + additional_image_keys`. Set the workspace
config (`${ISAAC_ROS_WS}/.isaac-ros-cli/config.yaml`) to the **renamed** key:

```yaml
docker:
  image:
    additional_image_keys:
      - isaac_ros_manipulation
#     - manymove_xarm   # optional, layer on top
```

The key must match the Dockerfile suffix: key `isaac_ros_manipulation` → `Dockerfile.isaac_ros_manipulation`.

## 3) Build + activate

```bash
isaac-ros activate --build-local        # add --verbose to debug Dockerfile resolution
```

## 4) FoundationPose + Isaac Sim — one-time model setup

The launch graph chains **RT-DETR → ESS (light_ess) → FoundationPose**, driven through
`isaac_ros_examples`. All three packages (and the `*-models-install` helpers) are baked into the
image; you only need to fetch + convert the models once, **inside the activated container**:

```bash
isaac-ros activate
src/isaac_ros_custom_bringup/isaac_ros_4/4.4/scripts/install_foundationpose_isaac_sim_models.sh
```

That produces (idempotently, skipped if already present):
- `isaac_ros_assets/models/foundationpose/{refine,score}_trt_engine.plan`
- `isaac_ros_assets/models/synthetica_detr/sdetr_grasp.plan` (RT-DETR, via `--eula`)
- `isaac_ros_assets/models/dnn_stereo_disparity/.../light_ess.engine` (ESS, via `--eula`)

Then launch (start Isaac Sim 5.0/5.1 per the Isaac Sim Setup Guide, press **Play**):

```bash
ros2 launch isaac_ros_foundationpose isaac_ros_foundationpose_isaac_sim.launch.py
```

> TensorRT engines are GPU/TensorRT-version specific. If you carried `isaac_ros_assets/` over from a
> different TensorRT (TRT errors like *"engine plan file is not compatible … expecting library version
> 10.13.3.9"*), force a rebuild — it deletes the stale `*.plan`/`*.engine` and regenerates them:
>
> ```bash
> FP_MODELS_FORCE=1 src/isaac_ros_custom_bringup/isaac_ros_4/4.4/scripts/install_foundationpose_isaac_sim_models.sh
> ```

## Runtime knobs (entrypoint hooks)

- `ISAAC_ROS_4_4_BOOTSTRAP=0` — disable both startup bootstrap hooks (image layers stay available).
- `ISAAC_ROS_4_4_USE_CYCLONEDDS=1` — opt into CycloneDDS. Otherwise the hook defaults
  `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` (unset → Fast DDS) for Isaac Sim compatibility, and warns
  because the Isaac for Manipulation docs recommend CycloneDDS for ROS workflows.
- Colcon layout defaults to `${ISAAC_ROS_WS}/{build,log,install}`; override with
  `ISAAC_ROS_COLCON_{BUILD,LOG,INSTALL}_BASE`.

The `manymove_xarm` hook expects `src/manymove` and `src/isaac_ros_custom_bringup` to exist, clones
`src/xarm_ros2` (`pastoriomarco/xarm_ros2 -b jazzy_no_gazebo --recursive`) and `src/Groot` when missing.
