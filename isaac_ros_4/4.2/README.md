# Isaac Manipulation layer (Isaac ROS 4.2) — `isaac-ros` CLI integration

This directory contains custom Isaac ROS CLI image layers:

- `Dockerfile.isaac_manipulation`
- `90-isaac-manipulation-bootstrap.user.sh` (copied into the image as an entrypoint hook)
- `Dockerfile.manymove_xarm`
- `91-manymove-xarm-bootstrap.user.sh` (optional layer that extends `isaac_manipulation`)

The `isaac-ros` CLI will only build a layer if:

1. it can **find** the corresponding Dockerfile (Dockerfile search path), and
2. the matching key is in the CLI **build sequence** (`additional_image_keys`).

The `4.2` bootstrap hooks rerun the required setup in-container and skip `git clone` when the target
directory already exists. The `manymove_xarm` hook expects `src/manymove` and `src/isaac_ros_custom_bringup`
to already exist in the mounted workspace, clones `src/xarm_ros2` from
`https://github.com/pastoriomarco/xarm_ros2.git -b jazzy_no_gazebo --recursive` when missing, and also bootstraps
`src/Groot` following ManyMove's workspace instructions.

The 4.0 folder solves (1) by writing a `.isaac_ros_common-config` that extends
`CONFIG_DOCKER_SEARCH_DIRS`. Do the same for `4.2/`.

## 1) Add this folder to the Dockerfile search path

`isaac-ros` looks for a shell config at:

- `${ISAAC_ROS_WS}/../scripts/.isaac_ros_common-config` (highest precedence), then
- `/etc/isaac-ros-cli/.isaac_ros_common-config`

Create or edit `${ISAAC_ROS_WS}/../scripts/.isaac_ros_common-config` and ensure it includes this directory:

```bash
mkdir -p "${ISAAC_ROS_WS}/../scripts"
cat > "${ISAAC_ROS_WS}/../scripts/.isaac_ros_common-config" <<'EOF'
CONFIG_DOCKER_SEARCH_DIRS=(/etc/isaac-ros-cli/docker ${ISAAC_ROS_WS}/docker ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4/4.2)
EOF
```

Notes:
- If you **also** have the `4.1/` or `4.0/` directory in `CONFIG_DOCKER_SEARCH_DIRS`, put `4.2/` **before** them
  (both contain `Dockerfile.isaac_manipulation`, and the first match wins).
- Keep `/etc/isaac-ros-cli/docker` in the list so the default CLI Dockerfiles remain available.

## 2) Add image keys to the `isaac-ros` CLI build sequence

Add `isaac_manipulation` under `docker.image.additional_image_keys`.
To keep the current startup behavior, leave the config as-is. To enable the ManyMove/xArm layer as well,
append `manymove_xarm` after `isaac_manipulation`.

Recommended (workspace-scoped) config location:

- `${ISAAC_ROS_WS}/.isaac-ros-cli/config.yaml`

Minimal example:

```yaml
docker:
  image:
    additional_image_keys:
      - isaac_manipulation
```

ManyMove/xArm example:

```yaml
docker:
  image:
    additional_image_keys:
      - isaac_manipulation
      - manymove_xarm
```

If you already have other keys (e.g., `realsense`, `zed`), keep them and append the new keys to the list in the
order you want them layered.

## 3) Build the image

From your workspace (with `ISAAC_ROS_WS` set), run:

```bash
isaac-ros activate --build-local
```

For troubleshooting Dockerfile resolution, rerun with verbose logging:

```bash
isaac-ros activate --build-local --verbose
```
