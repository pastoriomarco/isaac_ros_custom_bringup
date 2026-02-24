# Isaac Manipulation layer (Isaac ROS 4.2) — `isaac-ros` CLI integration

This directory contains a custom Isaac ROS CLI image layer:

- `Dockerfile.isaac_manipulation`
- `90-isaac-manipulation-bootstrap.user.sh` (copied into the image as an entrypoint hook)

The `isaac-ros` CLI will only build this layer if:

1. it can **find** `Dockerfile.isaac_manipulation` (Dockerfile search path), and
2. the `isaac_manipulation` key is in the CLI **build sequence** (`additional_image_keys`).

The `4.2` bootstrap hook applies the `isaac-ros-cli` `pip_shim_constraints.txt` tensordict patch, reruns the
required apt installs in-container, and skips `git clone` when the target directory already exists.

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

## 2) Add `isaac_manipulation` to the `isaac-ros` CLI build sequence

Add `isaac_manipulation` under `docker.image.additional_image_keys`.

Recommended (workspace-scoped) config location:

- `${ISAAC_ROS_WS}/.isaac-ros-cli/config.yaml`

Minimal example:

```yaml
docker:
  image:
    additional_image_keys:
      - isaac_manipulation
```

If you already have other keys (e.g., `realsense`, `zed`), keep them and append `isaac_manipulation` to the list.

## 3) Build the image

From your workspace (with `ISAAC_ROS_WS` set), run:

```bash
isaac-ros activate --build-local
```

For troubleshooting Dockerfile resolution, rerun with verbose logging:

```bash
isaac-ros activate --build-local --verbose
```
