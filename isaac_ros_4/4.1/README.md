# Isaac Manipulation (Isaac ROS 4.1) — dev-image customization

This folder provides custom Isaac ROS CLI image layers and scripts to reach the same “ready-to-run” state as the
upstream Isaac Manipulation setup guides (Isaac Sim), with a lighter + more workspace-scoped bootstrap.

## Key differences vs `4.0/`

- Bootstrap writes **workspace-scoped** CLI config by default (`${ISAAC_ROS_WS}/.isaac-ros-cli/config.yaml`).
- RealSense is **optional** (enable only if you need a RealSense camera).
- Source workflow is **pinned by default** to `release-4.1`, but supports `--latest` to follow the newest Isaac ROS 4.x minor.

## Quick start

1. Bootstrap the `isaac-ros` CLI files (workspace-scoped):

   ```bash
   bash ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4/4.1/scripts/bootstrap_isaac_ros_cli_files.sh
   ```

   Common options:
   - `--source`: build `isaac_manipulator` from source at container start.
   - `--latest`: resolve the latest Isaac ROS 4.x minor and checkout `release-4.<minor>` for `isaac_manipulator`.
   - `--realsense`: include the `realsense` layer (only needed for RealSense camera workflows).

2. Build and start the dev container:

   ```bash
   isaac-ros activate --build-local
   ```

## Optional: prefetch NGC “quickstart” assets on the host

```bash
bash ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4/4.1/scripts/prefetch_quickstart_assets_host.sh
```

## Notes

- Jetson/Thor: during image build, `scripts/ensure_isaac_ros_apt_repo.sh` will auto-toggle the Isaac ROS apt suite
  (`noble` ↔ `noble-jetpack`) if core packages (e.g., `ros-jazzy-isaac-ros-common`) are not discoverable.
- Auto-setup: `/usr/local/bin/isaac-manipulation-setup.sh` installs the core model bundles and then runs
  `setup_perception_models.py --models all` (disable with `ISAAC_ROS_MANIPULATION_SETUP_PERCEPTION_MODELS=0`).
- RealSense: making the `realsense` layer optional does not change the manipulation install logic; it only affects
  whether librealsense + realsense-ros are built into the dev image. You still need the host udev rules from the
  upstream RealSense setup guide when using a physical camera.
