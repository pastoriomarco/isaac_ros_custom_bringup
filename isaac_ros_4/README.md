# Isaac ROS 4.x (Jazzy) — custom bringup helpers

This directory is versioned by **Isaac ROS minor release**:

- `4.4/`: **recommended** — Isaac ROS 4.4 rename (`isaac_manipulator` → `isaac_ros_manipulation`)
  + FoundationPose / RT-DETR / ESS baked in for the "FoundationPose with Isaac Sim" example
- `4.3/`: previous (workspace-root `build/`, `log/`, `install/` + idempotent clones)
- `4.2/`: previous recommended layout
- `4.1/`: workspace-scoped `isaac-ros-cli` config + lighter bootstrap
- `4.0/`: legacy bootstrap (writes user-global `~/.config/isaac-ros-cli/config.yaml` and `~/.isaac_ros_dev-dockerargs`)

## Quick start (4.4)

```bash
# After the host is on release-4.4 and CONFIG_DOCKER_SEARCH_DIRS points at 4.4/ (see 4.4/README.md):
isaac-ros activate --build-local
```

Notes:
- **4.4 is a breaking rename.** Configure the Dockerfile search path + the renamed
  `additional_image_keys: [isaac_ros_manipulation]`, and move the host to `release-4.4` first —
  full steps in [4.4/README.md](4.4/README.md).
- FoundationPose + Isaac Sim model setup is a one-time in-container step
  (`4.4/scripts/install_foundationpose_isaac_sim_models.sh`).
- RealSense support notes for the older helper flow remain in `4.1/README.md`.
