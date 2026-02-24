# Isaac ROS 4.x (Jazzy) — custom bringup helpers

This directory is versioned by **Isaac ROS minor release**:

- `4.2/`: **recommended** (`pip_shim_constraints` patch + in-container bootstrap sequence + idempotent clones)
- `4.1/`: previous workspace-scoped `isaac-ros-cli` config + lighter bootstrap
- `4.0/`: legacy bootstrap (writes user-global `~/.config/isaac-ros-cli/config.yaml` and `~/.isaac_ros_dev-dockerargs`)

## Quick start (4.2)

```bash
isaac-ros activate --build-local
```

Notes:
- Configure Dockerfile search path + `additional_image_keys` as described in `4.2/README.md`.
- RealSense support notes for the previous helper flow remain in `4.1/README.md`.
