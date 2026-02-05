# Isaac ROS 4.x (Jazzy) — custom bringup helpers

This directory is versioned by **Isaac ROS minor release**:

- `4.1/`: **recommended** (workspace-scoped `isaac-ros-cli` config + lighter bootstrap)
- `4.0/`: legacy bootstrap (writes user-global `~/.config/isaac-ros-cli/config.yaml` and `~/.isaac_ros_dev-dockerargs`)

## Quick start (4.1)

```bash
bash ${ISAAC_ROS_WS}/src/isaac_ros_custom_bringup/isaac_ros_4/4.1/scripts/bootstrap_isaac_ros_cli_files.sh
isaac-ros activate --build-local
```

Notes:
- RealSense support is optional in 4.1; enable it via the 4.1 bootstrap flags (see `4.1/README.md`).

