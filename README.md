isaac_ros_custom_bringup
========================

This repository contains a ROS 2 package (`isaac_ros_custom_bringup`) plus a small set of helper tools/docs used around Isaac ROS.
Content is grouped by Isaac ROS major release, with a few utilities that are useful regardless of ROS distro.

## Quick links

- Isaac ROS 3 (Humble): [isaac_ros_3/README.md](isaac_ros_3/README.md)
- Isaac ROS 4 (Jazzy): [isaac_ros_4/README.md](isaac_ros_4/README.md)
- Jetson Orin NVMe storage: [jetson_orin_storage/README.md](jetson_orin_storage/README.md)
- Jetson remote screen: [jetson_quick_remote_screen/README.md](jetson_quick_remote_screen/README.md)

## Repository layout

### `isaac_ros_3/` (ROS 2 Humble / Isaac ROS 3.x)

- Bringup launch graphs for perception pipelines that combine Isaac ROS packages without modifying upstream repos.
- Includes YOLOv8 inference and YOLOv8 → FoundationPose pipelines (including a variant that subscribes to a remote RealSense stream).
- Entry points are launch files under `isaac_ros_3/launch/` (e.g., YOLOv8-only and FoundationPose integration launches).
- Documentation and full end-to-end setup notes live in [isaac_ros_3/README.md](isaac_ros_3/README.md).

### `isaac_ros_4/` (ROS 2 Jazzy / Isaac ROS 4.x)

- A dev-image customization layer for Isaac Manipulation (Dockerfile + optional entrypoint hooks) to install tutorial packages and models.
- Versioned by Isaac ROS minor: `isaac_ros_4/4.1/` (recommended) and `isaac_ros_4/4.0/` (legacy).
- Config-driven asset/model selection via `isaac_ros_4/<version>/config/isaac_manipulation_assets.yaml` (used by host prefetch + in-container setup).
- Host helper scripts for Isaac ROS CLI bootstrap and NGC quickstart asset prefetch live in `isaac_ros_4/<version>/scripts/`.
- Full usage and rationale are in [isaac_ros_4/README.md](isaac_ros_4/README.md).

### `jetson_quick_remote_screen/` (utility)

- Step-by-step guide to mirror/control the Jetson’s real `:0` desktop from an Ubuntu laptop using `x11vnc` + SSH port forwarding + TigerVNC.
- Convenience scripts `connect_thor.sh` and `connect_orin.sh` automate tunnel + `x11vnc` startup (both support `--ip` overrides).
- See [jetson_quick_remote_screen/README.md](jetson_quick_remote_screen/README.md) for prerequisites and troubleshooting.

### `jetson_orin_storage/` (utility)

- Orin-specific notes for using the 1 TB NVMe mounted at `/mnt/nova_ssd` for Docker, containerd, apt archives, `/usr/local`, temp directories, VS Code, downloads, and common caches.
- Extends NVIDIA’s official [Isaac ROS Jetson Storage Setup](https://nvidia-isaac-ros.github.io/getting_started/compute/jetson_storage.html) with the extra bind mounts and GNOME desktop cleanup used on this machine.
- See [jetson_orin_storage/README.md](jetson_orin_storage/README.md) for verification and cleanup commands.
