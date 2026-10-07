isaac_ros_custom_bringup
========================

This repository contains a ROS 2 package (`isaac_ros_custom_bringup`) plus a small set of helper tools/docs used around Isaac ROS.
Content is grouped by Isaac ROS major release, with a few utilities that are useful regardless of ROS distro.

## Quick links

- Isaac ROS 3 (Humble): [isaac_ros_3/README.md](isaac_ros_3/README.md)
- Isaac ROS 4 (Jazzy): [isaac_ros_4/README.md](isaac_ros_4/README.md)
- Isaac ROS 5 (Lyrical): [isaac_ros_5/README.md](isaac_ros_5/README.md) — image and stock FoundationPose smoke check verified on Thor; custom YOLO/simulator launch **not validated**.
- Jetson Orin NVMe storage: [jetson_orin_storage/README.md](jetson_orin_storage/README.md)
- Jetson remote screen: [jetson_quick_remote_screen/README.md](jetson_quick_remote_screen/README.md)

## Repository layout

### `isaac_ros_3/` (ROS 2 Humble / Isaac ROS 3.x)

- Bringup launch graphs for perception pipelines that combine Isaac ROS packages without modifying upstream repos.
- Includes YOLOv8 inference and YOLOv8 → FoundationPose pipelines (including a variant that subscribes to a remote RealSense stream).
- Entry points are launch files under `isaac_ros_3/launch/` (e.g., YOLOv8-only and FoundationPose integration launches).
- Documentation and full end-to-end setup notes live in [isaac_ros_3/README.md](isaac_ros_3/README.md).

### `isaac_ros_4/` (ROS 2 Jazzy / Isaac ROS 4.x)

- A dev-image customization layer for Isaac ROS Manipulation + FoundationPose (Dockerfile + optional entrypoint hooks) to install tutorial packages and models.
- Versioned by Isaac ROS minor: [4.6](isaac_ros_4/4.6/README.md) is a **NOT VALIDATED** adaptation of 4.4; `isaac_ros_4/4.4/` remains the existing reference, with older helpers down to `4.0/` (legacy).
- Isaac ROS 4.4 renamed `isaac_manipulator` → `isaac_ros_manipulation`; the `4.4/` layer carries that rename through the image key, Dockerfile, bootstrap, and apt package, and bakes in the FoundationPose / RT-DETR / ESS packages for the Isaac Sim example.
- The `4.4/` layer activates with `isaac-ros activate --build-local` once the host is on `release-4.4`; FoundationPose models install via `isaac_ros_4/4.4/scripts/install_foundationpose_isaac_sim_models.sh`.
- Full usage and rationale are in [isaac_ros_4/README.md](isaac_ros_4/README.md).

### `isaac_ros_5/` (ROS 2 Lyrical / Isaac ROS 5.0)

- A focused perception image extension for FoundationPose, custom YOLO and optional visualization, using NVIDIA's existing development base.
- Documents the actual 5.0 launch changes and required versus optional dependencies. The 4.x launch is not yet ported to 5.0.
- Does not install the complete manipulation workflow or add ManyMove/ManyForge images, drivers, startup builds or model downloads.
- See [the dependency review and validation steps](isaac_ros_5/README.md).

### `jetson_quick_remote_screen/` (utility)

- Step-by-step guide to mirror/control the Jetson’s real `:0` desktop from an Ubuntu laptop using `x11vnc` + SSH port forwarding + TigerVNC.
- Convenience scripts `connect_thor.sh` and `connect_orin.sh` automate tunnel + `x11vnc` startup (both support `--ip` overrides).
- See [jetson_quick_remote_screen/README.md](jetson_quick_remote_screen/README.md) for prerequisites and troubleshooting.

### `jetson_orin_storage/` (utility)

- Orin-specific notes for using the 1 TB NVMe mounted at `/mnt/nova_ssd` for Docker, containerd, apt archives, `/usr/local`, temp directories, VS Code, downloads, and common caches.
- Extends NVIDIA’s official [Isaac ROS Jetson Storage Setup](https://nvidia-isaac-ros.github.io/getting_started/compute/jetson_storage.html) with the extra bind mounts and GNOME desktop cleanup used on this machine.
- See [jetson_orin_storage/README.md](jetson_orin_storage/README.md) for verification and cleanup commands.

## Troubleshooting: `~/.gitconfig` is a directory

If Git or `isaac-ros activate` reports `unable to access .../.gitconfig: Is a directory`,
check the path on the **host**. Git expects a file. Docker's `-v` bind-mount syntax
[creates a directory when the host source is missing](https://docs.docker.com/engine/storage/bind-mounts/#syntax),
so a launcher mounting a nonexistent `.gitconfig` can cause this.

Before activation, create the file if missing, or replace an **empty** directory:

```bash
if [ -d "$HOME/.gitconfig" ]; then
    rmdir -- "$HOME/.gitconfig" && touch "$HOME/.gitconfig"
elif [ ! -e "$HOME/.gitconfig" ]; then
    touch "$HOME/.gitconfig"
fi
```

`rmdir` refuses a nonempty directory; inspect its contents instead of using
recursive deletion. Existing config files are left untouched. If a running
container already mounted the incorrect directory, recreate that container
after fixing the host path; rebuilding its image is unnecessary.
