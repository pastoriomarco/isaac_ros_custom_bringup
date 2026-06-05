# Jetson Orin NVMe storage setup

This note documents the storage layout used on the Jetson Orin Developer Kit for Isaac ROS work.

It extends the official NVIDIA Isaac ROS Jetson storage guide:

<https://nvidia-isaac-ros.github.io/getting_started/compute/jetson_storage.html>

The NVIDIA guide covers the base SSD flow: mount the NVMe at `/mnt/nova_ssd`, make that mount persistent with `/etc/fstab`, change ownership with `sudo chown ${USER}:${USER} /mnt/nova_ssd`, and move Docker to `/mnt/nova_ssd/docker`. This document records the Orin-specific setup used here, including extra bind mounts for large developer directories and desktop cleanup details.

The Orin has:

- eMMC root filesystem: `/`, about 64 GB
- NVMe SSD: `/mnt/nova_ssd`, about 1 TB

The goal is to keep the operating system on eMMC while putting large developer/runtime data on the NVMe. This avoids filling the eMMC with Docker images, Isaac ROS assets, CUDA local installs, VS Code server files, downloads, and caches.

If the Jetson boots from one sufficiently large SSD/NVMe root filesystem, this
extra eMMC-relief layout is usually not needed. In that case, use the standard
Docker/NVIDIA/runtime setup and keep model caches on the root SSD. This document
is for the split-storage case: small system disk plus large secondary NVMe.

## Fresh Machine Bootstrap

Use this sequence on a newly formatted Orin before applying the
machine-specific record below. Do **not** copy the UUID from this document onto a
different SSD; discover the UUID on the target machine.

1. Identify the NVMe partition:

   ```bash
   lsblk -o NAME,SIZE,FSTYPE,LABEL,UUID,MOUNTPOINTS
   ```

   Set the partition you intend to use:

   ```bash
   export NVME_PART=/dev/nvme0n1p1
   ```

2. Format only when you intentionally want to erase that partition:

   ```bash
   sudo mkfs.ext4 -L nova_ssd "${NVME_PART}"
   ```

3. Mount it persistently at `/mnt/nova_ssd`:

   ```bash
   sudo mkdir -p /mnt/nova_ssd
   NVME_UUID="$(sudo blkid -s UUID -o value "${NVME_PART}")"
   echo "UUID=${NVME_UUID} /mnt/nova_ssd ext4 defaults,noatime 0 2" | sudo tee -a /etc/fstab
   sudo mount /mnt/nova_ssd
   sudo chown "${USER}:${USER}" /mnt/nova_ssd
   ```

4. Create the load-bearing directories used by the ManyForge/NemoClaw Orin
   stack:

   ```bash
   mkdir -p \
     /mnt/nova_ssd/docker \
     /mnt/nova_ssd/containerd \
     /mnt/nova_ssd/apt/archives \
     /mnt/nova_ssd/opt \
     /mnt/nova_ssd/hf-cache-orin \
     /mnt/nova_ssd/llama-cpp-cache \
     /mnt/nova_ssd/nemoclaw-state/nemoclaw \
     /mnt/nova_ssd/home-bind \
     /mnt/nova_ssd/user-bind \
     /mnt/nova_ssd/system-bind \
     /mnt/nova_ssd/tmp \
     /mnt/nova_ssd/var-tmp
   sudo chmod 1777 /mnt/nova_ssd/tmp /mnt/nova_ssd/var-tmp
   ```

5. After Docker and the NVIDIA Container Toolkit are installed, move Docker to
   the NVMe and keep the NVIDIA runtime as default. If
   `/etc/docker/daemon.json` already contains site-specific settings, merge
   these keys instead of replacing the file:

   ```bash
   sudo mkdir -p /etc/docker
   sudo tee /etc/docker/daemon.json >/dev/null <<'EOF'
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     },
     "default-runtime": "nvidia",
     "data-root": "/mnt/nova_ssd/docker"
   }
   EOF
   sudo systemctl restart docker
   sudo usermod -aG docker "${USER}"
   ```

   Log out/in before relying on Docker group membership.

6. Add bind mounts only for paths that would otherwise fill the eMMC. At minimum
   for the ManyForge/NemoClaw Orin stack, keep `~/.cache`, `~/.local`, `/tmp`,
   `/var/tmp`, and `~/.nemoclaw` on the NVMe. Copy existing contents first, then
   add fstab bind entries using the pattern below:

   ```bash
   sudo apt-get install -y rsync

   USER_NAME="$(id -un)"
   HOME_DIR="$(getent passwd "${USER_NAME}" | cut -d: -f6)"

   mkdir -p \
     "/mnt/nova_ssd/user-bind/${USER_NAME}-cache" \
     "/mnt/nova_ssd/user-bind/${USER_NAME}-local" \
     "/mnt/nova_ssd/nemoclaw-state/nemoclaw"

   mkdir -p "${HOME_DIR}/.cache" "${HOME_DIR}/.local" "${HOME_DIR}/.nemoclaw"
   sudo rsync -aHAX "${HOME_DIR}/.cache/" "/mnt/nova_ssd/user-bind/${USER_NAME}-cache/"
   sudo rsync -aHAX "${HOME_DIR}/.local/" "/mnt/nova_ssd/user-bind/${USER_NAME}-local/"
   sudo rsync -aHAX "${HOME_DIR}/.nemoclaw/" "/mnt/nova_ssd/nemoclaw-state/nemoclaw/"

   sudo mount --bind "/mnt/nova_ssd/user-bind/${USER_NAME}-cache" "${HOME_DIR}/.cache"
   sudo mount --bind "/mnt/nova_ssd/user-bind/${USER_NAME}-local" "${HOME_DIR}/.local"
   sudo mount --bind /mnt/nova_ssd/nemoclaw-state/nemoclaw "${HOME_DIR}/.nemoclaw"

   {
     echo "/mnt/nova_ssd/user-bind/${USER_NAME}-cache ${HOME_DIR}/.cache none bind,nofail,x-gvfs-hide,x-systemd.requires-mounts-for=/mnt/nova_ssd 0 0"
     echo "/mnt/nova_ssd/user-bind/${USER_NAME}-local ${HOME_DIR}/.local none bind,nofail,x-gvfs-hide,x-systemd.requires-mounts-for=/mnt/nova_ssd 0 0"
     echo "/mnt/nova_ssd/nemoclaw-state/nemoclaw ${HOME_DIR}/.nemoclaw none bind,nofail,x-gvfs-hide,x-systemd.requires-mounts-for=/mnt/nova_ssd 0 0"
   } | sudo tee -a /etc/fstab
   ```

   ```fstab
   /mnt/nova_ssd/user-bind/<user>-cache /home/<user>/.cache none bind,nofail,x-gvfs-hide,x-systemd.requires-mounts-for=/mnt/nova_ssd 0 0
   ```

   `~/.nemoclaw` must be a real directory or bind mount, never a symlink;
   NemoClaw rejects symlinked config directories.

7. Verify before continuing with the stack install:

   ```bash
   findmnt /mnt/nova_ssd
   docker info | grep -E 'Docker Root Dir|Default Runtime|Runtimes'
   df -h / /mnt/nova_ssd
   findmnt -T "$HOME/.cache"
   findmnt -T "$HOME/.local"
   test ! -L "$HOME/.nemoclaw"
   ```

## Scope

This setup is specific to the Orin system where the NVMe partition has UUID:

```text
33b718a7-eae4-4cc5-99be-5d2d23563a36
```

If the NVMe is reformatted, the UUID will change and `/etc/fstab` must be updated.

This setup does not move the whole root filesystem to NVMe. Package-managed system files under paths such as `/usr/lib`, `/etc`, `/boot`, and `/var/lib/dpkg` remain on eMMC. That is intentional. To make every system package file live on NVMe, boot the root filesystem from NVMe instead.

## Main Mount

The NVMe is mounted at the same path used by NVIDIA Isaac ROS Jetson storage instructions:

```text
/mnt/nova_ssd
```

The base `/etc/fstab` line is:

```fstab
UUID=33b718a7-eae4-4cc5-99be-5d2d23563a36 /mnt/nova_ssd ext4 defaults,noatime 0 2
```

Ownership was assigned to the main user:

```bash
sudo chown ${USER}:${USER} /mnt/nova_ssd
```

This matches the ownership step in NVIDIA's Jetson storage setup.

## Docker And Containerd

Docker data is stored on the NVMe:

```text
/mnt/nova_ssd/docker
```

The Docker daemon config is `/etc/docker/daemon.json`:

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia",
  "data-root": "/mnt/nova_ssd/docker"
}
```

This follows NVIDIA's Isaac ROS storage recommendation to add Docker `"data-root": "/mnt/nova_ssd/docker"` and keep the NVIDIA runtime as the default runtime.

Docker 29 uses the containerd snapshotter, so containerd was also moved:

```text
/mnt/nova_ssd/containerd
```

The relevant line in `/etc/containerd/config.toml` is:

```toml
root = "/mnt/nova_ssd/containerd"
```

The user was added to the Docker group:

```bash
sudo usermod -aG docker ${USER}
```

After logging out/in, this should work without `sudo`:

```bash
docker ps
```

## Apt Archives

Downloaded `.deb` archives are stored on the NVMe:

```text
/mnt/nova_ssd/apt/archives
```

The config file is `/etc/apt/apt.conf.d/99archives-on-nvme`:

```aptconf
Dir::Cache::archives "/mnt/nova_ssd/apt/archives";
```

This moves downloaded package archives only. Installed package files still go to their normal system paths.

## Bind Mounts

Large directories are copied to the NVMe and mounted back at their original paths. This keeps the system and applications using normal paths while the storage is actually on the NVMe.

The current bind mounts are:

| Original path | NVMe backing path |
| --- | --- |
| `/usr/local` | `/mnt/nova_ssd/system-bind/usr-local` |
| `/opt/nvidia/nsight-systems` | `/mnt/nova_ssd/system-bind/opt-nvidia-nsight-systems` |
| `/opt/nvidia/nsight-compute` | `/mnt/nova_ssd/system-bind/opt-nvidia-nsight-compute` |
| `/home/tndlux/.vscode-server` | `/mnt/nova_ssd/home-bind/tndlux-vscode-server` |
| `/home/tndlux/Downloads` | `/mnt/nova_ssd/user-bind/tndlux-Downloads` |
| `/home/tndlux/Desktop` | `/mnt/nova_ssd/user-bind/tndlux-Desktop` |
| `/home/tndlux/Documents` | `/mnt/nova_ssd/user-bind/tndlux-Documents` |
| `/home/tndlux/Pictures` | `/mnt/nova_ssd/user-bind/tndlux-Pictures` |
| `/home/tndlux/Videos` | `/mnt/nova_ssd/user-bind/tndlux-Videos` |
| `/home/tndlux/Music` | `/mnt/nova_ssd/user-bind/tndlux-Music` |
| `/home/tndlux/.cache` | `/mnt/nova_ssd/user-bind/tndlux-cache` |
| `/home/tndlux/.local` | `/mnt/nova_ssd/user-bind/tndlux-local` |
| `/home/tndlux/.vscode` | `/mnt/nova_ssd/user-bind/tndlux-vscode` |
| `/tmp` | `/mnt/nova_ssd/tmp` |
| `/var/tmp` | `/mnt/nova_ssd/var-tmp` |
| `/var/cache` | `/mnt/nova_ssd/system-bind/var-cache` |
| `/var/lib/snapd/cache` | `/mnt/nova_ssd/system-bind/var-lib-snapd-cache` |

`/usr/local` is mounted as a whole so future CUDA local installs such as `/usr/local/cuda-13.3` land on the NVMe automatically.

The bind mount entries include `x-gvfs-hide` to keep GNOME/Nautilus from showing every bind-mounted folder as a separate disk in "Devices & Locations".

Example `/etc/fstab` bind entry:

```fstab
/mnt/nova_ssd/system-bind/usr-local /usr/local none bind,nofail,x-gvfs-hide,x-systemd.requires-mounts-for=/mnt/nova_ssd 0 0
```

`/tmp` and `/var/tmp` must preserve sticky-bit permissions:

```bash
sudo chmod 1777 /mnt/nova_ssd/tmp /mnt/nova_ssd/var-tmp
```

## Desktop Launcher Trust

After moving `~/Desktop`, GNOME may mark copied `.desktop` files as untrusted and show red error overlays.

Restore trust with:

```bash
for f in "${HOME}"/Desktop/*.desktop; do
  chmod +x "$f"
  gio set "$f" metadata::trusted true
done
```

Refresh the desktop icons extension:

```bash
gnome-extensions disable ding@rastersoft.com
sleep 1
gnome-extensions enable ding@rastersoft.com
```

If Nautilus still shows stale device entries, restart the GVFS disk monitor or log out/in:

```bash
systemctl --user restart gvfs-udisks2-volume-monitor.service
```

## Verification

Check that the NVMe is mounted:

```bash
findmnt /mnt/nova_ssd
```

Check key bind mounts:

```bash
findmnt /usr/local
findmnt /tmp
findmnt /var/tmp
findmnt /home/tndlux/Downloads
findmnt /home/tndlux/.vscode-server
findmnt /var/cache
findmnt /var/lib/snapd/cache
```

Check Docker:

```bash
docker info | grep -E 'Docker Root Dir|Default Runtime|Storage Driver|driver-type'
```

Expected:

```text
Docker Root Dir: /mnt/nova_ssd/docker
Default Runtime: nvidia
```

Check containerd:

```bash
grep -E '^root = ' /etc/containerd/config.toml
```

Expected:

```text
root = "/mnt/nova_ssd/containerd"
```

Check apt archives:

```bash
apt-config dump | grep -E '^Dir::Cache::archives'
```

Expected:

```text
Dir::Cache::archives "/mnt/nova_ssd/apt/archives";
```

Check CUDA:

```bash
readlink -f /usr/local/cuda
/usr/local/cuda/bin/nvcc --version
```

Check GNOME visible volumes:

```bash
gio mount -li | grep -E 'Volume\\(|Mount\\(|Videos|Pictures|Downloads|Desktop|Documents|Music|nova|nsight|usr-local|cache|tmp' || true
```

The internal bind mounts should not appear as separate GNOME volumes. The `L4T-README` loop volume may still appear; that is normal on Jetson images.

Validate `/etc/fstab`:

```bash
sudo findmnt --verify --verbose
```

The Jetson root entry may produce warnings for `/dev/root`, and swap may produce a warning about `/swapfile`. There should be no parse errors and no real mount errors.

## Current Result

After the migration, typical storage usage was:

```text
/              about 22 GB used, 30 GB free
/mnt/nova_ssd  about 13 GB used, 857 GB free
```

The most important remaining eMMC usage is package-managed system content, mostly under `/usr`. That is expected for an eMMC-root system.

## Cleanup Checks

The migration used temporary backup names such as `.emmc-old`, `docker.old`, and `fstab.bak-*`. Those were removed after verification.

Check for leftovers with:

```bash
sudo find / -xdev \
  \( -name '*.emmc-old' -o -name '*.nvme-old' -o -name 'docker.old' -o -name 'fstab.bak-*' -o -name 'config.toml.bak-nvme' \) \
  -print 2>/dev/null
```

An empty result is expected.

There can be tiny hidden pre-mount files underneath `/tmp` and `/var/tmp` if those bind mounts were activated while the desktop was running. They are not visible to applications after the bind mount and were only about a few hundred KB on this Orin.
