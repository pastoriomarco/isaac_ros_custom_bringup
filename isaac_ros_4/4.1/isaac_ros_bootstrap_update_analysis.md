# Isaac ROS install/bootstrap analysis (focus: Isaac ROS 4.x / Jazzy)

## Scope

This document originally analyzed the pre-split `src/isaac_ros_custom_bringup/isaac_ros_4/` bootstrap as an
install/helper for **Isaac ROS 4 (ROS 2 Jazzy)**.

**Update (2026-02-05)**: this repo is now split by minor release:

- `src/isaac_ros_custom_bringup/isaac_ros_4/4.0/`: legacy bootstrap (user-global config writes)
- `src/isaac_ros_custom_bringup/isaac_ros_4/4.1/`: recommended bootstrap (workspace-scoped config by default)

The “why” and most of the considerations below still apply, but when the text says “current state” it is usually
describing what is now preserved under `isaac_ros_4/4.0/`.

It uses “Isaac ROS 4” to mean the **major** release train (e.g., `release-4`), which can either:

- float to the latest minor (e.g., APT repo `release-4`), or
- be pinned to a specific minor (e.g., `release-4.0`, `release-4.1`).

The goal is to keep **all** the moving pieces (CLI base image, Debian packages, NGC assets, and any source checkouts)
coherent with the selected strategy.

- **Coherently target Isaac ROS 4** (either floating `release-4` or pinned `release-4.X`)
- **Less invasive** (workspace-scoped vs global host config edits)
- **More “CLI-native”** (leaning on `isaac-ros` CLI behavior, fewer side effects)

Not covered yet (deferred):

- Automating the “build from source” instructions for Isaac Sim setup guides
- A dedicated “clone repos only” script (requested; referenced below as a recommended change, but not implemented here)

## Upstream baseline: Isaac ROS CLI (what matters for this repo)

### APT repo release selection (`release-4` vs `release-4.X`)

The “Getting Started” docs explicitly support *either* pinning to a specific minor (e.g., `release-4.1`) *or* tracking
the latest minor (e.g., `release-4`). See the “Configure Isaac ROS Apt Repository” section in:

- `src/NVIDIA-ISAAC-ROS.github.io/public/getting_started/index.html`

This is the upstream “source of truth” for what “targeting Isaac ROS 4” means on a host: the **APT repo stanza you
choose** controls which Isaac ROS CLI version and Debian packages you’ll pull over time.

For custom layers, the safest pattern is: **do not override APT release selection inside custom Dockerfiles**; rely on
whatever the Isaac ROS CLI base image layers already configured.

### Image keys (4.0-style vs 4.1-style)

The Isaac ROS CLI composes the dev image from `Dockerfile.<image_key>` layers.

- **Isaac ROS 4.0 docs**: default base image key sequence is `["noble", "ros2_jazzy"]`; `realsense` is an optional
  additional key. See `src/NVIDIA-ISAAC-ROS.github.io/public/v/release-4.0/_sources/concepts/dev_env/index.rst.txt`.
- **Isaac ROS 4.1+ docs**: default base is `["isaac_ros"]`; `noble`/`ros2_jazzy` are marked deprecated.
  See `src/NVIDIA-ISAAC-ROS.github.io/public/_sources/concepts/dev_env/index.rst.txt`.

This matters because `isaac_ros_custom_bringup/isaac_ros_4` should **not** assume a specific base-key scheme. The repo
should work whether the CLI is using the 4.0-style base keys or the newer 4.1-style `isaac_ros` base key.

### Where the CLI finds Dockerfiles

The CLI’s image build logic (via `/usr/lib/isaac-ros-cli/build_image_layers.py`) discovers Dockerfiles by searching
directories listed in `CONFIG_DOCKER_SEARCH_DIRS` from a shell config file named `.isaac_ros_common-config`.

Lookup order includes:

- `$ISAAC_ROS_WS/../scripts/.isaac_ros_common-config` (workspace-adjacent override)
- `/etc/isaac-ros-cli/.isaac_ros_common-config` (system default)

So, for custom image keys to work, you either:

1. Place custom Dockerfiles in an already-searched directory, or
2. Provide a workspace-adjacent `.isaac_ros_common-config` that adds your custom Dockerfile directory.

### Where extra `docker run` args come from

The CLI appends additional `docker run` arguments from `.isaac_ros_dev-dockerargs`, with support for:

- `~/.isaac_ros_dev-dockerargs` (user-global)
- `$ISAAC_ROS_WS/../scripts/.isaac_ros_dev-dockerargs` (workspace-adjacent)
- `/etc/isaac-ros-cli/.isaac_ros_dev-dockerargs` (system default)

This repo currently uses `~/.isaac_ros_dev-dockerargs` to inject env vars that trigger auto-build/auto-setup behavior.

## Current situation: `src/isaac_ros_custom_bringup/isaac_ros_4/`

### What it provides

**Custom CLI image layers**

- `Dockerfile.isaac_manipulation`: installs Debian packages for Isaac Manipulation (cuMotion/nvblox/perception stacks)
  plus optional entrypoint hook for assets/model setup.
- `Dockerfile.isaac_manipulation_source`: prepares a “build from source” environment by importing repos into a *temporary*
  workspace at image build time and running `rosdep install` (but defers `colcon build` to container start).
- `Dockerfile.isaac_manipulation_rsl_rl`: optional RL dependencies layer.

**Config + scripts**

- `scripts/bootstrap_isaac_ros_cli_files.sh`: generates:
  - `$ISAAC_ROS_WS/../scripts/.isaac_ros_common-config` (adds Dockerfile search dirs so the CLI can find these layers)
  - `~/.config/isaac-ros-cli/config.yaml` (adds `realsense` + `isaac_manipulation*` to `additional_image_keys`)
  - `~/.isaac_ros_dev-dockerargs` (sets env vars to trigger runtime automation)
- `scripts/prefetch_quickstart_assets_host.sh`: host-side download/extract of NGC `quickstart.tar.gz` bundles into
  `${ISAAC_ROS_WS}/isaac_ros_assets` (defaults to Isaac ROS 4.1 bundles; override with `--major/--minor/--latest`).
- `scripts/isaac-manipulation-build.sh`: in-container script that can `vcs import` the source repos, optionally `vcs pull`,
  and `colcon build` a subset of packages (based on what exists in the workspace).
- `scripts/isaac-manipulation-setup.sh`: in-container model + asset setup (prefers upstream “models_install” scripts).
- `scripts/apply-isaac-manipulation-fixes.sh`: applies a small set of file overrides into checked out repos and rebuilds
  affected packages.

**Source repo pinning**

- `source/isaac_ros_manipulation.repos` pins NVIDIA repos to `release-4.0` (a specific minor), but leaves some third-party
  repos floating (e.g., `ros2_robotiq_gripper: main`).

### What this means in practice today

If you follow `src/isaac_ros_custom_bringup/isaac_ros_4/README.md`, you end up with:

- Custom Dockerfile search dirs via `.isaac_ros_common-config`
- A user-global CLI config that adds extra image keys
- A user-global dockerargs file that:
  - enables auto-build at container start (source layer), and
  - enables auto-setup at container start (assets/models),
  - **and can default EULA acceptance to “on”** (depending on bootstrap flags/defaults)

This is convenient for “one workspace, one machine” setups, but it is more invasive than it needs to be, and it is easy
to accidentally become **incoherent** about which Isaac ROS 4 minor you’re actually targeting (or whether you intend to
float to the latest minor).

## Isaac ROS 4 (major) gaps / risks

### 1) Release coherence should follow APT/NGC configuration (not hardcoded)

Per Getting Started, a user can intentionally choose:

- **Pinned minor**: APT repo `release-4.X` (e.g., `release-4.1`)
- **Rolling minor**: APT repo `release-4` (latest minor within major 4)

This repo should “let the CLI do its job” and behave coherently with that choice:

- **Custom Dockerfiles** should remain version-agnostic and rely on the base layers’ APT setup (current state is good).
- **Assets** should be selected by NGC versioning (current default is major `4`, mode `latest`, which matches the rolling
  `release-4` intent).
- **Source checkouts** are the current mismatch: `source/isaac_ros_manipulation.repos` is pinned to `release-4.0`.
  That is coherent only if you’re truly targeting 4.0; it is *not* coherent with a rolling `release-4` host, or with a
  pinned `release-4.1` host.

So the “gap” is not that the CLI might pick 4.1+; it’s that the repo currently mixes:

- rolling APT (`release-4`) and rolling NGC assets (major 4, latest), **with**
- pinned source branches (`release-4.0`).

### 2) Bootstrap writes user-global config (not workspace-scoped)

`scripts/bootstrap_isaac_ros_cli_files.sh` writes to:

- `~/.config/isaac-ros-cli/config.yaml`
- `~/.isaac_ros_dev-dockerargs`

That impacts *every* workspace on the machine, and can clobber existing CLI customization unless the user carefully
merges changes by hand.

For Isaac ROS 4 “bootstrap my workspace” workflows, it is cleaner to generate **workspace-scoped** config instead.
This can (and should) include the `realsense` layer as well.

### 3) EULA acceptance is a default assumption for this repo

The purpose of this repo is to automate install/setup of Isaac ROS workflows that require NGC assets and model installs.
In practice, that implies users intend to accept the applicable EULAs. Treat EULA acceptance as the default behavior
(e.g., `ISAAC_ROS_ACCEPT_EULA=1`) so automation works out of the box.

### 4) A lot happens at container start (keep as-is for now)

Auto-build + auto-setup at container start is intentionally part of the experience for now. Keep the current behavior and
env-var toggles; defer “make it opt-in” to a later iteration.

### 5) “Floating” is intentional: don’t fight NGC or upstream instructions

Third-party repos are meant to be cloned per upstream instructions and can remain floating. NGC is the source of truth
for which quickstart assets should be downloaded, so the workflow should generally follow NGC versioning (and not try to
pin packages/assets “the other way around”).

The only thing that needs tightening for coherence is the *NVIDIA* source branches in the `.repos` list (see #1).

## Recommended changes to `isaac_ros_custom_bringup` for Isaac ROS 4 (major)

### A) Let `isaac-ros` decide the base image; make source branches follow the chosen release strategy

The CLI should continue to decide the base image keys and APT release selection. The missing piece is a coherent way to
choose which `release-4.X` branch to clone when you want to build from source.

Concretely, `isaac_ros_custom_bringup/isaac_ros_4` should support two modes:

- **Pinned minor mode**: user passes `--minor 0|1|...` (or `--release 4.0|4.1|...`) and the repo list uses
  `release-4.<minor>` for NVIDIA repos.
- **Rolling minor mode**: user wants “whatever `release-4` currently means”. This cannot be *perfectly* automatic unless
  you have a reliable way to map “latest minor” → `release-4.X` for Git branches. Two practical approaches:
  - **Best-effort auto**: reuse the existing NGC “latest version” resolution logic (already used for quickstart assets)
    to infer the minor, then pick `release-4.X` branches.
  - **Manual-with-defaults**: default to a documented minor (e.g., current latest) but allow override via flags/env vars.

In either case, the important change is: stop hardcoding `release-4.0` in `source/isaac_ros_manipulation.repos` as the
only option.

### B) Make generated files workspace-scoped by default (non-invasive)

Prefer these outputs:

- CLI YAML config: `${ISAAC_ROS_WS}/.isaac-ros-cli/config.yaml` (workspace scope)
- Docker run args: `${ISAAC_ROS_WS}/../scripts/.isaac_ros_dev-dockerargs` (workspace-adjacent)
- Dockerfile search dirs: `${ISAAC_ROS_WS}/../scripts/.isaac_ros_common-config` (already used)

Keep user-global writes (`~/.config/...`, `~/.isaac_ros_dev-dockerargs`) behind an explicit `--user-scope` flag.

### E) Split “clone repos” from “build repos” (requested by user)

Today `scripts/isaac-manipulation-build.sh` will import repos and build as one workflow.

Recommended additions:

- New script (host or in-container) that does **only** `vcs import` (no `colcon build`)
- Keep `isaac-manipulation-build.sh` as “build-only” by default; make repo import opt-in

### F) Keep manual fixes/patching as-is (for now)

`scripts/isaac-manipulation-build.sh` edits an upstream CMake file to toggle asset install behavior, and
`scripts/apply-isaac-manipulation-fixes.sh` overlays upstream files.

These fixes are currently applied manually and are part of making the workflow work today; keep them unchanged for now.

## Bottom line (for Isaac ROS 4 major)

`src/isaac_ros_custom_bringup/isaac_ros_4/` already contains useful building blocks (custom image keys, asset setup,
source repo list, bootstrap script), but it needs a small refactor to be a clean Isaac ROS 4 bootstrap:

- Keep the CLI in charge of base images/APT versioning, and make `.repos` selection coherent (pinned vs rolling minor)
- Make bootstrap outputs workspace-scoped by default (including `realsense`)
- Add a clone-only script so users can fetch all repos without building
