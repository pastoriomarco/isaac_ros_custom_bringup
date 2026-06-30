# DDS transport tuning — multi-machine Isaac Sim ↔ Isaac ROS (no LAN flooding)

## Symptom & root cause
When Isaac Sim streams camera topics, the **whole LAN** slows down (ping to `8.8.8.8`
jumps ~10 ms → ~1700 ms). Cause: **ROS 2/DDS uses multicast for participant discovery**,
and an L2 switch **without IGMP snooping floods multicast to every port** — including the
gateway uplink. Here Thor (`enP2p1s0` = `192.168.1.136`), the Isaac Sim host, and the
gateway (`192.168.1.1`) are all on the same `/24`, so the flood hits the internet path.

**Both Isaac Sim 5.x and the Isaac ROS container default to Fast DDS** (`rmw_fastrtps_cpp`),
so they already match — do **not** switch one to CycloneDDS (mixing vendors breaks discovery).
Raw 1280×720 RGB + depth @30 Hz is ~1.5 Gbps; unicast keeps it off the gateway, but a single
1 GbE link can still saturate — consider a dedicated NIC/subnet or lower camera res for stability.

## Fix A — quickest (env vars only, no XML). Try this first.
Set on **both** the Isaac Sim host **and** inside the Isaac ROS container (same values):
```bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST          # stop subnet-wide multicast discovery
export ROS_STATIC_PEERS='192.168.1.136;<ISAAC_SIM_IP>'  # explicit unicast peers (Thor;Sim)
```
`LOCALHOST` suppresses the subnet multicast sweep; `ROS_STATIC_PEERS` still lets the two named
machines find each other (Jazzy "Improved Dynamic Discovery"). No file needed.

## Fix B — robust (XML profile). Use if A isn't enough or for stable production streaming.
`fastdds_unicast.xml` (in this folder) forces unicast discovery, disables multicast, and drops
shared-memory. Edit the `<initialPeersList>` IPs (Thor is pre-filled; set the Isaac Sim host),
then on **both** machines:
```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/abs/path/to/fastdds_unicast.xml
export ROS_DOMAIN_ID=0
```

### Isaac Sim host
Set the env vars **in the shell before launching** (or in the App Selector's inherited
environment) — the selector's "additional args" set startup extensions, **not** DDS env vars,
so they must be real environment variables:
```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/.ros/fastdds_unicast.xml   # copy this file there
export ROS_DOMAIN_ID=0
./isaac-sim.sh
```
> The RMW/library env "can only be set once per terminal" (Isaac Sim docs) — set it in a clean
> shell, once, before launch. Copy `fastdds_unicast.xml` to the Sim host and set its `<address>`
> whitelist (if used) to the Sim host's own LAN IP.

### Isaac ROS container (Thor)
The container runs `--network host`, so it shares `enP2p1s0`/`192.168.1.136`. Point it at this
file (it lives in the mounted workspace) by adding to the dev-container Docker args
(`/etc/isaac-ros-cli/.isaac_ros_dev-dockerargs`, or export inside the container before launching nodes):
```
-e RMW_IMPLEMENTATION=rmw_fastrtps_cpp
-e ROS_DOMAIN_ID=0
-e FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/src/isaac_ros_custom_bringup/isaac_ros_4/dds/fastdds_unicast.xml
```
The shipped `/etc/isaac-ros-cli/docker/middleware_profiles/rtps_udp_profile.xml` only disables
shared-memory — it does **not** stop multicast — so override it with this profile.

## OS-level receive buffer (Isaac ROS recommended, for heavy topics)
```bash
sudo sysctl -w net.core.rmem_max=2147483647     # add to /etc/sysctl.d to persist
```

## If you keep any multicast: enable **IGMP snooping** on the switch/router (hardware fix).

## Symptom 2 — `ros2 topic echo`/`hz` can't find the topic once FoundationPose runs
Ranked causes + how to tell them apart (run in the container):
```bash
echo $RMW_IMPLEMENTATION                     # must match the publisher's RMW
ros2 topic info /your/topic --verbose        # check publisher Reliability (BEST_EFFORT?)
ros2 topic echo /your/topic --qos-reliability best_effort --qos-durability volatile
ros2 daemon stop && ros2 daemon start && ros2 topic list   # clears stale discovery cache
ros2 topic list --no-daemon                  # bypass the cache
ros2 doctor --report
```
- Plain echo fails but `--qos-reliability best_effort` works → **QoS** (camera = `SensorDataQoS`
  = BEST_EFFORT; echo defaults to RELIABLE). Data was fine.
- Topic gone from `ros2 topic list` but returns after `daemon stop/start`, correlated with the
  ping spike → **discovery loss from the multicast flood** → apply Fix A/B.
- Only `*/nitros`-suffixed topics are silent while the pipeline clearly runs → **NITROS type
  negotiation**, expected (those never `echo`); check the non-nitros sibling with correct QoS.
- `RMW_IMPLEMENTATION` differs between machines → **RMW mismatch** → unify on Fast DDS.

Sources: Isaac Sim 5.1 ROS 2 Installation; Isaac ROS Nvblox "ROS Communication Issues";
eProsima Fast DDS (interface whitelist, shared-memory, ros2_configure); ROS 2 Jazzy
"Improved Dynamic Discovery"; Cyclone DDS config reference.
