#!/usr/bin/env bash
# usage: live_run.sh TAG DURATION "LAUNCH ARGS"   (domain 0, subnet discovery; read-only on sim topics)
TAG=$1; DUR=$2; LARGS=$3
B=/workspaces/isaac_ros-dev/worktrees/isaac_ros5_trocar-build; T=$B/test; O=$T/out/$TAG; mkdir -p "$O"
source /opt/ros/lyrical/setup.bash; source $B/install/setup.bash
unset ROS_STATIC_PEERS ROS_LOCALHOST_ONLY FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI
export ROS_DOMAIN_ID=0 ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET RMW_IMPLEMENTATION=rmw_fastrtps_cpp
setsid ros2 launch isaac_ros_custom_bringup yolov8_foundationpose_isaac_sim.launch.py \
  yolov8_engine_file_path:=/workspaces/isaac_ros-dev/isaac_ros_assets/models/trocar_yolov8/engines/trocar_yolov8_thor_trt10.16.2_fp16.plan \
  $LARGS > "$O/launch.log" 2>&1 &
LP=$!; echo $LP > "$O/launch.pid"
for i in $(seq 1 90); do sleep 1; grep -q -E "Failed to load|Traceback" "$O/launch.log" && break; kill -0 $LP 2>/dev/null || break
  [ "$(grep -c 'Loaded node' "$O/launch.log")" -ge "${EXPECT_NODES:-17}" ] && break; done
echo "loaded nodes: $(grep -c 'Loaded node' "$O/launch.log") after ${i}s"
sleep ${WARMUP:-15}
CP=$(pgrep -f "component_container.*yolov8_foundationpose_container" | head -1); echo "container pid $CP"
( top -b -d 2 -n $((DUR/2)) -p $CP > "$O/top.log" 2>&1 & )
python3 -u $T/live_monitor.py --duration $DUR --out "$O" ${MON_ARGS} > "$O/monitor.log" 2>&1
cat "$O/monitor.log"
if [ -z "$KEEP" ]; then kill -INT -- -$LP 2>/dev/null; sleep 8; kill -KILL -- -$LP 2>/dev/null; sleep 1
  pgrep -af "component_container|ros2 launch|live_monitor" || echo "all test processes stopped"; fi
