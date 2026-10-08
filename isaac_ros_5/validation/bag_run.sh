#!/usr/bin/env bash
# Offline replay of the recorded simulator inputs (isolated: domain 78, localhost).
TAG=$1; B=/workspaces/isaac_ros-dev/worktrees/isaac_ros5_trocar-build; T=$B/test; O=$T/out/$TAG; mkdir -p "$O"
BAG=/workspaces/isaac_ros-dev/isaac_ros_assets/recordings/trocar_sim_20261007
source /opt/ros/lyrical/setup.bash; source $B/install/setup.bash
unset ROS_STATIC_PEERS ROS_LOCALHOST_ONLY FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI
export ROS_DOMAIN_ID=78 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
setsid ros2 launch isaac_ros_custom_bringup yolov8_foundationpose_isaac_sim.launch.py \
  yolov8_engine_file_path:=/workspaces/isaac_ros-dev/isaac_ros_assets/models/trocar_yolov8/engines/trocar_yolov8_thor_trt10.16.2_fp16.plan \
  camera_drop_drop_count:=0 camera_drop_window:=1 camera_drop_input_qos:=DEFAULT > "$O/launch.log" 2>&1 &
LP=$!
for i in $(seq 1 60); do sleep 1; [ "$(grep -c 'Loaded node' "$O/launch.log")" -ge 17 ] && break; done; sleep 5
python3 -u $T/live_monitor.py --duration 30 --out "$O" > "$O/monitor.log" 2>&1 &
MP=$!; sleep 2
ros2 bag play $BAG --qos-profile-overrides-path $T/replay_qos.yaml --topics /image_rect /camera_info /depth > "$O/play.log" 2>&1
wait $MP; cat "$O/monitor.log"
kill -INT -- -$LP 2>/dev/null; sleep 15; kill -KILL -- -$LP 2>/dev/null; sleep 1
pgrep -af "component_container|ros2 launch|live_monitor|bag play" || echo "all test processes stopped"
