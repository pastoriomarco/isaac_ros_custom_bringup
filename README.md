Isaac ROS YOLOv8 + FoundationPose Bringup
=========================================

This package provides standalone launch graphs to run a fine‑tuned YOLOv8 detector and wire it into the Isaac ROS FoundationPose pipeline using an Intel RealSense RGB‑D camera. It keeps upstream packages unmodified.

Contents
- `launch/yolov8_inference.launch.py`: YOLOv8 encoder → TensorRT → decoder (Detection2DArray)
- `launch/yolov8_foundationpose_realsense.launch.py`: RealSense → encoder → YOLOv8 → Detection2DToMask → resize → FoundationPose (+ tracking)

Build
1) From the workspace root:

```bash
   colcon build --packages-select isaac_ros_custom_bringup
   source install/setup.bash
```

YOLOv8 Only
Example with Ultralytics export defaults (`images`/`output0`):

```bash
  ros2 launch isaac_ros_custom_bringup yolov8_inference.launch.py \
    model_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/yolov8/your_finetuned.onnx \
    engine_file_path:='' \
    input_tensor_names:='["input_tensor"]' \
    input_binding_names:='["images"]' \
    output_tensor_names:='["output_tensor"]' \
    output_binding_names:='["output0"]' \
    confidence_threshold:=0.25 nms_threshold:=0.45 num_classes:=1 \
    image_input_topic:=/rgb/image_rect_color camera_info_input_topic:=/rgb/camera_info
```

YOLOv8 → FoundationPose (RealSense)

```bash
  ros2 launch isaac_ros_custom_bringup yolov8_foundationpose_realsense.launch.py \
    yolov8_model_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/yolov8/your_finetuned.onnx \
    yolov8_engine_file_path:='' \
    input_tensor_names:='["input_tensor"]' \
    input_binding_names:='["images"]' \
    output_tensor_names:='["output_tensor"]' \
    output_binding_names:='["output0"]' \
    confidence_threshold:=0.25 nms_threshold:=0.45 num_classes:=1 \
    mesh_file_path:=$HOME/meshes/workpiece.obj \
    texture_path:=$HOME/meshes/flat_gray.png \
    refine_model_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/foundationpose/refine_model.onnx \
    refine_engine_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan \
    score_model_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/foundationpose/score_model.onnx \
    score_engine_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/foundationpose/score_trt_engine.plan \
    launch_rviz:=True
```

ros2 launch isaac_ros_custom_bringup yolov8_foundationpose_bag.launch.py \
  yolov8_model_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/yolov8/td06_c.onnx \
  yolov8_engine_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/yolov8/td06_c.plan \
  input_tensor_names:='["input_tensor"]' \
  input_binding_names:='["images"]' \
  output_tensor_names:='["output_tensor"]' \
  output_binding_names:='["output0"]' \
  confidence_threshold:=0.25 nms_threshold:=0.45 num_classes:=1 \
  mesh_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/objects/TD06/TDNS06.obj \
  texture_path:=$ISAAC_ROS_WS/isaac_ros_assets/objects/TD06/gray.png \
  refine_model_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/foundationpose/refine_model.onnx \
  refine_engine_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan \
  score_model_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/foundationpose/score_model.onnx \
  score_engine_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/foundationpose/score_trt_engine.plan \
  image_input_topic:=/image_rect camera_info_input_topic:=/camera_info_rect depth_image_topic:=/depth/image_rect

Notes
- The encoder letterboxes 1280×720 → 640×640. The mask is created at 640×360 (valid content) and resized to 1280×720 so RGB/depth/mask match (required by FoundationPose).
- Depth is converted from uint16 mm → float32 meters using `ConvertMetricNode`.
- Detection2DToMask takes the highest‑score detection. For multiple instances, consider multiple FoundationPose nodes or a custom mask node.



## Walkthrough YoloV8 only:

#Follow instructions from 
https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_yolov8/index.html

#Differences:

Download Quickstart Assets
2. Use custom model instead
3. The script in my custom sdg repo converts it automatically (ADD REFERENCE HERE), but it must be copied in 
${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/ , and the commands must refer to it correctly if renamed

Build isaac_ros_yolov8
1. change the command:
cd ${ISAAC_ROS_WS}/src && \
   git clone -b release-3.2 https://github.com/pastoriomarco/isaac_ros_object_detection.git isaac_ros_object_detection

Run Launch File
2. Use the following launchers or change accordingly

#with engine update, model _a

```bash
ros2 launch isaac_ros_custom_bringup yolov8_inference.launch.py \
  model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/td06_a.onnx \
  engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/td06_a.plan \
  network_image_width:=640 network_image_height:=640 \
  input_tensor_names:="['input_tensor']" \
  input_binding_names:="['images']" \
  output_tensor_names:="['output_tensor']" \
  output_binding_names:="['output0']" \
  num_classes:=1 \
  force_engine_update:=True \
  verbose:=True
```

#threshold, no engine update, model _c

```bash
ros2 launch isaac_ros_custom_bringup yolov8_inference.launch.py \
  model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/td06_c.onnx \
  engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/td06_c.plan \
  network_image_width:=640 \
  network_image_height:=640 \
  input_tensor_names:="['input_tensor']" \
  input_binding_names:="['images']" \
  output_tensor_names:="['output_tensor']" \
  output_binding_names:="['output0']" \
  num_classes:=1 \
  verbose:=True \
  confidence_threshold:=0.7 \
  nms_threshold:=0.5 
```

4. Don't run the rosbag if you publish from Isaac SIM