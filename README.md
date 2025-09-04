Isaac ROS YOLOv8 + FoundationPose Bringup
=========================================

This package provides standalone launch graphs to run a fine‑tuned YOLOv8 detector and wire it into the Isaac ROS FoundationPose pipeline using an Intel RealSense RGB‑D camera. It keeps upstream packages unmodified.

Contents
- `launch/yolov8_inference.launch.py`: YOLOv8 encoder → TensorRT → decoder (Detection2DArray)
- `launch/yolov8_foundationpose_realsense.launch.py`: RealSense → encoder → YOLOv8 → Detection2DToMask → resize → FoundationPose (+ tracking)

Build
1) From the workspace root:

   colcon build --packages-select isaac_ros_custom_bringup
   source install/setup.bash

YOLOv8 Only
Example with Ultralytics export defaults (`images`/`output0`):

  ros2 launch isaac_ros_custom_bringup yolov8_inference.launch.py \
    model_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/yolov8/your_finetuned.onnx \
    engine_file_path:='' \
    input_tensor_names:='["input_tensor"]' \
    input_binding_names:='["images"]' \
    output_tensor_names:='["output_tensor"]' \
    output_binding_names:='["output0"]' \
    confidence_threshold:=0.25 nms_threshold:=0.45 num_classes:=80 \
    image_input_topic:=/rgb/image_rect_color camera_info_input_topic:=/rgb/camera_info

YOLOv8 → FoundationPose (RealSense)

  ros2 launch isaac_ros_custom_bringup yolov8_foundationpose_realsense.launch.py \
    yolov8_model_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/yolov8/your_finetuned.onnx \
    yolov8_engine_file_path:='' \
    input_tensor_names:='["input_tensor"]' \
    input_binding_names:='["images"]' \
    output_tensor_names:='["output_tensor"]' \
    output_binding_names:='["output0"]' \
    confidence_threshold:=0.25 nms_threshold:=0.45 num_classes:=80 \
    mesh_file_path:=$HOME/meshes/workpiece.obj \
    texture_path:=$HOME/meshes/flat_gray.png \
    refine_model_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/foundationpose/refine_model.onnx \
    refine_engine_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan \
    score_model_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/foundationpose/score_model.onnx \
    score_engine_file_path:=$ISAAC_ROS_WS/isaac_ros_assets/models/foundationpose/score_trt_engine.plan \
    launch_rviz:=True

Notes
- The encoder letterboxes 1280×720 → 640×640. The mask is created at 640×360 (valid content) and resized to 1280×720 so RGB/depth/mask match (required by FoundationPose).
- Depth is converted from uint16 mm → float32 meters using `ConvertMetricNode`.
- Detection2DToMask takes the highest‑score detection. For multiple instances, consider multiple FoundationPose nodes or a custom mask node.
