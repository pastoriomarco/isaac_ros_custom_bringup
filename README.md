Isaac ROS YOLOv8 + FoundationPose Bringup
=========================================

This package provides standalone launch graphs to run a fine‑tuned YOLOv8 detector and wire it into the Isaac ROS FoundationPose pipeline using an Intel RealSense RGB‑D camera. It keeps upstream packages unmodified.

## PREREQUISITES

### Generate SDG and train model

Follow the instructions on [sdg_training_custom GitHub repo](https://github.com/pastoriomarco/sdg_training_custom).  

**ATTENTION**: You need at least 16GB VRAM to run `isaac_ros_foundationpose`. I run the FoundationPose pipeline on a Jetson Orin AGX Developer Kit, then use a laptop with 8GB VRAM RTX GPU to stream Isaac SIM camera/scene simulation.

You will also need a `obj` model and a `png` texture as described in [isaac_ros_foundationpose tutorial](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/#try-more-examples).

## SETUP & LAUNCH

### Step 1: Developer Environment Setup

First, set up the compute and developer environment by following Nvidia’s instructions:

* [Compute Setup](https://nvidia-isaac-ros.github.io/getting_started/hardware_setup/compute/index.html)
* [Developer Environment Setup](https://nvidia-isaac-ros.github.io/getting_started/dev_env_setup.html)

It’ very important that you completely follow the above setup steps for the platform you are going to use. Don’t skip the steps for [Jetson Platform](https://nvidia-isaac-ros.github.io/getting_started/hardware_setup/compute/index.html#jetson-platforms) if that’s what you are using, including [VPI](https://nvidia-isaac-ros.github.io/getting_started/hardware_setup/compute/jetson_vpi.html).

### Step 2: Download and run the setup script

Download [manymove_isaac_ros_startup.sh](https://github.com/pastoriomarco/manymove/blob/humble/manymove_planner/config/isaac_ros/manymove_isaac_ros_startup.sh), make it executable and run it. It will download all the packages tested with ManyMove and Isaac ROS.

### Step 3: [OPTIONAL] Check isaac_ros_common-config

Since I’m working with realsense cameras, the current config includes the realsense package.  
**If you are using RealSense cameras too, *skip this step*.**

If you don’t need it, modify the following file with your favorite editor:

```
${ISAAC_ROS_WS}/src/isaac_ros_common/scripts/.isaac_ros_common-config
```

Remove `.realsense` step from the config line. It will go from:

```
CONFIG_IMAGE_KEY=ros2_humble.realsense.manymove
```

To:

```
CONFIG_IMAGE_KEY=ros2_humble.manymove
```

### Step 4: Prepare YOLO model and object model

You need to place the fine-tuned YOLOV8 model inside ${ISAAC_ROS_WS} folder for the docker environment to be able to access it.  
Export the variables below to be able to use the example commands correctly.

```bash
export YOLO_MODEL_NAME=your_model_name
mkdir -p ${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/
cp /PATH/TO/YOUR/${YOLO_MODEL_NAME}.onnx ${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/
```

### Step 5: Launch the Docker container

Launch the docker container using the `run_dev.sh` script:

```
cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
   ./scripts/run_dev.sh
```

This will also build all the required packages.  
**IMPORTANT**: if you want to rebuild, remove `/build`, `/install` and `/log` folders in `${ISAAC_ROS_WS}/` before launching the docker.

### Step 6: Test Installation

*From inside the container:*  
You'll need to complete the [isaac_ros_foundationpose tutorial](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html#run-launch-file) and the [isaac_ros_yolov8 tutorial](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_yolov8/index.html#run-launch-file): on each, start from **`Run Launch File`** section, as once inside the docker container you'll already have the repos built from source.

### Step 7: Run example

The following example requires all the required variables to be set correctly:

```bash
export YOLO_MODEL_NAME=your_model_name
export MESH_FILE_PATH=/path/to/your/object.obj
export TEXTURE_PATH=/path/to/object/texture.png
ros2 launch isaac_ros_custom_bringup yolov8_foundationpose_realsense_remote.launch.py \
  yolov8_model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/${YOLO_MODEL_NAME}.onnx \
  yolov8_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/${YOLO_MODEL_NAME}.plan \
  input_tensor_names:='["input_tensor"]' input_binding_names:='["images"]' \
  output_tensor_names:='["output_tensor"]' output_binding_names:='["output0"]' \
  confidence_threshold:=0.25 nms_threshold:=0.45 num_classes:=1 \
  mesh_file_path:=${MESH_FILE_PATH} \
  texture_path:=${TEXTURE_PATH} \
  refine_model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_model.onnx \
  refine_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan \
  score_model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_model.onnx \
  score_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_trt_engine.plan \
  remote_color_image_topic:=/image_rect \
  remote_color_info_topic:=/camera_info \
  remote_depth_aligned_topic:=/depth \
  depth_is_float:=True \
  launch_rviz:=True
```

---

## INFO

### Contents
- `launch/yolov8_inference.launch.py`: YOLOv8 encoder → TensorRT → decoder (Detection2DArray)
- `launch/yolov8_foundationpose_realsense.launch.py`: RealSense → encoder → YOLOv8 → Detection2DToMask → resize → FoundationPose (+ tracking)
- `launch/yolov8_foundationpose_realsense_remote.launch.py`: Same as RealSense pipeline but assumes the RealSense runs on another machine; subscribes to its topics (no driver node started locally)  

### Build
1) From the workspace root:
```bash
colcon build --packages-select isaac_ros_custom_bringup
source install/setup.bash
```
### YOLOv8 Only
Example with Ultralytics export defaults (`images`/`output0`):
```bash
ros2 launch isaac_ros_custom_bringup yolov8_inference.launch.py \
  model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/your_finetuned.onnx \
  engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/your_finetuned.plan \
  input_tensor_names:='["input_tensor"]' \
  input_binding_names:='["images"]' \
  output_tensor_names:='["output_tensor"]' \
  output_binding_names:='["output0"]' \
  confidence_threshold:=0.25 nms_threshold:=0.45 num_classes:=1 \
  image_input_topic:=/rgb/image_rect_color camera_info_input_topic:=/rgb/camera_info
```
### YOLOv8 → FoundationPose (RealSense)
```bash
ros2 launch isaac_ros_custom_bringup yolov8_foundationpose_realsense.launch.py \
  yolov8_model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/td06_d.onnx \
  yolov8_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/td06_d.plan \
  input_tensor_names:='["input_tensor"]' \
  input_binding_names:='["images"]' \
  output_tensor_names:='["output_tensor"]' \
  output_binding_names:='["output0"]' \
  confidence_threshold:=0.25 nms_threshold:=0.45 num_classes:=1 \
  mesh_file_path:=$HOME/meshes/workpiece.obj \
  texture_path:=$HOME/meshes/flat_gray.png \
  refine_model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_model.onnx \
  refine_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan \
  score_model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_model.onnx \
  score_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_trt_engine.plan \
  launch_rviz:=True
```
### YOLOv8 → FoundationPose (Remote RealSense)
```bash
ros2 launch isaac_ros_custom_bringup yolov8_foundationpose_realsense_remote.launch.py \
  yolov8_model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/td06_c.onnx \
  yolov8_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/td06_c.plan \
  input_tensor_names:='["input_tensor"]' input_binding_names:='["images"]' \
  output_tensor_names:='["output_tensor"]' output_binding_names:='["output0"]' \
  confidence_threshold:=0.25 nms_threshold:=0.45 num_classes:=1 \
  mesh_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/objects/TD06/TDNS06.obj \
  texture_path:=${ISAAC_ROS_WS}/isaac_ros_assets/objects/TD06/gray.png \
  refine_model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_model.onnx \
  refine_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan \
  score_model_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_model.onnx \
  score_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_trt_engine.plan \
  remote_color_image_topic:=/image_rect \
  remote_color_info_topic:=/camera_info \
  remote_depth_aligned_topic:=/depth \
  depth_is_float:=True \
  launch_rviz:=True
```
Required incoming topics (from remote RealSense):
- `/color/image_raw`: RGB image (sensor_msgs/Image, rgb8)
- `/color/camera_info`: Camera intrinsics (sensor_msgs/CameraInfo)
- `/aligned_depth_to_color/image_raw`: Depth aligned to color (sensor_msgs/Image)
  - If 16UC1 in mm (RealSense-like), use default settings
  - If 32FC1 in meters (Isaac Sim-like), add `depth_is_float:=True`
What FoundationPose subscribes to in this graph:
- `pose_estimation/image`: RGB image (nitros_image_rgb8) — produced via encoder/drop remap
- `pose_estimation/camera_info`: Camera model (nitros_camera_info)
- `pose_estimation/depth_image`: Depth image (nitros_image_32FC1, meters) — `ConvertMetricNode` converts from 16UC1 mm
- `pose_estimation/segmentation`: Binary mask (nitros_image_mono8) — from YOLOv8 detections via `Detection2DToMask` and resize
Tracking node subscriptions (if enabled):
- `tracking/image`, `tracking/camera_info`, `tracking/depth_image` — same modalities as above
- `tracking/pose_input` — initial pose from the selector when a new detection arrives

Notes
- The encoder letterboxes 1280×720 → 640×640. The mask is created at 640×360 (valid content) and resized to 1280×720 so RGB/depth/mask match (required by FoundationPose).
- Depth is converted from uint16 mm → float32 meters using `ConvertMetricNode`.
- Detection2DToMask takes the highest‑score detection. For multiple instances, consider multiple FoundationPose nodes or a custom mask node.

---

## YoloV8 only Walkthrough:

**Follow instructions** from 
[isaac_ros_object_detection GitHub repo for isaac_ros_yolov8](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_yolov8/index.html), but with the following **differences**:

### Download Quickstart Assets

#### Section 1.
Download Quickstart Assets as instructed.

#### Section 2. 
***Use custom model*** instead of the default one.  
To obtain a custom model, generate synthetic data with Isaac SIM and train your custom yolov8 following [the instructions in THIS REPO](https://github.com/pastoriomarco/sdg_training_custom.git).

#### Section 3.
The [script](https://github.com/pastoriomarco/sdg_training_custom/blob/main/custom_sdg/custom_train_yolov8.sh) in my custom sdg repo converts the `.pt` model to `.onnx` automatically, but it should then be copied in `${ISAAC_ROS_WS}/isaac_ros_assets/models/yolov8/`, and the commands must refer to its name correctly.

If you interrupt the training of the model before the selected number of epochs, you'll skip the conversion too: you can still follow the original instructions to convert the best model obtained from the script.

### Build isaac_ros_yolov8

#### Use the Build from Source tab

#### Section 1. 

Clone the original repo, but also add my custom bringup. You might need isaac_ros_foundationpose's repo too for dependencies.  
Here's the full sequence of command:
```bash
cd ${ISAAC_ROS_WS}/src
git clone -b release-3.2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection.git isaac_ros_object_detection
git clone -b release-3.2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation.git isaac_ros_pose_estimation
git clone https://github.com/pastoriomarco/isaac_ros_custom_bringup.git
```

#### Section 2. to 5.

Follow original instructions.

### Run Launch File

#### Section 1. 

Enter the docker container:

```bash
cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
   ./scripts/run_dev.sh
```

#### Rosbag tab

##### Section 1.

```bash
sudo apt-get update && \
sudo apt-get install -y ros-humble-isaac-ros-examples
```

##### Section 2.

Use the following launchers or change accordingly

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

##### Section 3.

```bash
cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
./scripts/run_dev.sh
```

##### Section 4. 
If you **publish from Isaac SIM** skip this section and **don't run the rosbag** 

### Visualize Results

Follow original instructions

---