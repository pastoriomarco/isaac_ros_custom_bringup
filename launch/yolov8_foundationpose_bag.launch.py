#!/usr/bin/env python3

import os, json

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


# Defaults (mirrors yolov8 encoder input size)
YOLOV8_MODEL_INPUT = 640
VISUALIZATION_DOWNSCALING_FACTOR = 10

# FoundationPose defaults (can be overridden by args)
REFINE_MODEL_PATH = '/tmp/refine_model.onnx'
REFINE_ENGINE_PATH = '/tmp/refine_trt_engine.plan'
SCORE_MODEL_PATH = '/tmp/score_model.onnx'
SCORE_ENGINE_PATH = '/tmp/score_trt_engine.plan'


def _load_specs(path: str):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {"camera_resolution": {"width": 1280, "height": 720}}


def generate_launch_description():
    # Interface specs (resolution for bag/Isaac Sim)
    default_specs = os.path.join(
        os.environ.get('ISAAC_ROS_WS', '/workspaces/isaac_ros-dev'),
        'isaac_ros_assets', 'isaac_ros_foundationpose', 'quickstart_interface_specs.json'
    )
    specs_path = os.path.expandvars(os.path.expanduser(
        os.getenv('INTERFACE_SPECS_FILE', default_specs)
    ))
    specs = _load_specs(specs_path)
    cam_w = int(specs.get('camera_resolution', {}).get('width', 1280))
    cam_h = int(specs.get('camera_resolution', {}).get('height', 720))

    # Compute YOLO letterboxed mask size (same logic as RealSense launcher)
    ratio = max(1, cam_w // YOLOV8_MODEL_INPUT) if cam_w >= YOLOV8_MODEL_INPUT else 1
    mask_w = int(cam_w / ratio)
    mask_h = int(cam_h / ratio)

    # Common args
    mesh_file_path = LaunchConfiguration('mesh_file_path')
    texture_path = LaunchConfiguration('texture_path')
    refine_model_file_path = LaunchConfiguration('refine_model_file_path')
    refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
    score_model_file_path = LaunchConfiguration('score_model_file_path')
    score_engine_file_path = LaunchConfiguration('score_engine_file_path')

    yolov8_model_file_path = LaunchConfiguration('yolov8_model_file_path')
    yolov8_engine_file_path = LaunchConfiguration('yolov8_engine_file_path')
    input_tensor_names = LaunchConfiguration('input_tensor_names')
    input_binding_names = LaunchConfiguration('input_binding_names')
    output_tensor_names = LaunchConfiguration('output_tensor_names')
    output_binding_names = LaunchConfiguration('output_binding_names')
    verbose = LaunchConfiguration('verbose')
    force_engine_update = LaunchConfiguration('force_engine_update')

    image_input_topic = LaunchConfiguration('image_input_topic')
    camera_info_input_topic = LaunchConfiguration('camera_info_input_topic')
    depth_image_topic = LaunchConfiguration('depth_image_topic')

    input_encoding = LaunchConfiguration('input_encoding')
    image_mean = LaunchConfiguration('image_mean')
    image_stddev = LaunchConfiguration('image_stddev')

    launch_rviz = LaunchConfiguration('launch_rviz')
    container_name = LaunchConfiguration('container_name')

    rviz_config_path = os.path.join(
        get_package_share_directory('isaac_ros_foundationpose'),
        'rviz', 'foundationpose_realsense.rviz')

    # Encoder (letterbox WxH -> 640x640)
    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    yolov8_encoder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')
        ]),
        launch_arguments={
            'input_image_width': str(cam_w),
            'input_image_height': str(cam_h),
            'network_image_width': str(YOLOV8_MODEL_INPUT),
            'network_image_height': str(YOLOV8_MODEL_INPUT),
            'image_mean': image_mean,
            'image_stddev': image_stddev,
            'attach_to_shared_component_container': 'True',
            'component_container_name': container_name,
            'dnn_image_encoder_namespace': 'yolov8_encoder',
            'image_input_topic': image_input_topic,
            'camera_info_input_topic': camera_info_input_topic,
            'tensor_output_topic': '/tensor_pub',
            'keep_aspect_ratio': 'True',
            'input_encoding': input_encoding,
        }.items()
    )

    # If depth is 16UC1 in bag/Isaac Sim, convert to 32FC1
    convert_metric_node = ComposableNode(
        package='isaac_ros_depth_image_proc',
        plugin='nvidia::isaac_ros::depth_image_proc::ConvertMetricNode',
        remappings=[('image_raw', depth_image_topic), ('image', 'depth_image')]
    )

    # YOLOv8 inference
    tensor_rt_node = ComposableNode(
        name='yolov8_tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'model_file_path': yolov8_model_file_path,
            'engine_file_path': yolov8_engine_file_path,
            'input_tensor_names': input_tensor_names,
            'input_binding_names': input_binding_names,
            'output_tensor_names': output_tensor_names,
            'output_binding_names': output_binding_names,
            'verbose': verbose,
            'force_engine_update': force_engine_update,
            'relaxed_dimension_check': True,
            'max_batch_size': 1,
        }],
        remappings=[('tensor_input', '/tensor_pub')]
    )

    yolov8_decoder_node = ComposableNode(
        name='yolov8_decoder',
        package='isaac_ros_yolov8',
        plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
        parameters=[{
            'confidence_threshold': ParameterValue(LaunchConfiguration('confidence_threshold'), value_type=float),
            'nms_threshold': ParameterValue(LaunchConfiguration('nms_threshold'), value_type=float),
            'num_classes': ParameterValue(LaunchConfiguration('num_classes'), value_type=int),
            'tensor_name': 'output_tensor',
        }]
    )

    # Convert 2D detection to segmentation mask (letterboxed size)
    detection2_d_to_mask_node = ComposableNode(
        name='detection2_d_to_mask',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
        parameters=[{
            'mask_width': mask_w,
            'mask_height': mask_h,
        }],
        remappings=[('detection2_d_array', 'detections_output'),
                    ('segmentation', 'yolov8_segmentation_small')]
    )

    # Scale segmentation back to camera resolution
    resize_mask_node = ComposableNode(
        name='resize_mask_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': mask_w,
            'input_height': mask_h,
            'output_width': cam_w,
            'output_height': cam_h,
            'keep_aspect_ratio': False,
            'disable_padding': False,
        }],
        remappings=[('image', 'yolov8_segmentation_small'),
                    ('camera_info', camera_info_input_topic),
                    ('resize/image', 'segmentation'),
                    ('resize/camera_info', 'camera_info_segmentation')]
    )

    # Selector to gate between detection/tracking
    selector_node = ComposableNode(
        name='selector_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Selector',
        parameters=[{'reset_period': 10000}],
        remappings=[('depth_image', 'depth_image'),
                    ('image', image_input_topic),
                    ('camera_info', camera_info_input_topic),
                    ('segmentation', 'segmentation')]
    )

    foundationpose_node = ComposableNode(
        name='foundationpose_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
        parameters=[{
            'mesh_file_path': mesh_file_path,
            'texture_path': texture_path,
            'refine_model_file_path': refine_model_file_path,
            'refine_engine_file_path': refine_engine_file_path,
            'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'refine_input_binding_names': ['input1', 'input2'],
            'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
            'refine_output_binding_names': ['output1', 'output2'],
            'score_model_file_path': score_model_file_path,
            'score_engine_file_path': score_engine_file_path,
            'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'score_input_binding_names': ['input1', 'input2'],
            'score_output_tensor_names': ['output_tensor'],
            'score_output_binding_names': ['output1'],
        }]
    )

    foundationpose_tracking_node = ComposableNode(
        name='foundationpose_tracking_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode',
        parameters=[{
            'mesh_file_path': mesh_file_path,
            'texture_path': texture_path,
            'refine_model_file_path': refine_model_file_path,
            'refine_engine_file_path': refine_engine_file_path,
            'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'refine_input_binding_names': ['input1', 'input2'],
            'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
            'refine_output_binding_names': ['output1', 'output2'],
        }]
    )

    # Small image for RViz
    resize_left_viz = ComposableNode(
        name='resize_left_viz',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': cam_w,
            'input_height': cam_h,
            'output_width': int(cam_w / VISUALIZATION_DOWNSCALING_FACTOR),
            'output_height': int(cam_h / VISUALIZATION_DOWNSCALING_FACTOR),
            'keep_aspect_ratio': False,
            'encoding_desired': 'rgb8',
            'disable_padding': False,
        }],
        remappings=[('image', image_input_topic),
                    ('camera_info', camera_info_input_topic),
                    ('resize/image', 'image_rect_viz'),
                    ('resize/camera_info', 'camera_info_viz')]
    )

    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_config_path], condition=IfCondition(launch_rviz)
    )

    container = ComposableNodeContainer(
        name=LaunchConfiguration('container_name'),
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            convert_metric_node,
            tensor_rt_node,
            yolov8_decoder_node,
            detection2_d_to_mask_node,
            resize_mask_node,
            selector_node,
            foundationpose_node,
            foundationpose_tracking_node,
            resize_left_viz,
        ],
        output='screen'
    )

    return LaunchDescription([
        # IO and env
        DeclareLaunchArgument('interface_specs_file', default_value=default_specs),

        # Input topics (from rosbag / Isaac Sim)
        DeclareLaunchArgument('image_input_topic', default_value='/image_rect'),
        DeclareLaunchArgument('camera_info_input_topic', default_value='/camera_info_rect'),
        DeclareLaunchArgument('depth_image_topic', default_value='/depth/image_rect'),
        DeclareLaunchArgument('input_encoding', default_value='rgb8'),
        DeclareLaunchArgument('image_mean', default_value='[0.0, 0.0, 0.0]'),
        DeclareLaunchArgument('image_stddev', default_value='[1.0, 1.0, 1.0]'),

        # FoundationPose assets
        DeclareLaunchArgument('mesh_file_path', default_value=''),
        DeclareLaunchArgument('texture_path', default_value=''),
        DeclareLaunchArgument('refine_model_file_path', default_value=REFINE_MODEL_PATH),
        DeclareLaunchArgument('refine_engine_file_path', default_value=REFINE_ENGINE_PATH),
        DeclareLaunchArgument('score_model_file_path', default_value=SCORE_MODEL_PATH),
        DeclareLaunchArgument('score_engine_file_path', default_value=SCORE_ENGINE_PATH),

        # YOLOv8 engines
        DeclareLaunchArgument('yolov8_model_file_path', default_value=''),
        DeclareLaunchArgument('yolov8_engine_file_path', default_value=''),
        DeclareLaunchArgument('input_tensor_names', default_value='["input_tensor"]'),
        DeclareLaunchArgument('input_binding_names', default_value='["images"]'),
        DeclareLaunchArgument('output_tensor_names', default_value='["output_tensor"]'),
        DeclareLaunchArgument('output_binding_names', default_value='["output0"]'),
        DeclareLaunchArgument('verbose', default_value='False'),
        DeclareLaunchArgument('force_engine_update', default_value='False'),

        # Decoder params
        DeclareLaunchArgument('confidence_threshold', default_value='0.25'),
        DeclareLaunchArgument('nms_threshold', default_value='0.45'),
        DeclareLaunchArgument('num_classes', default_value='80'),

        # UI / container
        DeclareLaunchArgument('launch_rviz', default_value='False'),
        DeclareLaunchArgument('container_name', default_value='yolov8_foundationpose_container'),

        # Nodes
        container,
        yolov8_encoder_launch,
        rviz_node,
    ])

