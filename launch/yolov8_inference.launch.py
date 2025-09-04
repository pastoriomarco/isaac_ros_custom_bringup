#!/usr/bin/env python3

import os, json
from typing import Dict, Any

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def _load_specs(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {"camera_resolution": {"width": 1280, "height": 720}}


def generate_launch_description():
    interface_specs_file = LaunchConfiguration('interface_specs_file')
    model_file_path = LaunchConfiguration('model_file_path')
    engine_file_path = LaunchConfiguration('engine_file_path')
    input_tensor_names = LaunchConfiguration('input_tensor_names')
    input_binding_names = LaunchConfiguration('input_binding_names')
    output_tensor_names = LaunchConfiguration('output_tensor_names')
    output_binding_names = LaunchConfiguration('output_binding_names')
    verbose = LaunchConfiguration('verbose')
    force_engine_update = LaunchConfiguration('force_engine_update')

    confidence_threshold = LaunchConfiguration('confidence_threshold')
    nms_threshold = LaunchConfiguration('nms_threshold')
    num_classes = LaunchConfiguration('num_classes')

    network_image_width = LaunchConfiguration('network_image_width')
    network_image_height = LaunchConfiguration('network_image_height')
    input_encoding = LaunchConfiguration('input_encoding')
    image_mean = LaunchConfiguration('image_mean')
    image_stddev = LaunchConfiguration('image_stddev')
    image_input_topic = LaunchConfiguration('image_input_topic')
    camera_info_input_topic = LaunchConfiguration('camera_info_input_topic')

    specs_path = os.environ.get("ISAAC_ROS_WS", "/workspaces/isaac_ros-dev") + "/isaac_ros_assets/isaac_ros_yolov8/quickstart_interface_specs.json"
    default_interface_specs_file = specs_path
    specs = _load_specs(os.path.expandvars(os.path.expanduser(os.getenv("INTERFACE_SPECS_FILE", default_interface_specs_file))))
    cam_w = str(specs.get("camera_resolution", {}).get("width", 1280))
    cam_h = str(specs.get("camera_resolution", {}).get("height", 720))

    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')

    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'model_file_path': model_file_path,
            'engine_file_path': engine_file_path,
            'output_binding_names': output_binding_names,
            'output_tensor_names': output_tensor_names,
            'input_tensor_names': input_tensor_names,
            'input_binding_names': input_binding_names,
            'verbose': verbose,
            'force_engine_update': force_engine_update,
            'relaxed_dimension_check': True,
            'max_batch_size': 1
        }]
    )

    yolov8_decoder_node = ComposableNode(
        name='yolov8_decoder_node',
        package='isaac_ros_yolov8',
        plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
        parameters=[{
            'confidence_threshold': ParameterValue(confidence_threshold, value_type=float),
            'nms_threshold': ParameterValue(nms_threshold, value_type=float),
            'num_classes': ParameterValue(num_classes, value_type=int),
            'tensor_name': 'output_tensor',
        }]
    )

    encoder_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]),
        launch_arguments={
            'input_image_width': cam_w,
            'input_image_height': cam_h,
            'network_image_width': network_image_width,
            'network_image_height': network_image_height,
            'image_mean': image_mean,
            'image_stddev': image_stddev,
            'attach_to_shared_component_container': 'True',
            'component_container_name': '/yolov8_container',
            'dnn_image_encoder_namespace': 'yolov8_encoder',
            'image_input_topic': image_input_topic,
            'camera_info_input_topic': camera_info_input_topic,
            'tensor_output_topic': '/tensor_pub',
            'input_encoding': input_encoding,
        }.items(),
    )

    container = ComposableNodeContainer(
        package='rclcpp_components',
        name='yolov8_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=[tensor_rt_node, yolov8_decoder_node],
        arguments=['--ros-args', '--log-level', 'INFO'],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('interface_specs_file', default_value=default_interface_specs_file),
        DeclareLaunchArgument('model_file_path'),
        DeclareLaunchArgument('engine_file_path'),
        DeclareLaunchArgument('input_tensor_names', default_value='["input_tensor"]'),
        DeclareLaunchArgument('input_binding_names', default_value='["images"]'),
        DeclareLaunchArgument('output_tensor_names', default_value='["output_tensor"]'),
        DeclareLaunchArgument('output_binding_names', default_value='["output0"]'),
        DeclareLaunchArgument('verbose', default_value='False'),
        DeclareLaunchArgument('force_engine_update', default_value='False'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.25'),
        DeclareLaunchArgument('nms_threshold', default_value='0.45'),
        DeclareLaunchArgument('num_classes', default_value='80'),
        DeclareLaunchArgument('network_image_width', default_value='640'),
        DeclareLaunchArgument('network_image_height', default_value='640'),
        DeclareLaunchArgument('input_encoding', default_value='rgb8'),
        DeclareLaunchArgument('image_mean', default_value='[0.0, 0.0, 0.0]'),
        DeclareLaunchArgument('image_stddev', default_value='[1.0, 1.0, 1.0]'),
        DeclareLaunchArgument('image_input_topic', default_value='/image_rect'),
        DeclareLaunchArgument('camera_info_input_topic', default_value='/camera_info_rect'),
        encoder_include,
        container
    ])

