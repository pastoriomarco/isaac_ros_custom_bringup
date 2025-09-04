#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


REALSENSE_IMAGE_WIDTH = 1280
REALSENSE_IMAGE_HEIGHT = 720
YOLOV8_MODEL_INPUT = 640
REALSENSE_TO_YOLO_RATIO = REALSENSE_IMAGE_WIDTH / YOLOV8_MODEL_INPUT
VISUALIZATION_DOWNSCALING_FACTOR = 10

REFINE_MODEL_PATH = '/tmp/refine_model.onnx'
REFINE_ENGINE_PATH = '/tmp/refine_trt_engine.plan'
SCORE_MODEL_PATH = '/tmp/score_model.onnx'
SCORE_ENGINE_PATH = '/tmp/score_trt_engine.plan'


def generate_launch_description():
    rviz_config_path = os.path.join(
        get_package_share_directory('isaac_ros_foundationpose'),
        'rviz', 'foundationpose_realsense.rviz')

    launch_args = [
        # Remote RealSense topics (from a different machine)
        DeclareLaunchArgument('remote_color_image_topic', default_value='/color/image_raw'),
        DeclareLaunchArgument('remote_color_info_topic', default_value='/color/camera_info'),
        DeclareLaunchArgument('remote_depth_aligned_topic', default_value='/aligned_depth_to_color/image_raw'),

        # Match upstream RealSense drop/frequency behavior
        DeclareLaunchArgument('hawk_expect_freq', default_value='28'),
        DeclareLaunchArgument('input_images_drop_freq', default_value='30'),

        # Assets / engines
        DeclareLaunchArgument('mesh_file_path', default_value=''),
        DeclareLaunchArgument('texture_path', default_value=''),
        DeclareLaunchArgument('refine_model_file_path', default_value=REFINE_MODEL_PATH),
        DeclareLaunchArgument('refine_engine_file_path', default_value=REFINE_ENGINE_PATH),
        DeclareLaunchArgument('score_model_file_path', default_value=SCORE_MODEL_PATH),
        DeclareLaunchArgument('score_engine_file_path', default_value=SCORE_ENGINE_PATH),
        DeclareLaunchArgument('yolov8_model_file_path', default_value=''),
        DeclareLaunchArgument('yolov8_engine_file_path', default_value=''),
        DeclareLaunchArgument('input_tensor_names', default_value='["input_tensor"]'),
        DeclareLaunchArgument('input_binding_names', default_value='["images"]'),
        DeclareLaunchArgument('output_tensor_names', default_value='["output_tensor"]'),
        DeclareLaunchArgument('output_binding_names', default_value='["output0"]'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.25'),
        DeclareLaunchArgument('nms_threshold', default_value='0.45'),
        DeclareLaunchArgument('num_classes', default_value='80'),
        DeclareLaunchArgument('launch_rviz', default_value='False'),
        DeclareLaunchArgument('container_name', default_value='yolov8_foundationpose_container'),
        # Depth handling: set True if remote depth is 32FC1 (meters)
        DeclareLaunchArgument('depth_is_float', default_value='False'),

        # Optional tracking (enables selector + tracking together)
        DeclareLaunchArgument('enable_tracking', default_value='False'),
    ]

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
    launch_rviz = LaunchConfiguration('launch_rviz')
    container_name = LaunchConfiguration('container_name')

    remote_color_image_topic = LaunchConfiguration('remote_color_image_topic')
    remote_color_info_topic = LaunchConfiguration('remote_color_info_topic')
    remote_depth_aligned_topic = LaunchConfiguration('remote_depth_aligned_topic')
    depth_is_float = LaunchConfiguration('depth_is_float')
    hawk_expect_freq = LaunchConfiguration('hawk_expect_freq')
    input_images_drop_freq = LaunchConfiguration('input_images_drop_freq')
    enable_tracking = LaunchConfiguration('enable_tracking')

    # Drop nodes: choose based on depth encoding
    drop_node_uint16 = ComposableNode(
        name='drop_node',
        package='isaac_ros_nitros_topic_tools',
        plugin='nvidia::isaac_ros::nitros::NitrosCameraDropNode',
        parameters=[{
            'X': hawk_expect_freq,
            'Y': input_images_drop_freq,
            'mode': 'mono+depth',
            'depth_format_string': 'nitros_image_mono16'}],
        remappings=[('image_1', remote_color_image_topic),
                    ('camera_info_1', remote_color_info_topic),
                    ('depth_1', remote_depth_aligned_topic),
                    ('image_1_drop', 'rgb/image_rect_color'),
                    ('camera_info_1_drop', 'rgb/camera_info'),
                    ('depth_1_drop', 'depth_uint16')],
        condition=UnlessCondition(depth_is_float))

    drop_node_float = ComposableNode(
        name='drop_node',
        package='isaac_ros_nitros_topic_tools',
        plugin='nvidia::isaac_ros::nitros::NitrosCameraDropNode',
        parameters=[{
            'X': hawk_expect_freq,
            'Y': input_images_drop_freq,
            'mode': 'mono+depth',
            'depth_format_string': 'nitros_image_32FC1'}],
        remappings=[('image_1', remote_color_image_topic),
                    ('camera_info_1', remote_color_info_topic),
                    ('depth_1', remote_depth_aligned_topic),
                    ('image_1_drop', 'rgb/image_rect_color'),
                    ('camera_info_1_drop', 'rgb/camera_info'),
                    ('depth_1_drop', 'depth_image')],
        condition=IfCondition(depth_is_float))

    convert_metric_node = ComposableNode(
        package='isaac_ros_depth_image_proc',
        plugin='nvidia::isaac_ros::depth_image_proc::ConvertMetricNode',
        remappings=[('image_raw', 'depth_uint16'), ('image', 'depth_image')],
        condition=UnlessCondition(depth_is_float))

    # Encoder (letterbox 1280x720 -> 640x640)
    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    yolov8_encoder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]),
        launch_arguments={
            'input_image_width': str(REALSENSE_IMAGE_WIDTH),
            'input_image_height': str(REALSENSE_IMAGE_HEIGHT),
            'network_image_width': str(YOLOV8_MODEL_INPUT),
            'network_image_height': str(YOLOV8_MODEL_INPUT),
            'image_mean': '[0.0, 0.0, 0.0]',
            'image_stddev': '[1.0, 1.0, 1.0]',
            'attach_to_shared_component_container': 'True',
            'component_container_name': container_name,
            'dnn_image_encoder_namespace': 'yolov8_encoder',
            'image_input_topic': '/rgb/image_rect_color',
            'camera_info_input_topic': '/rgb/camera_info',
            'tensor_output_topic': '/tensor_pub',
            'keep_aspect_ratio': 'True',
        }.items())

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
            'verbose': False,
            'force_engine_update': False,
            'relaxed_dimension_check': True,
            'max_batch_size': 1,
        }],
        # Consume the encoder's NCHW planar tensor directly
        remappings=[('tensor_input', '/yolov8_encoder/planar_tensor')])

    yolov8_decoder_node = ComposableNode(
        name='yolov8_decoder',
        package='isaac_ros_yolov8',
        plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
        parameters=[{
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'nms_threshold': LaunchConfiguration('nms_threshold'),
            'num_classes': LaunchConfiguration('num_classes'),
            'tensor_name': 'output_tensor',
        }])

    detection2_d_to_mask_node = ComposableNode(
        name='detection2_d_to_mask',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
        parameters=[{
            'mask_width': int(REALSENSE_IMAGE_WIDTH / REALSENSE_TO_YOLO_RATIO),
            'mask_height': int(REALSENSE_IMAGE_HEIGHT / REALSENSE_TO_YOLO_RATIO),
        }],
        remappings=[('detection2_d_array', 'detections_output'),
                    ('segmentation', 'yolov8_segmentation_small')])

    resize_mask_node = ComposableNode(
        name='resize_mask_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': int(REALSENSE_IMAGE_WIDTH / REALSENSE_TO_YOLO_RATIO),
            'input_height': int(REALSENSE_IMAGE_HEIGHT / REALSENSE_TO_YOLO_RATIO),
            'output_width': REALSENSE_IMAGE_WIDTH,
            'output_height': REALSENSE_IMAGE_HEIGHT,
            'keep_aspect_ratio': False,
            'disable_padding': False,
            # Ensure mask encoding matches FoundationPose expectation
            'encoding_desired': 'mono8',
        }],
        remappings=[('image', 'yolov8_segmentation_small'),
                    ('camera_info', 'yolov8_encoder/resize/camera_info'),
                    ('resize/image', 'segmentation'),
                    ('resize/camera_info', 'camera_info_segmentation')])

    selector_node = ComposableNode(
        name='selector_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Selector',
        parameters=[{'reset_period': 10000}],
        remappings=[('depth_image', 'depth_image'),
                    ('image', 'rgb/image_rect_color'),
                    ('camera_info', 'rgb/camera_info'),
                    ('segmentation', 'segmentation')],
        condition=IfCondition(enable_tracking))

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
        }],
        remappings=[
            ('pose_estimation/depth_image', 'depth_image'),
            ('pose_estimation/image', 'rgb/image_rect_color'),
            ('pose_estimation/camera_info', 'rgb/camera_info'),
            ('pose_estimation/segmentation', 'segmentation'),
            ('pose_estimation/output', 'output')
        ])

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
        }],
        condition=IfCondition(enable_tracking))

    resize_left_viz = ComposableNode(
        name='resize_left_viz',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': REALSENSE_IMAGE_WIDTH,
            'input_height': REALSENSE_IMAGE_HEIGHT,
            'output_width': int(REALSENSE_IMAGE_WIDTH/ VISUALIZATION_DOWNSCALING_FACTOR),
            'output_height': int(REALSENSE_IMAGE_HEIGHT/ VISUALIZATION_DOWNSCALING_FACTOR),
            'keep_aspect_ratio': False,
            'encoding_desired': 'rgb8',
            'disable_padding': False,
        }],
        remappings=[('image', 'rgb/image_rect_color'),
                    ('camera_info', 'rgb/camera_info'),
                    ('resize/image', 'rgb/image_rect_color_viz'),
                    ('resize/camera_info', 'rgb/camera_info_viz')])

    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_config_path], condition=IfCondition(launch_rviz))

    container = ComposableNodeContainer(
        name=container_name,
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            # No RealSense driver here. It runs remotely.
            drop_node_uint16,
            drop_node_float,
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
        output='screen')

    return launch.LaunchDescription(launch_args + [container, yolov8_encoder_launch, rviz_node])
