#!/usr/bin/env python3
# SPDX-FileCopyrightText: community-created port of the isaac_ros_3 trocar example.
# Isaac ROS 5.0 (ROS 2 Lyrical) port of
#   isaac_ros_4/launch/yolov8_foundationpose_isaac_sim.launch.py.
#
# Pipeline: Isaac Sim RGB + aligned depth + camera_info ->
#   CameraDropNode -> DNN image encoder (letterbox) -> TensorRT (custom YOLOv8) -> YoloV8 decoder
#   -> Detection2DArrayFilter -> Detection2DToMask -> unletterbox crop -> resize to camera res
#   -> FoundationPose (+ separate refine/score TensorRT nodes), optional Selector + tracking.
#
# 5.0 deltas vs the 4.x launch (verified against the installed release-5.0 packages):
#   * isaac_ros_nitros_topic_tools/NitrosCameraDropNode -> isaac_ros_topic_tools/CameraDropNode.
#     X/Y means DROP X of every Y (evenly spread). `depth_format_string` is gone (plain
#     sensor_msgs/Image). Args: camera_drop_drop_count (X) / camera_drop_window (Y).
#   * FoundationPose no longer owns TensorRT: refine/score (and tracking refine) run in separate
#     TensorRTNode components wired via refine|score|tracking_refine/tensor_{pub,sub}; engine
#     paths and batch sizes (42/252/1) go to those nodes, as in isaac_ros_foundationpose_core /
#     _tracking_core.launch.py. `texture_path` was removed: the OBJ's MTL/textures are loaded
#     through mesh_file_path (keep .obj, .mtl and textures together).
#   * dnn_image_encoder.launch.py now defaults its tensor name to `output_tensor`; we pass
#     tensor_name=input_tensor to keep the YOLO TensorRT input binding mapping.
#   * ResizeNode/CropNode are CV-CUDA nodes: ResizeNode has no input_width/input_height/
#     encoding_desired params (input dims come from the message). Letterbox geometry is now
#     derived from image_width/image_height launch args (ResizeNode pads centered).
#   * The 4.x crop node republished onto the encoder's resize/camera_info topic (a loop);
#     it now publishes its own camera_info topic.
#   * With enable_tracking, FoundationPose/tracking use the Selector's default
#     pose_estimation/* and tracking/* topics (the 4.x launch remapped FoundationPose straight
#     to the camera topics, bypassing the Selector). Outputs: `output` (pose estimation) and
#     `tracking/output` (tracking).

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

YOLOV8_MODEL_INPUT = 640
VISUALIZATION_DOWNSCALING_FACTOR = 10

TROCAR_MESH_PATH = '/workspaces/isaac_ros-dev/src/isaac_sim_custom_examples/trocar_short.obj'
REFINE_ENGINE_PATH = '/reference_assets/models/foundationpose/refine_trt_engine.plan'
SCORE_ENGINE_PATH = '/reference_assets/models/foundationpose/score_trt_engine.plan'


def letterbox_content_dims(width, height, size):
    """Replicate isaac_ros_image_proc ResizeNode::CalculateOutputDims (5.0, keep_aspect_ratio)."""
    height_factor = size / height
    width_factor = size / width
    out_w, out_h = size, size
    if height_factor < width_factor:
        out_w = int(width * height_factor)
        out_w += out_w % 2
    elif width_factor < height_factor:
        out_h = int(height * width_factor)
        out_h += out_h % 2
    return out_w, out_h


def launch_setup(context, *args, **kwargs):
    image_width = int(LaunchConfiguration('image_width').perform(context))
    image_height = int(LaunchConfiguration('image_height').perform(context))
    # Content area inside the 640x640 letterbox; ResizeNode pads it centered.
    content_w, content_h = letterbox_content_dims(image_width, image_height, YOLOV8_MODEL_INPUT)
    pad_left = (YOLOV8_MODEL_INPUT - content_w) // 2
    pad_top = (YOLOV8_MODEL_INPUT - content_h) // 2

    container_name = LaunchConfiguration('container_name')
    depth_is_float = LaunchConfiguration('depth_is_float')
    enable_tracking = LaunchConfiguration('enable_tracking')
    mesh_file_path = LaunchConfiguration('mesh_file_path')
    refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
    refine_model_file_path = LaunchConfiguration('refine_model_file_path')
    score_engine_file_path = LaunchConfiguration('score_engine_file_path')
    score_model_file_path = LaunchConfiguration('score_model_file_path')

    # Drop X (camera_drop_drop_count) of every Y (camera_drop_window) synchronized RGB+info+depth
    # sets. mono+depth mode: image_1 + camera_info_1 + depth_1 (exact-time sync).
    def drop_node(depth_out, condition):
        return ComposableNode(
            name='drop_node',
            package='isaac_ros_topic_tools',
            plugin='nvidia::isaac_ros::topic_tools::CameraDropNode',
            parameters=[{
                'input_qos': LaunchConfiguration('camera_drop_input_qos'),
                'output_qos': 'DEFAULT',
                'X': LaunchConfiguration('camera_drop_drop_count'),
                'Y': LaunchConfiguration('camera_drop_window'),
                'mode': 'mono+depth',
                'sync_queue_size': 100,
            }],
            remappings=[('image_1', LaunchConfiguration('remote_color_image_topic')),
                        ('camera_info_1', LaunchConfiguration('remote_color_info_topic')),
                        ('depth_1', LaunchConfiguration('remote_depth_aligned_topic')),
                        ('image_1_drop', 'rgb/image_rect_color'),
                        ('camera_info_1_drop', 'rgb/camera_info'),
                        ('depth_1_drop', depth_out)],
            condition=condition)

    drop_node_uint16 = drop_node('depth_uint16', UnlessCondition(depth_is_float))
    drop_node_float = drop_node('depth_image', IfCondition(depth_is_float))

    # 16UC1 mm -> 32FC1 m (only when depth_is_float is False)
    convert_metric_node = ComposableNode(
        name='convert_metric_node',
        package='isaac_ros_depth_image_proc',
        plugin='nvidia::isaac_ros::depth_image_proc::ConvertMetricNode',
        remappings=[('image_raw', 'depth_uint16'), ('image', 'depth_image')],
        condition=UnlessCondition(depth_is_float))

    # Letterbox camera image -> 640x640 planar NCHW tensor on /tensor_pub.
    # mean=0 / stddev=1 keep YOLOv8's 0..1 scaling (NOT the encoder's 0.5 default).
    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    yolov8_encoder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]),
        launch_arguments={
            'input_image_width': str(image_width),
            'input_image_height': str(image_height),
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
            'tensor_name': 'input_tensor',
        }.items())

    # Custom YOLOv8 engine: /tensor_pub -> /tensor_sub (default topics).
    tensor_rt_node = ComposableNode(
        name='yolov8_tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'model_file_path': LaunchConfiguration('yolov8_model_file_path'),
            'engine_file_path': LaunchConfiguration('yolov8_engine_file_path'),
            'input_tensor_names': LaunchConfiguration('input_tensor_names'),
            'input_binding_names': LaunchConfiguration('input_binding_names'),
            'output_tensor_names': LaunchConfiguration('output_tensor_names'),
            'output_binding_names': LaunchConfiguration('output_binding_names'),
            'verbose': False,
            'force_engine_update': LaunchConfiguration('force_engine_update'),
        }])

    # tensor_sub -> detections_output (boxes in 640x640 letterboxed network coordinates)
    yolov8_decoder_node = ComposableNode(
        name='yolov8_decoder',
        package='isaac_ros_yolov8',
        plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
        parameters=[{
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'nms_threshold': LaunchConfiguration('nms_threshold'),
            'num_classes': LaunchConfiguration('num_classes'),
            'tensor_name': 'output_tensor',  # must equal output_tensor_names[0]
        }])

    # Detection2DArray -> single Detection2D (highest confidence; class-agnostic).
    # Publishes nothing when there is no detection.
    detection2_d_array_filter_node = ComposableNode(
        name='detection2_d_array_filter',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Detection2DArrayFilter',
        parameters=[{'desired_class_id': ''}],
        remappings=[('detection2_d_array', 'detections_output'),
                    ('detection2_d', 'detection2_d')])

    # Binary mono8 mask at YOLO network resolution (640x640 letterboxed)
    detection2_d_to_mask_node = ComposableNode(
        name='detection2_d_to_mask',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
        parameters=[{
            'mask_width': YOLOV8_MODEL_INPUT,
            'mask_height': YOLOV8_MODEL_INPUT,
        }],
        remappings=[('detection2_d', 'detection2_d'),
                    ('segmentation', 'yolov8_segmentation_small')])

    # Remove the letterbox padding (e.g. 1280x720: 640x640 -> 640x360 at y=140).
    unletterbox_mask_node = ComposableNode(
        name='unletterbox_mask',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::CropNode',
        parameters=[{
            'input_width': YOLOV8_MODEL_INPUT,
            'input_height': YOLOV8_MODEL_INPUT,
            'crop_width': content_w,
            'crop_height': content_h,
            'roi_top_left_x': pad_left,
            'roi_top_left_y': pad_top,
            'crop_mode': 'BBOX',
        }],
        remappings=[('image', 'yolov8_segmentation_small'),
                    ('camera_info', 'yolov8_encoder/resize/camera_info'),
                    ('crop/image', 'yolov8_segmentation_small_unpadded'),
                    ('crop/camera_info', 'yolov8_segmentation_small_unpadded/camera_info')])

    # Upscale the mask to full camera resolution so RGB/depth/mask match for FoundationPose.
    resize_mask_node = ComposableNode(
        name='resize_mask_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': image_width,
            'output_height': image_height,
            'keep_aspect_ratio': False,
            'disable_padding': False,
        }],
        remappings=[('image', 'yolov8_segmentation_small_unpadded'),
                    ('camera_info', 'yolov8_segmentation_small_unpadded/camera_info'),
                    ('resize/image', 'segmentation'),
                    ('resize/camera_info', 'camera_info_segmentation')])

    def trt_node(name, prefix, engine, model, outputs, bindings, batch, condition=None):
        return ComposableNode(
            name=name,
            package='isaac_ros_tensor_rt',
            plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
            parameters=[{
                'model_file_path': model,
                'engine_file_path': engine,
                'input_tensor_names': ['input_tensor1', 'input_tensor2'],
                'input_binding_names': ['input1', 'input2'],
                'output_tensor_names': outputs,
                'output_binding_names': bindings,
                'force_engine_update': False,
                'verbose': False,
                'max_batch_size': batch,
            }],
            remappings=[('tensor_pub', prefix + '/tensor_pub'),
                        ('tensor_sub', prefix + '/tensor_sub')],
            condition=condition)

    refine_trt_node = trt_node(
        'refine_trt', 'refine', refine_engine_file_path, refine_model_file_path,
        ['output_tensor1', 'output_tensor2'], ['output1', 'output2'], 42)
    score_trt_node = trt_node(
        'score_trt', 'score', score_engine_file_path, score_model_file_path,
        ['output_tensor'], ['output1'], 252)
    tracking_refine_trt_node = trt_node(
        'tracking_refine_trt', 'tracking_refine', refine_engine_file_path,
        refine_model_file_path, ['output_tensor1', 'output_tensor2'],
        ['output1', 'output2'], 1, IfCondition(enable_tracking))

    fp_params = {
        'mesh_file_path': mesh_file_path,
        'symmetry_axes': LaunchConfiguration('symmetry_axes'),
        'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
        'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
    }

    # Without tracking: FoundationPose consumes the dropped camera set + mask directly.
    foundationpose_node_direct = ComposableNode(
        name='foundationpose_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
        parameters=[fp_params],
        remappings=[
            ('pose_estimation/depth_image', 'depth_image'),
            ('pose_estimation/image', 'rgb/image_rect_color'),
            ('pose_estimation/camera_info', 'rgb/camera_info'),
            ('pose_estimation/segmentation', 'segmentation'),
            ('pose_estimation/output', 'output'),
        ],
        condition=UnlessCondition(enable_tracking))

    # With tracking: Selector routes frames to pose_estimation/* or tracking/* (upstream
    # isaac_ros_foundationpose_tracking_core.launch.py wiring).
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

    foundationpose_node_via_selector = ComposableNode(
        name='foundationpose_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
        parameters=[fp_params],
        remappings=[('pose_estimation/output', 'output')],
        condition=IfCondition(enable_tracking))

    foundationpose_tracking_node = ComposableNode(
        name='foundationpose_tracking_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode',
        parameters=[{
            'mesh_file_path': mesh_file_path,
            'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
        }],
        condition=IfCondition(enable_tracking))

    resize_left_viz = ComposableNode(
        name='resize_left_viz',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': int(image_width / VISUALIZATION_DOWNSCALING_FACTOR),
            'output_height': int(image_height / VISUALIZATION_DOWNSCALING_FACTOR),
            'keep_aspect_ratio': False,
            'disable_padding': False,
        }],
        remappings=[('image', 'rgb/image_rect_color'),
                    ('camera_info', 'rgb/camera_info'),
                    ('resize/image', 'rgb/image_rect_color_viz'),
                    ('resize/camera_info', 'rgb/camera_info_viz')])

    rviz_config_path = os.path.join(
        get_package_share_directory('isaac_ros_foundationpose'),
        'rviz', 'foundationpose_realsense.rviz')
    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_config_path],
        condition=IfCondition(LaunchConfiguration('launch_rviz')))

    container = ComposableNodeContainer(
        name=container_name,
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            drop_node_uint16,
            drop_node_float,
            convert_metric_node,
            tensor_rt_node,
            yolov8_decoder_node,
            detection2_d_array_filter_node,
            detection2_d_to_mask_node,
            unletterbox_mask_node,
            resize_mask_node,
            refine_trt_node,
            score_trt_node,
            tracking_refine_trt_node,
            foundationpose_node_direct,
            selector_node,
            foundationpose_node_via_selector,
            foundationpose_tracking_node,
            resize_left_viz,
        ],
        output='screen')

    return [container, yolov8_encoder_launch, rviz_node]


def generate_launch_description():
    launch_args = [
        # Isaac Sim sim-RealSense topics (override to match your scene's OmniGraph)
        DeclareLaunchArgument('remote_color_image_topic', default_value='/image_rect'),
        DeclareLaunchArgument('remote_color_info_topic', default_value='/camera_info'),
        DeclareLaunchArgument('remote_depth_aligned_topic', default_value='/depth'),
        # Must match the published camera stream; letterbox/crop/resize geometry derives from it.
        DeclareLaunchArgument('image_width', default_value='1280'),
        DeclareLaunchArgument('image_height', default_value='720'),

        # CameraDropNode: drop camera_drop_drop_count of every camera_drop_window synced sets.
        # 28 of 30 keeps 2 of 30 (2 Hz at 30 Hz input).
        DeclareLaunchArgument('camera_drop_drop_count', default_value='28',
                              description='CameraDropNode X: frames DROPPED per window'),
        DeclareLaunchArgument('camera_drop_window', default_value='30',
                              description='CameraDropNode Y: window size in frames'),
        # SENSOR_DATA (best effort) accepts reliable and best-effort publishers; use DEFAULT
        # (reliable) to avoid losing large frames when the publisher is reliable.
        DeclareLaunchArgument('camera_drop_input_qos', default_value='SENSOR_DATA'),

        # FoundationPose mesh: OBJ with its MTL and textures in the same directory.
        DeclareLaunchArgument('mesh_file_path', default_value=TROCAR_MESH_PATH),
        DeclareLaunchArgument('refine_engine_file_path', default_value=REFINE_ENGINE_PATH),
        DeclareLaunchArgument('score_engine_file_path', default_value=SCORE_ENGINE_PATH),
        # Only used by TensorRT if the engine file is missing (empty: never rebuild).
        DeclareLaunchArgument('refine_model_file_path', default_value=''),
        DeclareLaunchArgument('score_model_file_path', default_value=''),

        # Custom YOLOv8 (trocar model: 1 class, images[1,3,640,640] -> output0[1,5,8400]).
        DeclareLaunchArgument('yolov8_engine_file_path', default_value=''),
        DeclareLaunchArgument('yolov8_model_file_path', default_value=''),
        DeclareLaunchArgument('input_tensor_names', default_value='["input_tensor"]'),
        DeclareLaunchArgument('input_binding_names', default_value='["images"]'),
        DeclareLaunchArgument('output_tensor_names', default_value='["output_tensor"]'),
        DeclareLaunchArgument('output_binding_names', default_value='["output0"]'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.25'),
        DeclareLaunchArgument('nms_threshold', default_value='0.45'),
        DeclareLaunchArgument('num_classes', default_value='1'),
        # Keep False: True rebuilds and overwrites the YOLOv8 engine from yolov8_model_file_path.
        DeclareLaunchArgument('force_engine_update', default_value='False'),

        DeclareLaunchArgument('launch_rviz', default_value='False'),
        DeclareLaunchArgument('container_name', default_value='yolov8_foundationpose_container'),
        # Depth handling: True if depth is 32FC1 meters (Isaac Sim). False for 16UC1 mm.
        DeclareLaunchArgument('depth_is_float', default_value='True'),
        # Optional tracking (Selector + tracking node + tracking refine TensorRT)
        DeclareLaunchArgument('enable_tracking', default_value='False'),
        # FoundationPose orientation symmetry (list-like string), e.g. "['x_full']"
        DeclareLaunchArgument('symmetry_axes', default_value="['x_full']"),
    ]

    return launch.LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
