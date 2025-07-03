#!/usr/bin/env python3
"""
感知流水线启动文件
支持DNN+TPU和CV+CPU两种模式
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory, get_package_prefix

def generate_launch_description():
    # 声明启动参数
    pipeline_mode_arg = DeclareLaunchArgument(
        'pipeline_mode',
        default_value='cv_cpu',
        description='Pipeline mode: dnn_tpu or cv_cpu'
    )
    
    use_simulation_arg = DeclareLaunchArgument(
        'use_simulation',
        default_value='true',
        description='Use simulation data instead of real camera'
    )
    
    target_fps_arg = DeclareLaunchArgument(
        'target_fps',
        default_value='30.0',
        description='Target FPS for the pipeline'
    )
    
    image_width_arg = DeclareLaunchArgument(
        'image_width',
        default_value='640',
        description='Processed image width'
    )
    
    image_height_arg = DeclareLaunchArgument(
        'image_height',
        default_value='480',
        description='Processed image height'
    )
    
    def launch_setup(context, *args, **kwargs):
        # 获取参数值
        pipeline_mode = LaunchConfiguration('pipeline_mode').perform(context)
        use_simulation = LaunchConfiguration('use_simulation').perform(context)
        target_fps = LaunchConfiguration('target_fps').perform(context)
        image_width = LaunchConfiguration('image_width').perform(context)
        image_height = LaunchConfiguration('image_height').perform(context)
        
        # 获取包的安装路径
        package_prefix = get_package_prefix('perception_pipeline')
        executable_path = os.path.join(package_prefix, 'bin')
        
        # 根据模式设置参数
        if pipeline_mode == 'dnn_tpu':
            detection_method = 'dnn'
            segmentation_method = 'dnn'
            depth_method = 'stereo_cnn'
            use_tpu = True
        else:
            detection_method = 'cv'
            segmentation_method = 'cv'
            depth_method = 'stereo_cv'
            use_tpu = False
        
        # 节点列表
        nodes = []
        
        # 1. 传感器输入节点
        sensor_input_node = Node(
            package='perception_pipeline',
            executable='sensor_input',
            name='sensor_input_node',
            parameters=[{
                'use_real_camera': use_simulation.lower() != 'true',
                'simulation_mode': use_simulation.lower() == 'true',
                'image_width': 1920,
                'image_height': 1080,
                'fps': float(target_fps),
            }],
            output='screen'
        )
        nodes.append(sensor_input_node)
        
        # 2. 图像预处理节点
        image_preprocessor_node = Node(
            package='perception_pipeline',
            executable='image_preprocessor',
            name='image_preprocessor_node',
            parameters=[{
                'target_width': int(image_width),
                'target_height': int(image_height),
                'enable_denoising': True,
                'enable_enhancement': True,
            }],
            output='screen'
        )
        nodes.append(image_preprocessor_node)
        
        # 3. 深度估计节点
        depth_estimator_node = Node(
            package='perception_pipeline',
            executable='depth_estimator',
            name='depth_estimator_node',
            parameters=[{
                'estimation_method': depth_method,
                'stereo_algorithm': 'SGBM',
                'min_disparity': 0,
                'num_disparities': 64,
                'block_size': 11,
            }],
            output='screen'
        )
        nodes.append(depth_estimator_node)
        
        # 4. 目标检测节点
        object_detector_node = Node(
            package='perception_pipeline',
            executable='object_detector',
            name='object_detector_node',
            parameters=[{
                'detection_method': detection_method,
                'dnn_model': 'yolov4',
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4,
                'use_tpu': use_tpu,
            }],
            output='screen'
        )
        nodes.append(object_detector_node)
        
        # 5. 语义分割节点
        semantic_segmentor_node = Node(
            package='perception_pipeline',
            executable='semantic_segmentor',
            name='semantic_segmentor_node',
            parameters=[{
                'segmentation_method': segmentation_method,
                'dnn_model': 'fast_scnn',
                'use_tpu': use_tpu,
            }],
            output='screen'
        )
        nodes.append(semantic_segmentor_node)
        
        # 6. 感知管理节点
        perception_manager_node = Node(
            package='perception_pipeline',
            executable='perception_manager',
            name='perception_manager_node',
            parameters=[{
                'pipeline_mode': pipeline_mode,
                'sync_timeout': 0.1,
            }],
            output='screen'
        )
        nodes.append(perception_manager_node)
        
        # 7. 时间分析节点
        timing_analyzer_node = Node(
            package='perception_pipeline',
            executable='timing_analyzer',
            name='timing_analyzer_node',
            parameters=[{
                'analysis_window': 100,
                'report_interval': 10.0,
                'target_fps': float(target_fps),
            }],
            output='screen'
        )
        nodes.append(timing_analyzer_node)
        
        return nodes
    
    return LaunchDescription([
        pipeline_mode_arg,
        use_simulation_arg,
        target_fps_arg,
        image_width_arg,
        image_height_arg,
        OpaqueFunction(function=launch_setup)
    ]) 