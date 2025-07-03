#!/usr/bin/env python3
"""
感知管理节点 - 整合所有感知流水线的结果
协调深度估计、目标检测、语义分割的输出
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from perception_interfaces.msg import (
    DepthImage, DetectedObjects, SegmentedImage, 
    PerceptionResult, TimingInfo
)
import time
import threading

class PerceptionManagerNode(Node):
    def __init__(self):
        super().__init__('perception_manager_node')
        
        # 参数声明
        self.declare_parameter('pipeline_mode', 'cv_cpu')  # 'dnn_tpu' or 'cv_cpu'
        self.declare_parameter('sync_timeout', 0.1)  # 同步超时时间（秒）
        
        # 获取参数
        self.pipeline_mode = self.get_parameter('pipeline_mode').value
        self.sync_timeout = self.get_parameter('sync_timeout').value
        
        # 订阅器
        self.left_image_sub = self.create_subscription(
            Image, '/stereo/left/image_processed', self.left_image_callback, 10)
        self.right_image_sub = self.create_subscription(
            Image, '/stereo/right/image_processed', self.right_image_callback, 10)
        self.depth_sub = self.create_subscription(
            DepthImage, '/perception/depth_estimation', self.depth_callback, 10)
        self.detection_sub = self.create_subscription(
            DetectedObjects, '/perception/object_detection', self.detection_callback, 10)
        self.segmentation_sub = self.create_subscription(
            SegmentedImage, '/perception/semantic_segmentation', self.segmentation_callback, 10)
        
        # 发布器
        self.perception_pub = self.create_publisher(PerceptionResult, '/perception/result', 10)
        self.timing_pub = self.create_publisher(TimingInfo, '/perception/timing', 10)
        
        # 数据存储
        self.latest_data = {
            'left_image': None,
            'right_image': None,
            'depth': None,
            'detection': None,
            'segmentation': None,
            'timestamps': {}
        }
        
        # 线程锁
        self.data_lock = threading.Lock()
        
        # 时间戳记录
        self.pipeline_start_time = None
        self.stage_times = {}
        
        self.get_logger().info(f"感知管理节点已启动 - 模式: {self.pipeline_mode}")
    
    def left_image_callback(self, msg):
        """左图像回调"""
        with self.data_lock:
            self.latest_data['left_image'] = msg
            self.latest_data['timestamps']['left_image'] = time.time()
            
            # 标记流水线开始
            if self.pipeline_start_time is None:
                self.pipeline_start_time = time.time()
                self.stage_times['sensor_input'] = time.time()
    
    def right_image_callback(self, msg):
        """右图像回调"""
        with self.data_lock:
            self.latest_data['right_image'] = msg
            self.latest_data['timestamps']['right_image'] = time.time()
    
    def depth_callback(self, msg):
        """深度估计回调"""
        with self.data_lock:
            self.latest_data['depth'] = msg
            self.latest_data['timestamps']['depth'] = time.time()
            self.stage_times['depth_estimation'] = time.time()
            self.try_publish_result()
    
    def detection_callback(self, msg):
        """目标检测回调"""
        with self.data_lock:
            self.latest_data['detection'] = msg
            self.latest_data['timestamps']['detection'] = time.time()
            self.stage_times['object_detection'] = time.time()
            self.try_publish_result()
    
    def segmentation_callback(self, msg):
        """语义分割回调"""
        with self.data_lock:
            self.latest_data['segmentation'] = msg
            self.latest_data['timestamps']['segmentation'] = time.time()
            self.stage_times['semantic_segmentation'] = time.time()
            self.try_publish_result()
    
    def try_publish_result(self):
        """尝试发布整合结果"""
        # 检查是否所有数据都已到达
        required_data = ['left_image', 'right_image', 'depth', 'detection', 'segmentation']
        
        if not all(self.latest_data[key] is not None for key in required_data):
            return
        
        # 检查数据时间戳是否在合理范围内
        current_time = time.time()
        timestamps = self.latest_data['timestamps']
        
        # 检查所有数据是否在同步窗口内
        if not all(current_time - timestamps.get(key, 0) < self.sync_timeout for key in required_data):
            self.get_logger().debug("数据同步超时，跳过本次发布")
            return
        
        # 创建整合结果
        result_msg = PerceptionResult()
        result_msg.header.stamp = self.get_clock().now().to_msg()
        result_msg.header.frame_id = "perception_result"
        
        # 填充数据
        result_msg.left_image = self.latest_data['left_image']
        result_msg.right_image = self.latest_data['right_image']
        result_msg.depth_result = self.latest_data['depth']
        result_msg.detection_result = self.latest_data['detection']
        result_msg.segmentation_result = self.latest_data['segmentation']
        result_msg.pipeline_mode = self.pipeline_mode
        
        # 计算时间信息
        timing_info = self.calculate_timing_info()
        result_msg.timing_info = timing_info
        
        # 发布结果
        self.perception_pub.publish(result_msg)
        self.timing_pub.publish(timing_info)
        
        # 重置数据（防止重复发布）
        self.reset_data()
        
        # 性能日志
        total_time = timing_info.total_pipeline_time
        self.get_logger().info(f"感知流水线完成 - 总时间: {total_time:.2f}ms")
    
    def calculate_timing_info(self):
        """计算时间信息"""
        timing_msg = TimingInfo()
        timing_msg.header.stamp = self.get_clock().now().to_msg()
        timing_msg.header.frame_id = "timing_info"
        timing_msg.hardware_config = self.pipeline_mode
        
        current_time = time.time()
        
        if self.pipeline_start_time is not None:
            # 总流水线时间
            timing_msg.total_pipeline_time = (current_time - self.pipeline_start_time) * 1000
            
            # 各阶段时间（简化计算）
            timing_msg.sensor_input_time = 5.0  # 假设传感器输入时间
            timing_msg.preprocessing_time = 10.0  # 假设预处理时间
            
            # 从stage_times计算实际时间
            if 'depth_estimation' in self.stage_times and self.pipeline_start_time:
                timing_msg.depth_estimation_time = (
                    self.stage_times['depth_estimation'] - self.pipeline_start_time) * 1000
            else:
                timing_msg.depth_estimation_time = 0.0
                
            if 'object_detection' in self.stage_times and self.pipeline_start_time:
                timing_msg.object_detection_time = (
                    self.stage_times['object_detection'] - self.pipeline_start_time) * 1000
            else:
                timing_msg.object_detection_time = 0.0
                
            if 'semantic_segmentation' in self.stage_times and self.pipeline_start_time:
                timing_msg.semantic_segmentation_time = (
                    self.stage_times['semantic_segmentation'] - self.pipeline_start_time) * 1000
            else:
                timing_msg.semantic_segmentation_time = 0.0
            
            timing_msg.postprocessing_time = 2.0  # 假设后处理时间
        else:
            # 如果没有开始时间，设置默认值
            timing_msg.total_pipeline_time = 0.0
            timing_msg.sensor_input_time = 0.0
            timing_msg.preprocessing_time = 0.0
            timing_msg.depth_estimation_time = 0.0
            timing_msg.object_detection_time = 0.0
            timing_msg.semantic_segmentation_time = 0.0
            timing_msg.postprocessing_time = 0.0
        
        return timing_msg
    
    def reset_data(self):
        """重置数据"""
        self.latest_data = {
            'left_image': None,
            'right_image': None,
            'depth': None,
            'detection': None,
            'segmentation': None,
            'timestamps': {}
        }
        self.pipeline_start_time = None
        self.stage_times = {}

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionManagerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 