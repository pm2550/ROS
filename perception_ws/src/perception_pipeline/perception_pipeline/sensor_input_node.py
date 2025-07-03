#!/usr/bin/env python3
"""
传感器输入节点 - 负责双目相机数据采集
支持模拟数据和真实相机数据
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
import os

class SensorInputNode(Node):
    def __init__(self):
        super().__init__('sensor_input_node')
        
        # 参数声明
        self.declare_parameter('use_real_camera', False)
        self.declare_parameter('camera_device_left', 0)
        self.declare_parameter('camera_device_right', 1)
        self.declare_parameter('image_width', 1920)
        self.declare_parameter('image_height', 1080)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('simulation_mode', True)
        
        # 获取参数
        self.use_real_camera = self.get_parameter('use_real_camera').value
        self.camera_device_left = self.get_parameter('camera_device_left').value
        self.camera_device_right = self.get_parameter('camera_device_right').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.fps = self.get_parameter('fps').value
        self.simulation_mode = self.get_parameter('simulation_mode').value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 发布器
        self.left_image_pub = self.create_publisher(Image, '/stereo/left/image_raw', 10)
        self.right_image_pub = self.create_publisher(Image, '/stereo/right/image_raw', 10)
        self.left_camera_info_pub = self.create_publisher(CameraInfo, '/stereo/left/camera_info', 10)
        self.right_camera_info_pub = self.create_publisher(CameraInfo, '/stereo/right/camera_info', 10)
        
        # 相机设置
        self.setup_cameras()
        
        # 定时器
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.capture_and_publish)
        
        self.get_logger().info(f"传感器输入节点已启动 - 模式: {'真实相机' if self.use_real_camera else '模拟数据'}")
        
    def setup_cameras(self):
        """设置相机"""
        if self.use_real_camera:
            try:
                self.cap_left = cv2.VideoCapture(self.camera_device_left)
                self.cap_right = cv2.VideoCapture(self.camera_device_right)
                
                # 设置分辨率
                self.cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
                self.cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
                self.cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
                self.cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
                
                # 设置FPS
                self.cap_left.set(cv2.CAP_PROP_FPS, self.fps)
                self.cap_right.set(cv2.CAP_PROP_FPS, self.fps)
                
                self.get_logger().info("真实相机初始化成功")
            except Exception as e:
                self.get_logger().error(f"相机初始化失败: {e}")
                self.use_real_camera = False
                self.simulation_mode = True
        
        # 相机信息设置
        self.setup_camera_info()
        
    def setup_camera_info(self):
        """设置相机标定信息"""
        self.camera_info_left = CameraInfo()
        self.camera_info_right = CameraInfo()
        
        # 模拟的相机内参（需要根据实际相机标定结果调整）
        fx = fy = 800.0  # 焦距
        cx = self.image_width / 2.0  # 主点x
        cy = self.image_height / 2.0  # 主点y
        baseline = 0.12  # 基线距离（米）
        
        for camera_info in [self.camera_info_left, self.camera_info_right]:
            camera_info.width = self.image_width
            camera_info.height = self.image_height
            camera_info.distortion_model = "plumb_bob"
            camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # 畸变参数
            camera_info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]  # 内参矩阵
            camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # 旋转矩阵
            camera_info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]  # 投影矩阵
        
        # 右相机的投影矩阵包含基线信息
        self.camera_info_right.p[3] = -fx * baseline
        
    def generate_simulation_data(self):
        """生成模拟双目数据"""
        # 创建带有不同视差的测试图像
        timestamp = time.time()
        
        # 左图像：基础图像
        left_img = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        # 添加一些测试对象
        # 矩形
        cv2.rectangle(left_img, (100, 100), (300, 300), (255, 0, 0), -1)
        # 圆形
        cv2.circle(left_img, (600, 200), 50, (0, 255, 0), -1)
        # 文字
        cv2.putText(left_img, f"Frame: {int(timestamp)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 右图像：添加视差
        right_img = left_img.copy()
        disparity = 20  # 像素视差
        
        # 简单的视差模拟 - 将对象向左移动
        M = np.float32([[1, 0, -disparity], [0, 1, 0]])
        right_img = cv2.warpAffine(right_img, M, (self.image_width, self.image_height))
        
        return left_img, right_img
        
    def capture_and_publish(self):
        """捕获并发布图像"""
        start_time = time.time()
        
        if self.use_real_camera and hasattr(self, 'cap_left') and hasattr(self, 'cap_right'):
            # 真实相机数据
            ret_left, left_frame = self.cap_left.read()
            ret_right, right_frame = self.cap_right.read()
            
            if not (ret_left and ret_right):
                self.get_logger().warning("无法读取相机数据，切换到模拟模式")
                self.use_real_camera = False
                self.simulation_mode = True
                return
        else:
            # 模拟数据
            left_frame, right_frame = self.generate_simulation_data()
        
        # 创建ROS消息
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "stereo_camera"
        
        # 转换为ROS Image消息
        try:
            left_msg = self.bridge.cv2_to_imgmsg(left_frame, "bgr8")
            right_msg = self.bridge.cv2_to_imgmsg(right_frame, "bgr8")
            
            left_msg.header = header
            right_msg.header = header
            
            # 相机信息
            self.camera_info_left.header = header
            self.camera_info_right.header = header
            
            # 发布
            self.left_image_pub.publish(left_msg)
            self.right_image_pub.publish(right_msg)
            self.left_camera_info_pub.publish(self.camera_info_left)
            self.right_camera_info_pub.publish(self.camera_info_right)
            
            # 性能日志
            processing_time = (time.time() - start_time) * 1000
            if processing_time > 5.0:  # 只在处理时间超过5ms时记录
                self.get_logger().debug(f"传感器输入处理时间: {processing_time:.2f}ms")
                
        except Exception as e:
            self.get_logger().error(f"图像转换错误: {e}")
    
    def destroy_node(self):
        """清理资源"""
        if hasattr(self, 'cap_left'):
            self.cap_left.release()
        if hasattr(self, 'cap_right'):
            self.cap_right.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SensorInputNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 