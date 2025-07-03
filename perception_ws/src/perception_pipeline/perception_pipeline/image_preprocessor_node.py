#!/usr/bin/env python3
"""
图像预处理节点 - 负责图像缩放、去噪、增强等预处理操作
针对RPI5 CPU优化
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import cv2
from cv_bridge import CvBridge
import numpy as np
import time

class ImagePreprocessorNode(Node):
    def __init__(self):
        super().__init__('image_preprocessor_node')
        
        # 参数声明
        self.declare_parameter('target_width', 640)
        self.declare_parameter('target_height', 480)
        self.declare_parameter('enable_denoising', True)
        self.declare_parameter('enable_enhancement', True)
        self.declare_parameter('gaussian_blur_kernel', 3)
        self.declare_parameter('clahe_clip_limit', 2.0)
        
        # 获取参数
        self.target_width = self.get_parameter('target_width').value
        self.target_height = self.get_parameter('target_height').value
        self.enable_denoising = self.get_parameter('enable_denoising').value
        self.enable_enhancement = self.get_parameter('enable_enhancement').value
        self.gaussian_blur_kernel = self.get_parameter('gaussian_blur_kernel').value
        self.clahe_clip_limit = self.get_parameter('clahe_clip_limit').value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 订阅器
        self.left_image_sub = self.create_subscription(
            Image, '/stereo/left/image_raw', self.left_image_callback, 10)
        self.right_image_sub = self.create_subscription(
            Image, '/stereo/right/image_raw', self.right_image_callback, 10)
        
        # 发布器
        self.left_processed_pub = self.create_publisher(Image, '/stereo/left/image_processed', 10)
        self.right_processed_pub = self.create_publisher(Image, '/stereo/right/image_processed', 10)
        
        # CLAHE对象（对比度限制自适应直方图均衡化）
        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8,8))
        
        # 性能统计
        self.processing_times = []
        self.frame_count = 0
        
        self.get_logger().info(f"图像预处理节点已启动 - 目标分辨率: {self.target_width}x{self.target_height}")
        
    def preprocess_image(self, cv_image):
        """预处理单张图像"""
        start_time = time.time()
        
        # 1. 图像缩放
        resized = cv2.resize(cv_image, (self.target_width, self.target_height), 
                           interpolation=cv2.INTER_LINEAR)
        
        # 2. 转换为LAB色彩空间用于处理
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 3. 去噪（如果启用）
        if self.enable_denoising:
            # 使用双边滤波保持边缘
            l = cv2.bilateralFilter(l, 9, 75, 75)
        
        # 4. 图像增强（如果启用）
        if self.enable_enhancement:
            # 对亮度通道应用CLAHE
            l = self.clahe.apply(l)
        
        # 5. 合并通道并转换回BGR
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 6. 轻微的锐化
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced_bgr, -1, kernel)
        
        # 7. 混合原图和锐化图像
        result = cv2.addWeighted(enhanced_bgr, 0.8, sharpened, 0.2, 0)
        
        # 性能统计
        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)
        
        # 每100帧统计一次性能
        if len(self.processing_times) >= 100:
            avg_time = np.mean(self.processing_times)
            self.get_logger().info(f"图像预处理平均时间: {avg_time:.2f}ms")
            self.processing_times = []
        
        return result
    
    def left_image_callback(self, msg):
        """左图像回调"""
        try:
            # 转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 预处理
            processed_image = self.preprocess_image(cv_image)
            
            # 转换回ROS消息
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
            processed_msg.header = msg.header
            
            # 发布
            self.left_processed_pub.publish(processed_msg)
            
        except Exception as e:
            self.get_logger().error(f"左图像预处理错误: {e}")
    
    def right_image_callback(self, msg):
        """右图像回调"""
        try:
            # 转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 预处理
            processed_image = self.preprocess_image(cv_image)
            
            # 转换回ROS消息
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
            processed_msg.header = msg.header
            
            # 发布
            self.right_processed_pub.publish(processed_msg)
            
        except Exception as e:
            self.get_logger().error(f"右图像预处理错误: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImagePreprocessorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 