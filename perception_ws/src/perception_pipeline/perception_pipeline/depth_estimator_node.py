#!/usr/bin/env python3
"""
深度估计节点 - 支持经典CV立体匹配和基于CNN的深度估计
包含StereoBM、StereoSGBM算法和深度学习模型支持
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from perception_interfaces.msg import DepthImage
from std_msgs.msg import Header
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
import threading

# 尝试导入深度学习相关库
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch未安装，将仅使用经典CV方法")

# 导入TPU推理类
try:
    from .tpu_inference import TPUDepthEstimator
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    print("TPU推理模块不可用")

class DepthEstimatorNode(Node):
    def __init__(self):
        super().__init__('depth_estimator_node')
        
        # 参数声明
        self.declare_parameter('estimation_method', 'tpu_depth')  # 'stereo_cv', 'stereo_cnn', 'tpu_depth'
        self.declare_parameter('stereo_algorithm', 'SGBM')  # 'BM' or 'SGBM'
        self.declare_parameter('min_disparity', 0)
        self.declare_parameter('num_disparities', 64)
        self.declare_parameter('block_size', 11)
        self.declare_parameter('enable_tpu', True)
        self.declare_parameter('tpu_model_path', '/home/10210/Desktop/ROS/perception_ws/src/models/fastdepth_quantized_edgetpu.tflite')
        
        # 获取参数
        self.estimation_method = self.get_parameter('estimation_method').value
        self.stereo_algorithm = self.get_parameter('stereo_algorithm').value
        self.min_disparity = self.get_parameter('min_disparity').value
        self.num_disparities = self.get_parameter('num_disparities').value
        self.block_size = self.get_parameter('block_size').value
        self.enable_tpu = self.get_parameter('enable_tpu').value
        self.tpu_model_path = self.get_parameter('tpu_model_path').value
        
        # TPU深度估计器
        self.tpu_depth_estimator = None
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 订阅器
        self.left_image_sub = self.create_subscription(
            Image, '/stereo/left/image_processed', self.left_image_callback, 10)
        self.right_image_sub = self.create_subscription(
            Image, '/stereo/right/image_processed', self.right_image_callback, 10)
        self.left_camera_info_sub = self.create_subscription(
            CameraInfo, '/stereo/left/camera_info', self.left_camera_info_callback, 10)
        
        # 发布器
        self.depth_pub = self.create_publisher(DepthImage, '/perception/depth_estimation', 10)
        
        # 存储最新的图像和相机信息
        self.left_image = None
        self.right_image = None
        self.camera_info = None
        self.latest_header = None
        
        # 线程锁
        self.image_lock = threading.Lock()
        
        # 设置深度估计器
        self.setup_depth_estimator()
        
        # 性能统计
        self.processing_times = []
        
        self.get_logger().info(f"深度估计节点已启动 - 方法: {self.estimation_method}")
        
    def setup_depth_estimator(self):
        """设置深度估计器"""
        if self.estimation_method == 'tpu_depth':
            if TPU_AVAILABLE and self.enable_tpu:
                self.setup_tpu_depth()
            else:
                self.get_logger().warning("TPU不可用，切换到经典CV方法")
                self.estimation_method = 'stereo_cv'
                self.setup_stereo_cv()
        elif self.estimation_method == 'stereo_cv':
            self.setup_stereo_cv()
        elif self.estimation_method == 'stereo_cnn':
            if TORCH_AVAILABLE:
                self.setup_stereo_cnn()
            else:
                self.get_logger().warning("PyTorch不可用，切换到经典CV方法")
                self.estimation_method = 'stereo_cv'
                self.setup_stereo_cv()
    
    def setup_stereo_cv(self):
        """设置经典CV立体匹配"""
        if self.stereo_algorithm == 'BM':
            self.stereo_matcher = cv2.StereoBM_create(
                numDisparities=self.num_disparities,
                blockSize=self.block_size
            )
            # 优化参数
            self.stereo_matcher.setMinDisparity(self.min_disparity)
            self.stereo_matcher.setUniquenessRatio(10)
            self.stereo_matcher.setSpeckleWindowSize(100)
            self.stereo_matcher.setSpeckleRange(32)
            self.stereo_matcher.setDisp12MaxDiff(1)
            
        elif self.stereo_algorithm == 'SGBM':
            self.stereo_matcher = cv2.StereoSGBM_create(
                minDisparity=self.min_disparity,
                numDisparities=self.num_disparities,
                blockSize=self.block_size,
                P1=8 * 3 * self.block_size ** 2,
                P2=32 * 3 * self.block_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32
            )
        
        self.get_logger().info(f"经典CV立体匹配器初始化完成 - 算法: {self.stereo_algorithm}")
    
    def setup_tpu_depth(self):
        """设置TPU深度估计"""
        try:
            self.tpu_depth_estimator = TPUDepthEstimator(self.tpu_model_path)
            self.get_logger().info(f"TPU深度估计器初始化完成 - 模型: {self.tpu_model_path}")
        except Exception as e:
            self.get_logger().error(f"TPU深度估计器初始化失败: {e}")
            # 回退到经典CV方法
            self.estimation_method = 'stereo_cv'
            self.setup_stereo_cv()
    
    def setup_stereo_cnn(self):
        """设置CNN深度估计（占位符，可集成实际模型）"""
        # 这里可以加载预训练的深度估计模型
        # 例如：MonoDepth, DispNet, PSMNet等
        self.get_logger().info("CNN深度估计器初始化完成")
        
        # 示例：简单的CNN前处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def estimate_depth_cv(self, left_img, right_img):
        """使用经典CV方法估计深度"""
        start_time = time.time()
        
        # 转换为灰度图
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        # 计算视差
        disparity = self.stereo_matcher.compute(left_gray, right_gray)
        
        # 后处理
        disparity = disparity.astype(np.float32) / 16.0  # SGBM输出需要除以16
        
        # 滤波去噪
        disparity = cv2.medianBlur(disparity.astype(np.uint8), 5).astype(np.float32)
        
        # 生成深度图
        # 简化的深度计算：depth = baseline * focal_length / disparity
        if self.camera_info is not None:
            focal_length = self.camera_info.k[0]  # fx
            baseline = 0.12  # 假设基线12cm，应从相机标定获取
            
            # 避免除零
            valid_disparity = disparity > 0
            depth_map = np.zeros_like(disparity)
            depth_map[valid_disparity] = (focal_length * baseline) / disparity[valid_disparity]
            
            # 限制深度范围
            depth_map = np.clip(depth_map, 0, 10.0)  # 最大10米
        else:
            depth_map = disparity / np.max(disparity) * 5.0  # 归一化到5米
        
        processing_time = (time.time() - start_time) * 1000
        return depth_map, disparity, processing_time
    
    def estimate_depth_cnn(self, left_img, right_img):
        """使用CNN方法估计深度（占位符实现）"""
        start_time = time.time()
        
        # 这里应该是实际的CNN推理
        # 暂时使用简化的处理作为示例
        
        # 转换为灰度并模拟深度估计
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        
        # 模拟CNN输出（实际应该是模型推理）
        # 使用简单的梯度来模拟深度
        grad_x = cv2.Sobel(left_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(left_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 模拟深度图
        depth_map = 5.0 - (gradient_magnitude / np.max(gradient_magnitude)) * 4.0
        disparity = depth_map * 10  # 模拟视差
        
        processing_time = (time.time() - start_time) * 1000
        return depth_map, disparity, processing_time
    
    def generate_point_cloud(self, depth_map):
        """从深度图生成点云"""
        if self.camera_info is None:
            return []
        
        points = []
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]
        
        height, width = depth_map.shape
        
        # 采样点云（每隔n个像素取一个点以减少计算量）
        step = 10
        for v in range(0, height, step):
            for u in range(0, width, step):
                depth = depth_map[v, u]
                if depth > 0:
                    x = (u - cx) * depth / fx
                    y = (v - cy) * depth / fy
                    z = depth
                    points.append([x, y, z])
        
        return points[:1000]  # 限制点云大小
    
    def left_image_callback(self, msg):
        """左图像回调"""
        with self.image_lock:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_header = msg.header
            self.process_stereo_pair()
    
    def right_image_callback(self, msg):
        """右图像回调"""
        with self.image_lock:
            self.right_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_stereo_pair()
    
    def left_camera_info_callback(self, msg):
        """相机信息回调"""
        self.camera_info = msg
    
    def process_stereo_pair(self):
        """处理立体图像对"""
        if self.left_image is None or self.right_image is None:
            return
        
        try:
            # 确保图像尺寸一致
            if self.left_image.shape != self.right_image.shape:
                return
            
            # 估计深度
            if self.estimation_method == 'stereo_cv':
                depth_map, disparity, processing_time = self.estimate_depth_cv(
                    self.left_image, self.right_image)
            else:
                depth_map, disparity, processing_time = self.estimate_depth_cnn(
                    self.left_image, self.right_image)
            
            # 生成点云
            point_cloud_points = self.generate_point_cloud(depth_map)
            
            # 创建消息
            depth_msg = DepthImage()
            depth_msg.header = self.latest_header if self.latest_header else Header()
            depth_msg.header.stamp = self.get_clock().now().to_msg()
            
            # 转换深度图为ROS Image消息
            depth_msg.depth_image = self.bridge.cv2_to_imgmsg(
                (depth_map * 1000).astype(np.uint16), "mono16")
            depth_msg.disparity_image = self.bridge.cv2_to_imgmsg(
                disparity.astype(np.uint8), "mono8")
            
            depth_msg.estimation_method = self.estimation_method
            
            if self.camera_info:
                depth_msg.baseline = 0.12  # 应从相机标定获取
                depth_msg.focal_length = self.camera_info.k[0]
            
            # 简化点云（转换为geometry_msgs/Point列表）
            from geometry_msgs.msg import Point
            depth_msg.point_cloud = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) 
                                   for p in point_cloud_points]
            
            # 发布
            self.depth_pub.publish(depth_msg)
            
            # 性能统计
            self.processing_times.append(processing_time)
            if len(self.processing_times) >= 50:
                avg_time = np.mean(self.processing_times)
                self.get_logger().info(f"深度估计平均时间: {avg_time:.2f}ms")
                self.processing_times = []
                
        except Exception as e:
            self.get_logger().error(f"深度估计错误: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimatorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 