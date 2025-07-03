#!/usr/bin/env python3
"""
语义分割节点 - 为每个像素分配类别标签
支持经典CV方法和深度学习方法
修复连接延迟问题
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from perception_interfaces.msg import SegmentedImage
import cv2
from cv_bridge import CvBridge
import numpy as np
import time

class SemanticSegmentorNode(Node):
    def __init__(self):
        super().__init__('semantic_segmentor_node')
        
        # 参数声明
        self.declare_parameter('segmentation_method', 'cv')  # 'cv' or 'dnn'
        self.declare_parameter('dnn_model', 'fast_scnn')  # 'fast_scnn', 'bisenet'
        self.declare_parameter('use_tpu', False)
        self.declare_parameter('model_path', '/models/')
        
        # 获取参数
        self.segmentation_method = self.get_parameter('segmentation_method').value
        self.dnn_model = self.get_parameter('dnn_model').value
        self.use_tpu = self.get_parameter('use_tpu').value
        self.model_path = self.get_parameter('model_path').value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 连接状态追踪
        self.publisher_connected = False
        self.first_image_received = False
        self.startup_time = time.time()
        
        # 订阅器 - 使用兼容的QoS设置
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,  # 改为RELIABLE确保连接
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        self.image_sub = self.create_subscription(
            Image, '/stereo/left/image_processed', self.image_callback, qos_profile)
        
        # 发布器
        self.segmentation_pub = self.create_publisher(SegmentedImage, '/perception/semantic_segmentation', 10)
        
        # 连接检测定时器
        self.connection_timer = self.create_timer(2.0, self.check_connections)
        
        # 类别定义（简化的城市场景）
        self.class_names = [
            'background', 'road', 'sidewalk', 'building', 'wall', 'fence',
            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
            'motorcycle', 'bicycle'
        ]
        
        # 类别颜色映射
        self.class_colors = self.generate_class_colors()
        
        # 设置分割器
        self.setup_segmentor()
        
        # 性能统计
        self.processing_times = []
        
        self.get_logger().info(f"语义分割节点已启动 - 方法: {self.segmentation_method}")
        self.get_logger().info("等待图像预处理节点连接...")
    
    def check_connections(self):
        """检查话题连接状态"""
        # 检查订阅的话题是否有发布者
        topic_names_and_types = self.get_topic_names_and_types()
        publishers_info = self.get_publishers_info_by_topic('/stereo/left/image_processed')
        
        current_time = time.time()
        elapsed = current_time - self.startup_time
        
        if len(publishers_info) > 0 and not self.publisher_connected:
            self.publisher_connected = True
            self.get_logger().info(f"✅ 图像预处理节点已连接！(启动后 {elapsed:.1f}s)")
        elif len(publishers_info) == 0:
            self.get_logger().info(f"⏳ 等待图像预处理节点连接... (已等待 {elapsed:.1f}s)")
        
        # 如果连接了但还没收到图像
        if self.publisher_connected and not self.first_image_received:
            self.get_logger().info(f"📡 已连接，等待第一帧图像... (启动后 {elapsed:.1f}s)")
    
    def generate_class_colors(self):
        """生成类别颜色映射"""
        colors = []
        np.random.seed(42)  # 固定随机种子确保颜色一致
        for i in range(len(self.class_names)):
            color = np.random.randint(0, 255, 3).tolist()
            colors.append(color)
        return colors
    
    def setup_segmentor(self):
        """设置分割器"""
        if self.segmentation_method == 'dnn':
            self.setup_dnn_segmentor()
        else:
            self.setup_cv_segmentor()
    
    def setup_dnn_segmentor(self):
        """设置DNN分割器"""
        try:
            # 这里应该加载实际的分割模型
            # 由于模型文件可能不存在，我们使用简化的实现
            self.get_logger().warning("DNN分割模型未加载，使用经典CV方法")
            self.segmentation_method = 'cv'
            self.setup_cv_segmentor()
        except Exception as e:
            self.get_logger().warning(f"DNN分割器初始化失败: {e}，切换到经典CV方法")
            self.segmentation_method = 'cv'
            self.setup_cv_segmentor()
    
    def setup_cv_segmentor(self):
        """设置经典CV分割器"""
        # 设置各种分割参数 - 优化性能
        self.kmeans_k = 4  # 减少聚类数，从8改为4
        self.kmeans_max_iter = 10  # 限制最大迭代次数
        self.watershed_markers = 50
        
        self.get_logger().info("经典CV分割器初始化完成")
    
    def segment_image_cv(self, image):
        """使用极简化CV方法进行语义分割 - 保持640x480分辨率"""
        start_time = time.time()
        
        height, width = image.shape[:2]
        
        # 创建分割结果 (直接在640x480上处理)
        segmented = np.zeros((height, width), dtype=np.uint8)
        
        # 一次性转换到HSV (最耗时的操作)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 策略：基于位置的快速分割，最小化cv2.inRange调用
        
        # 1. 天空区域 (只检测图像上半部分)
        sky_region = hsv[:height//2, :]  # 只处理上半部分
        sky_mask = cv2.inRange(sky_region, np.array([100, 50, 50]), np.array([130, 255, 255]))
        segmented[:height//2, :][sky_mask > 0] = self.class_names.index('sky')
        
        # 2. 道路区域 (只检测图像下半部分的灰色)
        road_start = int(height * 0.7)
        road_region = hsv[road_start:, :]
        # 简化检测：只用亮度通道，避免HSV范围检测
        road_v = road_region[:, :, 2]  # V通道
        road_mask = (road_v > 50) & (road_v < 150)  # 中等亮度
        segmented[road_start:, :][road_mask] = self.class_names.index('road')
        
        # 3. 植被区域 (只检测绿色，简化范围)
        vegetation_h = hsv[:, :, 0]  # H通道
        vegetation_s = hsv[:, :, 1]  # S通道
        vegetation_mask = (vegetation_h >= 35) & (vegetation_h <= 85) & (vegetation_s > 40)
        segmented[vegetation_mask] = self.class_names.index('vegetation')
        
        # 4. 建筑物 (图像中上部分的高亮度区域)
        building_end = int(height * 0.6)
        building_v = hsv[:building_end, :, 2]
        building_mask = building_v > 120  # 高亮度
        segmented[:building_end, :][building_mask] = self.class_names.index('building')
        
        # 5. 背景填充 (一次性操作)
        background_mask = (segmented == 0)
        segmented[background_mask] = self.class_names.index('background')
        
        processing_time = (time.time() - start_time) * 1000
        return segmented, processing_time
    
    def create_color_map(self, segmented_image):
        """创建彩色分割图"""
        height, width = segmented_image.shape
        color_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        for class_id in range(len(self.class_names)):
            mask = (segmented_image == class_id)
            color_map[mask] = self.class_colors[class_id]
        
        return color_map
    
    def image_callback(self, msg):
        """图像回调"""
        try:
            # 标记收到第一帧图像
            if not self.first_image_received:
                elapsed = time.time() - self.startup_time
                self.first_image_received = True
                self.get_logger().info(f"🎉 收到第一帧图像！(启动后 {elapsed:.1f}s)")
                # 停止连接检测定时器
                self.connection_timer.cancel()
            
            # 转换图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 语义分割
            if self.segmentation_method == 'dnn':
                # DNN分割（占位符）
                segmented, processing_time = self.segment_image_cv(cv_image)
            else:
                segmented, processing_time = self.segment_image_cv(cv_image)
            
            # 创建彩色分割图
            color_map = self.create_color_map(segmented)
            
            # 创建分割结果消息
            segmentation_msg = SegmentedImage()
            segmentation_msg.header = msg.header
            
            # 转换分割图像
            segmentation_msg.segmented_image = self.bridge.cv2_to_imgmsg(
                segmented, "mono8")
            segmentation_msg.color_map = self.bridge.cv2_to_imgmsg(
                color_map, "bgr8")
            
            segmentation_msg.class_names = self.class_names
            segmentation_msg.class_ids = list(range(len(self.class_names)))
            segmentation_msg.segmentation_method = self.segmentation_method
            
            # 发布
            self.segmentation_pub.publish(segmentation_msg)
            
            # 性能统计
            self.processing_times.append(processing_time)
            if len(self.processing_times) >= 3:  # 改为3帧就输出，更快看到结果
                avg_time = np.mean(self.processing_times)
                unique_classes = len(np.unique(segmented))
                self.get_logger().info(f"🎯 语义分割平均时间: {avg_time:.2f}ms, 检测到 {unique_classes} 个类别")
                self.processing_times = []
                
        except Exception as e:
            self.get_logger().error(f"语义分割错误: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SemanticSegmentorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 