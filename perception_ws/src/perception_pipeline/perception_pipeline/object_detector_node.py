#!/usr/bin/env python3
"""
目标检测节点 - 支持经典CV和深度学习方法
包含YOLO、MobileNet-SSD、经典特征检测等
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from perception_interfaces.msg import DetectedObjects, DetectedObject
import cv2
from cv_bridge import CvBridge
import numpy as np
import time

class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector_node')
        
        # 参数声明
        self.declare_parameter('detection_method', 'dnn')  # 'cv' or 'dnn'
        self.declare_parameter('dnn_model', 'yolov4')  # 'yolov4', 'yolov5', 'mobilenet_ssd'
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.4)
        self.declare_parameter('use_tpu', False)
        self.declare_parameter('model_path', '/models/')
        
        # 获取参数
        self.detection_method = self.get_parameter('detection_method').value
        self.dnn_model = self.get_parameter('dnn_model').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.nms_threshold = self.get_parameter('nms_threshold').value
        self.use_tpu = self.get_parameter('use_tpu').value
        self.model_path = self.get_parameter('model_path').value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 订阅器
        self.image_sub = self.create_subscription(
            Image, '/stereo/left/image_processed', self.image_callback, 10)
        
        # 发布器
        self.detection_pub = self.create_publisher(DetectedObjects, '/perception/object_detection', 10)
        
        # COCO类别名称
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee'
        ]
        
        # 设置检测器
        self.setup_detector()
        
        # 性能统计
        self.processing_times = []
        
        self.get_logger().info(f"目标检测节点已启动 - 方法: {self.detection_method}")
        
    def setup_detector(self):
        """设置检测器"""
        if self.detection_method == 'dnn':
            self.setup_dnn_detector()
        else:
            self.setup_cv_detector()
    
    def setup_dnn_detector(self):
        """设置DNN检测器"""
        try:
            if self.dnn_model == 'yolov4':
                # YOLOv4配置
                weights_path = f"{self.model_path}/yolov4.weights"
                config_path = f"{self.model_path}/yolov4.cfg"
                
                # 如果模型文件不存在，使用OpenCV DNN的内置模型
                try:
                    self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
                except:
                    # 使用预训练的MobileNet-SSD作为备选
                    self.setup_mobilenet_ssd()
                    return
                    
            elif self.dnn_model == 'mobilenet_ssd':
                self.setup_mobilenet_ssd()
            
            # 设置DNN后端
            if self.use_tpu:
                # 尝试使用TPU/神经网络加速器
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
            self.get_logger().info(f"DNN检测器初始化完成 - 模型: {self.dnn_model}")
            
        except Exception as e:
            self.get_logger().warning(f"DNN初始化失败: {e}，切换到经典CV方法")
            self.detection_method = 'cv'
            self.setup_cv_detector()
    
    def setup_mobilenet_ssd(self):
        """设置MobileNet-SSD"""
        # 使用OpenCV内置的MobileNet-SSD
        prototxt_path = f"{self.model_path}/MobileNetSSD_deploy.prototxt"
        model_path = f"{self.model_path}/MobileNetSSD_deploy.caffemodel"
        
        try:
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        except:
            # 如果文件不存在，创建一个简化的检测器
            self.get_logger().warning("MobileNet-SSD模型文件不存在，使用简化检测")
            self.net = None
    
    def setup_cv_detector(self):
        """设置经典CV检测器"""
        # 设置各种特征检测器
        
        # HOG人体检测器
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # 级联分类器（人脸检测）
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            self.face_cascade = None
            
        # ORB特征检测器
        self.orb = cv2.ORB_create()
        
        # 轮廓检测参数
        self.contour_area_threshold = 1000
        
        self.get_logger().info("经典CV检测器初始化完成")
    
    def detect_objects_dnn(self, image):
        """使用DNN检测目标"""
        if self.net is None:
            return []
            
        start_time = time.time()
        
        # 预处理
        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        
        # 推理
        detections = self.net.forward()
        
        height, width = image.shape[:2]
        objects = []
        
        # 解析检测结果
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                class_id = int(detections[0, 0, i, 1])
                
                # 边界框坐标
                left = int(detections[0, 0, i, 3] * width)
                top = int(detections[0, 0, i, 4] * height)
                right = int(detections[0, 0, i, 5] * width)
                bottom = int(detections[0, 0, i, 6] * height)
                
                # 创建检测对象
                obj = DetectedObject()
                obj.class_id = class_id
                obj.class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                obj.confidence = float(confidence)
                
                # 几何信息
                from geometry_msgs.msg import Point
                obj.center = Point(x=float((left + right) / 2), y=float((top + bottom) / 2), z=0.0)
                obj.min_bound = Point(x=float(left), y=float(top), z=0.0)
                obj.max_bound = Point(x=float(right), y=float(bottom), z=0.0)
                obj.pixel_area = (right - left) * (bottom - top)
                
                objects.append(obj)
        
        processing_time = (time.time() - start_time) * 1000
        return objects, processing_time
    
    def detect_objects_cv(self, image):
        """使用经典CV方法检测目标"""
        start_time = time.time()
        objects = []
        
        # 1. HOG人体检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        try:
            # 检测人体
            bodies, weights = self.hog.detectMultiScale(gray, winStride=(8,8), padding=(32,32), scale=1.05)
            
            for i, (x, y, w, h) in enumerate(bodies):
                obj = DetectedObject()
                obj.class_id = 0  # person
                obj.class_name = "person"
                obj.confidence = float(weights[i][0]) if len(weights) > 0 else 0.8
                
                from geometry_msgs.msg import Point
                obj.center = Point(x=float(x + w/2), y=float(y + h/2), z=0.0)
                obj.min_bound = Point(x=float(x), y=float(y), z=0.0)
                obj.max_bound = Point(x=float(x + w), y=float(y + h), z=0.0)
                obj.pixel_area = w * h
                
                objects.append(obj)
        except Exception as e:
            self.get_logger().debug(f"HOG检测错误: {e}")
        
        # 2. 人脸检测
        if self.face_cascade is not None:
            try:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    obj = DetectedObject()
                    obj.class_id = 100  # face (自定义类别)
                    obj.class_name = "face"
                    obj.confidence = 0.9
                    
                    from geometry_msgs.msg import Point
                    obj.center = Point(x=float(x + w/2), y=float(y + h/2), z=0.0)
                    obj.min_bound = Point(x=float(x), y=float(y), z=0.0)
                    obj.max_bound = Point(x=float(x + w), y=float(y + h), z=0.0)
                    obj.pixel_area = w * h
                    
                    objects.append(obj)
            except Exception as e:
                self.get_logger().debug(f"人脸检测错误: {e}")
        
        # 3. 轮廓检测（检测大型对象）
        try:
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.contour_area_threshold:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 简单的形状分类
                    aspect_ratio = float(w) / h
                    if 0.8 < aspect_ratio < 1.2:
                        class_name = "square_object"
                        class_id = 101
                    elif aspect_ratio > 2.0:
                        class_name = "rectangular_object"
                        class_id = 102
                    else:
                        class_name = "unknown_object"
                        class_id = 103
                    
                    obj = DetectedObject()
                    obj.class_id = class_id
                    obj.class_name = class_name
                    obj.confidence = 0.6
                    
                    from geometry_msgs.msg import Point
                    obj.center = Point(x=float(x + w/2), y=float(y + h/2), z=0.0)
                    obj.min_bound = Point(x=float(x), y=float(y), z=0.0)
                    obj.max_bound = Point(x=float(x + w), y=float(y + h), z=0.0)
                    obj.pixel_area = w * h
                    
                    objects.append(obj)
        except Exception as e:
            self.get_logger().debug(f"轮廓检测错误: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        return objects, processing_time
    
    def image_callback(self, msg):
        """图像回调"""
        try:
            # 转换图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 检测目标
            if self.detection_method == 'dnn':
                objects, processing_time = self.detect_objects_dnn(cv_image)
            else:
                objects, processing_time = self.detect_objects_cv(cv_image)
            
            # 创建检测结果消息
            detection_msg = DetectedObjects()
            detection_msg.header = msg.header
            detection_msg.objects = objects
            detection_msg.total_objects = len(objects)
            
            # 发布
            self.detection_pub.publish(detection_msg)
            
            # 性能统计
            self.processing_times.append(processing_time)
            if len(self.processing_times) >= 50:
                avg_time = np.mean(self.processing_times)
                self.get_logger().info(f"目标检测平均时间: {avg_time:.2f}ms, 检测到 {len(objects)} 个对象")
                self.processing_times = []
                
        except Exception as e:
            self.get_logger().error(f"目标检测错误: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 