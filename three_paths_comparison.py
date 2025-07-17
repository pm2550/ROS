#!/usr/bin/env python3
"""
三路径感知管道对比测试
========================

路径1：AART混合方式 - 使用AARTware现有组件 + 传统CV补充
路径2：全CPU TFLite - 所有推理都使用CPU TensorFlow Lite
路径3：全TPU - 所有推理都在EdgeTPU上执行

测试完整感知流水线的三种不同实现方式
"""

import os
import sys
import time
import numpy as np  
import cv2
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

# 添加AARTware路径
sys.path.append('/ros2_ws/src/workspace/AARTware-main/cores/perception/yolov8_ros2/yolo_detect_opponent/yolo_detect_opponent')

# 导入依赖
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 不可用")

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    print("⚠️  TFLite Runtime 不可用")

try:
    from pycoral.utils import edgetpu
    from pycoral.adapters import common, detect, segment
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    print("⚠️  EdgeTPU 不可用")

class SyntheticDataGenerator:
    """合成测试数据生成器"""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
    
    def generate_test_scene(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载真实测试图片而不是合成数据"""
        # 尝试加载真实图片
        real_img_path = "/ros2_ws/test_image.jpg"
        if os.path.exists(real_img_path):
            img = cv2.imread(real_img_path)
            if img is not None:
                print(f"✅ 使用真实图片: {img.shape}")
                # 生成对应的深度图（简化版）
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                depth_map = 255 - gray  # 简单的深度估计
                return img, depth_map
        
        # 如果真实图片不存在，创建基础场景
        print("⚠️  真实图片不存在，使用合成数据")
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # 添加道路背景
        cv2.rectangle(img, (0, self.height//2), (self.width, self.height), (100, 100, 100), -1)
        
        # 添加车辆 (红色矩形)
        cv2.rectangle(img, (100, 200), (200, 300), (0, 0, 255), -1)
        cv2.rectangle(img, (400, 180), (520, 320), (0, 0, 255), -1)
        
        # 添加行人 (绿色椭圆)
        cv2.ellipse(img, (300, 350), (30, 80), 0, 0, 360, (0, 255, 0), -1)
        
        # 添加噪声
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # 生成对应的深度图
        depth_map = np.full((self.height, self.width), 100, dtype=np.uint8)
        depth_map[200:300, 100:200] = 50  # 车辆1深度
        depth_map[180:320, 400:520] = 60  # 车辆2深度
        depth_map[270:430, 270:330] = 30  # 行人深度
        
        return img, depth_map

class Path1_AARTMixedProcessor:
    """路径1：AART混合方式处理器"""
    
    def __init__(self):
        self.name = "路径1：AART混合方式"
        self.description = "使用AARTware现有组件 + 传统CV补充"
        
        # 初始化AARTware YOLO检测器
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                # 优先使用AARTware自定义模型
                model_paths = [
                    '/ros2_ws/yolov8n.pt',
                    '/ros2_ws/AARTware-main/cores/perception/yolov8_ros2/yolo_detect_opponent/yolo_detect_opponent/yolov8n.pt',
                    '/ros2_ws/AARTware-main/cores/perception/yolov8_ros2/yolo_detect_opponent/yolo_detect_opponent/best.pt'
                ]
                
                for model_path in model_paths:
                    if os.path.exists(model_path):
                        self.yolo_model = YOLO(model_path)
                        print(f"✅ 路径1：加载YOLOv8模型 {model_path}")
                        break
                else:
                    print("⚠️  路径1：YOLOv8模型文件不存在")
            except Exception as e:
                print(f"⚠️  路径1：YOLO模型加载失败: {e}")
        
        print(f"🔧 初始化完成：{self.name}")
    
    def process_image(self, img: np.ndarray, depth_map: np.ndarray) -> Dict[str, Any]:
        """处理图像 - AART混合方式"""
        results = {
            'path': self.name,
            'object_detection': [],
            'semantic_segmentation': None,
            'depth_estimation': None,
            'processing_times': {}
        }
        
        # 1. 目标检测 - 使用AARTware YOLOv8
        start_time = time.perf_counter()
        if self.yolo_model is not None:
            try:
                detections = self.yolo_model(img, verbose=False)
                for detection in detections:
                    boxes = detection.boxes
                    if boxes is not None:
                        for box in boxes:
                            conf = float(box.conf[0])
                            if conf > 0.5:
                                cls = int(box.cls[0])
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                results['object_detection'].append({
                                    'class': f'object_{cls}',
                                    'confidence': conf,
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'method': 'aartware_yolo'
                                })
            except Exception as e:
                print(f"路径1 YOLO检测失败: {e}")
        
        # 传统CV作为补充
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 过滤小对象
                x, y, w, h = cv2.boundingRect(contour)
                results['object_detection'].append({
                    'class': 'traditional_cv_object',
                    'confidence': 0.8,
                    'bbox': [x, y, x+w, y+h],
                    'method': 'traditional_cv'
                })
        
        detection_time = (time.perf_counter() - start_time) * 1000
        results['processing_times']['object_detection'] = detection_time
        
        # 2. 语义分割 - 使用传统CV方法（AART没有此功能）
        start_time = time.perf_counter()
        segmentation_map = self._traditional_semantic_segmentation(img)
        segmentation_time = (time.perf_counter() - start_time) * 1000
        results['semantic_segmentation'] = segmentation_map
        results['processing_times']['semantic_segmentation'] = segmentation_time
        
        # 3. 深度估计 - 使用传统CV方法
        start_time = time.perf_counter()
        depth_estimation = self._traditional_depth_estimation(img)
        depth_time = (time.perf_counter() - start_time) * 1000
        results['depth_estimation'] = depth_estimation
        results['processing_times']['depth_estimation'] = depth_time
        
        # 计算总时间
        total_time = sum(results['processing_times'].values())
        results['processing_times']['total'] = total_time
        
        return results
    
    def _traditional_semantic_segmentation(self, img: np.ndarray) -> np.ndarray:
        """传统CV语义分割"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 创建分割掩码
        segmentation = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # 道路区域（灰色）
        road_mask = cv2.inRange(hsv, (0, 0, 50), (180, 50, 150))
        segmentation[road_mask > 0] = 1
        
        # 车辆区域（红色）
        car_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        segmentation[car_mask > 0] = 2
        
        # 行人区域（绿色）
        person_mask = cv2.inRange(hsv, (50, 100, 100), (70, 255, 255))
        segmentation[person_mask > 0] = 3
        
        return segmentation
    
    def _traditional_depth_estimation(self, img: np.ndarray) -> np.ndarray:
        """传统CV深度估计（简化版）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用梯度强度估计深度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 归一化到深度范围
        depth = 255 - cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return depth

class Path2_CPUTFLiteProcessor:
    """路径2：全CPU TFLite处理器"""
    
    def __init__(self):
        self.name = "路径2：全CPU TFLite"
        self.description = "所有推理都使用CPU TensorFlow Lite"
        
        # 模型路径 - 使用models_unified统一模型
        self.detection_model_path = "/ros2_ws/models_unified/cpu/ssd_mobilenet_v2_300_cpu.tflite"
        self.segmentation_model_path = "/ros2_ws/models_unified/cpu/deeplabv3_mnv2_513_cpu.tflite"
        
        # 初始化模型
        self.detection_interpreter = None
        self.segmentation_interpreter = None
        
        if TFLITE_AVAILABLE:
            try:
                # 加载检测模型
                if os.path.exists(self.detection_model_path):
                    self.detection_interpreter = tflite.Interpreter(model_path=self.detection_model_path)
                    self.detection_interpreter.allocate_tensors()
                    print("✅ 路径2：加载CPU检测模型")
                    
                    # CPU预热
                    input_details = self.detection_interpreter.get_input_details()
                    dummy_input = np.zeros(input_details[0]['shape'], dtype=input_details[0]['dtype'])
                    for _ in range(3):
                        self.detection_interpreter.set_tensor(input_details[0]['index'], dummy_input)
                        self.detection_interpreter.invoke()
                    print("✅ 路径2：CPU检测模型预热完成")
                
                # 加载分割模型
                if os.path.exists(self.segmentation_model_path):
                    self.segmentation_interpreter = tflite.Interpreter(model_path=self.segmentation_model_path)
                    self.segmentation_interpreter.allocate_tensors()
                    print("✅ 路径2：加载CPU分割模型")
                    
                    # CPU预热
                    input_details = self.segmentation_interpreter.get_input_details()
                    dummy_input = np.zeros(input_details[0]['shape'], dtype=input_details[0]['dtype'])
                    for _ in range(3):
                        self.segmentation_interpreter.set_tensor(input_details[0]['index'], dummy_input)
                        self.segmentation_interpreter.invoke()
                    print("✅ 路径2：CPU分割模型预热完成")
                    
            except Exception as e:
                print(f"⚠️  路径2：模型加载失败: {e}")
        
        print(f"🔧 初始化完成：{self.name}")
    
    def process_image(self, img: np.ndarray, depth_map: np.ndarray) -> Dict[str, Any]:
        """处理图像 - 全CPU TFLite"""
        results = {
            'path': self.name,
            'object_detection': [],
            'semantic_segmentation': None,
            'depth_estimation': None,
            'processing_times': {},
            'detailed_times': {}
        }
        
        # 1. 目标检测 - CPU TFLite
        if self.detection_interpreter is not None:
            detection_result = self._cpu_object_detection_detailed(img)
            results['object_detection'] = detection_result['detections']
            results['processing_times']['object_detection'] = detection_result['total_time']
            results['detailed_times']['object_detection'] = detection_result['detailed_times']
        else:
            results['processing_times']['object_detection'] = 0
            results['detailed_times']['object_detection'] = {}
        
        # 2. 语义分割 - CPU TFLite
        if self.segmentation_interpreter is not None:
            segmentation_result = self._cpu_semantic_segmentation_detailed(img)
            results['semantic_segmentation'] = segmentation_result['segmentation']
            results['processing_times']['semantic_segmentation'] = segmentation_result['total_time']
            results['detailed_times']['semantic_segmentation'] = segmentation_result['detailed_times']
        else:
            results['processing_times']['semantic_segmentation'] = 0
            results['detailed_times']['semantic_segmentation'] = {}
        
        # 3. 深度估计 - 使用相同的传统CV方法控制变量
        start_time = time.perf_counter()
        depth_estimation = self._cpu_depth_estimation(img)
        depth_time = (time.perf_counter() - start_time) * 1000
        results['depth_estimation'] = depth_estimation
        results['processing_times']['depth_estimation'] = depth_time
        
        # 深度估计使用相同的传统CV方法，实际测量时间，不需要详细分解
        
        # 计算总时间
        total_time = sum(results['processing_times'].values())
        results['processing_times']['total'] = total_time
        
        return results
    
    def _cpu_object_detection_detailed(self, img: np.ndarray) -> Dict[str, Any]:
        """CPU TFLite目标检测 - 详细时间统计"""
        try:
            detailed_times = {}
            total_start = time.perf_counter()
            
            # 获取输入输出详情
            input_details = self.detection_interpreter.get_input_details()
            output_details = self.detection_interpreter.get_output_details()
            
            # 1. 预处理
            preprocessing_start = time.perf_counter()
            input_shape = input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            img_resized = cv2.resize(img, (width, height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            if input_details[0]['dtype'] == np.uint8:
                input_data = np.expand_dims(img_rgb.astype(np.uint8), axis=0)
            else:
                img_normalized = img_rgb.astype(np.float32) / 255.0
                input_data = np.expand_dims(img_normalized, axis=0)
            
            detailed_times['preprocessing'] = (time.perf_counter() - preprocessing_start) * 1000
            
            # 2. 数据传输
            data_transfer_start = time.perf_counter()
            self.detection_interpreter.set_tensor(input_details[0]['index'], input_data)
            detailed_times['data_transfer'] = (time.perf_counter() - data_transfer_start) * 1000
            
            # 3. 纯推理
            inference_start = time.perf_counter()
            self.detection_interpreter.invoke()
            detailed_times['inference'] = (time.perf_counter() - inference_start) * 1000
            
            # 4. 结果获取
            result_fetch_start = time.perf_counter()
            boxes = self.detection_interpreter.get_tensor(output_details[0]['index'])[0]
            classes = self.detection_interpreter.get_tensor(output_details[1]['index'])[0]
            scores = self.detection_interpreter.get_tensor(output_details[2]['index'])[0]
            detailed_times['result_fetch'] = (time.perf_counter() - result_fetch_start) * 1000
            
            # 5. 后处理
            postprocessing_start = time.perf_counter()
            detections = []
            for i in range(len(scores)):
                if scores[i] > 0.3:  # 降低置信度阈值
                    y1, x1, y2, x2 = boxes[i]
                    x1, y1, x2, y2 = int(x1 * img.shape[1]), int(y1 * img.shape[0]), int(x2 * img.shape[1]), int(y2 * img.shape[0])
                    detections.append({
                        'class': int(classes[i]),
                        'score': float(scores[i]),
                        'bbox': [x1, y1, x2, y2]
                    })
            detailed_times['postprocessing'] = (time.perf_counter() - postprocessing_start) * 1000
            
            # 总时间
            total_time = (time.perf_counter() - total_start) * 1000
            
            return {
                'detections': detections,
                'total_time': total_time,
                'detailed_times': detailed_times
            }
            
        except Exception as e:
            print(f"CPU检测失败: {e}")
            return {
                'detections': [],
                'total_time': 0,
                'detailed_times': {}
            }
    
    def _cpu_object_detection(self, img: np.ndarray) -> List[Dict]:
        """CPU TFLite目标检测"""
        try:
            # 获取输入输出详情
            input_details = self.detection_interpreter.get_input_details()
            output_details = self.detection_interpreter.get_output_details()
            
            # 预处理图像
            input_shape = input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            
            # 调整图像大小并处理量化模型
            img_resized = cv2.resize(img, (width, height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # 检查模型期望的数据类型
            if input_details[0]['dtype'] == np.uint8:
                # 量化模型，使用uint8输入
                input_data = np.expand_dims(img_rgb.astype(np.uint8), axis=0)
            else:
                # 浮点模型，使用归一化输入
                img_normalized = img_rgb.astype(np.float32) / 255.0
                input_data = np.expand_dims(img_normalized, axis=0)
            
            # 运行推理
            self.detection_interpreter.set_tensor(input_details[0]['index'], input_data)
            self.detection_interpreter.invoke()
            
            # 获取输出
            boxes = self.detection_interpreter.get_tensor(output_details[0]['index'])[0]
            classes = self.detection_interpreter.get_tensor(output_details[1]['index'])[0]
            scores = self.detection_interpreter.get_tensor(output_details[2]['index'])[0]
            
            # 处理检测结果
            detections = []
            for i in range(len(scores)):
                if scores[i] > 0.3:  # 降低置信度阈值
                    y1, x1, y2, x2 = boxes[i]
                    x1, y1, x2, y2 = int(x1 * img.shape[1]), int(y1 * img.shape[0]), int(x2 * img.shape[1]), int(y2 * img.shape[0])
                    
                    detections.append({
                        'class': f'class_{int(classes[i])}',
                        'confidence': float(scores[i]),
                        'bbox': [x1, y1, x2, y2],
                        'method': 'cpu_tflite'
                    })
            
            return detections
        
        except Exception as e:
            print(f"路径2 CPU检测失败: {e}")
            return []
    
    def _cpu_semantic_segmentation_detailed(self, img: np.ndarray) -> Dict[str, Any]:
        """CPU TFLite语义分割 - 详细时间统计"""
        try:
            detailed_times = {}
            total_start = time.perf_counter()
            
            # 获取输入输出详情
            input_details = self.segmentation_interpreter.get_input_details()
            output_details = self.segmentation_interpreter.get_output_details()
            
            # 1. 预处理
            preprocessing_start = time.perf_counter()
            input_shape = input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            img_resized = cv2.resize(img, (width, height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            if input_details[0]['dtype'] == np.uint8:
                input_data = np.expand_dims(img_rgb.astype(np.uint8), axis=0)
            else:
                img_normalized = img_rgb.astype(np.float32) / 127.5 - 1.0
                input_data = np.expand_dims(img_normalized, axis=0)
            
            detailed_times['preprocessing'] = (time.perf_counter() - preprocessing_start) * 1000
            
            # 2. 数据传输
            data_transfer_start = time.perf_counter()
            self.segmentation_interpreter.set_tensor(input_details[0]['index'], input_data)
            detailed_times['data_transfer'] = (time.perf_counter() - data_transfer_start) * 1000
            
            # 3. 纯推理
            inference_start = time.perf_counter()
            self.segmentation_interpreter.invoke()
            detailed_times['inference'] = (time.perf_counter() - inference_start) * 1000
            
            # 4. 结果获取
            result_fetch_start = time.perf_counter()
            output_data = self.segmentation_interpreter.get_tensor(output_details[0]['index'])[0]
            detailed_times['result_fetch'] = (time.perf_counter() - result_fetch_start) * 1000
            
            # 5. 后处理
            postprocessing_start = time.perf_counter()
            # 输出已经是分割掩码，无需argmax
            segmentation_map = output_data  # output_data已经是[513,513]格式
            segmentation_resized = cv2.resize(segmentation_map.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            detailed_times['postprocessing'] = (time.perf_counter() - postprocessing_start) * 1000
            
            # 总时间
            total_time = (time.perf_counter() - total_start) * 1000
            
            return {
                'segmentation': segmentation_resized,
                'total_time': total_time,
                'detailed_times': detailed_times
            }
            
        except Exception as e:
            print(f"路径2 CPU分割失败: {e}")
            return {
                'segmentation': np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8),
                'total_time': 0,
                'detailed_times': {}
            }
    
    def _cpu_semantic_segmentation(self, img: np.ndarray) -> np.ndarray:
        """CPU TFLite语义分割"""
        try:
            # 获取输入输出详情
            input_details = self.segmentation_interpreter.get_input_details()
            output_details = self.segmentation_interpreter.get_output_details()
            
            # 预处理图像
            input_shape = input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            
            # 调整图像大小并处理量化模型
            img_resized = cv2.resize(img, (width, height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # 检查模型期望的数据类型
            if input_details[0]['dtype'] == np.uint8:
                # 量化模型，使用uint8输入
                input_data = np.expand_dims(img_rgb.astype(np.uint8), axis=0)
            else:
                # 浮点模型，使用归一化输入
                img_normalized = img_rgb.astype(np.float32) / 127.5 - 1.0
                input_data = np.expand_dims(img_normalized, axis=0)
            
            # 运行推理
            self.segmentation_interpreter.set_tensor(input_details[0]['index'], input_data)
            self.segmentation_interpreter.invoke()
            
            # 获取输出
            output_data = self.segmentation_interpreter.get_tensor(output_details[0]['index'])[0]
            
            # 输出已经是分割掩码，无需argmax
            segmentation_map = output_data  # output_data已经是[513,513]格式
            
            # 调整回原始尺寸
            segmentation_resized = cv2.resize(segmentation_map.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            return segmentation_resized
            
        except Exception as e:
            print(f"路径2 CPU分割失败: {e}")
            return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    def _cpu_depth_estimation(self, img: np.ndarray) -> np.ndarray:
        """CPU深度估计（统一使用Sobel方法）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用梯度强度估计深度（与Path 1相同）
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 归一化到深度范围
        depth = 255 - cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return depth

class Path3_TPUProcessor:
    """路径3：全TPU处理器"""
    
    def __init__(self):
        self.name = "路径3：全TPU"
        self.description = "所有推理都在EdgeTPU上执行"
        
        # 模型路径 - 使用models_unified统一模型
        self.detection_model_path = "/ros2_ws/models_unified/edgetpu/ssd_mobilenet_v2_300_edgetpu.tflite"
        self.segmentation_model_path = "/ros2_ws/models_unified/edgetpu/deeplabv3_mnv2_513_edgetpu.tflite"
        
        # 初始化模型
        self.detection_interpreter = None
        self.segmentation_interpreter = None
        
        if TPU_AVAILABLE:
            try:
                # 加载检测模型
                if os.path.exists(self.detection_model_path):
                    self.detection_interpreter = tflite.Interpreter(
                        model_path=self.detection_model_path,
                        experimental_delegates=[edgetpu.load_edgetpu_delegate()]
                    )
                    self.detection_interpreter.allocate_tensors()
                    print("✅ 路径3：加载TPU检测模型")
                    
                    # EdgeTPU预热
                    input_details = self.detection_interpreter.get_input_details()
                    dummy_input = np.zeros(input_details[0]['shape'], dtype=input_details[0]['dtype'])
                    for _ in range(5):
                        self.detection_interpreter.set_tensor(input_details[0]['index'], dummy_input)
                        self.detection_interpreter.invoke()
                    print("✅ 路径3：TPU检测模型预热完成")
                
                # 加载分割模型
                if os.path.exists(self.segmentation_model_path):
                    self.segmentation_interpreter = tflite.Interpreter(
                        model_path=self.segmentation_model_path,
                        experimental_delegates=[edgetpu.load_edgetpu_delegate()]
                    )
                    self.segmentation_interpreter.allocate_tensors()
                    print("✅ 路径3：加载TPU分割模型")
                    
                    # EdgeTPU预热
                    input_details = self.segmentation_interpreter.get_input_details()
                    dummy_input = np.zeros(input_details[0]['shape'], dtype=input_details[0]['dtype'])
                    for _ in range(5):
                        self.segmentation_interpreter.set_tensor(input_details[0]['index'], dummy_input)
                        self.segmentation_interpreter.invoke()
                    print("✅ 路径3：TPU分割模型预热完成")
                    
            except Exception as e:
                print(f"⚠️  路径3：TPU模型加载失败，回退到CPU: {e}")
                # 回退到CPU
                try:
                    if os.path.exists(self.detection_model_path):
                        self.detection_interpreter = tflite.Interpreter(model_path=self.detection_model_path)
                        self.detection_interpreter.allocate_tensors()
                        print("✅ 路径3：使用CPU模式")
                except Exception as e2:
                    print(f"⚠️  路径3：CPU回退也失败: {e2}")
        
        print(f"🔧 初始化完成：{self.name}")
    
    def process_image(self, img: np.ndarray, depth_map: np.ndarray) -> Dict[str, Any]:
        """处理图像 - 全TPU"""
        results = {
            'path': self.name,
            'object_detection': [],
            'semantic_segmentation': None,
            'depth_estimation': None,
            'processing_times': {},
            'detailed_times': {}
        }
        
        # 1. 目标检测 - TPU
        if self.detection_interpreter is not None:
            detection_result = self._tpu_object_detection_detailed(img)
            results['object_detection'] = detection_result['detections']
            results['processing_times']['object_detection'] = detection_result['total_time']
            results['detailed_times']['object_detection'] = detection_result['detailed_times']
        else:
            results['processing_times']['object_detection'] = 0
            results['detailed_times']['object_detection'] = {}
        
        # 2. 语义分割 - TPU
        if self.segmentation_interpreter is not None:
            segmentation_result = self._tpu_semantic_segmentation_detailed(img)
            results['semantic_segmentation'] = segmentation_result['segmentation']
            results['processing_times']['semantic_segmentation'] = segmentation_result['total_time']
            results['detailed_times']['semantic_segmentation'] = segmentation_result['detailed_times']
        else:
            results['processing_times']['semantic_segmentation'] = 0
            results['detailed_times']['semantic_segmentation'] = {}
        
        # 3. 深度估计 - 使用相同的传统CV方法控制变量
        start_time = time.perf_counter()
        depth_estimation = self._tpu_depth_estimation(img)
        depth_time = (time.perf_counter() - start_time) * 1000
        results['depth_estimation'] = depth_estimation
        results['processing_times']['depth_estimation'] = depth_time
        
        # 深度估计使用相同的传统CV方法，实际测量时间，不需要详细分解
        
        # 计算总时间
        total_time = sum(results['processing_times'].values())
        results['processing_times']['total'] = total_time
        
        return results
    
    def _tpu_object_detection_detailed(self, img: np.ndarray) -> Dict[str, Any]:
        """TPU目标检测 - 详细时间统计"""
        try:
            detailed_times = {}
            total_start = time.perf_counter()
            
            # 获取输入输出详情
            input_details = self.detection_interpreter.get_input_details()
            output_details = self.detection_interpreter.get_output_details()
            
            # 1. 预处理
            preprocessing_start = time.perf_counter()
            input_shape = input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            img_resized = cv2.resize(img, (width, height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            if input_details[0]['dtype'] == np.uint8:
                input_data = np.expand_dims(img_rgb.astype(np.uint8), axis=0)
            else:
                img_normalized = img_rgb.astype(np.float32) / 255.0
                input_data = np.expand_dims(img_normalized, axis=0)
            
            detailed_times['preprocessing'] = (time.perf_counter() - preprocessing_start) * 1000
            
            # 2. 数据传输
            data_transfer_start = time.perf_counter()
            self.detection_interpreter.set_tensor(input_details[0]['index'], input_data)
            detailed_times['data_transfer'] = (time.perf_counter() - data_transfer_start) * 1000
            
            # 3. 纯推理
            inference_start = time.perf_counter()
            self.detection_interpreter.invoke()
            detailed_times['inference'] = (time.perf_counter() - inference_start) * 1000
            
            # 4. 结果获取
            result_fetch_start = time.perf_counter()
            boxes = self.detection_interpreter.get_tensor(output_details[0]['index'])[0]
            classes = self.detection_interpreter.get_tensor(output_details[1]['index'])[0]
            scores = self.detection_interpreter.get_tensor(output_details[2]['index'])[0]
            detailed_times['result_fetch'] = (time.perf_counter() - result_fetch_start) * 1000
            
            # 5. 后处理
            postprocessing_start = time.perf_counter()
            detections = []
            for i in range(len(scores)):
                if scores[i] > 0.3:  # 降低置信度阈值
                    y1, x1, y2, x2 = boxes[i]
                    x1, y1, x2, y2 = int(x1 * img.shape[1]), int(y1 * img.shape[0]), int(x2 * img.shape[1]), int(y2 * img.shape[0])
                    detections.append({
                        'class': f'class_{int(classes[i])}',
                        'confidence': float(scores[i]),
                        'bbox': [x1, y1, x2, y2],
                        'method': 'edgetpu'
                    })
            detailed_times['postprocessing'] = (time.perf_counter() - postprocessing_start) * 1000
            
            # 总时间
            total_time = (time.perf_counter() - total_start) * 1000
            
            # 6. 立即再运行一遍测量理想性能（无预热开销）
            ideal_start = time.perf_counter()
            self.detection_interpreter.set_tensor(input_details[0]['index'], input_data)
            self.detection_interpreter.invoke()
            ideal_time = (time.perf_counter() - ideal_start) * 1000
            detailed_times['ideal_total'] = ideal_time
            
            # 单独测量理想推理时间
            ideal_inference_start = time.perf_counter()
            self.detection_interpreter.invoke()
            ideal_inference_time = (time.perf_counter() - ideal_inference_start) * 1000
            detailed_times['ideal_inference'] = ideal_inference_time
            
            return {
                'detections': detections,
                'total_time': total_time,
                'detailed_times': detailed_times
            }
            
        except Exception as e:
            print(f"路径3 TPU检测失败: {e}")
            return {
                'detections': [],
                'total_time': 0,
                'detailed_times': {}
            }
    
    def _tpu_object_detection(self, img: np.ndarray) -> List[Dict]:
        """TPU目标检测"""
        try:
            # 获取输入输出详情
            input_details = self.detection_interpreter.get_input_details()
            output_details = self.detection_interpreter.get_output_details()
            
            # 预处理图像
            input_shape = input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            
            # 调整图像大小并处理量化模型
            img_resized = cv2.resize(img, (width, height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # 检查模型期望的数据类型
            if input_details[0]['dtype'] == np.uint8:
                # 量化模型，使用uint8输入
                input_data = np.expand_dims(img_rgb.astype(np.uint8), axis=0)
            else:
                # 浮点模型，使用归一化输入
                img_normalized = img_rgb.astype(np.float32) / 255.0
                input_data = np.expand_dims(img_normalized, axis=0)
            
            # 运行推理
            self.detection_interpreter.set_tensor(input_details[0]['index'], input_data)
            self.detection_interpreter.invoke()
            
            # 获取输出
            boxes = self.detection_interpreter.get_tensor(output_details[0]['index'])[0]
            classes = self.detection_interpreter.get_tensor(output_details[1]['index'])[0]
            scores = self.detection_interpreter.get_tensor(output_details[2]['index'])[0]
            
            # 处理检测结果
            detections = []
            for i in range(len(scores)):
                if scores[i] > 0.3:  # 降低置信度阈值
                    y1, x1, y2, x2 = boxes[i]
                    x1, y1, x2, y2 = int(x1 * img.shape[1]), int(y1 * img.shape[0]), int(x2 * img.shape[1]), int(y2 * img.shape[0])
                    
                    detections.append({
                        'class': f'class_{int(classes[i])}',
                        'confidence': float(scores[i]),
                        'bbox': [x1, y1, x2, y2],
                        'method': 'tpu_edgetpu'
                    })
            
            return detections
        
        except Exception as e:
            print(f"路径3 TPU检测失败: {e}")
            return []
    
    def _tpu_semantic_segmentation_detailed(self, img: np.ndarray) -> Dict[str, Any]:
        """TPU语义分割 - 详细时间统计"""
        try:
            detailed_times = {}
            total_start = time.perf_counter()
            
            # 获取输入输出详情
            input_details = self.segmentation_interpreter.get_input_details()
            output_details = self.segmentation_interpreter.get_output_details()
            
            # 1. 预处理
            preprocessing_start = time.perf_counter()
            input_shape = input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            img_resized = cv2.resize(img, (width, height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            if input_details[0]['dtype'] == np.uint8:
                input_data = np.expand_dims(img_rgb.astype(np.uint8), axis=0)
            else:
                img_normalized = img_rgb.astype(np.float32) / 127.5 - 1.0
                input_data = np.expand_dims(img_normalized, axis=0)
            
            detailed_times['preprocessing'] = (time.perf_counter() - preprocessing_start) * 1000
            
            # 2. 数据传输
            data_transfer_start = time.perf_counter()
            self.segmentation_interpreter.set_tensor(input_details[0]['index'], input_data)
            detailed_times['data_transfer'] = (time.perf_counter() - data_transfer_start) * 1000
            
            # 3. 纯推理
            inference_start = time.perf_counter()
            self.segmentation_interpreter.invoke()
            detailed_times['inference'] = (time.perf_counter() - inference_start) * 1000
            
            # 4. 结果获取
            result_fetch_start = time.perf_counter()
            output_data = self.segmentation_interpreter.get_tensor(output_details[0]['index'])[0]
            detailed_times['result_fetch'] = (time.perf_counter() - result_fetch_start) * 1000
            
            # 5. 后处理
            postprocessing_start = time.perf_counter()
            # 输出已经是分割掩码，无需argmax
            segmentation_map = output_data  # output_data已经是[513,513]格式
            segmentation_resized = cv2.resize(segmentation_map.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            detailed_times['postprocessing'] = (time.perf_counter() - postprocessing_start) * 1000
            
            # 总时间
            total_time = (time.perf_counter() - total_start) * 1000
            
            # 6. 立即再运行一遍测量理想性能（无预热开销）
            ideal_start = time.perf_counter()
            self.segmentation_interpreter.set_tensor(input_details[0]['index'], input_data)
            self.segmentation_interpreter.invoke()
            ideal_time = (time.perf_counter() - ideal_start) * 1000
            detailed_times['ideal_total'] = ideal_time
            
            # 单独测量理想推理时间
            ideal_inference_start = time.perf_counter()
            self.segmentation_interpreter.invoke()
            ideal_inference_time = (time.perf_counter() - ideal_inference_start) * 1000
            detailed_times['ideal_inference'] = ideal_inference_time
            
            return {
                'segmentation': segmentation_resized,
                'total_time': total_time,
                'detailed_times': detailed_times
            }
            
        except Exception as e:
            print(f"路径3 TPU分割失败: {e}")
            return {
                'segmentation': np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8),
                'total_time': 0,
                'detailed_times': {}
            }
    
    def _tpu_semantic_segmentation(self, img: np.ndarray) -> np.ndarray:
        """TPU语义分割"""
        try:
            # 获取输入输出详情
            input_details = self.segmentation_interpreter.get_input_details()
            output_details = self.segmentation_interpreter.get_output_details()
            
            # 预处理图像
            input_shape = input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            
            # 调整图像大小并处理量化模型
            img_resized = cv2.resize(img, (width, height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # 检查模型期望的数据类型
            if input_details[0]['dtype'] == np.uint8:
                # 量化模型，使用uint8输入
                input_data = np.expand_dims(img_rgb.astype(np.uint8), axis=0)
            else:
                # 浮点模型，使用归一化输入
                img_normalized = img_rgb.astype(np.float32) / 127.5 - 1.0
                input_data = np.expand_dims(img_normalized, axis=0)
            
            # 运行推理
            self.segmentation_interpreter.set_tensor(input_details[0]['index'], input_data)
            self.segmentation_interpreter.invoke()
            
            # 获取输出
            output_data = self.segmentation_interpreter.get_tensor(output_details[0]['index'])[0]
            
            # 输出已经是分割掩码，无需argmax
            segmentation_map = output_data  # output_data已经是[513,513]格式
            
            # 调整回原始尺寸
            segmentation_resized = cv2.resize(segmentation_map.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            return segmentation_resized
            
        except Exception as e:
            print(f"路径3 TPU分割失败: {e}")
            return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    def _tpu_depth_estimation(self, img: np.ndarray) -> np.ndarray:
        """TPU深度估计（统一使用Sobel方法）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用梯度强度估计深度（与Path 1、2相同）
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 归一化到深度范围
        depth = 255 - cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return depth

class ThreePathsComparison:
    """三路径对比测试器"""
    
    def __init__(self):
        self.data_generator = SyntheticDataGenerator()
        
        # 初始化三个处理器
        self.path1_processor = Path1_AARTMixedProcessor()
        self.path2_processor = Path2_CPUTFLiteProcessor()
        self.path3_processor = Path3_TPUProcessor()
        
        self.processors = [
            self.path1_processor,
            self.path2_processor,
            self.path3_processor
        ]
    
    def run_comparison(self, num_tests: int = 3, save_results: bool = True) -> Dict[str, Any]:
        """运行三路径对比测试"""
        print("\n" + "="*60)
        print("🚀 三路径感知管道对比测试")
        print("="*60)
        
        all_results = {}
        
        for processor in self.processors:
            print(f"\n🔧 测试 {processor.name}")
            print(f"   描述: {processor.description}")
            
            path_results = {
                'processor': processor,
                'results': [],
                'avg_times': {},
                'total_detections': 0
            }
            
            # 运行多次测试
            for i in range(num_tests):
                print(f"   测试轮次 {i+1}/{num_tests}")
                
                # 生成测试数据
                test_img, test_depth = self.data_generator.generate_test_scene()
                
                # 运行处理
                try:
                    result = processor.process_image(test_img, test_depth)
                    path_results['results'].append(result)
                    path_results['total_detections'] += len(result['object_detection'])
                    
                    # 打印本轮结果
                    total_time = result['processing_times']['total']
                    detection_count = len(result['object_detection'])
                    print(f"     处理时间: {total_time:.2f}ms, 检测数量: {detection_count}")
                    
                except Exception as e:
                    print(f"     ❌ 处理失败: {e}")
                    continue
            
            # 计算平均时间
            if path_results['results']:
                avg_times = {}
                for key in path_results['results'][0]['processing_times'].keys():
                    times = [r['processing_times'][key] for r in path_results['results']]
                    avg_times[key] = sum(times) / len(times)
                path_results['avg_times'] = avg_times
            
            all_results[processor.name] = path_results
        
        # 打印对比结果
        self._print_comparison_results(all_results)
        
        # 保存结果
        if save_results:
            self._save_results(all_results)
        
        return all_results
    
    def _print_comparison_results(self, results: Dict[str, Any]):
        """打印对比结果"""
        print("\n" + "="*60)
        print("📊 三路径性能对比结果")
        print("="*60)
        
        print(f"{'路径':<20} {'总时间(ms)':<12} {'检测时间(ms)':<14} {'分割时间(ms)':<14} {'深度时间(ms)':<14} {'检测数量':<8}")
        print("-" * 95)
        
        for path_name, path_data in results.items():
            if path_data['avg_times']:
                avg_times = path_data['avg_times']
                total_time = avg_times.get('total', 0)
                detection_time = avg_times.get('object_detection', 0)
                segmentation_time = avg_times.get('semantic_segmentation', 0)
                depth_time = avg_times.get('depth_estimation', 0)
                total_detections = path_data['total_detections']
                
                print(f"{path_name:<20} {total_time:<12.2f} {detection_time:<14.2f} {segmentation_time:<14.2f} {depth_time:<14.2f} {total_detections:<8}")
        
        # 添加详细时间分解
        self._print_detailed_timing_breakdown(results)
        
        print("\n🎯 性能分析:")
        
        # 找出最快的路径
        fastest_path = min(results.items(), 
                          key=lambda x: x[1]['avg_times'].get('total', float('inf')) if x[1]['avg_times'] else float('inf'))
        
        if fastest_path[1]['avg_times']:
            print(f"   最快路径: {fastest_path[0]} ({fastest_path[1]['avg_times']['total']:.2f}ms)")
        
        # 功能对比
        print("\n🔍 功能对比:")
        print("   路径1 (AART混合)：✅ 目标检测(AARTware) ✅ 语义分割(传统CV) ✅ 深度估计(传统CV)")
        print("   路径2 (CPU TFLite)：✅ 目标检测(TFLite) ✅ 语义分割(TFLite) ✅ 深度估计(传统CV)")
        print("   路径3 (TPU)：✅ 目标检测(EdgeTPU) ✅ 语义分割(EdgeTPU) ✅ 深度估计(快速算法)")
        
        print("\n📈 推荐使用场景:")
        print("   路径1：开发原型，使用现有AARTware代码")
        print("   路径2：生产部署，CPU设备，功能完整")
        print("   路径3：高性能需求，有EdgeTPU硬件")
    
    def _print_detailed_timing_breakdown(self, results: Dict[str, Any]):
        """打印详细时间分解 - 按用户要求的格式"""
        print("\n" + "="*80)
        print("🔍 分步功能模块详细时间分解分析")
        print("="*80)
        
        # 获取各路径数据
        path1_data = results.get('路径1：AART混合方式', {})
        path2_data = results.get('路径2：全CPU TFLite', {})  
        path3_data = results.get('路径3：全TPU', {})
        
        if not (path1_data or path2_data or path3_data):
            print("⚠️  没有详细时间数据可供分析")
            return
            
        # 步骤1：目标检测
        print("\n📊 步骤1：目标检测 (SSD MobileNet V2)")
        print("="*60)
        self._print_step_comparison('object_detection', path1_data, path2_data, path3_data)
        
        # 步骤2：语义分割
        print("\n📊 步骤2：语义分割 (DeepLab V3)")
        print("="*60)
        self._print_step_comparison('semantic_segmentation', path1_data, path2_data, path3_data)
        
        # 步骤3：深度估计
        print("\n📊 步骤3：深度估计 (传统CV)")
        print("="*60)
        self._print_step_comparison('depth_estimation', path1_data, path2_data, path3_data)
        
        # 最后：总时间汇总
        self._print_final_summary(results)
    
    def _print_step_comparison(self, step_key: str, path1_data: Dict, path2_data: Dict, path3_data: Dict):
        """按步骤打印三路径对比"""
        
        # 路径1：AART混合方式
        path1_time = path1_data.get('avg_times', {}).get(step_key, 0)
        print(f"路径1 (AART YOLO+CV): {path1_time:.2f}ms")
        
        # 路径2：CPU TFLite
        path2_detailed = path2_data.get('detailed_times', {}).get(step_key, {})
        path2_total = sum(v for k, v in path2_detailed.items() if not k.startswith('ideal'))
        if path2_detailed:
            print(f"路径2 (CPU TFLite):")
            print(f"  ├── 预处理: {path2_detailed.get('preprocessing', 0):.2f}ms")
            print(f"  ├── 数据传输: {path2_detailed.get('data_transfer', 0):.2f}ms")
            print(f"  ├── 纯推理: {path2_detailed.get('inference', 0):.2f}ms")
            print(f"  ├── 结果获取: {path2_detailed.get('result_fetch', 0):.2f}ms")
            print(f"  ├── 后处理: {path2_detailed.get('postprocessing', 0):.2f}ms")
            print(f"  └── 总时间: {path2_total:.2f}ms")
        else:
            path2_time = path2_data.get('avg_times', {}).get(step_key, 0)
            print(f"路径2 (CPU TFLite): {path2_time:.2f}ms")
        
        # 路径3：EdgeTPU
        path3_detailed = path3_data.get('detailed_times', {}).get(step_key, {})
        path3_total = sum(v for k, v in path3_detailed.items() if not k.startswith('ideal'))
        if path3_detailed:
            print(f"路径3 (EdgeTPU):")
            print(f"  ├── 预处理: {path3_detailed.get('preprocessing', 0):.2f}ms")
            print(f"  ├── 数据传输: {path3_detailed.get('data_transfer', 0):.2f}ms")
            print(f"  ├── 纯推理: {path3_detailed.get('inference', 0):.2f}ms")
            print(f"  ├── 结果获取: {path3_detailed.get('result_fetch', 0):.2f}ms")
            print(f"  ├── 后处理: {path3_detailed.get('postprocessing', 0):.2f}ms")
            print(f"  ├── 总时间: {path3_total:.2f}ms")
            
            # 理想性能（热加载）
            ideal_total = path3_detailed.get('ideal_total', 0)
            ideal_inference = path3_detailed.get('ideal_inference', 0)
            if ideal_total > 0:
                print(f"  🚀 理想总时间: {ideal_total:.2f}ms")
            if ideal_inference > 0:
                print(f"  🚀 理想推理: {ideal_inference:.2f}ms")
        else:
            path3_time = path3_data.get('avg_times', {}).get(step_key, 0)
            print(f"路径3 (EdgeTPU): {path3_time:.2f}ms")
        
        # 性能对比
        if path2_total > 0 and path3_total > 0:
            speedup = path2_total / path3_total
            print(f"\n🚀 CPU vs EdgeTPU: {speedup:.2f}x 加速")
            
        # 理想性能对比
        if path3_detailed and 'ideal_total' in path3_detailed and path2_total > 0:
            ideal_speedup = path2_total / path3_detailed['ideal_total']
            print(f"🚀 理想加速比: {ideal_speedup:.2f}x")
    
    def _print_final_summary(self, results: Dict[str, Any]):
        """最终总时间汇总"""
        print("\n" + "="*80)
        print("📊 最终总时间汇总")
        print("="*80)
        
        for path_name, path_data in results.items():
            if not path_data.get('avg_times'):
                continue
                
            avg_times = path_data['avg_times']
            detection_time = avg_times.get('object_detection', 0)
            segmentation_time = avg_times.get('semantic_segmentation', 0)
            depth_time = avg_times.get('depth_estimation', 0)
            total_time = detection_time + segmentation_time + depth_time
            
            print(f"\n{path_name}:")
            print(f"  ├── 目标检测: {detection_time:.2f}ms")
            print(f"  ├── 语义分割: {segmentation_time:.2f}ms")
            print(f"  ├── 深度估计: {depth_time:.2f}ms")
            print(f"  └── 总时间: {total_time:.2f}ms")
            
            # 路径3的理想性能
            if path_name == '路径3：全TPU':
                detailed_times = path_data.get('detailed_times', {})
                ideal_detection = detailed_times.get('object_detection', {}).get('ideal_total', 0)
                ideal_segmentation = detailed_times.get('semantic_segmentation', {}).get('ideal_total', 0)
                ideal_total = ideal_detection + ideal_segmentation + depth_time
                
                if ideal_detection > 0 or ideal_segmentation > 0:
                    print(f"  🚀 理想状态:")
                    print(f"     ├── 目标检测: {ideal_detection:.2f}ms")
                    print(f"     ├── 语义分割: {ideal_segmentation:.2f}ms")
                    print(f"     ├── 深度估计: {depth_time:.2f}ms")
                    print(f"     └── 理想总时间: {ideal_total:.2f}ms")
    
    def _print_module_comparison(self, title: str, module_key: str, path1_data: Dict, path2_data: Dict, path3_data: Dict):
        """打印单个模块的对比"""
        print(f"\n{'='*80}")
        print(f"📊 {title}")
        print('='*80)
        
        # 路径1时间（总时间，无详细分解）
        path1_time = path1_data.get('avg_times', {}).get(module_key, 0)
        if path1_time > 0:
            print(f"路径1 (AARTware混合): {path1_time:.2f}ms")
        else:
            print(f"路径1 (AARTware混合): 无数据")
        
        # 路径2详细时间
        path2_detailed = path2_data.get('detailed_times', {}).get(module_key, {})
        if path2_detailed:
            print(f"\n路径2 (CPU TFLite) 详细分解:")
            print(f"  ├── 预处理: {path2_detailed.get('preprocessing', 0):.2f}ms")
            print(f"  ├── 数据传输: {path2_detailed.get('data_transfer', 0):.2f}ms")
            print(f"  ├── 纯推理: {path2_detailed.get('inference', 0):.2f}ms")
            print(f"  ├── 结果获取: {path2_detailed.get('result_fetch', 0):.2f}ms")
            print(f"  ├── 后处理: {path2_detailed.get('postprocessing', 0):.2f}ms")
            path2_total = sum(v for k, v in path2_detailed.items() if not k.startswith('ideal'))
            print(f"  └── 总计: {path2_total:.2f}ms")
        else:
            path2_time = path2_data.get('avg_times', {}).get(module_key, 0)
            if path2_time > 0:
                print(f"\n路径2 (CPU TFLite): {path2_time:.2f}ms")
                path2_total = path2_time
            else:
                print(f"\n路径2 (CPU TFLite): 无数据")
                path2_total = 0
        
        # 路径3详细时间
        path3_detailed = path3_data.get('detailed_times', {}).get(module_key, {})
        if path3_detailed:
            print(f"\n路径3 (EdgeTPU) 详细分解:")
            print(f"  ├── 预处理: {path3_detailed.get('preprocessing', 0):.2f}ms")
            print(f"  ├── 数据传输: {path3_detailed.get('data_transfer', 0):.2f}ms")
            print(f"  ├── 纯推理: {path3_detailed.get('inference', 0):.2f}ms")
            print(f"  ├── 结果获取: {path3_detailed.get('result_fetch', 0):.2f}ms")
            print(f"  ├── 后处理: {path3_detailed.get('postprocessing', 0):.2f}ms")
            path3_total = sum(v for k, v in path3_detailed.items() if not k.startswith('ideal'))
            print(f"  └── 总计: {path3_total:.2f}ms")
            
            # 理想性能
            if 'ideal_total' in path3_detailed or 'ideal_inference' in path3_detailed:
                print(f"\n路径3 (EdgeTPU) 理想性能:")
                if 'ideal_total' in path3_detailed:
                    print(f"  🚀 理想总时间: {path3_detailed.get('ideal_total', 0):.2f}ms")
                if 'ideal_inference' in path3_detailed:
                    print(f"  🚀 理想推理: {path3_detailed.get('ideal_inference', 0):.2f}ms")
        else:
            path3_time = path3_data.get('avg_times', {}).get(module_key, 0)
            if path3_time > 0:
                print(f"\n路径3 (EdgeTPU): {path3_time:.2f}ms")
                path3_total = path3_time
            else:
                print(f"\n路径3 (EdgeTPU): 无数据")
                path3_total = 0
        
        # 性能对比分析
        print(f"\n🎯 {title.split('：')[1]} 性能对比:")
        print("-" * 70)
        
        if path1_time > 0 and path2_total > 0:
            print(f"路径1 vs 路径2: {path1_time:.2f}ms vs {path2_total:.2f}ms = {path1_time/path2_total:.2f}x")
        if path1_time > 0 and path3_total > 0:
            print(f"路径1 vs 路径3: {path1_time:.2f}ms vs {path3_total:.2f}ms = {path1_time/path3_total:.2f}x")
        
        if path2_total > 0 and path3_total > 0:
            cpu_vs_tpu = path2_total / path3_total
            print(f"路径2 vs 路径3: {path2_total:.2f}ms vs {path3_total:.2f}ms = {cpu_vs_tpu:.2f}x 加速")
        
        # 理想性能对比
        if path3_detailed and 'ideal_inference' in path3_detailed and path2_detailed and 'inference' in path2_detailed:
            ideal_speedup = path2_detailed.get('inference', 0) / path3_detailed.get('ideal_inference', 1)
            print(f"🚀 CPU vs EdgeTPU理想: {path2_detailed.get('inference', 0):.2f}ms vs {path3_detailed.get('ideal_inference', 0):.2f}ms = {ideal_speedup:.2f}x 理想加速")
        
        # 如果所有数据都为0，显示提示
        if path1_time == 0 and path2_total == 0 and path3_total == 0:
            print("⚠️  该模块暂无性能数据")
    
    def _print_total_pipeline_comparison(self, results: Dict[str, Any]):
        """打印总流程对比"""
        print(f"\n{'='*80}")
        print("📊 总流程对比分析")
        print('='*80)
        
        print("流程描述:")
        print("  路径1: YOLOv8检测 → 传统CV分割 → 传统CV深度估计")
        print("  路径2: CPU TFLite检测 → CPU TFLite分割 → 传统CV深度估计") 
        print("  路径3: EdgeTPU检测 → EdgeTPU分割 → 快速深度估计")
        
        print(f"\n各路径总时间对比:")
        print("-" * 70)
        
        for path_name, path_data in results.items():
            if path_data.get('avg_time') is not None:
                avg_time = path_data['avg_time']
                std_time = path_data.get('std_time', 0)
                detections = path_data.get('total_detections', 0)
                
                print(f"{path_name}:")
                print(f"  ├── 平均时间: {avg_time:.2f}±{std_time:.2f}ms")
                print(f"  ├── 检测总数: {detections}")
                
                # 显示各步骤时间
                if 'avg_times' in path_data:
                    step_times = path_data['avg_times']
                    print(f"  ├── 目标检测: {step_times.get('object_detection', 0):.2f}ms")
                    print(f"  ├── 语义分割: {step_times.get('semantic_segmentation', 0):.2f}ms")
                    print(f"  └── 深度估计: {step_times.get('depth_estimation', 0):.2f}ms")
                
                # 理想性能汇总
                if path_name == '路径3：全TPU' and path_data.get('detailed_times'):
                    ideal_detection = path_data['detailed_times'].get('object_detection', {}).get('ideal_total', 0)
                    ideal_segmentation = path_data['detailed_times'].get('semantic_segmentation', {}).get('ideal_total', 0)
                    ideal_depth = step_times.get('depth_estimation', 0)  # 深度估计没有理想版本
                    ideal_total = ideal_detection + ideal_segmentation + ideal_depth
                    
                    if ideal_total > 0:
                        print(f"  🚀 理想总时间: {ideal_total:.2f}ms (检测{ideal_detection:.2f} + 分割{ideal_segmentation:.2f} + 深度{ideal_depth:.2f})")
        
        # 最终性能对比  
        path_times = {}
        ideal_time = 0
        
        for path_name, path_data in results.items():
            if path_data.get('avg_time') is not None:
                path_times[path_name] = path_data['avg_time']
                
                # 计算理想时间
                if path_name == '路径3：全TPU':
                    detailed = path_data.get('detailed_times', {})
                    ideal_detection = detailed.get('object_detection', {}).get('ideal_total', 0)
                    ideal_segmentation = detailed.get('semantic_segmentation', {}).get('ideal_total', 0)
                    ideal_depth = path_data.get('avg_times', {}).get('depth_estimation', 0)
                    ideal_time = ideal_detection + ideal_segmentation + ideal_depth
        
        print(f"\n🎯 总体性能排名:")
        print("-" * 70)
        sorted_paths = sorted([(name, time) for name, time in path_times.items()], key=lambda x: x[1])
        
        for i, (path_name, time) in enumerate(sorted_paths, 1):
            print(f"{i}. {path_name}: {time:.2f}ms")
        
        if ideal_time > 0:
            print(f"🚀 理想EdgeTPU: {ideal_time:.2f}ms")
        
        # 加速比对比
        if len(path_times) >= 2:
            print(f"\n🚀 加速比分析:")
            print("-" * 70)
            slowest_time = max(path_times.values())
            for path_name, time in path_times.items():
                speedup = slowest_time / time
                print(f"{path_name} 相对最慢路径: {speedup:.2f}x")
            
            if ideal_time > 0:
                ideal_speedup = slowest_time / ideal_time
                print(f"🚀 理想EdgeTPU 相对最慢路径: {ideal_speedup:.2f}x")
    
    def _save_results(self, results: Dict[str, Any]):
        """保存结果到文件"""
        timestamp = int(time.time())
        filename = f"/ros2_ws/three_paths_comparison_{timestamp}.json"
        
        # 准备可序列化的数据
        serializable_results = {}
        for path_name, path_data in results.items():
            serializable_results[path_name] = {
                'description': path_data['processor'].description,
                'avg_times': path_data['avg_times'],
                'total_detections': path_data['total_detections'],
                'num_tests': len(path_data['results'])
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存到: {filename}")

def test_individual_path(path_name: str, num_tests: int = 3):
    """单独测试一个路径，避免内存问题"""
    print(f"\n🚀 开始测试 {path_name}")
    print("="*40)
    
    try:
        # 只初始化指定路径的处理器
        if path_name == "路径1":
            processor = Path1_AARTMixedProcessor()
        elif path_name == "路径2": 
            processor = Path2_CPUTFLiteProcessor()
        elif path_name == "路径3":
            processor = Path3_TPUProcessor()
        else:
            print(f"❌ 未知路径: {path_name}")
            return None
        
        print(f"✅ {path_name} ({processor.description}) 初始化完成")
        
        # 创建测试数据
        data_generator = SyntheticDataGenerator()
        
        # 运行测试
        results = []
        times = []
        
        for i in range(num_tests):
            print(f"   运行第 {i+1}/{num_tests} 次测试...")
            
            # 生成测试数据
            test_img, test_depth = data_generator.generate_test_scene()
            
            start_time = time.perf_counter()
            result = processor.process_image(test_img, test_depth)
            end_time = time.perf_counter()
            
            processing_time = (end_time - start_time) * 1000
            times.append(processing_time)
            results.append(result)
            
            detection_count = len(result['object_detection']) if 'object_detection' in result else 0
            print(f"   第{i+1}次: {processing_time:.2f}ms, 检测到 {detection_count} 个物体")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        total_detections = sum(len(r.get('object_detection', [])) for r in results)
        
        # 计算详细时间的平均值
        detailed_times = {}
        if results and 'detailed_times' in results[0]:
            for task in ['object_detection', 'semantic_segmentation', 'depth_estimation']:
                if task in results[0]['detailed_times']:
                    detailed_times[task] = {}
                    for timing_type in ['preprocessing', 'data_transfer', 'inference', 'result_fetch', 'postprocessing', 'ideal_total', 'ideal_inference']:
                        timing_values = [r['detailed_times'][task].get(timing_type, 0) for r in results if r.get('detailed_times', {}).get(task, {}).get(timing_type, 0) > 0]
                        if timing_values:
                            detailed_times[task][timing_type] = np.mean(timing_values)
        
        # 对于路径1，计算processing_times的平均值
        processing_times = {}
        if results and 'processing_times' in results[0]:
            for task in ['object_detection', 'semantic_segmentation', 'depth_estimation']:
                if task in results[0]['processing_times']:
                    timing_values = [r['processing_times'][task] for r in results if r.get('processing_times', {}).get(task, 0) > 0]
                    if timing_values:
                        processing_times[task] = np.mean(timing_values)
        
        summary = {
            'path_name': path_name,
            'description': processor.description,
            'times': times,
            'avg_time': avg_time,
            'std_time': std_time,
            'total_detections': total_detections,
            'num_tests': num_tests,
            'detailed_times': detailed_times,
            'processing_times': processing_times
        }
        
        print(f"✅ {path_name} 完成 - 平均耗时: {avg_time:.2f}±{std_time:.2f}ms")
        print(f"   总检测数: {total_detections}")
        
        # 清理内存
        del processor
        del data_generator
        import gc
        gc.collect()
        
        return summary
        
    except Exception as e:
        print(f"❌ {path_name} 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数 - 分步测试模式"""
    print("🎯 三路径感知管道对比测试 (内存友好模式)")
    print("="*50)
    
    # 检查依赖
    print("📋 检查依赖:")
    print(f"   YOLOv8: {'✅' if YOLO_AVAILABLE else '❌'}")
    print(f"   TFLite: {'✅' if TFLITE_AVAILABLE else '❌'}")
    print(f"   EdgeTPU: {'✅' if TPU_AVAILABLE else '❌'}")
    
    # 分步测试每个路径
    test_paths = ["路径2", "路径3", "路径1"]  # 先测试轻量级的
    results = {}
    
    for path_name in test_paths:
        result = test_individual_path(path_name, num_tests=3)
        if result:
            results[path_name] = result
        
        # 每次测试后等待内存回收
        print("🕐 等待内存回收...")
        time.sleep(2)
    
    # 按用户要求的格式输出
    if len(results) > 1:
        print("\n" + "="*50)
        print("测试结果")
        print("="*50)
        
        # 步骤1：检测
        print("\n步骤1：检测")
        print("-" * 30)
        
        if "路径1" in results:
            # 路径1使用processing_times结构
            if 'processing_times' in results["路径1"]:
                detection_time = results["路径1"]['processing_times'].get('object_detection', 0)
                print(f"路径1(AART yolo+cv): {detection_time:.2f}ms")
            else:
                print("路径1(AART yolo+cv): 数据不可用")
        
        if "路径2" in results:
            path2_detection = results["路径2"].get('detailed_times', {}).get('object_detection', {})
            if path2_detection:
                preprocessing = path2_detection.get('preprocessing', 0)
                data_transfer = path2_detection.get('data_transfer', 0)
                inference = path2_detection.get('inference', 0)
                result_fetch = path2_detection.get('result_fetch', 0)
                postprocessing = path2_detection.get('postprocessing', 0)
                total = preprocessing + data_transfer + inference + result_fetch + postprocessing
                print(f"路径2(cpu+tflite): {total:.2f}ms")
            else:
                print("路径2(cpu+tflite): 数据不可用")
        
        if "路径3" in results:
            path3_detection = results["路径3"].get('detailed_times', {}).get('object_detection', {})
            if path3_detection:
                preprocessing = path3_detection.get('preprocessing', 0)
                data_transfer = path3_detection.get('data_transfer', 0)
                inference = path3_detection.get('inference', 0)
                result_fetch = path3_detection.get('result_fetch', 0)
                postprocessing = path3_detection.get('postprocessing', 0)
                total = preprocessing + data_transfer + inference + result_fetch + postprocessing
                
                ideal_inference = path3_detection.get('ideal_inference', 0)
                ideal_total = preprocessing + data_transfer + ideal_inference + result_fetch + postprocessing
                
                print(f"路径3(tpu+tflite): 预处理{preprocessing:.2f}ms + 数据传输{data_transfer:.2f}ms + 推理{inference:.2f}ms(不需要加载{ideal_inference:.2f}ms) + 结果获取{result_fetch:.2f}ms + 后处理{postprocessing:.2f}ms = {total:.2f}ms/不需要加载{ideal_total:.2f}ms")
            else:
                print("路径3(tpu+tflite): 数据不可用")
        
        # 步骤2：语义分割
        print("\n步骤2：语义分割")
        print("-" * 30)
        
        if "路径1" in results:
            # 路径1使用processing_times结构
            if 'processing_times' in results["路径1"]:
                segmentation_time = results["路径1"]['processing_times'].get('semantic_segmentation', 0)
                print(f"路径1(AART yolo+cv): {segmentation_time:.2f}ms")
            else:
                print("路径1(AART yolo+cv): 数据不可用")
        
        if "路径2" in results:
            path2_segmentation = results["路径2"].get('detailed_times', {}).get('semantic_segmentation', {})
            if path2_segmentation:
                preprocessing = path2_segmentation.get('preprocessing', 0)
                data_transfer = path2_segmentation.get('data_transfer', 0)
                inference = path2_segmentation.get('inference', 0)
                result_fetch = path2_segmentation.get('result_fetch', 0)
                postprocessing = path2_segmentation.get('postprocessing', 0)
                total = preprocessing + data_transfer + inference + result_fetch + postprocessing
                print(f"路径2(cpu+tflite): {total:.2f}ms")
            else:
                print("路径2(cpu+tflite): 数据不可用")
        
        if "路径3" in results:
            path3_segmentation = results["路径3"].get('detailed_times', {}).get('semantic_segmentation', {})
            if path3_segmentation:
                preprocessing = path3_segmentation.get('preprocessing', 0)
                data_transfer = path3_segmentation.get('data_transfer', 0)
                inference = path3_segmentation.get('inference', 0)
                result_fetch = path3_segmentation.get('result_fetch', 0)
                postprocessing = path3_segmentation.get('postprocessing', 0)
                total = preprocessing + data_transfer + inference + result_fetch + postprocessing
                
                ideal_inference = path3_segmentation.get('ideal_inference', 0)
                ideal_total = preprocessing + data_transfer + ideal_inference + result_fetch + postprocessing
                
                print(f"路径3(tpu+tflite): 预处理{preprocessing:.2f}ms + 数据传输{data_transfer:.2f}ms + 推理{inference:.2f}ms(不需要加载{ideal_inference:.2f}ms) + 结果获取{result_fetch:.2f}ms + 后处理{postprocessing:.2f}ms = {total:.2f}ms/不需要加载{ideal_total:.2f}ms")
            else:
                print("路径3(tpu+tflite): 数据不可用")
        
        # 步骤3：深度估计（三种路径都使用相同的传统CV方法）
        print("\n步骤3：深度估计")
        print("-" * 30)
        
        if "路径1" in results:
            # 路径1使用processing_times结构
            if 'processing_times' in results["路径1"]:
                depth_time = results["路径1"]['processing_times'].get('depth_estimation', 0)
                print(f"路径1(AART yolo+cv): {depth_time:.2f}ms")
            else:
                print("路径1(AART yolo+cv): 数据不可用")
        
        if "路径2" in results:
            # 路径2使用processing_times结构（深度估计不使用TFLite）
            if 'processing_times' in results["路径2"]:
                depth_time = results["路径2"]['processing_times'].get('depth_estimation', 0)
                print(f"路径2(cpu+tflite): {depth_time:.2f}ms")
            else:
                print("路径2(cpu+tflite): 数据不可用")
        
        if "路径3" in results:
            # 路径3使用processing_times结构（深度估计不使用TPU）
            if 'processing_times' in results["路径3"]:
                depth_time = results["路径3"]['processing_times'].get('depth_estimation', 0)
                print(f"路径3(tpu+tflite): {depth_time:.2f}ms")
            else:
                print("路径3(tpu+tflite): 数据不可用")
        
        # 最后总时间
        print("\n最后总时间")
        print("-" * 30)
        
        if "路径1" in results:
            # 路径1使用processing_times结构
            if 'processing_times' in results["路径1"]:
                detection_time = results["路径1"]['processing_times'].get('object_detection', 0)
                segmentation_time = results["路径1"]['processing_times'].get('semantic_segmentation', 0)
                depth_time = results["路径1"]['processing_times'].get('depth_estimation', 0)
                total_time = detection_time + segmentation_time + depth_time
                print(f"路径1(AART yolo+cv): 检测{detection_time:.2f}ms + 语义分割{segmentation_time:.2f}ms + 深度估计{depth_time:.2f}ms = {total_time:.2f}ms")
            else:
                print("路径1(AART yolo+cv): 数据不可用")
        
        if "路径2" in results:
            path2_detection = results["路径2"].get('detailed_times', {}).get('object_detection', {})
            path2_segmentation = results["路径2"].get('detailed_times', {}).get('semantic_segmentation', {})
            
            detection_time = sum(path2_detection.values()) if path2_detection else 0
            segmentation_time = sum(path2_segmentation.values()) if path2_segmentation else 0
            depth_time = results["路径2"].get('processing_times', {}).get('depth_estimation', 0)
            total_time = detection_time + segmentation_time + depth_time
            
            print(f"路径2(cpu+tflite): 检测{detection_time:.2f}ms + 语义分割{segmentation_time:.2f}ms + 深度估计{depth_time:.2f}ms = {total_time:.2f}ms")
        
        if "路径3" in results:
            path3_detection = results["路径3"].get('detailed_times', {}).get('object_detection', {})
            path3_segmentation = results["路径3"].get('detailed_times', {}).get('semantic_segmentation', {})
            
            detection_time = sum(v for k, v in path3_detection.items() if not k.startswith('ideal')) if path3_detection else 0
            segmentation_time = sum(v for k, v in path3_segmentation.items() if not k.startswith('ideal')) if path3_segmentation else 0
            depth_time = results["路径3"].get('processing_times', {}).get('depth_estimation', 0)
            total_time = detection_time + segmentation_time + depth_time
            
            # 理想状态时间
            ideal_detection = (path3_detection.get('preprocessing', 0) + path3_detection.get('data_transfer', 0) + 
                              path3_detection.get('ideal_inference', 0) + path3_detection.get('result_fetch', 0) + 
                              path3_detection.get('postprocessing', 0)) if path3_detection else 0
            ideal_segmentation = (path3_segmentation.get('preprocessing', 0) + path3_segmentation.get('data_transfer', 0) + 
                                 path3_segmentation.get('ideal_inference', 0) + path3_segmentation.get('result_fetch', 0) + 
                                 path3_segmentation.get('postprocessing', 0)) if path3_segmentation else 0
            ideal_depth = depth_time  # 深度估计使用相同的传统CV方法，没有理想状态区别
            ideal_total = ideal_detection + ideal_segmentation + ideal_depth
            
            print(f"路径3(tpu+tflite): 检测{detection_time:.2f}ms + 语义分割{segmentation_time:.2f}ms + 深度估计{depth_time:.2f}ms = {total_time:.2f}ms, 理想状态: 检测{ideal_detection:.2f}ms + 语义分割{ideal_segmentation:.2f}ms + 深度估计{ideal_depth:.2f}ms = {ideal_total:.2f}ms")
    
    # 保存结果
    if results:
        timestamp = int(time.time())
        filename = f"/ros2_ws/three_paths_comparison_{timestamp}.json"
        
        # 准备可序列化的数据
        serializable_results = {}
        for path_name, data in results.items():
            serializable_results[path_name] = {
                'description': data['description'],
                'times': [float(t) for t in data['times']],
                'avg_time': float(data['avg_time']),
                'std_time': float(data['std_time']),
                'total_detections': data['total_detections'],
                'num_tests': data['num_tests']
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存到: {filename}")
    
    print("\n✅ 所有测试完成！")

if __name__ == "__main__":
    main() 