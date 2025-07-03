#!/usr/bin/env python3
"""
TPU推理工具类 - 封装EdgeTPU推理功能
支持深度估计、物体检测、语义分割
"""

import numpy as np
import cv2
import time
import os
from typing import Tuple, List, Dict, Any

try:
    from pycoral.utils import edgetpu
    from pycoral.utils import dataset
    from pycoral.adapters import common
    from pycoral.adapters import detect
    from pycoral.adapters import segment
    import tflite_runtime.interpreter as tflite
    TPU_AVAILABLE = True
except ImportError:
    print("警告: pycoral未安装，TPU推理将回退到CPU")
    TPU_AVAILABLE = False

class TPUInferenceEngine:
    """EdgeTPU推理引擎"""
    
    def __init__(self, model_path: str, model_type: str):
        """
        初始化TPU推理引擎
        
        Args:
            model_path: 模型文件路径
            model_type: 模型类型 ('depth', 'detection', 'segmentation')
        """
        self.model_path = model_path
        self.model_type = model_type
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = None
        self.is_tpu_available = TPU_AVAILABLE
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型"""
        try:
            if self.is_tpu_available and os.path.exists(self.model_path):
                # 尝试使用EdgeTPU
                self.interpreter = tflite.Interpreter(
                    model_path=self.model_path,
                    experimental_delegates=[edgetpu.make_edgetpu_delegate()]
                )
                print(f"✅ EdgeTPU已加载: {self.model_path}")
            else:
                # 回退到CPU
                self.interpreter = tflite.Interpreter(model_path=self.model_path)
                print(f"⚠️ 使用CPU推理: {self.model_path}")
                
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape']
            
            print(f"模型输入形状: {self.input_shape}")
            print(f"模型输出数量: {len(self.output_details)}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.interpreter = None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理输入图像"""
        if self.interpreter is None:
            return None
            
        # 获取目标尺寸
        if len(self.input_shape) == 4:  # [batch, height, width, channels]
            target_height = self.input_shape[1]
            target_width = self.input_shape[2]
        else:
            raise ValueError(f"不支持的输入形状: {self.input_shape}")
        
        # 缩放图像
        resized = cv2.resize(image, (target_width, target_height))
        
        # 归一化处理
        if self.input_details[0]['dtype'] == np.uint8:
            # 量化模型，保持0-255范围
            preprocessed = resized.astype(np.uint8)
        else:
            # 浮点模型，归一化到0-1
            preprocessed = resized.astype(np.float32) / 255.0
        
        # 添加batch维度
        return np.expand_dims(preprocessed, axis=0)
    
    def inference(self, input_data: np.ndarray) -> List[np.ndarray]:
        """执行推理"""
        if self.interpreter is None:
            return None
            
        start_time = time.time()
        
        # 设置输入
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # 执行推理
        self.interpreter.invoke()
        
        # 获取输出
        outputs = []
        for output_detail in self.output_details:
            output_data = self.interpreter.get_tensor(output_detail['index'])
            outputs.append(output_data)
        
        inference_time = (time.time() - start_time) * 1000
        print(f"TPU推理时间: {inference_time:.2f}ms")
        
        return outputs

class TPUDepthEstimator:
    """TPU深度估计器"""
    
    def __init__(self, model_path: str):
        self.engine = TPUInferenceEngine(model_path, 'depth')
        self.depth_scale = 1000.0  # 深度值缩放因子
    
    def estimate_depth(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """估计深度图"""
        start_time = time.time()
        
        # 预处理
        input_data = self.engine.preprocess_image(image)
        if input_data is None:
            return None, 0
        
        # 推理
        outputs = self.engine.inference(input_data)
        if outputs is None:
            return None, 0
        
        # 后处理
        depth_map = outputs[0][0]  # 移除batch维度
        
        # 如果输出是单通道
        if len(depth_map.shape) == 3 and depth_map.shape[2] == 1:
            depth_map = depth_map[:, :, 0]
        
        # 缩放到原始图像尺寸
        if depth_map.shape != image.shape[:2]:
            depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))
        
        # 归一化深度值
        depth_map = (depth_map * self.depth_scale).astype(np.float32)
        
        processing_time = (time.time() - start_time) * 1000
        return depth_map, processing_time

class TPUObjectDetector:
    """TPU物体检测器"""
    
    def __init__(self, model_path: str):
        self.engine = TPUInferenceEngine(model_path, 'detection')
        # COCO类别名称
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def detect_objects(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[List[Dict], float]:
        """检测物体"""
        start_time = time.time()
        
        # 预处理
        input_data = self.engine.preprocess_image(image)
        if input_data is None:
            return [], 0
        
        # 推理
        outputs = self.engine.inference(input_data)
        if outputs is None:
            return [], 0
        
        # 后处理 - SSD MobileNet输出格式
        # outputs[0]: 检测框 [1, 10, 4]
        # outputs[1]: 类别 [1, 10]  
        # outputs[2]: 置信度 [1, 10]
        # outputs[3]: 检测数量 [1]
        
        detections = []
        num_detections = int(outputs[3][0])
        
        img_height, img_width = image.shape[:2]
        
        for i in range(min(num_detections, 10)):  # 最多10个检测
            confidence = outputs[2][0][i]
            if confidence >= confidence_threshold:
                # 边界框坐标 (归一化)
                y1, x1, y2, x2 = outputs[0][0][i]
                
                # 转换到像素坐标
                x1 = int(x1 * img_width)
                y1 = int(y1 * img_height)
                x2 = int(x2 * img_width)
                y2 = int(y2 * img_height)
                
                # 类别ID
                class_id = int(outputs[1][0][i])
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confidence),
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        processing_time = (time.time() - start_time) * 1000
        return detections, processing_time

class TPUSemanticSegmentor:
    """TPU语义分割器"""
    
    def __init__(self, model_path: str):
        self.engine = TPUInferenceEngine(model_path, 'segmentation')
        # PASCAL VOC类别名称
        self.class_names = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor'
        ]
        
        # 类别颜色映射
        self.class_colors = self._generate_colors()
    
    def _generate_colors(self):
        """生成类别颜色"""
        colors = []
        np.random.seed(42)
        for i in range(len(self.class_names)):
            color = np.random.randint(0, 255, 3).tolist()
            colors.append(color)
        return colors
    
    def segment_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """语义分割"""
        start_time = time.time()
        
        # 预处理
        input_data = self.engine.preprocess_image(image)
        if input_data is None:
            return None, None, 0
        
        # 推理
        outputs = self.engine.inference(input_data)
        if outputs is None:
            return None, None, 0
        
        # 后处理
        # DeepLabV3输出: [1, 513, 513, 21] - 每个像素的类别概率
        segmentation_logits = outputs[0][0]  # 移除batch维度
        
        # 获取每个像素的最大概率类别
        segmentation_map = np.argmax(segmentation_logits, axis=2).astype(np.uint8)
        
        # 缩放到原始图像尺寸
        if segmentation_map.shape != image.shape[:2]:
            segmentation_map = cv2.resize(
                segmentation_map, (image.shape[1], image.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        
        # 创建彩色分割图
        color_map = self._create_color_map(segmentation_map)
        
        processing_time = (time.time() - start_time) * 1000
        return segmentation_map, color_map, processing_time
    
    def _create_color_map(self, segmentation_map: np.ndarray) -> np.ndarray:
        """创建彩色分割图"""
        height, width = segmentation_map.shape
        color_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        for class_id in range(len(self.class_names)):
            mask = (segmentation_map == class_id)
            color_map[mask] = self.class_colors[class_id]
        
        return color_map 