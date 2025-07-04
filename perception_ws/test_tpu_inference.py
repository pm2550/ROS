#!/usr/bin/env python3
"""
TPU推理测试脚本
测试三个EdgeTPU模型的推理性能
"""

import numpy as np
import cv2
import time
import os
import sys

# 添加模块路径
sys.path.append('src/perception_pipeline')

try:
    from pycoral.utils import edgetpu
    from pycoral.adapters import common
    import tflite_runtime.interpreter as tflite
    TPU_AVAILABLE = True
    print("✅ TPU库导入成功")
except ImportError as e:
    TPU_AVAILABLE = False
    print(f"❌ TPU库导入失败: {e}")
    sys.exit(1)

def load_tpu_model(model_path):
    """加载TPU模型"""
    try:
        # 尝试使用EdgeTPU
        interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[edgetpu.make_edgetpu_delegate()]
        )
        print(f"✅ EdgeTPU模式加载: {os.path.basename(model_path)}")
    except:
        # 回退到CPU
        interpreter = tflite.Interpreter(model_path=model_path)
        print(f"⚠️ CPU模式加载: {os.path.basename(model_path)}")
    
    interpreter.allocate_tensors()
    return interpreter

def test_depth_model():
    """测试深度估计模型"""
    print("\n🔍 测试深度估计模型...")
    model_path = "src/models/fastdepth_quantized_edgetpu.tflite"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    interpreter = load_tpu_model(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"输入形状: {input_details[0]['shape']}")
    print(f"输出形状: {output_details[0]['shape']}")
    
    # 创建测试图像
    input_shape = input_details[0]['shape']
    test_image = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    
    # 推理测试
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], test_image)
    interpreter.invoke()
    depth_output = interpreter.get_tensor(output_details[0]['index'])
    inference_time = (time.time() - start_time) * 1000
    
    print(f"⚡ 深度估计推理时间: {inference_time:.2f}ms")
    print(f"✅ 输出深度图形状: {depth_output.shape}")

def test_detection_model():
    """测试物体检测模型"""
    print("\n🎯 测试物体检测模型...")
    model_path = "src/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    interpreter = load_tpu_model(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"输入形状: {input_details[0]['shape']}")
    print(f"输出数量: {len(output_details)}")
    
    # 创建测试图像
    input_shape = input_details[0]['shape']
    test_image = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    
    # 推理测试
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], test_image)
    interpreter.invoke()
    
    # 获取所有输出
    outputs = []
    for i, output_detail in enumerate(output_details):
        output = interpreter.get_tensor(output_detail['index'])
        outputs.append(output)
        print(f"输出 {i} 形状: {output.shape}")
    
    inference_time = (time.time() - start_time) * 1000
    print(f"⚡ 物体检测推理时间: {inference_time:.2f}ms")

def test_segmentation_model():
    """测试语义分割模型"""
    print("\n🎨 测试语义分割模型...")
    model_path = "src/models/deeplabv3_mnv2_pascal_quant_edgetpu.tflite"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    interpreter = load_tpu_model(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"输入形状: {input_details[0]['shape']}")
    print(f"输出形状: {output_details[0]['shape']}")
    
    # 创建测试图像
    input_shape = input_details[0]['shape']
    test_image = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    
    # 推理测试
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], test_image)
    interpreter.invoke()
    segmentation_output = interpreter.get_tensor(output_details[0]['index'])
    inference_time = (time.time() - start_time) * 1000
    
    print(f"⚡ 语义分割推理时间: {inference_time:.2f}ms")
    print(f"✅ 输出分割图形状: {segmentation_output.shape}")
    
    # 分析分割结果
    seg_map = np.argmax(segmentation_output[0], axis=2)
    unique_classes = np.unique(seg_map)
    print(f"🏷️ 检测到 {len(unique_classes)} 个类别: {unique_classes}")

def test_real_image_inference():
    """使用真实图像进行端到端测试"""
    print("\n🖼️ 真实图像端到端测试...")
    
    # 创建一个640x480的测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 添加一些结构
    cv2.rectangle(test_image, (100, 100), (300, 200), (255, 0, 0), -1)  # 蓝色矩形
    cv2.circle(test_image, (400, 300), 50, (0, 255, 0), -1)  # 绿色圆形
    cv2.rectangle(test_image, (200, 300), (500, 400), (0, 0, 255), 3)  # 红色框
    
    print(f"测试图像形状: {test_image.shape}")
    
    # 测试每个模型处理相同图像的时间
    models = [
        ("深度估计", "src/models/fastdepth_quantized_edgetpu.tflite", (480, 640, 3)),
        ("物体检测", "src/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite", (300, 300, 3)),
        ("语义分割", "src/models/deeplabv3_mnv2_pascal_quant_edgetpu.tflite", (513, 513, 3))
    ]
    
    total_time = 0
    
    for model_name, model_path, input_size in models:
        if not os.path.exists(model_path):
            print(f"❌ 跳过 {model_name}: 模型文件不存在")
            continue
        
        # 缩放图像到模型输入尺寸
        resized_image = cv2.resize(test_image, (input_size[1], input_size[0]))
        resized_image = np.expand_dims(resized_image, axis=0).astype(np.uint8)
        
        # 加载模型并推理
        interpreter = load_tpu_model(model_path)
        input_details = interpreter.get_input_details()
        
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], resized_image)
        interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000
        
        total_time += inference_time
        print(f"⚡ {model_name}: {inference_time:.2f}ms")
    
    print(f"\n🚀 总推理时间: {total_time:.2f}ms")
    print(f"📊 理论FPS: {1000/total_time:.1f}")

def main():
    """主函数"""
    print("🚀 EdgeTPU推理性能测试")
    print("=" * 50)
    
    # 检查TPU设备
    devices = edgetpu.list_edge_tpus()
    if devices:
        print(f"✅ 检测到 {len(devices)} 个EdgeTPU设备")
        for i, device in enumerate(devices):
            print(f"   设备 {i}: {device}")
    else:
        print("⚠️ 未检测到EdgeTPU设备，将使用CPU推理")
    
    # 逐个测试模型
    try:
        test_depth_model()
        test_detection_model()
        test_segmentation_model()
        test_real_image_inference()
        
        print("\n🎉 所有测试完成！")
        print("💡 提示: 如果看到CPU模式，请连接EdgeTPU设备以获得更好性能")
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 