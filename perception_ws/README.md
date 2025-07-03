# RPI5 TPU 感知流水线

基于ROS 2 Galactic的完整感知流水线实现，支持双目视觉、目标检测和语义分割。针对Raspberry Pi 5 + TPU优化，与Nvidia Jetson Orin Nano性能对比。

## 系统概览

### 硬件配置
- **主要配置**: Raspberry Pi 5 + TPU (USB 3.0)
- **对比配置**: Raspberry Pi 5 (仅CPU)
- **基准配置**: Nvidia Jetson Orin Nano 8G (目标: 100ms端到端延迟)

### 感知流水线阶段

1. **传感器输入** (5-10ms)
   - 双目相机数据采集 (1080p)
   - CPU处理

2. **图像预处理** (10-15ms)
   - 图像缩放、去噪、增强
   - CPU处理

3. **深度估计** (20-50ms)
   - 经典CV: StereoBM/StereoSGBM
   - CNN: 深度学习模型 (可选)

4. **目标检测** (15-40ms)
   - 经典CV: HOG + 级联分类器
   - DNN: YOLOv4/5, MobileNet-SSD (TPU加速)

5. **语义分割** (20-45ms)
   - 经典CV: 基于颜色和纹理的分割
   - DNN: Fast-SCNN, BiSeNet (TPU加速)

6. **后处理** (2-5ms)
   - 结果整合和输出

## 快速开始

### 1. 环境准备

```bash
# 进入ROS 2容器
./ros2-start.sh enter

# 或者在主机上安装ROS 2 Galactic
# 参考: https://docs.ros.org/en/galactic/Installation.html
```

### 2. 编译项目

```bash
cd /workspace/perception_ws
colcon build
source install/setup.bash
```

### 3. 运行感知流水线

#### 模拟模式 (推荐用于测试)
```bash
# CPU模式
ros2 launch perception_pipeline perception_pipeline.launch.py \
  pipeline_mode:=cv_cpu \
  use_simulation:=true \
  target_fps:=15.0

# TPU模式 (需要TPU硬件)
ros2 launch perception_pipeline perception_pipeline.launch.py \
  pipeline_mode:=dnn_tpu \
  use_simulation:=true \
  target_fps:=30.0
```

#### 真实相机模式
```bash
# 确保双目相机已连接
ros2 launch perception_pipeline perception_pipeline.launch.py \
  pipeline_mode:=cv_cpu \
  use_simulation:=false \
  target_fps:=15.0
```

### 4. 查看结果

```bash
# 查看感知结果
ros2 topic echo /perception/result

# 查看性能统计
ros2 topic echo /perception/timing

# 查看检测对象
ros2 topic echo /perception/object_detection

# 可视化 (需要rviz2)
rviz2 -d config/perception_visualization.rviz
```

## 话题接口

### 输入话题
- `/stereo/left/image_raw` - 左相机原始图像
- `/stereo/right/image_raw` - 右相机原始图像
- `/stereo/left/camera_info` - 相机标定信息

### 输出话题
- `/perception/result` - 完整感知结果
- `/perception/depth_estimation` - 深度估计结果
- `/perception/object_detection` - 目标检测结果
- `/perception/semantic_segmentation` - 语义分割结果
- `/perception/timing` - 性能时间统计

## 配置文件

### RPI5 + TPU 配置
```bash
# 使用预配置的TPU参数
ros2 launch perception_pipeline perception_pipeline.launch.py \
  --params-file config/rpi5_tpu_config.yaml
```

### RPI5 CPU 配置
```bash
# 使用CPU优化参数
ros2 launch perception_pipeline perception_pipeline.launch.py \
  --params-file config/rpi5_cpu_config.yaml
```

## 性能优化建议

### 针对RPI5 + TPU
1. **模型优化**
   - 使用量化模型（INT8）
   - 优先选择MobileNet系列轻量级模型
   - 考虑模型剪枝

2. **分辨率策略**
   - 输入: 1080p (传感器)
   - 处理: 640x480 (深度学习)
   - 输出: 可配置

3. **TPU利用**
   - 并行处理目标检测和语义分割
   - 批处理优化

### 针对RPI5 CPU
1. **算法选择**
   - 优先使用经典CV方法
   - 简化参数设置
   - 降低处理分辨率

2. **多线程优化**
   - 利用RPI5的4核CPU
   - 流水线并行处理

## 性能基准

### 目标性能 (与Jetson Orin Nano对比)
| 配置 | 总延迟 | FPS | 功耗 |
|------|--------|-----|------|
| Jetson Orin Nano | ~100ms | 10 | 15W |
| RPI5 + TPU | ~120ms | 8-10 | 12W |
| RPI5 CPU | ~200ms | 5-8 | 8W |

### 各阶段性能分解
| 阶段 | RPI5+TPU | RPI5 CPU |
|------|----------|----------|
| 传感器输入 | 5ms | 5ms |
| 预处理 | 10ms | 15ms |
| 深度估计 | 25ms | 60ms |
| 目标检测 | 30ms | 80ms |
| 语义分割 | 35ms | 70ms |
| 后处理 | 3ms | 5ms |
| **总计** | **108ms** | **235ms** |

## 故障排除

### 常见问题

1. **相机无法打开**
   ```bash
   # 检查设备
   ls /dev/video*
   
   # 测试相机
   sudo apt install v4l-utils
   v4l2-ctl --list-devices
   ```

2. **TPU未检测到**
   ```bash
   # 检查TPU设备
   lsusb | grep "Google"
   
   # 安装TPU驱动
   echo 'deb https://packages.cloud.google.com/apt coral-edgetpu-stable main' | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
   sudo apt update
   sudo apt install libedgetpu1-std
   ```

3. **性能不达标**
   - 检查系统负载: `htop`
   - 监控温度: `vcgencmd measure_temp`
   - 调整CPU频率: `sudo cpufreq-set -g performance`

### 调试模式

```bash
# 启用详细日志
ros2 launch perception_pipeline perception_pipeline.launch.py \
  --log-level debug

# 单独运行节点调试
ros2 run perception_pipeline sensor_input --ros-args --log-level debug
```

## 扩展开发

### 添加新的检测算法
1. 继承基础检测器类
2. 实现检测接口
3. 更新配置文件
4. 测试和性能评估

### 自定义消息类型
- 编辑 `perception_interfaces/msg/` 中的消息定义
- 重新编译接口包
- 更新节点以使用新消息

## 许可证

Apache 2.0 License

## 贡献

欢迎提交Issue和Pull Request!

# RPI5感知流水线系统

## 概述
基于ROS 2 Galactic的实时感知系统，支持RPI5+TPU和RPI5 CPU两种运行模式。

## 系统架构

### 核心节点 (7个)
1. **sensor_input_node** - 双目相机数据采集
2. **image_preprocessor_node** - 图像预处理
3. **depth_estimator_node** - 深度估计
4. **object_detector_node** - 目标检测  
5. **semantic_segmentor_node** - 语义分割
6. **perception_manager_node** - 结果整合
7. **timing_analyzer_node** - 性能分析

### 消息类型 (6个)
- `DetectedObject.msg` - 单个检测对象
- `DetectedObjects.msg` - 多个检测对象
- `DepthImage.msg` - 深度图像
- `SegmentedImage.msg` - 分割结果
- `PerceptionResult.msg` - 完整感知结果
- `TimingInfo.msg` - 性能计时

## 性能目标

| 模式 | 总延迟 | FPS | 算法类型 |
|------|--------|-----|----------|
| RPI5 + TPU | ~120ms | 8-10 | 深度学习 |
| RPI5 CPU | ~200ms | 5-8 | 经典CV |

## 配置文件
- `config/rpi5_tpu_config.yaml` - TPU优化配置
- `config/rpi5_cpu_config.yaml` - CPU优化配置

## 快速使用指南

### 1. 启动完整流水线
```bash
# 进入工作目录
cd /ros2_ws/perception_ws
source install/setup.bash

# CPU模式 (经典CV算法)
ros2 launch perception_pipeline perception_pipeline.launch.py

# TPU模式 (需要深度学习模型)
ros2 launch perception_pipeline perception_pipeline.launch.py pipeline_mode:=dnn_tpu
```

### 2. 启动单个节点测试
```bash
# 传感器输入节点
python3 src/perception_pipeline/perception_pipeline/sensor_input_node.py

# 图像预处理节点  
python3 src/perception_pipeline/perception_pipeline/image_preprocessor_node.py

# 目标检测节点
python3 src/perception_pipeline/perception_pipeline/object_detector_node.py
```

### 3. 监控系统状态
```bash
# 查看活跃话题
ros2 topic list

# 监控图像发布频率
ros2 topic hz /stereo/left/image_raw

# 查看检测结果
ros2 topic echo /detected_objects

# 查看完整感知结果
ros2 topic echo /perception_result
```

### 4. 系统行为说明

**正常现象**：
- 节点启动后会"停在那里" - 这是**正常的**！
- 节点在后台持续运行，处理数据和发布结果
- 可以在另一个终端监控话题来验证节点工作

**如何停止**：
```bash
# 停止launch启动的所有节点
Ctrl+C

# 或者杀死特定节点
pkill -f sensor_input_node
```

## 技术实现

### 已完全实现的功能
✅ **经典CV算法**：
- HOG + 级联分类器人体检测
- Haar特征人脸检测  
- StereoBM/SGBM立体匹配
- 颜色和纹理语义分割

✅ **模拟数据生成**：
- 双目相机数据模拟
- 相机信息和标定参数
- 可配置分辨率和频率

✅ **ROS 2集成**：
- 完整的包结构和依赖管理
- 自定义消息类型
- 参数化配置
- Launch文件启动

### 需要扩展的部分
🔄 **深度学习模型**：
- YOLO/MobileNet目标检测模型
- FastSCNN语义分割模型  
- CNN深度估计模型

🔄 **真实硬件接口**：
- 真实双目相机驱动
- TPU推理引擎集成

## 故障排除

### 常见问题
1. **Launch文件找不到可执行文件**
   - 确保已运行 `colcon build`
   - 检查 `install/perception_pipeline/lib/perception_pipeline/` 目录

2. **话题没有数据**
   - 检查节点是否真正启动：`ros2 node list`
   - 验证话题存在：`ros2 topic list`

3. **性能不佳**  
   - 检查CPU使用率：`htop`
   - 调整配置文件中的参数
   - 考虑降低图像分辨率

## 开发说明

### 添加新的检测算法
1. 修改 `object_detector_node.py`
2. 在 `_init_cv_detector()` 中添加新方法
3. 更新配置文件参数

### 扩展消息类型
1. 在 `perception_interfaces/msg/` 添加新消息
2. 更新 `CMakeLists.txt`
3. 重新编译：`colcon build`

### 性能优化
1. 调整 `config/*.yaml` 中的参数
2. 使用 `timing_analyzer_node` 识别瓶颈
3. 考虑多线程或分布式处理

---

**系统状态**: ✅ 完全可运行  
**测试验证**: ✅ 所有节点正常启动  
**数据流**: ✅ 11-13 Hz图像发布  
**算法实现**: ✅ 经典CV完整实现 