#!/usr/bin/env python3
"""
时间分析节点 - 分析和报告感知流水线的性能
提供详细的性能统计和瓶颈分析
"""

import rclpy
from rclpy.node import Node
from perception_interfaces.msg import TimingInfo
import numpy as np
import time
from collections import deque

class TimingAnalyzerNode(Node):
    def __init__(self):
        super().__init__('timing_analyzer_node')
        
        # 参数声明
        self.declare_parameter('analysis_window', 100)  # 分析窗口大小
        self.declare_parameter('report_interval', 10.0)  # 报告间隔（秒）
        self.declare_parameter('target_fps', 30.0)  # 目标FPS
        
        # 获取参数
        self.analysis_window = self.get_parameter('analysis_window').value
        self.report_interval = self.get_parameter('report_interval').value
        self.target_fps = self.get_parameter('target_fps').value
        
        # 计算目标时间
        self.target_time_ms = 1000.0 / self.target_fps
        
        # 订阅器
        self.timing_sub = self.create_subscription(
            TimingInfo, '/perception/timing', self.timing_callback, 10)
        
        # 时间数据存储
        self.timing_data = {
            'total_pipeline_time': deque(maxlen=self.analysis_window),
            'sensor_input_time': deque(maxlen=self.analysis_window),
            'preprocessing_time': deque(maxlen=self.analysis_window),
            'depth_estimation_time': deque(maxlen=self.analysis_window),
            'object_detection_time': deque(maxlen=self.analysis_window),
            'semantic_segmentation_time': deque(maxlen=self.analysis_window),
            'postprocessing_time': deque(maxlen=self.analysis_window),
        }
        
        # 统计信息
        self.frame_count = 0
        self.last_report_time = time.time()
        
        # 定时报告
        self.report_timer = self.create_timer(self.report_interval, self.generate_performance_report)
        
        self.get_logger().info(f"时间分析节点已启动 - 目标FPS: {self.target_fps}")
    
    def timing_callback(self, msg):
        """时间信息回调"""
        # 存储时间数据
        self.timing_data['total_pipeline_time'].append(msg.total_pipeline_time)
        self.timing_data['sensor_input_time'].append(msg.sensor_input_time)
        self.timing_data['preprocessing_time'].append(msg.preprocessing_time)
        self.timing_data['depth_estimation_time'].append(msg.depth_estimation_time)
        self.timing_data['object_detection_time'].append(msg.object_detection_time)
        self.timing_data['semantic_segmentation_time'].append(msg.semantic_segmentation_time)
        self.timing_data['postprocessing_time'].append(msg.postprocessing_time)
        
        self.frame_count += 1
        
        # 实时性能检查
        if msg.total_pipeline_time > self.target_time_ms:
            self.get_logger().warning(
                f"流水线时间超标: {msg.total_pipeline_time:.2f}ms > {self.target_time_ms:.2f}ms"
            )
    
    def calculate_statistics(self, data_list):
        """计算统计信息"""
        if len(data_list) == 0:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
        
        data_array = np.array(data_list)
        return {
            'mean': np.mean(data_array),
            'std': np.std(data_array),
            'min': np.min(data_array),
            'max': np.max(data_array),
            'median': np.median(data_array)
        }
    
    def identify_bottlenecks(self):
        """识别性能瓶颈"""
        bottlenecks = []
        
        # 计算各阶段的平均时间
        stage_times = {}
        for stage, data in self.timing_data.items():
            if stage != 'total_pipeline_time' and len(data) > 0:
                stage_times[stage] = np.mean(data)
        
        # 找出最耗时的阶段
        if stage_times:
            max_stage = max(stage_times, key=stage_times.get)
            max_time = stage_times[max_stage]
            
            # 如果某个阶段占用了过多时间，标记为瓶颈
            total_avg = np.mean(self.timing_data['total_pipeline_time']) if len(self.timing_data['total_pipeline_time']) > 0 else 1
            if max_time / total_avg > 0.4:  # 如果某阶段占用超过40%的时间
                bottlenecks.append(f"{max_stage}: {max_time:.2f}ms")
        
        # 检查总时间是否超标
        if len(self.timing_data['total_pipeline_time']) > 0:
            avg_total = np.mean(self.timing_data['total_pipeline_time'])
            if avg_total > self.target_time_ms:
                bottlenecks.append(f"总时间超标: {avg_total:.2f}ms > {self.target_time_ms:.2f}ms")
        
        return bottlenecks
    
    def calculate_fps(self):
        """计算实际FPS"""
        current_time = time.time()
        time_elapsed = current_time - self.last_report_time
        
        if time_elapsed > 0:
            fps = len(self.timing_data['total_pipeline_time']) / time_elapsed
            return fps
        return 0.0
    
    def generate_performance_report(self):
        """生成性能报告"""
        if len(self.timing_data['total_pipeline_time']) == 0:
            self.get_logger().info("等待性能数据...")
            return
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("感知流水线性能报告")
        self.get_logger().info("=" * 60)
        
        # 总体统计
        total_stats = self.calculate_statistics(self.timing_data['total_pipeline_time'])
        self.get_logger().info(f"总流水线时间统计 (ms):")
        self.get_logger().info(f"  平均: {total_stats['mean']:.2f}")
        self.get_logger().info(f"  标准差: {total_stats['std']:.2f}")
        self.get_logger().info(f"  最小: {total_stats['min']:.2f}")
        self.get_logger().info(f"  最大: {total_stats['max']:.2f}")
        self.get_logger().info(f"  中位数: {total_stats['median']:.2f}")
        
        # FPS统计
        actual_fps = self.calculate_fps()
        self.get_logger().info(f"实际FPS: {actual_fps:.2f} / 目标FPS: {self.target_fps:.2f}")
        
        # 各阶段详细统计
        self.get_logger().info("\n各阶段时间统计 (ms):")
        for stage, data in self.timing_data.items():
            if stage != 'total_pipeline_time' and len(data) > 0:
                stats = self.calculate_statistics(data)
                percentage = (stats['mean'] / total_stats['mean']) * 100 if total_stats['mean'] > 0 else 0
                self.get_logger().info(f"  {stage}: {stats['mean']:.2f} ({percentage:.1f}%)")
        
        # 瓶颈分析
        bottlenecks = self.identify_bottlenecks()
        if bottlenecks:
            self.get_logger().info("\n发现的性能瓶颈:")
            for bottleneck in bottlenecks:
                self.get_logger().info(f"  - {bottleneck}")
        else:
            self.get_logger().info("\n未发现明显性能瓶颈")
        
        # 性能建议
        self.generate_optimization_suggestions(total_stats, actual_fps)
        
        self.get_logger().info("=" * 60)
        
        # 重置报告时间
        self.last_report_time = time.time()
    
    def generate_optimization_suggestions(self, total_stats, actual_fps):
        """生成优化建议"""
        suggestions = []
        
        # FPS相关建议
        if actual_fps < self.target_fps * 0.8:
            suggestions.append("考虑降低图像分辨率或简化算法复杂度")
        
        # 时间相关建议
        if total_stats['mean'] > self.target_time_ms:
            suggestions.append("流水线总时间超标，需要优化")
        
        # 稳定性建议
        if total_stats['std'] / total_stats['mean'] > 0.3:
            suggestions.append("处理时间波动较大，检查系统负载")
        
        # 具体阶段建议
        depth_avg = np.mean(self.timing_data['depth_estimation_time']) if len(self.timing_data['depth_estimation_time']) > 0 else 0
        detection_avg = np.mean(self.timing_data['object_detection_time']) if len(self.timing_data['object_detection_time']) > 0 else 0
        segmentation_avg = np.mean(self.timing_data['semantic_segmentation_time']) if len(self.timing_data['semantic_segmentation_time']) > 0 else 0
        
        if depth_avg > 30:
            suggestions.append("深度估计耗时较长，考虑使用更快的算法或降低精度")
        if detection_avg > 20:
            suggestions.append("目标检测耗时较长，考虑使用轻量级模型")
        if segmentation_avg > 25:
            suggestions.append("语义分割耗时较长，考虑简化分割策略")
        
        if suggestions:
            self.get_logger().info("\n性能优化建议:")
            for suggestion in suggestions:
                self.get_logger().info(f"  - {suggestion}")

def main(args=None):
    rclpy.init(args=args)
    node = TimingAnalyzerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 