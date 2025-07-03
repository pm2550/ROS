#!/bin/bash

# RPI5 感知流水线快速启动脚本
# 支持多种运行模式和配置

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印信息函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo -e "${BLUE}RPI5 感知流水线启动脚本${NC}"
    echo ""
    echo "用法: $0 [模式] [选项]"
    echo ""
    echo "模式:"
    echo "  tpu        - 使用TPU加速的DNN模式 (推荐)"
    echo "  cpu        - 纯CPU的经典CV模式"
    echo "  sim-tpu    - TPU模式 + 模拟数据"
    echo "  sim-cpu    - CPU模式 + 模拟数据 (默认)"
    echo "  build      - 编译项目"
    echo "  clean      - 清理构建"
    echo ""
    echo "选项:"
    echo "  --fps N    - 设置目标FPS (默认: 15)"
    echo "  --width N  - 设置处理图像宽度 (默认: 640)"
    echo "  --height N - 设置处理图像高度 (默认: 480)"
    echo "  --debug    - 启用调试模式"
    echo "  --help     - 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 sim-cpu              # 运行CPU模式模拟"
    echo "  $0 tpu --fps 30         # 运行TPU模式，30FPS"
    echo "  $0 cpu --debug          # 运行CPU模式调试"
    echo "  $0 build                # 编译项目"
}

# 检查ROS 2环境
check_ros2_env() {
    if [ -z "$ROS_DISTRO" ]; then
        print_error "ROS 2环境未设置，请先source ROS 2环境"
        exit 1
    fi
    
    if [ "$ROS_DISTRO" != "galactic" ]; then
        print_warning "检测到ROS $ROS_DISTRO，推荐使用ROS 2 Galactic"
    fi
}

# 编译项目
build_project() {
    print_info "编译感知流水线项目..."
    
    if [ ! -d "src" ]; then
        print_error "请在工作空间根目录运行此脚本"
        exit 1
    fi
    
    # 编译
    colcon build --packages-select perception_interfaces perception_pipeline \
        --cmake-args -DCMAKE_BUILD_TYPE=Release
    
    if [ $? -eq 0 ]; then
        print_success "编译完成"
        print_info "请运行: source install/setup.bash"
    else
        print_error "编译失败"
        exit 1
    fi
}

# 清理构建
clean_build() {
    print_info "清理构建文件..."
    rm -rf build/ install/ log/
    print_success "清理完成"
}

# 检查设备
check_devices() {
    local mode=$1
    
    if [[ "$mode" == *"tpu"* ]]; then
        # 检查TPU
        if lsusb | grep -q "Google"; then
            print_success "检测到TPU设备"
        else
            print_warning "未检测到TPU设备，将回退到CPU模式"
            return 1
        fi
    fi
    
    if [[ "$mode" != *"sim"* ]]; then
        # 检查相机
        if ls /dev/video* &> /dev/null; then
            print_success "检测到相机设备"
        else
            print_warning "未检测到相机设备，建议使用模拟模式"
            return 1
        fi
    fi
    
    return 0
}

# 运行感知流水线
run_perception() {
    local mode=$1
    local fps=$2
    local width=$3
    local height=$4
    local debug=$5
    
    # 检查环境
    check_ros2_env
    
    # 检查是否已编译
    if [ ! -d "install" ]; then
        print_warning "项目未编译，正在编译..."
        build_project
    fi
    
    # Source环境
    source install/setup.bash
    
    # 检查设备
    check_devices "$mode"
    device_ok=$?
    
    # 设置参数
    case "$mode" in
        "tpu"|"dnn_tpu")
            pipeline_mode="dnn_tpu"
            use_simulation="false"
            ;;
        "cpu"|"cv_cpu")
            pipeline_mode="cv_cpu"
            use_simulation="false"
            ;;
        "sim-tpu")
            pipeline_mode="dnn_tpu"
            use_simulation="true"
            ;;
        "sim-cpu")
            pipeline_mode="cv_cpu"
            use_simulation="true"
            ;;
        *)
            print_error "未知模式: $mode"
            show_help
            exit 1
            ;;
    esac
    
    # 如果设备检查失败且不是模拟模式，强制使用模拟
    if [ $device_ok -ne 0 ] && [ "$use_simulation" == "false" ]; then
        print_warning "设备检查失败，切换到模拟模式"
        use_simulation="true"
    fi
    
    # 构建启动命令
    local launch_cmd="ros2 launch perception_pipeline perception_pipeline.launch.py"
    launch_cmd="$launch_cmd pipeline_mode:=$pipeline_mode"
    launch_cmd="$launch_cmd use_simulation:=$use_simulation"
    launch_cmd="$launch_cmd target_fps:=$fps"
    launch_cmd="$launch_cmd image_width:=$width"
    launch_cmd="$launch_cmd image_height:=$height"
    
    if [ "$debug" == "true" ]; then
        launch_cmd="$launch_cmd --log-level debug"
    fi
    
    # 显示启动信息
    print_info "启动感知流水线..."
    print_info "模式: $pipeline_mode"
    print_info "数据源: $([ "$use_simulation" == "true" ] && echo "模拟" || echo "真实相机")"
    print_info "目标FPS: $fps"
    print_info "处理分辨率: ${width}x${height}"
    
    # 启动
    print_info "执行命令: $launch_cmd"
    eval $launch_cmd
}

# 主函数
main() {
    # 默认参数
    local mode="sim-cpu"
    local fps=15
    local width=640
    local height=480
    local debug="false"
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            tpu|cpu|sim-tpu|sim-cpu|dnn_tpu|cv_cpu)
                mode="$1"
                shift
                ;;
            build)
                build_project
                exit 0
                ;;
            clean)
                clean_build
                exit 0
                ;;
            --fps)
                fps="$2"
                shift 2
                ;;
            --width)
                width="$2"
                shift 2
                ;;
            --height)
                height="$2"
                shift 2
                ;;
            --debug)
                debug="true"
                shift
                ;;
            --help|-h|help)
                show_help
                exit 0
                ;;
            *)
                print_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 验证参数
    if ! [[ "$fps" =~ ^[0-9]+$ ]] || [ "$fps" -lt 1 ] || [ "$fps" -gt 60 ]; then
        print_error "无效的FPS值: $fps (范围: 1-60)"
        exit 1
    fi
    
    if ! [[ "$width" =~ ^[0-9]+$ ]] || [ "$width" -lt 160 ]; then
        print_error "无效的宽度值: $width (最小: 160)"
        exit 1
    fi
    
    if ! [[ "$height" =~ ^[0-9]+$ ]] || [ "$height" -lt 120 ]; then
        print_error "无效的高度值: $height (最小: 120)"
        exit 1
    fi
    
    # 运行感知流水线
    run_perception "$mode" "$fps" "$width" "$height" "$debug"
}

# 运行主函数
main "$@" 