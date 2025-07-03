#!/bin/bash

# ROS 2 Galactic Docker 启动脚本
# 作者: AI助手
# 日期: $(date)

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印彩色信息
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

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装。请先安装Docker。"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose未安装。请先安装Docker Compose。"
        exit 1
    fi
}

# 检查X11转发权限
setup_x11() {
    print_info "设置X11转发权限..."
    xhost +local:docker || print_warning "无法设置X11权限，GUI应用可能无法显示"
}

# 构建Docker镜像
build_image() {
    print_info "构建ROS 2 Galactic Docker镜像..."
    docker-compose build ros2-galactic
    print_success "Docker镜像构建完成"
}

# 启动容器
start_container() {
    print_info "启动ROS 2 Galactic容器..."
    docker-compose up -d ros2-galactic
    print_success "容器启动成功"
}

# 进入容器
enter_container() {
    print_info "进入ROS 2容器..."
    docker-compose exec ros2-galactic bash
}

# 停止容器
stop_container() {
    print_info "停止ROS 2容器..."
    docker-compose down
    print_success "容器已停止"
}

# 查看容器状态
status_container() {
    print_info "容器状态:"
    docker-compose ps
}

# 查看容器日志
logs_container() {
    print_info "查看容器日志:"
    docker-compose logs -f ros2-galactic
}

# 启动ROS 2工具
start_tools() {
    print_info "启动ROS 2管理工具..."
    setup_x11
    docker-compose --profile tools up -d ros2-tools
    print_success "ROS 2工具启动成功"
}

# 显示帮助信息
show_help() {
    echo -e "${BLUE}ROS 2 Galactic Docker 管理脚本${NC}"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "可用命令:"
    echo "  build     - 构建Docker镜像"
    echo "  start     - 启动容器"
    echo "  stop      - 停止容器"
    echo "  restart   - 重启容器"
    echo "  enter     - 进入容器终端"
    echo "  status    - 查看容器状态"
    echo "  logs      - 查看容器日志"
    echo "  tools     - 启动ROS 2管理工具"
    echo "  setup     - 一键设置和启动"
    echo "  clean     - 清理容器和镜像"
    echo "  help      - 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 setup     # 构建镜像并启动容器"
    echo "  $0 enter     # 进入正在运行的容器"
    echo "  $0 tools     # 启动RQT等GUI工具"
}

# 一键设置
setup_all() {
    print_info "开始一键设置ROS 2 Galactic环境..."
    check_docker
    setup_x11
    build_image
    start_container
    print_success "ROS 2环境设置完成！"
    print_info "使用 '$0 enter' 进入容器"
    print_info "使用 '$0 tools' 启动GUI工具"
}

# 清理环境
clean_all() {
    print_warning "这将删除所有相关的容器和镜像"
    read -p "确定要继续吗? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "清理Docker环境..."
        docker-compose down -v --remove-orphans
        docker rmi ros2-galactic:latest 2>/dev/null || true
        print_success "清理完成"
    else
        print_info "操作已取消"
    fi
}

# 重启容器
restart_container() {
    print_info "重启ROS 2容器..."
    docker-compose restart ros2-galactic
    print_success "容器重启完成"
}

# 主要逻辑
main() {
    case "${1:-help}" in
        build)
            check_docker
            build_image
            ;;
        start)
            check_docker
            setup_x11
            start_container
            ;;
        stop)
            stop_container
            ;;
        restart)
            restart_container
            ;;
        enter)
            enter_container
            ;;
        status)
            status_container
            ;;
        logs)
            logs_container
            ;;
        tools)
            check_docker
            start_tools
            ;;
        setup)
            setup_all
            ;;
        clean)
            clean_all
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@" 