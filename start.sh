#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# ROS 2 Galactic Docker 管理脚本（Edge-TPU MAX 版）
# ----------------------------------------------------------------------------
# build     构建镜像
# start     启动容器
# stop      停止+删除容器
# restart   重启容器
# enter     进入 bash
# status    查看状态
# logs      实时日志
# tools     启动 GUI/RQT
# setup     一键构建+启动
# clean     删除镜像/卷/容器
# help      帮助
# ----------------------------------------------------------------------------
set -euo pipefail

# ====== 可调参数 ======
SERVICE_NAME="ros2-galactic-max"      # docker-compose.yml 中的服务名
IMAGE_TAG="${SERVICE_NAME}:latest"    # 镜像 tag
COMPOSE_FILE="docker-compose.yml"    # Compose 文件
# ======================

# -------- 颜色 --------
RED='[0;31m'; GREEN='[0;32m'; YELLOW='[1;33m'; BLUE='[0;34m'; NC='[0m'
log()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail(){ echo -e "${RED}[ERR]${NC}  $*"; exit 1; }

# -------- Compose CLI 检测 --------
if command -v docker-compose &>/dev/null; then          # 独立二进制优先
  DC=(docker-compose -f "$COMPOSE_FILE")
elif docker compose version &>/dev/null; then           # CLI 插件
  DC=(docker compose -f "$COMPOSE_FILE")
else
  fail "Docker Compose 未安装 (docker-compose 或 docker compose plugin)"
fi

check_docker() { command -v docker &>/dev/null || fail "Docker 未安装"; }
setup_x11()    { [[ -n ${DISPLAY:-} ]] && xhost +local:docker &>/dev/null || warn "GUI 可能无法显示"; }

build_image()      { log "构建镜像";    "${DC[@]}" build "$SERVICE_NAME"; ok "完成"; }
start_container()  { log "启动容器";    "${DC[@]}" up -d "$SERVICE_NAME"; ok "已启动"; }
enter_container()  { "${DC[@]}" exec "$SERVICE_NAME" bash; }
stop_container()   { log "停止容器";    "${DC[@]}" down; ok "已停止"; }
restart_container(){ log "重启容器";    "${DC[@]}" restart "$SERVICE_NAME"; }
status_container() { log "容器状态";    "${DC[@]}" ps; }
logs_container()   { "${DC[@]}" logs -f "$SERVICE_NAME"; }
start_tools()     { setup_x11; "${DC[@]}" --profile tools up -d ros2-tools; }

clean_all() {
  warn "将删除容器和镜像 ($IMAGE_TAG)"; read -rp "继续? (y/N): " ans
  [[ $ans =~ ^[Yy]$ ]] || { log "取消"; return; }
  "${DC[@]}" down -v --remove-orphans || true
  docker rmi "$IMAGE_TAG" 2>/dev/null || true
  ok "已清理";
}

setup_all() { check_docker; setup_x11; build_image; start_container; ok "环境就绪，使用 '$0 enter'"; }

help_msg(){ cat <<EOF
用法: $0 <command>
命令: build start stop restart enter status logs tools setup clean help
EOF
}

case ${1:-help} in
  build)   build_image ;;
  start)   start_container ;;
  stop)    stop_container ;;
  restart) restart_container ;;
  enter)   enter_container ;;
  status)  status_container ;;
  logs)    logs_container ;;
  tools)   start_tools ;;
  setup)   setup_all ;;
  clean)   clean_all ;;
  help|--help|-h) help_msg ;;
  *) fail "未知命令: $1" ;;
esac
