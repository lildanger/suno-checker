#!/bin/bash
# ==============================================================================
# Suno Checker 本地 macOS 一键打包编译脚本 (支持 macOS 12/13/14/15)
# ==============================================================================

# 确保脚本在遇到任何错误时立刻退出，防止后续步骤出错
set -e

# 核心提示
echo -e "\033[1;36m==================================================\033[0m"
echo -e "\033[1;36m    Suno Checker 本地 macOS 一键打包程序启动...\033[0m"
echo -e "\033[1;36m==================================================\033[0m"

# 1. 检查 Python 3 环境
if ! command -v python3 &> /dev/null; then
    echo -e "\033[1;31m[错误] 未检测到 python3 运行环境！\033[0m"
    echo -e "请先前往苹果官网或使用 brew 安装 Python 3 (推荐使用 Python 3.11)。"
    exit 1
fi

PYTHON_VER=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "\033[1;32m[检测] 发现本地 Python 版本: $PYTHON_VER\033[0m"

# 2. 安装/更新依赖
echo -e "\033[1;33m[步骤 1/3] 正在检测并安装 Python 依赖库...\033[0m"
python3 -m pip install --upgrade pip

# 锁定numpy<2以及onnxruntime版本以保证绝对兼容
python3 -m pip install pyinstaller librosa "numpy<2" scipy onnxruntime==1.16.3 pyqt5 pyloudnorm pillow

# 3. 设置编译部署兼容目标（向后兼容至 macOS 12.0 Monterey，全面覆盖 13.x 14.x 等系统）
echo -e "\033[1;33m[步骤 2/3] 正在进行 PyInstaller 打包配置...\033[0m"
export MACOSX_DEPLOYMENT_TARGET=12.0

# 清理历史编译遗留
if [ -d "build" ]; then rm -rf build; fi
if [ -d "dist" ]; then rm -rf dist; fi

# 执行打包
pyinstaller predict.spec

# 4. 将生成的 .app 打包为标准的 .dmg 磁盘镜像
echo -e "\033[1;33m[步骤 3/3] 正在将可执行文件转换为 .dmg 安装包...\033[0m"

APP_PATH="dist/Suno Checker.app"
DMG_PATH="dist/suno-checker-macos.dmg"

if [ ! -d "$APP_PATH" ]; then
    echo -e "\033[1;31m[错误] 未能成功生成 Suno Checker.app，请检查上方 PyInstaller 报错信息！\033[0m"
    exit 1
fi

# 如果旧的 dmg 存在，先删除
if [ -f "$DMG_PATH" ]; then
    rm "$DMG_PATH"
fi

# 制作 dmg
hdiutil create -volname "Suno Checker" -srcfolder "$APP_PATH" -ov -format UDZO "$DMG_PATH"

echo -e "\033[1;32m==================================================\033[0m"
echo -e "\033[1;32m             ✨ 恭喜！本地编译成功！ ✨\033[0m"
echo -e "\033[1;32m==================================================\033[0m"
echo -e "全兼容 macOS 安装包生成成功，路径位于："
echo -e "\033[1;36m$(pwd)/$DMG_PATH\033[0m"
echo -e "您可以直接双击该 DMG 文件，将 Suno Checker 拖入应用程序直接使用！"
