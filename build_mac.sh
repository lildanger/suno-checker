#!/bin/bash
# ==============================================================================
# Suno Checker 本地 macOS 一键打包编译脚本 (自适应沙盒隔离版)
# ==============================================================================

# 确保脚本在遇到任何错误时立刻退出，防止后续步骤出错
set -e

# 核心提示
echo -e "\033[1;36m==================================================\033[0m"
echo -e "\033[1;36m    Suno Checker 本地 macOS 一键打包程序启动...\033[0m"
echo -e "\033[1;36m==================================================\033[0m"

# 1. 优先寻找兼容的 Python 版本 (推荐 Python 3.11 或 3.12)
# 由于 Python 3.13 过于新，PyQt5 和 ONNX Runtime 官方尚未对其提供预编译支持
PYTHON_BIN=""
DEFAULT_PY_VER=""

if command -v python3 &> /dev/null; then
    DEFAULT_PY_VER=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
fi

for cmd in python3.11 python3.12 python3; do
    if command -v $cmd &> /dev/null; then
        VER=$($cmd -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        MAJOR=$(echo $VER | cut -d. -f1)
        MINOR=$(echo $VER | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 8 ] && [ "$MINOR" -le 12 ]; then
            PYTHON_BIN=$cmd
            PYTHON_VER=$VER
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo -e "\033[1;31m[错误] 未能在系统里找到兼容的 Python 环境！\033[0m"
    echo -e "您当前默认的 python3 版本是 \033[1;33m$DEFAULT_PY_VER\033[0m。"
    echo -e "由于 Python 3.13+ 太新，PyQt5 与 onnxruntime 官方尚未提供对应二进制支持包，导致依赖无法安装。"
    echo -e "\033[1;32m\n👉 请尝试通过以下任一方式安装兼容版本：\033[0m"
    echo -e "  * 方式 1 (Homebrew 极速安装):"
    echo -e "    \033[1;36mbrew install python@3.11\033[0m"
    echo -e "  * 方式 2 (Conda 虚拟环境):"
    echo -e "    \033[1;36mconda create -n suno python=3.11 -y && conda activate suno\033[0m"
    echo -e "\n安装完成后，重新在此处运行此脚本即可！"
    exit 1
fi

echo -e "\033[1;32m[检测] 发现并选用兼容 Python 版本: $PYTHON_VER (使用命令: $PYTHON_BIN)\033[0m"

# 2. 创建并启用隔离的 Python 虚拟环境 (venv)
# 这一步可以防止污染你的系统全局库，并完美绕过 macOS 强制的 externally-managed-environment 报错限制
echo -e "\033[1;33m[步骤 1/3] 正在初始化局部打包沙盒虚拟环境 (venv)...\033[0m"
if [ -d "venv" ]; then
    rm -rf venv
fi
$PYTHON_BIN -m venv venv
source venv/bin/activate

# 3. 安装依赖库
echo -e "\033[1;33m[步骤 2/3] 正在沙盒中安装编译依赖库...\033[0m"
pip install --upgrade pip
pip install pyinstaller librosa "numpy<2" scipy "onnxruntime>=1.16.3" pyqt5 pyloudnorm pillow

# 4. 设置编译部署兼容目标（向后兼容至 macOS 12.0，全面覆盖 13.x 14.x 15.x 等系统）
echo -e "\033[1;33m[步骤 3/3] 正在调用 PyInstaller 开始编译...\033[0m"
export MACOSX_DEPLOYMENT_TARGET=12.0

# 清理历史编译残留
if [ -d "build" ]; then rm -rf build; fi
if [ -d "dist" ]; then rm -rf dist; fi

# 执行打包
pyinstaller predict.spec

# 5. 将生成的 .app 打包为标准的 .dmg 磁盘镜像
APP_PATH="dist/Suno Checker.app"
DMG_PATH="dist/suno-checker-macos.dmg"

if [ ! -d "$APP_PATH" ]; then
    echo -e "\033[1;31m[错误] 未能成功生成 Suno Checker.app，请检查上方编译报错信息！\033[0m"
    deactivate
    exit 1
fi

if [ -f "$DMG_PATH" ]; then
    rm "$DMG_PATH"
fi

# 制作 dmg
echo -e "\033[1;33m[收尾] 正在封装 .dmg 磁盘镜像安装包...\033[0m"
hdiutil create -volname "Suno Checker" -srcfolder "$APP_PATH" -ov -format UDZO "$DMG_PATH"

# 销毁沙盒激活状态
deactivate
rm -rf venv

echo -e "\033[1;32m==================================================\033[0m"
echo -e "\033[1;32m             ✨ 恭喜！本地编译成功！ ✨\033[0m"
echo -e "\033[1;32m==================================================\033[0m"
echo -e "全兼容 macOS 安装包生成成功，路径位于："
echo -e "\033[1;36m$(pwd)/$DMG_PATH\033[0m"
echo -e "您可以直接双击该 DMG 文件，将 Suno Checker 拖入应用程序直接使用！"
