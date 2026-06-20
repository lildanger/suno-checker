# Suno Checker 编译打包与项目文件说明

本项目是一个用于检测 AI 音频（基于神经声码器反卷积伪影）的桌面客户端，使用 PyQt5 编写 GUI 界面，并由 PyInstaller 编译为三端（Windows、macOS、Linux）的可执行程序。

---

## 📂 核心文件结构与职责

在编译或开发本项目时，请知悉以下关键文件的作用：

### 1. 业务逻辑与算法文件
*   **[predict.py](file:///d:/Documents/GitHub/suno-checker/predict.py)**：
    *   **主程序入口**：负责初始化 PyQt5 图形界面、调度后台推理线程。
    *   **环境防御机制**：顶部以 `try-except` 包裹了对 ONNX Runtime 的导入。如果系统因缺少 VC++ 运行库或 CPU 缺少 AVX 指令集而加载失败，程序将启动友好报错界面，指引用户前往微软官网下载运行时，且拦截拖入分析逻辑，彻底规避闪退。
*   **[fakeprint.py](file:///d:/Documents/GitHub/suno-checker/fakeprint.py)**：
    *   **特征提取模块**：基于 ISMIR 2025 论文算法，提取音频的频谱反卷积伪影指纹（神经声码器指纹）。
*   **[ai_music_detector.onnx](file:///d:/Documents/GitHub/suno-checker/ai_music_detector.onnx)**：
    *   **推理模型**：预训练的轻量化 CNN 分类模型，用来判断指纹是否符合 AI 生成特征。

### 2. 编译配置文件（PyInstaller）
*   **[predict.spec](file:///d:/Documents/GitHub/suno-checker/predict.spec)**：
    *   **标准编译配置文件**：用于 Windows（文件夹秒开版）、macOS（DMG/APP版）和 Linux 的打包。
    *   自动处理 `onnxruntime` 二进制文件多目录双向分发（兼顾 Python 包相对导入与操作系统 DLL 寻找）。
    *   Windows 环境下打包时，自动提取编译机的 `vcruntime140_1.dll` 等微软运行库，实现依赖包补全。
*   **[predict-portable.spec](file:///d:/Documents/GitHub/suno-checker/predict-portable.spec)**：
    *   **Windows 单文件便携版配置文件**：通过将 `a.binaries` 与 `a.datas` 打包入 `EXE` 节点，编译出单一的独立 `suno-checker-portable.exe`，无需解压即可在其他电脑上直接运行。

### 3. 应用图标资产
*   **[app_icon.png](file:///d:/Documents/GitHub/suno-checker/app_icon.png)**：用于软件运行时窗口左上角及任务栏的图标。
*   **[app_icon.ico](file:///d:/Documents/GitHub/suno-checker/app_icon.ico)**：Windows 编译出的 `.exe` 可执行文件的外置图标。
*   **[app_icon.icns](file:///d:/Documents/GitHub/suno-checker/app_icon.icns)**：macOS 编译出的 `.app` 应用的图标。

### 4. 自动化构建流
*   **[.github/workflows/build.yml](file:///d:/Documents/GitHub/suno-checker/.github/workflows/build.yml)**：
    *   **GitHub Actions 流水线**：当向仓库推送 `v*` 格式的 tag（例如 `v0.4.1`）时，自动拉起多平台的虚拟机执行打包，生成 4 个平台的最终压缩包，并同步发布在 GitHub Release 页面。

---

## 🛠️ 如何在本地手动编译

如果您想在自己的开发机上进行本地编译，请参考以下流程：

### 1. 配置 Python 开发环境
推荐使用 **Python 3.11**。在项目根目录下，使用终端安装编译所需的依赖包：
```bash
pip install pyinstaller librosa "numpy<2" scipy onnxruntime==1.16.3 pyqt5 pyloudnorm pillow
```
> **注意**：此处锁定了 `onnxruntime==1.16.3` 与 `numpy<2` 以保证编译后的最高系统兼容性。

### 2. 执行编译命令

#### 编译标准文件夹版本 (Windows / macOS / Linux)
在终端运行：
```bash
pyinstaller predict.spec
```
*   **Windows / Linux**：输出产物为 `dist/suno-checker/` 文件夹。
*   **macOS**：输出产物为 `dist/Suno Checker.app`。

#### 编译单文件 Portable 版本 (仅 Windows 支持)
在 Windows 终端中运行：
```bash
pyinstaller predict-portable.spec
```
*   输出产物为 `dist/suno-checker-portable.exe`。

---

## ⚙️ 编译相关的进阶说明

1.  **大包剔除规则 (Excludes)**：
    在两个 `.spec` 文件中，我们利用了 `excludes` 机制移除了 `torch`、`matplotlib`、`pandas` 等与主逻辑无关但体积极大的第三方包。这能让发布包的体积精简 150MB+，请勿轻易移除这些 exclude 规则。
2.  **动态库自动搜寻**：
    `predict.py` 会在启动时自适应判断自己处于开发环境还是打包好的 `frozen` 沙盒环境（利用 `sys._MEIPASS`）。运行时会自动遍历同级和子级目录，将含 DLL/PYD 的路径添加至系统的 DLL 搜寻列表，为程序的平稳运行提供底层保证。
