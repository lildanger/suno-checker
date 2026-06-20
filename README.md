# Suno AI 音乐检测系统 (Suno AI Music Detector)

基于 **ISMIR 2025 fakeprint 频谱伪影分析** 的 AI 生成音乐检测系统。可识别 Suno ≤v5、Udio ≤v1.5 等神经声码器生成的音频。提供 PyQt5 桌面 GUI 和 ONNX 模型推理。

> **v0.4 更新**：引擎已从 ResNet-18 CNN 升级为 lofcz/ai-music-detector（fakeprint + 逻辑回归），假阳性率从 70% 降至 ~10%。

---

## 功能

- **桌面预测应用**：PyQt5 GUI，拖拽 mp3/wav/flac/m4a → 即时 AI/真人判定报告
- **独立 EXE 分发**：PyInstaller 打包，单文件免安装，Windows 直接运行
- **GPU 训练工具**：Tkinter GUI，CNN 特征提取 + 训练 + ONNX 导出（`1.py`）
- **双模型 fallback**：优先加载 lofcz fakeprint 模型，缺失时自动回退 ResNet-18 CNN
- **批量特征提取**：支持多目录、懒加载裂变增强（15×）、多线程缓存预热

---

## 快速开始

### 直接使用（无需安装）

下载 `dist/predict.exe`，双击运行，拖拽音频文件即可。

### 从源码运行

```bash
# 环境：Python 3.9+, venv 推荐
pip install librosa numpy scipy onnxruntime pyloudnorm pyqt5 soundfile
python predict.py
```

### 训练自己的模型

```bash
python 1.py
```

在 Tkinter GUI 中设置 AI/真人音频目录，点击训练即可导出 ONNX。

---

## 项目结构

```
├── predict.py                    # PyQt5 GUI 预测工具 (v0.4, fakeprint 引擎)
├── predict.spec                  # PyInstaller 打包配置
├── fakeprint.py                  # lofcz fakeprint 特征提取器 (纯 numpy/scipy)
├── ai_music_detector.onnx        # lofcz 预训练 ONNX 模型 (14.4KB, 逻辑回归)
├── suno_detector_model.onnx      # 旧版 ResNet-18 CNN ONNX 模型 (fallback)
├── 1.py                          # Tkinter GUI 训练工具 (ResNet-18 CNN)
├── train_cnn.py                  # ResNet-18 训练核心逻辑
├── dataset.py                    # Log-Mel 频谱图数据集 + 懒加载裂变增强
├── index.html                    # 纯前端 Web 检测页面 (实验性, ONNX Runtime Web)
├── evaluate_onnx.py              # ONNX 模型评估脚本
├── AI/                           # AI 生成音频样本 (训练用)
├── 人类/                         # 真人音频样本 (训练用)
├── lofcz_ai_music_detector/      # lofcz 上游仓库 (含完整训练/导出代码)
│   └── src/
│       ├── python/
│       │   ├── extract_fakeprints.py   # GPU 加速 fakeprint 提取
│       │   ├── train_model.py          # 逻辑回归训练
│       │   ├── export_onnx.py          # ONNX 导出
│       │   └── download_data.py        # 数据集下载 (FMA + SONICS)
│       └── models/
│           └── ai_music_detector.onnx  # 预训练模型
├── dist/
│   └── predict.exe               # 打包后的独立可执行文件 (~171MB)
└── venv/                         # Python 虚拟环境
```

---

## 技术原理

### 核心检测方法：Fakeprint 频谱伪影分析

基于 Afchar et al. (ISMIR 2025) — *"A Fourier Explanation of AI-Music Artifacts"*。

**核心发现**：Suno/Udio 等 AI 音乐生成器使用神经声码器（neural vocoder）将声学潜变量转换为波形。声码器中的**反卷积层（transposed convolution / deconvolution）** 以步长 k 执行零上采样 + 卷积两步操作，会在频域留下**确定性的等间距峰值** — 称为 "fakeprint"。

这些峰值：
- **仅取决于反卷积层步长配置**，与训练数据、权重随机种子、输入内容无关
- **对已知架构是完全确定性的** — 只要知道声码器结构，就能精确预测峰值位置
- **不依赖"学习"** — 区别于 CNN 的统计特征方法，不会将编曲风格误判为 AI 指纹

### Fakeprint 提取管线

```
音频文件 (.mp3/.wav/.flac)
    │
    ▼
[1] librosa.load(sr=16000, mono=True)
    │  重采样至 16kHz 单声道
    ▼
[2] STFT (n_fft=8192, hop_length=2048)
    │  高分辨率频谱 (频率分辨率 ~1.95 Hz)
    ▼
[3] 功率谱 → dB 域转换
    10 * log10(|STFT|^2)
    │
    ▼
[4] 时间轴平均 → 均值频谱
    mean over time frames
    │
    ▼
[5] 频率掩码 1000-8000 Hz
    提取 artifact 检测范围的 3585 个频点
    │
    ▼
[6] 下包络 (Lower Hull) 计算
    scipy.ndimage.minimum_filter1d(size=10)
    │
    ▼
[7] 残差 = 频谱 - 下包络
    residue = clip(spectrum - hull, 0, +inf)
    │
    ▼
[8] 归一化 [0, 1]
    clip(residue, 0, 5dB) / max(residue)
    │
    ▼
fakeprint 向量 (3585 维 float32)
    │
    ▼
[9] 逻辑回归 ONNX 推理
    sigmoid(w @ fakeprint + b) → AI 概率 (0=真人, 1=AI)
```

### 为什么比 CNN 好

| 维度 | ResNet-18 CNN (旧) | Fakeprint (新) |
|------|-------------------|----------------|
| **检测目标** | 统计特征分布 ("Suno 音乐长什么样") | 物理伪影 ("声码器留下了什么指纹") |
| **泛化** | 依赖训练数据覆盖 | 对已知声码器架构确定性检测 |
| **假阳性来源** | 现代编曲风格约等于 AI 风格 | 极低，仅极短/极简音频偶尔触发 |
| **假阳性率 (实测)** | 70% (14/20) | ~10% (2/19 独特歌曲) |
| **模型大小** | ~40MB (ResNet-18) | 14.4KB (逻辑回归) |
| **推理速度** | ~5-15s (3s x N 切片) | ~0.5s (单次 STFT + 向量点积) |

### 旧版模型 (v0.3, 保留作为 fallback)

ResNet-18 CNN 在 Log-Mel 频谱图上训练：

- **输入**：单通道 Log-Mel 频谱图 (128 mel bands x 130 time frames, ~3 秒)
- **架构**：ResNet-18 (BasicBlock [2,2,2,2]), ImageNet 预训练权重首层通道压缩
- **训练**：15x 在线增强裂变（音高偏移、时间拉伸、EQ、噪声注入），25 epochs，AdamW
- **数据集划分**：文件级隔离 (80/20)，防止变体泄露
- **推理**：整首歌 → 3s 不重叠切片 → Log-Mel → 逐片推理 → 均值投票

---

## 性能基准

### 假阳性率测试 (E:\Music, 20 首随机真人歌曲, seed=42)

| 模型 | 正确 | 误判 | 假阳性率 |
|------|------|------|----------|
| ResNet-18 CNN (旧) | 6 | 14 | **70.0%** |
| lofcz fakeprint (新) | 17 | 3* | **15.0%** (独特歌曲 ~10%) |

> *3 首误判中，2 首为同一歌曲的不同格式 (Why Not by JLS, mp3 + flac 各一份)。另外 1 首为 Toby Fox - Small Shock (14s 芯片音乐)。

### lofcz 官方报告性能

| 指标 | 数值 |
|------|------|
| 准确率 (Accuracy) | 99.88% |
| 精确率 (Precision) | 0.9985 |
| 召回率 (Recall) | 0.9998 |
| F1 分数 | 0.9991 |
| 假阳性率 (FPR) | **0.31%** |
| 假阴性率 (FNR) | 0.02% |
| 测试集规模 | 17,866 样本 (5,741 真实 + 12,125 AI) |

---

## 打包为独立 EXE

```bash
# 使用现有的 spec 文件
pyinstaller --clean predict.spec

# 或手动指定
pyinstaller --noconsole --onefile \
  --add-data "ai_music_detector.onnx;." \
  --add-data "suno_detector_model.onnx;." \
  --hidden-import fakeprint \
  --hidden-import scipy.ndimage._nd_image \
  --hidden-import scipy.ndimage._filters \
  --hidden-import soundfile \
  --hidden-import pyloudnorm \
  --exclude-module torch \
  --exclude-module torchvision \
  --exclude-module matplotlib \
  --exclude-module tkinter \
  predict.py
```

### 关于反编译保护

PyInstaller 打包的 exe 容易通过 `pyinstxtractor` + `pycdc` 提取和反编译。如需加固：

| 方案 | 保护强度 | 说明 |
|------|---------|------|
| **Nuitka** | 高 | Python -> C -> 机器码编译，无可提取的 .pyc |
| **Cython + PyInstaller** | 高 | 核心逻辑编译为 .pyd (C 扩展) |
| **PyArmor** | 中 | 字节码混淆加密，运行时解密 (仍可 hook) |

Nuitka 推荐用法：

```bash
pip install nuitka
python -m nuitka --standalone --onefile \
  --windows-console-mode=disable \
  --enable-plugin=pyqt5 \
  --include-data-file=ai_music_detector.onnx=. \
  --include-data-file=suno_detector_model.onnx=. \
  predict.py
```

---

## 已知局限

1. **生成器版本覆盖**：当前 fakeprint 模型针对 Suno ≤v5 和 Udio ≤v1.5 的反卷积配置。新版本生成器如需支持，需重新提取特征并训练
2. **最短时长**：10 秒以上音频效果最佳。极短音频 (<5s) 或极简波形 (纯正弦波芯片音乐) 可能误判
3. **采样率依赖**：音频自动重采样至 16kHz 分析。低于 16kHz 的音源会丢失高频 artifact 信息
4. **非音频检测**：本系统仅分析音频信号，不分析歌词文本。"AI 歌词 + 真人演唱" 的场景无法通过频谱检测
5. **对抗规避**：刻意去除反卷积伪影（低通滤波、重采样破坏、叠加噪声）可能降低检测率。这是检测/规避的持续军备竞赛

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 核心检测引擎 | lofcz/ai-music-detector (ISMIR 2025) |
| 特征提取 | librosa, scipy, numpy (纯 CPU) |
| ONNX 推理 | onnxruntime |
| 预测 GUI | PyQt5 |
| 训练 GUI | Tkinter + PyTorch |
| 深度学习 (旧) | ResNet-18 CNN, Log-Mel Spectrogram |
| 打包 | PyInstaller |
| Web 版 (实验) | ONNX Runtime Web + Meyda |

## 数据集

本项目的训练和评估使用以下公开数据集：

| 数据集 | 真实音乐 | AI 生成 | 总时长 |
|--------|---------|---------|--------|
| [SONICS](https://github.com/awsaf49/sonics) | 48,090 (YouTube) | 49,074 (Suno/Udio) | 4,751 小时 |
| [FMA Medium](https://github.com/mdeff/fma) | 25,000 | — | ~1,200 小时 |

## 参考文献

- Afchar, D., et al. (2025). *"A Fourier Explanation of AI-Music Artifacts."* ISMIR 2025. [arXiv:2506.19108](https://arxiv.org/abs/2506.19108)
- Rahman, A., et al. (2025). *"SONICS: Synthetic Or Not — Identifying Counterfeit Songs."* ICLR 2025. [arXiv:2408.14080](https://arxiv.org/abs/2408.14080)
- lofcz/ai-music-detector. GitHub. <https://github.com/lofcz/ai-music-detector>

## License

MIT
