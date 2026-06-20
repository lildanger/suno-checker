# Suno AI 音乐检测系统 (Suno AI Music Detector)

基于 **ISMIR 2025 fakeprint 频谱伪影分析** 的 AI 生成音乐检测系统。可识别 Suno ≤v5、Udio ≤v1.5 等神经声码器生成的音频。提供 PyQt5 桌面 GUI 和 ONNX 模型推理。

> **v0.4**：引擎为 lofcz/ai-music-detector（fakeprint + 逻辑回归），假阳性率 ~10%。

---

## 功能

- **桌面预测应用**：PyQt5 GUI，拖拽 mp3/wav/flac/m4a → 即时 AI/真人判定报告
- **独立 EXE 分发**：PyInstaller 打包，单文件免安装，Windows 直接运行
- **14.4KB 超轻量模型**：逻辑回归，推理 ~0.5s，纯 CPU

---

## 快速开始

```bash
pip install librosa numpy scipy onnxruntime pyqt5
python predict.py
```

---

## 项目结构

```
├── predict.py                 # PyQt5 GUI 推理工具
├── fakeprint.py               # fakeprint 特征提取器 (纯 numpy/scipy)
├── ai_music_detector.onnx     # 预训练 ONNX 模型 (14.4KB, 逻辑回归)
├── .gitignore
└── .nojekyll                  # GitHub Pages
```

---

## 技术原理

### Fakeprint 频谱伪影分析

基于 Afchar et al. (ISMIR 2025) — *"A Fourier Explanation of AI-Music Artifacts"*。

**核心发现**：Suno/Udio 等 AI 音乐生成器使用神经声码器（neural vocoder）将声学潜变量转换为波形。声码器中的**反卷积层（transposed convolution）**以步长 k 执行零上采样 + 卷积，会在频域留下**确定性的等间距峰值** — 称为 "fakeprint"。

这些峰值：
- **仅取决于反卷积层步长配置**，与训练数据、权重随机种子、输入内容无关
- **对已知架构是完全确定性的** — 只要知道声码器结构，就能精确预测峰值位置
- **不依赖"学习"** — 区别于 CNN 的统计特征方法，不会将编曲风格误判为 AI 指纹

### 提取管线

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
    │
    ▼
[5] 频率掩码 1000-8000 Hz
    │  提取 artifact 检测范围的 3585 个频点
    ▼
[6] 下包络 (Lower Hull) 计算
    │  scipy.ndimage.minimum_filter1d(size=10)
    ▼
[7] 残差 = 频谱 - 下包络
    │  residue = clip(spectrum - hull, 0, +inf)
    ▼
[8] 归一化 [0, 1]
    │  clip(residue, 0, 5dB) / max(residue)
    ▼
fakeprint 向量 (3585 维 float32)
    │
    ▼
[9] 逻辑回归 ONNX 推理
    sigmoid(w @ fakeprint + b) → AI 概率 (0=真人, 1=AI)
```

---

## 性能基准

### lofcz 官方报告

| 指标 | 数值 |
|------|------|
| 准确率 (Accuracy) | 99.88% |
| 精确率 (Precision) | 0.9985 |
| 召回率 (Recall) | 0.9998 |
| F1 分数 | 0.9991 |
| 假阳性率 (FPR) | 0.31% |
| 假阴性率 (FNR) | 0.02% |
| 测试集 | 17,866 样本 (5,741 真实 + 12,125 AI) |

### 本地实测 (20 首随机真人歌曲)

| 正确 | 误判 | 假阳性率 |
|------|------|----------|
| 17 | 3* | 15.0% (去重后 ~10%) |

> *3 首误判中，2 首为同一歌曲的 mp3 + flac 双格式。另 1 首为 14s 芯片音乐 (Toby Fox - Small Shock)。

---

## 打包为独立 EXE

```bash
pyinstaller --noconsole --onefile \
  --add-data "ai_music_detector.onnx;." \
  --hidden-import fakeprint \
  --hidden-import scipy.ndimage._nd_image \
  --hidden-import scipy.ndimage._filters \
  predict.py
```

### 反编译保护

PyInstaller 打包的 exe 可通过 `pyinstxtractor` + `pycdc` 提取反编译。如需加固：

| 方案 | 保护强度 | 说明 |
|------|---------|------|
| **Nuitka** | 高 | Python → C → 机器码，无可提取的 .pyc |
| **Cython + PyInstaller** | 高 | 核心逻辑编译为 .pyd (C 扩展) |
| **PyArmor** | 中 | 字节码混淆加密，运行时解密 |

```bash
pip install nuitka
python -m nuitka --standalone --onefile \
  --windows-console-mode=disable \
  --enable-plugin=pyqt5 \
  --include-data-file=ai_music_detector.onnx=. \
  predict.py
```

---

## 已知局限

1. **生成器版本覆盖**：当前模型针对 Suno ≤v5 和 Udio ≤v1.5 的反卷积配置，新版本需重新提取特征并训练
2. **最短时长**：10 秒以上音频效果最佳，极短音频 (<5s) 或极简波形可能误判
3. **采样率依赖**：音频自动重采样至 16kHz，低于 16kHz 的音源会丢失高频 artifact
4. **非音频检测**：仅分析音频信号，不分析歌词文本
5. **对抗规避**：刻意低通滤波、重采样破坏、叠加噪声可能降低检测率

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 检测引擎 | lofcz/ai-music-detector (ISMIR 2025) |
| 特征提取 | librosa, scipy, numpy |
| ONNX 推理 | onnxruntime |
| GUI | PyQt5 |
| 打包 | PyInstaller |

## 参考文献

- Afchar, D., et al. (2025). *"A Fourier Explanation of AI-Music Artifacts."* ISMIR 2025. [arXiv:2506.19108](https://arxiv.org/abs/2506.19108)
- Rahman, A., et al. (2025). *"SONICS: Synthetic Or Not — Identifying Counterfeit Songs."* ICLR 2025. [arXiv:2408.14080](https://arxiv.org/abs/2408.14080)
- lofcz/ai-music-detector. <https://github.com/lofcz/ai-music-detector>

## License

MIT
