# Suno Checker

**在浏览器里直接运行，无需上传音频，保护你的隐私。**

[English](#english) | [中文](#chinese)

---

<a id="english"></a>
## English

Suno Checker is a browser-based AI music detector that runs completely on your device. No server, no upload, no privacy risk.

### How It Works

Just drag and drop an audio file into the page. The tool analyzes the sound characteristics and tells you whether it's likely AI-generated or human-made music.

### Features

- **Runs in your browser**: No installation needed, works on Chrome, Edge, Firefox, Safari
- **Privacy first**: Your audio never leaves your device, everything happens locally
- **Fast analysis**: Usually takes 5 to 15 seconds to analyze a song
- **Clear results**: Shows a straightforward verdict with a confidence percentage
- **Small size**: The entire tool takes up less than 30 KB (not counting the model file)

### Model Performance

| Metric | Value |
|--------|:-----:|
| Accuracy | 88.81% |
| Training Samples | 534 |
| Model File Size | ~21 KB |

### How to Use

1. Open the page in a browser
2. Drag and drop an audio file (MP3, WAV, FLAC, M4A)
3. Wait 5 to 15 seconds for the analysis
4. Read the result and confidence percentage

### Technical Details

- **Frontend**: Vanilla HTML/CSS/JS, Web Audio API, custom audio processing
- **ML Engine**: ONNX Runtime Web (runs via WebAssembly in the browser)
- **Training**: Python with librosa, XGBoost, scikit-learn
- **Analysis**: Uses 61 acoustic features including frequency patterns, rhythm strength, and spectral characteristics

### Limitations

- Currently trained on **Suno 5.5** AI-generated music; other AI generators may produce different results
- Works best with music that has both vocals and instruments
- Not designed to detect AI vocals isolated from instrumental backing

---

<a id="chinese"></a>
## 中文

一个在浏览器里运行的 AI 音乐检测工具。不用上传文件，保护隐私。

### 它能做什么

拖一个音乐文件进去，工具会分析声音特征，告诉你这首歌更可能是 AI 生成的还是真人创作的。

### 特点

- **打开浏览器就能用**：不需要安装任何软件，Chrome、Edge、Firefox、Safari 都支持
- **保护隐私**：音频全程在本机处理，不会传到任何服务器
- **分析很快**：一首歌通常 5 到 15 秒就能出结果
- **结果直观**：直接显示"AI 生成"或"真人创作"的判断，附带置信度百分比
- **体积很小**：整个工具不到 30 KB（不含模型文件）

### 模型表现

| 指标 | 数值 |
|------|:---:|
| 准确率 | 88.81% |
| 训练样本数 | 534 |
| 模型文件大小 | ~21 KB |

### 怎么使用

1. 用浏览器打开页面
2. 把音频文件拖到检测区域（支持 MP3、WAV、FLAC、M4A）
3. 等 5 到 15 秒，工具会自动分析
4. 查看判定结果和置信度

### 技术细节

- **前端**：原生 HTML/CSS/JS，Web Audio API，自研音频处理算法
- **推理引擎**：ONNX Runtime Web（在浏览器里通过 WebAssembly 运行）
- **训练**：Python + librosa + XGBoost + scikit-learn
- **分析方式**：提取 61 维声音特征，包括频率分布、节奏强度、频谱特性等

### 局限性

- 目前只用 **Suno 5.5** 生成的音乐训练过，其他 AI 工具生成的音乐可能判断不准
- 对纯乐器演奏或者编曲比较简单的音乐，检测效果会打折扣
- 不适合单独检测 AI 生成的人声（没有乐器伴奏的情况）

---

## License

MIT
