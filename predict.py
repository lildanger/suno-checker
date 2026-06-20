import sys
import os
import warnings

# 强制添加脚本所在绝对目录至 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 解决 PyInstaller 单文件打包后 DLL 搜索路径问题
if getattr(sys, 'frozen', False):
    _meipass = sys._MEIPASS
    # 递归添加所有含 DLL/PYD 的目录，确保 onnxruntime 等原生扩展能找到依赖
    for _root, _dirs, _files in os.walk(_meipass):
        if any(f.endswith(('.dll', '.pyd', '.so')) for f in _files):
            try:
                os.add_dll_directory(_root)
            except (AttributeError, OSError):
                pass

# 优先导入 onnxruntime，避免与 PyQt5 产生的内存库锁定冲突
try:
    import onnxruntime as ort
    ort_available = True
    ort_error_msg = ""
except Exception as e:
    import traceback
    ort_available = False
    ort_error_msg = traceback.format_exc()

# ==========================================
# 修复 Windows venv 下找不到 Qt 插件的 Bug
# ==========================================
import PyQt5
plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
os.environ['QT_PLUGIN_PATH'] = plugin_path

# ==========================================
# 路径解析：兼容开发环境和 PyInstaller 打包后的路径
# ==========================================
def get_base_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))

import numpy as np
import librosa
import pyloudnorm as pyln

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QTextEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 基于 CNN 的音频时频图预处理逻辑 (扫描整首歌曲)
# ==========================================
def preprocess_audio_to_segments(file_path, sample_rate=22050, duration=3, n_mels=128):
    # 1. 载入完整音频
    y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    
    # 2. 对整首歌进行 LUFS 响度归一化 (-23.0 LUFS)
    try:
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(y)
        if loudness > -100 and not np.isnan(loudness) and not np.isinf(loudness):
            y = pyln.normalize.loudness(y, loudness, -23.0)
    except Exception:
        pass
        
    # 3. 按步长为 3 秒将整首歌切分为不重叠切片
    target_length = int(sample_rate * duration)
    total_len = len(y)
    
    segments = []
    step = target_length
    start = 0
    while start < total_len:
        end = start + target_length
        if end <= total_len:
            seg = y[start:end]
        else:
            # 最后一小段不足 3 秒，使用常数零填充
            seg = y[start:]
            seg = np.pad(seg, (0, target_length - len(seg)), mode='constant')
        segments.append(seg)
        start += step
        
    # 兜底：如果音频为空或极短，确保至少有一个 3 秒片段
    if len(segments) == 0:
        seg = np.zeros(target_length)
        segments.append(seg)
        
    input_tensors = []
    target_width = int(target_length / 512) + 1
    
    # 对每个切片提取 Log-Mel 频谱图
    for seg in segments:
        mel_spec = librosa.feature.melspectrogram(
            y=seg, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=n_mels
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 归一化到 [-1, 1]
        s_min = log_mel_spec.min()
        s_max = log_mel_spec.max()
        log_mel_spec = (log_mel_spec - s_min) / (s_max - s_min + 1e-6)
        log_mel_spec = (log_mel_spec * 2.0) - 1.0
        
        if log_mel_spec.shape[1] < target_width:
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, target_width - log_mel_spec.shape[1])), mode='constant')
        elif log_mel_spec.shape[1] > target_width:
            log_mel_spec = log_mel_spec[:, :target_width]
            
        input_tensor = np.expand_dims(log_mel_spec, axis=(0, 1)).astype(np.float32)
        input_tensors.append(input_tensor)
        
    return input_tensors

# ==========================================
# 后台工作线程
# ==========================================
class AnalysisThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int, str)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        filename = os.path.basename(self.file_path)
        try:
            self.log_signal.emit(f"[进程启动] 正在读取音频轨道: {filename}...\n(基于 ISMIR 2025 fakeprint 频谱伪影检测)")

            # 1. 查找 ONNX 模型文件 (优先新版 fakeprint 模型，fallback 旧版)
            base_dir = get_base_dir()
            onnx_path = os.path.join(base_dir, 'ai_music_detector.onnx')
            if not os.path.exists(onnx_path):
                onnx_path = 'ai_music_detector.onnx'
            if not os.path.exists(onnx_path):
                onnx_path = os.path.join(base_dir, 'suno_detector_model.onnx')
            if not os.path.exists(onnx_path):
                onnx_path = 'suno_detector_model.onnx'
            if not os.path.exists(onnx_path):
                self.log_signal.emit("[系统错误] ONNX 模型文件缺失！请确保 ai_music_detector.onnx 存在于程序目录。")
                self.finished_signal.emit(-1, "")
                return

            is_new_model = 'ai_music_detector' in onnx_path
            if is_new_model:
                self.log_signal.emit("[模型] lofcz/ai-music-detector (fakeprint + 逻辑回归, ISMIR 2025)")
            else:
                self.log_signal.emit("[模型] ResNet-18 CNN (旧版)")

            # 2. 推理
            if is_new_model:
                # --- 新版 fakeprint 模型 ---
                from fakeprint import extract_fakeprint
                t0 = __import__('time').time()
                fakeprint = extract_fakeprint(self.file_path)
                extract_time = __import__('time').time() - t0

                self.log_signal.emit(f"频谱伪影特征提取完成 (维度: {len(fakeprint)}, 耗时: {extract_time:.1f}s)")

                session = ort.InferenceSession(onnx_path)
                output = session.run(None, {"fakeprint": fakeprint.reshape(1, -1).astype(np.float32)})
                ai_prob_raw = float(output[0][0, 0])
                human_prob_raw = 1.0 - ai_prob_raw
                prediction = 1 if ai_prob_raw > 0.5 else 0
                ai_prob = ai_prob_raw * 100
                human_prob = human_prob_raw * 100
            else:
                # --- 旧版 ResNet-18 CNN 模型 ---
                input_tensors = preprocess_audio_to_segments(self.file_path)
                num_segments = len(input_tensors)
                self.log_signal.emit(f"已切分为 {num_segments} 个音频片段进行扫描...")

                session = ort.InferenceSession(onnx_path)
                input_name = session.get_inputs()[0].name

                all_probabilities = []
                for t in input_tensors:
                    raw_outputs = session.run(None, {input_name: t})[0]
                    exp_logits = np.exp(raw_outputs - np.max(raw_outputs, axis=1, keepdims=True))
                    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                    all_probabilities.append(probabilities[0])

                avg_probabilities = np.mean(all_probabilities, axis=0)
                human_prob_raw = float(avg_probabilities[0])
                ai_prob_raw = float(avg_probabilities[1])
                prediction = int(np.argmax(avg_probabilities))
                ai_prob = ai_prob_raw * 100
                human_prob = human_prob_raw * 100

            # 3. 生成判定报告 (HTML 格式)
            if prediction == 1:
                res = f"""
                <div style="margin: 5px 0; padding: 12px; border-radius: 8px; border: 1px solid #ff3333; background-color: rgba(255, 51, 51, 0.05);">
                    <div style="font-size: 15px; font-weight: bold; color: #ff4d4d; margin-bottom: 6px;">
                        [判定结果] AI 生成概率极高
                    </div>
                    <div style="font-size: 12px; color: #ffa4a4; margin-bottom: 12px;">
                        检测到神经声码器反卷积伪影 (fakeprint)，频谱存在等间距峰值特征。
                    </div>
                    <div style="border-top: 1px solid rgba(255, 51, 51, 0.15); padding-top: 8px;">
                        <table width="100%" style="font-family: monospace; font-size: 12px; color: #e0e0e0;">
                            <tr>
                                <td style="padding: 2px 0;">AI 声码器指纹匹配度:</td>
                                <td align="right" style="color: #ff4d4d; font-weight: bold; font-size: 13px;">{ai_prob:.2f}%</td>
                            </tr>
                            <tr>
                                <td style="padding: 2px 0;">真人音频特征匹配度:</td>
                                <td align="right" style="color: #888888;">{human_prob:.2f}%</td>
                            </tr>
                        </table>
                    </div>
                    <div style="margin-top: 8px; height: 5px; background-color: #222222; border-radius: 3px; overflow: hidden; display: flex;">
                        <div style="width: {ai_prob:.1f}%; background: linear-gradient(90deg, #ff8a80, #ff4d4d);"></div>
                        <div style="width: {human_prob:.1f}%; background-color: #333333;"></div>
                    </div>
                </div>
                """
            else:
                res = f"""
                <div style="margin: 5px 0; padding: 12px; border-radius: 8px; border: 1px solid #00e676; background-color: rgba(0, 230, 118, 0.05);">
                    <div style="font-size: 15px; font-weight: bold; color: #00e676; margin-bottom: 6px;">
                        [判定结果] 真人制作概率极高
                    </div>
                    <div style="font-size: 12px; color: #a7ffeb; margin-bottom: 12px;">
                        未检测到神经声码器反卷积伪影，频谱结构呈现自然声学特征。
                    </div>
                    <div style="border-top: 1px solid rgba(0, 230, 118, 0.15); padding-top: 8px;">
                        <table width="100%" style="font-family: monospace; font-size: 12px; color: #e0e0e0;">
                            <tr>
                                <td style="padding: 2px 0;">真人音频特征匹配度:</td>
                                <td align="right" style="color: #00e676; font-weight: bold; font-size: 13px;">{human_prob:.2f}%</td>
                            </tr>
                            <tr>
                                <td style="padding: 2px 0;">AI 声码器指纹匹配度:</td>
                                <td align="right" style="color: #888888;">{ai_prob:.2f}%</td>
                            </tr>
                        </table>
                    </div>
                    <div style="margin-top: 8px; height: 5px; background-color: #222222; border-radius: 3px; overflow: hidden; display: flex;">
                        <div style="width: {human_prob:.1f}%; background: linear-gradient(90deg, #b9f6ca, #00e676);"></div>
                        <div style="width: {ai_prob:.1f}%; background-color: #333333;"></div>
                    </div>
                </div>
                """

            self.finished_signal.emit(prediction, res)

        except Exception as e:
            self.log_signal.emit(f"[异常] 分析发生崩溃: {str(e)}")
            self.finished_signal.emit(-1, "")

# ==========================================
# 动画拖拽区域控件
# ==========================================
from PyQt5.QtWidgets import QGraphicsDropShadowEffect, QGraphicsOpacityEffect
from PyQt5.QtCore import QPropertyAnimation, pyqtProperty, QPoint
from PyQt5.QtGui import QPainter, QLinearGradient, QColor, QPen, QBrush
import math

class DropArea(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(220)
        self.setAcceptDrops(False)  # 拖拽事件由主窗口统一调度
        
        self._scan_y = 0.0
        self.is_analyzing = False
        
        # 阴影发光效果
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(15)
        self.shadow.setColor(QColor(0, 190, 255, 20))
        self.shadow.setOffset(0, 0)
        self.setGraphicsEffect(self.shadow)
        
        # 阴影颜色动画
        self.shadow_anim = QPropertyAnimation(self.shadow, b"color")
        self.shadow_anim.setDuration(250)
        
        # 扫描激光线位置动画
        self.scan_anim = QPropertyAnimation(self, b"scan_y")
        self.scan_anim.setDuration(1600)
        self.scan_anim.setStartValue(0.0)
        
        self.set_default_text()
        self.set_style_normal()

    def set_default_text(self):
        self.setText("↓\n拖拽音频文件至此\n支持格式: MP3 / WAV / FLAC")

    @pyqtProperty(float)
    def scan_y(self):
        return self._scan_y

    @scan_y.setter
    def scan_y(self, val):
        self._scan_y = val
        self.update()

    def set_style_normal(self):
        self.setStyleSheet("""
            QLabel {
                background-color: #11141d;
                border: 2px dashed #2c3547;
                border-radius: 12px;
                font-size: 15px;
                color: #8c9cb2;
                padding: 20px;
            }
        """)

    def set_style_hover(self):
        self.setStyleSheet("""
            QLabel {
                background-color: #152236;
                border: 2px dashed #00e5ff;
                border-radius: 12px;
                font-size: 15px;
                color: #00e5ff;
                font-weight: bold;
                padding: 20px;
            }
        """)

    def drag_enter(self):
        self.set_style_hover()
        self.shadow_anim.stop()
        self.shadow_anim.setStartValue(self.shadow.color())
        self.shadow_anim.setEndValue(QColor(0, 229, 255, 180))
        self.shadow_anim.start()

    def drag_leave(self):
        self.set_style_normal()
        self.shadow_anim.stop()
        self.shadow_anim.setStartValue(self.shadow.color())
        self.shadow_anim.setEndValue(QColor(0, 190, 255, 20))
        self.shadow_anim.start()

    def start_scan(self):
        self.is_analyzing = True
        self.setText("🔬\n正在检测音频指纹\nCNN 时频扫描分析中...")
        self.setStyleSheet("""
            QLabel {
                background-color: #0d1726;
                border: 2px solid #00e5ff;
                border-radius: 12px;
                font-size: 15px;
                color: #00e5ff;
                font-weight: bold;
                padding: 20px;
            }
        """)
        
        self.scan_anim.stop()
        self.scan_anim.setEndValue(float(self.height()))
        self.scan_anim.setLoopCount(-1)
        self.scan_anim.start()
        
        self.shadow_anim.stop()
        self.shadow_anim.setStartValue(self.shadow.color())
        self.shadow_anim.setEndValue(QColor(0, 229, 255, 200))
        self.shadow_anim.start()

    def stop_scan(self, prediction):
        self.scan_anim.stop()
        self.is_analyzing = False
        
        if prediction == 1: # AI
            self.setText("⚠️\n判定结果: AI 生成")
            self.setStyleSheet("""
                QLabel {
                    background-color: #241111;
                    border: 2px solid #ff4d4d;
                    border-radius: 12px;
                    font-size: 16px;
                    color: #ff4d4d;
                    font-weight: bold;
                    padding: 20px;
                }
            """)
            self.flash_pulse(QColor(255, 77, 77))
        else: # Human
            self.setText("✅\n判定结果: 真人制作")
            self.setStyleSheet("""
                QLabel {
                    background-color: #112415;
                    border: 2px solid #00e676;
                    border-radius: 12px;
                    font-size: 16px;
                    color: #00e676;
                    font-weight: bold;
                    padding: 20px;
                }
            """)
            self.flash_pulse(QColor(0, 230, 118))

    def flash_pulse(self, color):
        self.shadow_anim.stop()
        self.shadow_anim.setDuration(1200)
        self.shadow_anim.setStartValue(QColor(color.red(), color.green(), color.blue(), 255))
        self.shadow_anim.setKeyValueAt(0.0, QColor(color.red(), color.green(), color.blue(), 255))
        self.shadow_anim.setKeyValueAt(0.25, QColor(color.red(), color.green(), color.blue(), 30))
        self.shadow_anim.setKeyValueAt(0.5, QColor(color.red(), color.green(), color.blue(), 255))
        self.shadow_anim.setKeyValueAt(0.75, QColor(color.red(), color.green(), color.blue(), 30))
        self.shadow_anim.setKeyValueAt(1.0, QColor(color.red(), color.green(), color.blue(), 90))
        self.shadow_anim.start()

    def paintEvent(self, event):
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        
        # 1. 扫描中，绘制精美音量声波跳动
        if self.is_analyzing:
            painter.setPen(Qt.NoPen)
            bar_width = 5
            gap = 3
            num_bars = 24
            total_width = num_bars * bar_width + (num_bars - 1) * gap
            start_x = (w - total_width) // 2
            
            for i in range(num_bars):
                dist_from_center = abs(i - num_bars / 2.0) / (num_bars / 2.0)
                phase = (i * 0.4) + (self._scan_y * 0.06)
                amplitude = (math.sin(phase) + 1.0) / 2.0
                amplitude = amplitude * (1.0 - dist_from_center * 0.55)
                amplitude = amplitude * 0.75 + (i % 3) * 0.05 + 0.1
                
                bar_height = int(amplitude * 60) + 6
                x = start_x + i * (bar_width + gap)
                y = int(h * 0.62) - bar_height // 2
                
                color = QColor(0, 229, 255, int(amplitude * 160) + 70)
                painter.setBrush(QBrush(color))
                painter.drawRoundedRect(x, y, bar_width, bar_height, 2, 2)
            
            # 2. 绘制上下移动的激光扫描线
            scan_gradient = QLinearGradient(0, self._scan_y - 12, 0, self._scan_y + 12)
            scan_gradient.setColorAt(0.0, QColor(0, 229, 255, 0))
            scan_gradient.setColorAt(0.5, QColor(0, 229, 255, 180))
            scan_gradient.setColorAt(1.0, QColor(0, 229, 255, 0))
            painter.fillRect(2, int(self._scan_y - 12), w - 4, 24, scan_gradient)
            
            painter.setPen(QPen(QColor(0, 229, 255, 255), 2))
            painter.drawLine(2, int(self._scan_y), w - 2, int(self._scan_y))


# ==========================================
# GUI 主界面
# ==========================================
from PyQt5.QtWidgets import QHBoxLayout

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Suno AI 音乐检测 (Fakeprint ISMIR 2025) - DanJuan v0.4")
        self.resize(480, 600)
        self.setAcceptDrops(True)

        # 全局深色底色
        self.setStyleSheet("""
            QMainWindow {
                background-color: #08090c;
            }
            QWidget#centralWidget {
                background-color: #08090c;
            }
        """)

        font_family = "Microsoft YaHei" if sys.platform == "win32" else "sans-serif"
        font = QFont(font_family, 9)
        QApplication.setFont(font)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(12)

        # 1. 顶部 Header 装饰栏
        self.header = QWidget()
        self.header.setStyleSheet("background-color: #11141c; border-radius: 6px;")
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(12, 8, 12, 8)
        self.header.setLayout(header_layout)

        title_label = QLabel("DanJuan AI 音频检测系统")
        title_label.setStyleSheet("color: #ffffff; font-size: 13px; font-weight: bold; font-family: 'Microsoft YaHei';")
        
        self.status_dot = QLabel("● 引擎就绪")
        self.status_dot.setStyleSheet("color: #00e676; font-size: 11px; font-weight: bold; font-family: 'Microsoft YaHei';")
        
        # 呼吸灯动效
        self.dot_opacity = QGraphicsOpacityEffect(self.status_dot)
        self.status_dot.setGraphicsEffect(self.dot_opacity)
        self.dot_anim = QPropertyAnimation(self.dot_opacity, b"opacity")
        self.dot_anim.setDuration(1600)
        self.dot_anim.setStartValue(0.25)
        self.dot_anim.setEndValue(1.0)
        self.dot_anim.setKeyValueAt(0.5, 1.0)
        self.dot_anim.setKeyValueAt(1.0, 0.25)
        self.dot_anim.setLoopCount(-1)
        self.dot_anim.start()

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_dot)

        # 2. 拖拽区域
        self.drop_label = DropArea(self)

        # 3. 日志及卡片输出区
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Microsoft YaHei', monospace;
                font-size: 12px;
                background-color: #0b0c10;
                color: #c5c6c7;
                border: 1px solid #1f2833;
                border-radius: 8px;
                padding: 12px;
            }
            QScrollBar:vertical {
                border: none;
                background: #0b0c10;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #1f2833;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #45a29e;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        if ort_available:
            welcome_text = "<span style='color: #45a29e;'>[系统初始化完成] lofcz/ai-music-detector (ISMIR 2025)</span><br>模型: fakeprint + 逻辑回归 | Accuracy: 99.88% | FPR: 0.31%<br>基于神经声码器反卷积伪影检测 — 非统计特征学习<br>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            self.log_text.append(welcome_text)
        else:
            self.status_dot.setText("● 引擎异常")
            self.status_dot.setStyleSheet("color: #ff3333; font-size: 11px; font-weight: bold; font-family: 'Microsoft YaHei';")
            error_text = f"""
            <span style='color: #ff3333; font-weight: bold; font-size: 14px;'>[引擎初始化失败] ONNX 推理模块加载崩溃</span><br><br>
            <b>崩溃详细原因：</b><br>
            <span style='color: #ff8a80;'>{ort_error_msg.replace(chr(10), '<br>')}</span><br><br>
            <b>通常解决方法：</b><br>
            1. 您的电脑缺少微软 C++ 运行库，请下载安装：<br>
               <a href="https://aka.ms/vs/17/release/vc_redist.x64.exe" style="color: #00e5ff; font-weight: bold;">点击此处下载微软官方 VC++ 2015-2022 运行库 (x64)</a><br>
               (安装后请重启电脑再运行此程序)<br><br>
            2. 如果您的 CPU 极其老旧（不支持 AVX/AVX2 指令集，如早期的奔腾、赛扬等），将无法运行此版本的 ONNX 引擎。<br>
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>
            """
            self.log_text.append(error_text)

        # 组装布局
        self.main_layout.addWidget(self.header)
        self.main_layout.addWidget(self.drop_label)
        self.main_layout.addWidget(self.log_text)

        container = QWidget()
        container.setObjectName("centralWidget")
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                if file_path.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                    event.acceptProposedAction()
                    self.drop_label.drag_enter()
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.drop_label.drag_leave()

    def dropEvent(self, event):
        self.drop_label.drag_leave()
        file_path = event.mimeData().urls()[0].toLocalFile()
        self.start_analysis(file_path)

    def start_analysis(self, file_path):
        if not ort_available:
            self.log_text.append("\n❌ <span style='color: #ff3333; font-weight: bold;'>[错误] 引擎未就绪，无法进行音频分析。请先根据上述说明安装微软运行库。</span>")
            return

        if hasattr(self, 'worker') and self.worker.isRunning():
            self.log_text.append("\n⚠️ <span style='color: #ff3333;'>[忙碌] 核心正在处理上一任务，请稍后再试！</span>")
            return

        self.log_text.clear()
        self.log_text.append(f"📁 <span style='color: #45a29e; font-weight: bold;'>导入目标文件:</span> {os.path.basename(file_path)}<br>")
        
        # 更新状态灯为扫描态
        self.status_dot.setText("● 正在扫描")
        self.status_dot.setStyleSheet("color: #00e5ff; font-size: 11px; font-weight: bold; font-family: 'Microsoft YaHei';")
        
        self.drop_label.start_scan()
        
        self.worker = AnalysisThread(file_path)
        self.worker.log_signal.connect(self.update_log)
        self.worker.finished_signal.connect(self.on_analysis_finished)
        self.worker.start()

    def update_log(self, text):
        self.log_text.append(text)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_analysis_finished(self, prediction, res_html):
        # 恢复状态灯为就绪态
        self.status_dot.setText("● 引擎就绪")
        self.status_dot.setStyleSheet("color: #00e676; font-size: 11px; font-weight: bold; font-family: 'Microsoft YaHei';")
        
        if prediction == -1:
            self.drop_label.set_default_text()
            self.drop_label.set_style_normal()
            self.drop_label.scan_anim.stop()
            self.drop_label.is_analyzing = False
            self.drop_label.shadow_anim.stop()
            self.drop_label.shadow.setColor(QColor(0, 190, 255, 20))
            self.drop_label.update()
        else:
            self.drop_label.stop_scan(prediction)
            self.log_text.append(res_html)
            
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())