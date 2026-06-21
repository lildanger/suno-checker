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

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QTextEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
import qtawesome as qta

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 后台工作线程
# ==========================================
class AnalysisThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int, str)
    
    def __init__(self, file_path, session):
        super().__init__()
        self.file_path = file_path
        self.session = session

    def run(self):
        try:
            if self.session is None:
                self.log_signal.emit("[系统错误] ONNX 推理会话未就绪！无法进行推理。")
                self.finished_signal.emit(-1, "")
                return

            # --- 新版 fakeprint 模型 ---
            from fakeprint import extract_fakeprint
            t0 = __import__('time').time()
            fakeprint = extract_fakeprint(self.file_path)
            extract_time = __import__('time').time() - t0

            self.log_signal.emit(f"频谱伪影特征提取完成 (维度: {len(fakeprint)}, 耗时: {extract_time:.1f}s)")

            output = self.session.run(None, {"fakeprint": fakeprint.reshape(1, -1).astype(np.float32)})
            ai_prob_raw = float(output[0][0, 0])
            human_prob_raw = 1.0 - ai_prob_raw
            prediction = 1 if ai_prob_raw > 0.5 else 0
            ai_prob = ai_prob_raw * 100
            human_prob = human_prob_raw * 100

            # 3. 生成判定报告 (HTML 格式)
            if prediction == 1:
                res = f"""
                <div style="margin: 6px 0; padding: 12px; border-radius: 8px; border: 1px solid #ff3333; background-color: rgba(255, 51, 51, 0.05); font-family: 'Microsoft YaHei', sans-serif;">
                    <div style="font-size: 15px; font-weight: bold; color: #ff4d4d; margin-bottom: 6px;">
                        [判定结果] AI 生成概率极高
                    </div>
                    <div style="font-size: 12px; color: #ffa4a4; margin-bottom: 12px;">
                        检测到神经声码器反卷积伪影 (fakeprint)，频谱存在等间距峰值特征。
                    </div>
                    <div style="border-top: 1px dashed rgba(255, 51, 51, 0.15); padding-top: 8px;">
                        <table width="100%" style="font-family: monospace; font-size: 12px; color: #cbd5e1;">
                            <tr>
                                <td style="padding: 2px 0; color: #8c9cb2;">AI 声码器指纹匹配度:</td>
                                <td align="right" style="color: #ff4d4d; font-weight: bold; font-size: 13px;">{ai_prob:.2f}%</td>
                            </tr>
                            <tr>
                                <td style="padding: 2px 0; color: #8c9cb2;">真人音频特征匹配度:</td>
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
                <div style="margin: 6px 0; padding: 12px; border-radius: 8px; border: 1px solid #00e676; background-color: rgba(0, 230, 118, 0.05); font-family: 'Microsoft YaHei', sans-serif;">
                    <div style="font-size: 15px; font-weight: bold; color: #00e676; margin-bottom: 6px;">
                        [判定结果] 真人制作概率极高
                    </div>
                    <div style="font-size: 12px; color: #a7ffeb; margin-bottom: 12px;">
                        未检测到神经声码器反卷积伪影，频谱呈现自然声学特征。
                    </div>
                    <div style="border-top: 1px dashed rgba(0, 230, 118, 0.15); padding-top: 8px;">
                        <table width="100%" style="font-family: monospace; font-size: 12px; color: #cbd5e1;">
                            <tr>
                                <td style="padding: 2px 0; color: #8c9cb2;">真人音频特征匹配度:</td>
                                <td align="right" style="color: #00e676; font-weight: bold; font-size: 13px;">{human_prob:.2f}%</td>
                            </tr>
                            <tr>
                                <td style="padding: 2px 0; color: #8c9cb2;">AI 声码器指纹匹配度:</td>
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
        self.setObjectName("DropArea")
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
        
        # 矢量图标包
        import qtawesome as qta
        self.qta = qta
        
        # 子布局与组件，用于优雅地呈现矢量图标及下方文本
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 25, 15, 25)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)
        
        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet("background: transparent; border: none;")
        
        self.text_label = QLabel()
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setStyleSheet("background: transparent; border: none; font-size: 15px; color: #8c9cb2; font-family: 'Microsoft YaHei';")
        self.text_label.setWordWrap(True)
        
        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)
        
        self.set_default_text()
        self.set_style_normal()

    def set_default_text(self):
        # 使用上传云矢量图标
        icon = self.qta.icon('fa5s.cloud-upload-alt', color='#8c9cb2')
        self.icon_label.setPixmap(icon.pixmap(48, 48))
        self.text_label.setText("拖拽音频文件至此\n支持格式: MP3 / WAV / FLAC / M4A")
        self.text_label.setStyleSheet("background: transparent; border: none; font-size: 13px; color: #8c9cb2; font-family: 'Microsoft YaHei'; line-height: 1.5;")

    @pyqtProperty(float)
    def scan_y(self):
        return self._scan_y

    @scan_y.setter
    def scan_y(self, val):
        self._scan_y = val
        self.update()

    def set_style_normal(self):
        self.setStyleSheet("""
            QLabel#DropArea {
                background-color: #11141d;
                border: 2px dashed #2c3547;
                border-radius: 12px;
            }
        """)

    def set_style_hover(self):
        self.setStyleSheet("""
            QLabel#DropArea {
                background-color: #152236;
                border: 2px dashed #00e5ff;
                border-radius: 12px;
            }
        """)

    def drag_enter(self):
        self.set_style_hover()
        icon = self.qta.icon('fa5s.cloud-upload-alt', color='#00e5ff')
        self.icon_label.setPixmap(icon.pixmap(48, 48))
        self.text_label.setText("释放文件以开始分析")
        self.text_label.setStyleSheet("background: transparent; border: none; font-size: 13px; color: #00e5ff; font-weight: bold; font-family: 'Microsoft YaHei';")
        self.shadow_anim.stop()
        self.shadow_anim.setStartValue(self.shadow.color())
        self.shadow_anim.setEndValue(QColor(0, 229, 255, 180))
        self.shadow_anim.start()

    def drag_leave(self):
        self.set_default_text()
        self.set_style_normal()
        self.shadow_anim.stop()
        self.shadow_anim.setStartValue(self.shadow.color())
        self.shadow_anim.setEndValue(QColor(0, 190, 255, 20))
        self.shadow_anim.start()

    def start_scan(self):
        self.is_analyzing = True
        icon = self.qta.icon('fa5s.microscope', color='#00e5ff')
        self.icon_label.setPixmap(icon.pixmap(48, 48))
        self.text_label.setText("正在检测音频指纹\n频谱伪影分析中...")
        self.text_label.setStyleSheet("background: transparent; border: none; font-size: 13px; color: #ffffff; font-weight: bold; font-family: 'Microsoft YaHei';")
        self.setStyleSheet("""
            QLabel#DropArea {
                background-color: #0d1726;
                border: 2px solid #00e5ff;
                border-radius: 12px;
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
            icon = self.qta.icon('fa5s.exclamation-triangle', color='#ff4d4d')
            self.icon_label.setPixmap(icon.pixmap(48, 48))
            self.text_label.setText("判定结果: AI 生成")
            self.text_label.setStyleSheet("background: transparent; border: none; font-size: 14px; color: #ff4d4d; font-weight: bold; font-family: 'Microsoft YaHei';")
            self.setStyleSheet("""
                QLabel#DropArea {
                    background-color: #241111;
                    border: 2px solid #ff4d4d;
                    border-radius: 12px;
                }
            """)
            self.flash_pulse(QColor(255, 77, 77))
        else: # Human
            icon = self.qta.icon('fa5s.check-circle', color='#00e676')
            self.icon_label.setPixmap(icon.pixmap(48, 48))
            self.text_label.setText("判定结果: 真人制作")
            self.text_label.setStyleSheet("background: transparent; border: none; font-size: 14px; color: #00e676; font-weight: bold; font-family: 'Microsoft YaHei';")
            self.setStyleSheet("""
                QLabel#DropArea {
                    background-color: #112415;
                    border: 2px solid #00e676;
                    border-radius: 12px;
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
                y = int(h * 0.65) - bar_height // 2
                
                color = QColor(0, 229, 255, int(amplitude * 160) + 70)
                painter.setBrush(QBrush(color))
                painter.drawRoundedRect(x, y, bar_width, bar_height, 2, 2)
            
            # 2. 绘制上下移动的激光扫描线
            scan_gradient = QLinearGradient(0, self._scan_y - 12, 0, self._scan_y + 12)
            scan_gradient.setColorAt(0.0, QColor(0, 229, 255, 0))
            scan_gradient.setColorAt(0.5, QColor(0, 229, 255, 180))
            scan_gradient.setColorAt(1.0, QColor(0, 229, 255, 0))
            painter.fillRect(2, int(self._scan_y - 12), w - 4, 24, scan_gradient)
            
            # 两端淡出的发光激光线
            line_grad = QLinearGradient(0, self._scan_y, w, self._scan_y)
            line_grad.setColorAt(0.0, QColor(0, 229, 255, 0))
            line_grad.setColorAt(0.1, QColor(0, 229, 255, 100))
            line_grad.setColorAt(0.5, QColor(0, 229, 255, 255))
            line_grad.setColorAt(0.9, QColor(0, 229, 255, 100))
            line_grad.setColorAt(1.0, QColor(0, 229, 255, 0))
            
            painter.setPen(QPen(QBrush(line_grad), 2))
            painter.drawLine(2, int(self._scan_y), w - 2, int(self._scan_y))


# ==========================================
# GUI 主界面
# ==========================================
from PyQt5.QtWidgets import QHBoxLayout

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Suno AI 音乐检测 - DanJuan v0.4")
        self.resize(480, 600)
        self.setAcceptDrops(True)

        # 设置窗口图标
        icon_path = os.path.join(get_base_dir(), 'app_icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # 初始化 ONNX 推理会话 (仅加载一次)
        self.session = None
        self.worker = None
        global ort_available, ort_error_msg
        if ort_available:
            try:
                base_dir = get_base_dir()
                onnx_path = os.path.join(base_dir, 'ai_music_detector.onnx')
                if not os.path.exists(onnx_path):
                    onnx_path = 'ai_music_detector.onnx'
                if not os.path.exists(onnx_path):
                    onnx_path = os.path.join(base_dir, 'suno_detector_model.onnx')
                if not os.path.exists(onnx_path):
                    onnx_path = 'suno_detector_model.onnx'
                
                if os.path.exists(onnx_path):
                    self.session = ort.InferenceSession(onnx_path)
                else:
                    raise FileNotFoundError("未找到 ai_music_detector.onnx 或 suno_detector_model.onnx 模型文件！")
            except Exception as e:
                import traceback
                ort_available = False
                ort_error_msg = traceback.format_exc()

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

        # 带有副标题的标题排版
        title_container = QWidget()
        title_container.setStyleSheet("background: transparent; border: none;")
        title_vbox = QVBoxLayout()
        title_vbox.setContentsMargins(0, 0, 0, 0)
        title_vbox.setSpacing(2)
        title_container.setLayout(title_vbox)

        title_label = QLabel("DanJuan AI 音频检测系统")
        title_label.setStyleSheet("color: #ffffff; font-size: 13px; font-weight: bold; font-family: 'Microsoft YaHei'; background: transparent; border: none;")
        
        subtitle_label = QLabel("FAKAPRINT SPECTRAL DETECTOR v0.4")
        subtitle_label.setStyleSheet("color: #8c9cb2; font-size: 9px; font-family: 'Consolas', monospace; font-weight: bold; background: transparent; border: none;")
        
        title_vbox.addWidget(title_label)
        title_vbox.addWidget(subtitle_label)

        self.status_dot = QLabel("● 引擎就绪")
        self.status_dot.setStyleSheet("""
            color: #00e676;
            background-color: rgba(0, 230, 118, 0.08);
            border: 1px solid rgba(0, 230, 118, 0.25);
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 11px;
            font-weight: bold;
        """)
        
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

        header_layout.addWidget(title_container)
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
            welcome_text = """
            <span style='color: #00e5ff; font-weight: bold;'>[系统就绪] lofcz/ai-music-detector READY.</span><br>基于神经声码器反卷积伪影检测，可检测suno ≤ 5.5 / udio ≤ 1.5<br>
            模型: fakeprint + 逻辑回归 | 准确率: 99.88% | 误判率: 0.31%<br>
            
            <span style='color: #1f2833;'>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span><br>
            """
            self.log_text.append(welcome_text)
        else:
            self.status_dot.setText("● 引擎异常")
            self.status_dot.setStyleSheet("""
                color: #ff3333;
                background-color: rgba(255, 51, 51, 0.08);
                border: 1px solid rgba(255, 51, 51, 0.25);
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: bold;
            """)
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

        if self.worker is not None and self.worker.isRunning():
            self.log_text.append("\n⚠️ <span style='color: #ff3333;'>[忙碌] 核心正在处理上一任务，请稍后再试！</span>")
            return

        # 将 qtawesome 图标转为 Base64 在 HTML 中渲染
        from PyQt5.QtCore import QByteArray, QBuffer, QIODevice
        icon = qta.icon('fa5s.file-audio', color='#45a29e')
        pixmap = icon.pixmap(14, 14)
        ba = QByteArray()
        buffer = QBuffer(ba)
        buffer.open(QIODevice.WriteOnly)
        pixmap.save(buffer, "PNG")
        icon_base64 = ba.toBase64().data().decode("utf-8")

        self.log_text.clear()
        self.log_text.append(f'<img src="data:image/png;base64,{icon_base64}" style="vertical-align: middle;"> <span style="color: #45a29e; font-weight: bold; vertical-align: middle;">导入目标文件:</span> <b style="vertical-align: middle;">{os.path.basename(file_path)}</b><br>')
        
        # 更新状态灯为扫描态
        self.status_dot.setText("● 正在扫描")
        self.status_dot.setStyleSheet("""
            color: #00e5ff;
            background-color: rgba(0, 229, 255, 0.08);
            border: 1px solid rgba(0, 229, 255, 0.25);
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 11px;
            font-weight: bold;
        """)
        
        self.drop_label.start_scan()
        
        self.worker = AnalysisThread(file_path, self.session)
        self.worker.log_signal.connect(self.update_log)
        self.worker.finished_signal.connect(self.on_analysis_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_finished(self):
        self.worker = None

    def update_log(self, text):
        self.log_text.append(text)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_analysis_finished(self, prediction, res_html):
        # 恢复状态灯为就绪态
        self.status_dot.setText("● 引擎就绪")
        self.status_dot.setStyleSheet("""
            color: #00e676;
            background-color: rgba(0, 230, 118, 0.08);
            border: 1px solid rgba(0, 230, 118, 0.25);
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 11px;
            font-weight: bold;
        """)
        
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