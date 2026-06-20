import os
import sys

# 强制添加脚本所在绝对目录至 sys.path，防止在外部 runtime 下运行时找不到同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import threading
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import multiprocessing

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 数据集快速校验与 Dry-run
# ==========================================
def validate_dataset(ai_folder, human_folder, stop_event=None):
    print(f"--- 开始数据集 Dry-run 完整性校验与懒加载裂变预计算 ---")
    from dataset import AudioSpectrogramDataset, get_file_lists
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    file_list, labels = get_file_lists(ai_folder, human_folder)
    total_original = len(file_list)
    if total_original == 0:
        print("❌ 未在指定目录下找到任何有效的音频文件 (.mp3, .wav, .flac, .m4a)")
        return
        
    fission_factor = 15
    dataset = AudioSpectrogramDataset(file_list, labels, fission_factor=fission_factor)
    total = len(dataset)
    
    print(f"扫描完成。原始音频数: {total_original} | 裂变后总变体数: {total}")
    print("开始执行物理增强与 Log-Mel 提取 (已开启多线程火力全开，首次耗时稍长)...")
    
    success_count = 0
    fail_count = 0
    completed = 0
    
    def process_single(idx):
        if stop_event and stop_event.is_set():
            return idx, False, "Aborted"
            
        file_idx = idx // fission_factor
        aug_idx = idx % fission_factor
        file_path = dataset.file_list[file_idx]
        label_name = "AI" if dataset.labels[file_idx] == 1 else "真人"
        filename = os.path.basename(file_path)
        display_name = filename if aug_idx == 0 else f"{filename} (变体 {aug_idx})"
        
        try:
            _ = dataset._process_audio(file_path, aug_idx)
            return idx, True, f"[{label_name}] {display_name}"
        except Exception as e:
            err_msg = str(e) or type(e).__name__
            return idx, False, f"[{label_name}] {display_name} (处理失败: {err_msg})"

    # 动态获取 CPU 核心数，保留 1 个核心给操作系统避免卡死
    max_workers = max(1, os.cpu_count() - 1)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, i): i for i in range(total)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                res_idx, is_success, msg = future.result()
                if msg == "Aborted":
                    break
                
                completed += 1
                if is_success:
                    success_count += 1
                    print(f"[{completed}/{total}] ✓ {msg}")
                else:
                    fail_count += 1
                    print(f"[{completed}/{total}] ✗ {msg}")
            except Exception as e:
                fail_count += 1
                completed += 1
                print(f"[{completed}/{total}] ✗ 线程崩溃: {e}")
                
    if stop_event and stop_event.is_set():
        print("\n[中断] 用户中止了校验与裂变流程。")
        return

    print(f"\n{'='*50}")
    print(f"[裂变与校验结果] 总计虚拟样本: {total} | 正常: {success_count} | 失败/受损: {fail_count}")
    print(f"{'='*50}\n")

# ==========================================
# GUI 界面与逻辑
# ==========================================
class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)

    def flush(self):
        pass

class AudioFeatureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN 音频特征训练工具 (Suno 鉴伪)")
        self.root.geometry("600x560")
        self.root.resizable(False, False)

        self.ai_path_var = tk.StringVar()
        self.human_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.use_gpu_var = tk.BooleanVar(value=True)

        self._stop_event = threading.Event()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.create_widgets()
        sys.stdout = StdoutRedirector(self.log_text)

        # 加载与载入本地配置，记住上一次的目录修改
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        self.load_config()

    def _on_close(self):
        self.save_config()
        self._stop_event.set()
        self.root.destroy()

    def load_config(self):
        import json
        default_ai = r"G:\suno5.5 sa"
        default_human = os.path.join(os.getcwd(), "人类")
        default_output = os.path.join(os.getcwd(), "suno_detector_model.onnx")
        default_gpu = True
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                    self.ai_path_var.set(cfg.get("ai_path", default_ai))
                    self.human_path_var.set(cfg.get("human_path", default_human))
                    self.output_path_var.set(cfg.get("output_path", default_output))
                    self.use_gpu_var.set(cfg.get("use_gpu", default_gpu))
                    return
            except Exception:
                pass
                
        self.ai_path_var.set(default_ai)
        self.human_path_var.set(default_human)
        self.output_path_var.set(default_output)
        self.use_gpu_var.set(default_gpu)

    def save_config(self):
        import json
        try:
            cfg = {
                "ai_path": self.ai_path_var.get(),
                "human_path": self.human_path_var.get(),
                "output_path": self.output_path_var.get(),
                "use_gpu": self.use_gpu_var.get()
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=4)
        except Exception:
            pass

    def create_widgets(self):
        tk.Label(self.root, text="AI 生成样本目录:").place(x=20, y=20)
        tk.Entry(self.root, textvariable=self.ai_path_var, width=50).place(x=140, y=20)
        tk.Button(self.root, text="浏览", command=lambda: self.select_folder(self.ai_path_var)).place(x=510, y=16)

        tk.Label(self.root, text="真人/参考样本目录:").place(x=20, y=60)
        tk.Entry(self.root, textvariable=self.human_path_var, width=50).place(x=140, y=60)
        tk.Button(self.root, text="浏览", command=lambda: self.select_folder(self.human_path_var)).place(x=510, y=56)

        tk.Label(self.root, text="ONNX 模型保存路径:").place(x=20, y=100)
        tk.Entry(self.root, textvariable=self.output_path_var, width=50).place(x=140, y=100)
        tk.Button(self.root, text="浏览", command=self.select_save_file).place(x=510, y=96)

        self.run_btn = tk.Button(self.root, text="开始校验数据集", bg="SystemHighlight", fg="black", font=("Arial", 10, "bold"), command=self.start_processing)
        self.run_btn.place(x=150, y=140, width=120, height=35)

        self.train_btn = tk.Button(self.root, text="训练并导出 ONNX", bg="SystemHighlight", fg="black", font=("Arial", 10, "bold"), command=self.start_training)
        self.train_btn.place(x=310, y=140, width=150, height=35)

        self.auto_btn = tk.Button(self.root, text="全自动（校验+训练+关机）", bg="#FF4444", fg="white", font=("Arial", 10, "bold"), command=self.start_auto)
        self.auto_btn.place(x=150, y=185, width=310, height=35)

        self.gpu_cb = tk.Checkbutton(self.root, text="GPU 加速 (CUDA)", variable=self.use_gpu_var,
                                      fg="#333", font=("Arial", 9))
        self.gpu_cb.place(x=470, y=190)

        tk.Label(self.root, text="运行日志:").place(x=20, y=240)
        self.log_text = scrolledtext.ScrolledText(self.root, width=76, height=18, state=tk.DISABLED, bg="#F0F0F0")
        self.log_text.place(x=20, y=265)

    def check_paths_exist(self, path_str):
        if not path_str or not path_str.strip():
            return False
        unified = path_str.replace('；', ';').replace(',', ';').replace('，', ';')
        paths = []
        for p in unified.split(';'):
            p = p.strip().strip('"').strip("'")
            if p:
                paths.append(p)
        if not paths:
            return False
        return all(os.path.exists(p) for p in paths)

    def select_folder(self, var):
        folder = filedialog.askdirectory()
        if folder:
            current = var.get().strip()
            if current:
                from tkinter import messagebox
                ans = messagebox.askyesnocancel("选择目录", "是否将新选目录追加到已有路径后？\n\n[是]：追加（以分号分隔）\n[否]：替换已有路径\n[取消]：什么都不做")
                if ans is True:
                    unified = current.replace('；', ';').replace(',', ';').replace('，', ';')
                    paths = [p.strip() for p in unified.split(';') if p.strip()]
                    if folder not in paths:
                        var.set(current + ";" + folder)
                elif ans is False:
                    var.set(folder)
            else:
                var.set(folder)

    def select_save_file(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".onnx",
            filetypes=[("ONNX Files", "*.onnx")],
            initialfile="suno_detector_model.onnx"
        )
        if file_path:
            self.output_path_var.set(file_path)

    def start_processing(self):
        ai_folder = self.ai_path_var.get()
        human_folder = self.human_path_var.get()

        if not self.check_paths_exist(ai_folder) or not self.check_paths_exist(human_folder):
            messagebox.showwarning("参数缺失", "请指定存在且合法的 AI 样本目录和真人样本目录。若输入了多个路径，请确保每个路径都真实存在。")
            return

        self.save_config()
        self._stop_event.clear()
        self.run_btn.config(state=tk.DISABLED, text="正在校验...")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

        threading.Thread(target=self.run_thread, args=(ai_folder, human_folder), daemon=True).start()

    def run_thread(self, ai_folder, human_folder):
        try:
            validate_dataset(ai_folder, human_folder, stop_event=self._stop_event)
        except Exception as e:
            print(f"\n[严重错误] 校验过程中发生异常: {e}")
        finally:
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL, text="开始校验数据集"))

    def start_training(self):
        ai_folder = self.ai_path_var.get()
        human_folder = self.human_path_var.get()
        onnx_path = self.output_path_var.get()

        if not self.check_paths_exist(ai_folder) or not self.check_paths_exist(human_folder):
            messagebox.showwarning("目录缺失", "请确保 AI 样本目录和真人样本目录都存在。若输入了多个路径，请确保每个路径都真实存在。")
            return
        
        self.save_config()
        self.train_btn.config(state=tk.DISABLED, text="正在训练...")
        self.run_btn.config(state=tk.DISABLED)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        threading.Thread(target=self.run_train_thread, args=(ai_folder, human_folder, onnx_path), daemon=True).start()

    def _do_training(self, ai_folder, human_folder, onnx_path):
        """核心训练对接逻辑 (PyTorch CNN)"""
        print("🤖 开始端到端深度学习（Log-Mel 频谱图 + CNN）训练...")
        from train_cnn import train_and_export
        
        use_gpu = self.use_gpu_var.get()
        
        success = train_and_export(
            ai_dir=ai_folder, 
            human_dir=human_folder, 
            onnx_path=onnx_path, 
            log_callback=print,
            epochs=25,
            use_gpu=use_gpu
        )
        return success

    def run_train_thread(self, ai_folder, human_folder, onnx_path):
        try:
            self._do_training(ai_folder, human_folder, onnx_path)
        except Exception as e:
            print(f"\n[严重错误] 训练或导出过程中发生异常: {e}")
        finally:
            self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL, text="训练并导出 ONNX"))
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))

    def start_auto(self):
        ai_folder = self.ai_path_var.get()
        human_folder = self.human_path_var.get()
        onnx_path = self.output_path_var.get()

        if not self.check_paths_exist(ai_folder) or not self.check_paths_exist(human_folder):
            messagebox.showwarning("参数缺失", "请指定存在且合法的 AI 样本目录和真人样本目录。若输入了多个路径，请确保每个路径都真实存在。")
            return

        self.save_config()
        self._stop_event.clear()
        self.auto_btn.config(state=tk.DISABLED, text="全自动运行中...")
        self.run_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

        threading.Thread(target=self.run_auto_thread, args=(ai_folder, human_folder, onnx_path), daemon=True).start()

    def run_auto_thread(self, ai_folder, human_folder, onnx_path):
        success = True
        try:
            print("=" * 50)
            print("🔧 阶段 1/2: 数据集快速校验")
            print("=" * 50)
            validate_dataset(ai_folder, human_folder, stop_event=self._stop_event)
            if self._stop_event.is_set():
                return
            print("\n✅ 数据集校验完成！")
        except Exception as e:
            print(f"\n❌ 数据集校验失败: {e}")
            success = False

        if success and not self._stop_event.is_set():
            try:
                print("\n" + "=" * 50)
                print("🤖 阶段 2/2: CNN 模型训练与 ONNX 导出")
                print("=" * 50)
                train_ok = self._do_training(ai_folder, human_folder, onnx_path)
                if not train_ok:
                    success = False
                else:
                    print("\n✅ 训练及 ONNX 导出完成！")
            except Exception as e:
                print(f"\n❌ 训练失败: {e}")
                success = False

        if self._stop_event.is_set():
            return

        if success:
            print("\n" + "=" * 50)
            print("🎉 全自动流程完成！系统将在 60 秒后自动关机...")
            print("   在命令行执行 shutdown /a 可取消关机")
            print("=" * 50)
            os.system("shutdown /s /t 60")
        else:
            self.root.after(0, lambda: self.auto_btn.config(state=tk.NORMAL, text="全自动（校验+训练+关机）"))
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app_root = tk.Tk()
    app = AudioFeatureApp(app_root)
    app_root.mainloop()