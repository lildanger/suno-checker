import os
import torch
import librosa
import numpy as np
import pyloudnorm as pyln
import hashlib
from torch.utils.data import Dataset

def get_file_lists(ai_dir, human_dir):
    """独立的扫描方法，提取所有音频的绝对路径和标签，用于在训练前切分防泄露。支持分号或逗号分隔的多路径扫描。"""
    file_list = []
    labels = []
    
    def split_paths(path_str):
        if not path_str:
            return []
        unified = path_str.replace('；', ';').replace(',', ';').replace('，', ';')
        paths = []
        for p in unified.split(';'):
            p = p.strip().strip('"').strip("'")
            if p:
                paths.append(p)
        return paths
        
    ai_paths = split_paths(ai_dir)
    for path in ai_paths:
        if os.path.exists(path):
            for root, _, files in os.walk(path):
                for f in files:
                    if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                        file_list.append(os.path.abspath(os.path.join(root, f)))
                        labels.append(1) # 1 代表 AI
                        
    human_paths = split_paths(human_dir)
    for path in human_paths:
        if os.path.exists(path):
            for root, _, files in os.walk(path):
                for f in files:
                    if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                        file_list.append(os.path.abspath(os.path.join(root, f)))
                        labels.append(0) # 0 代表人类
                        
    return file_list, labels

class AudioSpectrogramDataset(Dataset):
    def __init__(self, file_list, labels, sample_rate=22050, duration=3, n_mels=128, use_cache=True, fission_factor=1):
        self.file_list = file_list
        self.labels = labels
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.target_length = int(sample_rate * duration)
        # 根据 hop_length=512 计算标准时间帧数
        self.target_width = int(self.target_length / 512) + 1
        self.meter = pyln.Meter(sample_rate)
        
        self.use_cache = use_cache
        self.fission_factor = max(1, fission_factor)
        self.cache_dir = os.path.join(os.getcwd(), ".spectrogram_cache")
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        # 通过 fission_factor 放大虚拟长度
        return len(self.file_list) * self.fission_factor

    def get_cache_filename(self, file_path, aug_idx):
        # 1. 路径无关缓存名（基于文件名和大小），解决目录改变/移动后无法复用缓存的问题
        basename = os.path.basename(file_path)
        try:
            file_size = os.path.getsize(file_path)
        except Exception:
            file_size = 0
        h = hashlib.md5(f"{basename}_{file_size}_{aug_idx}".encode('utf-8')).hexdigest()
        return f"{h}.pt"

    def get_old_cache_filename(self, file_path, aug_idx):
        # 2. 路径相关缓存名，用于兼容和读取老版缓存
        h = hashlib.md5(f"{file_path}_{aug_idx}".encode('utf-8')).hexdigest()
        return f"{h}.pt"

    def _process_audio(self, file_path, aug_idx):
        cache_file_path = None
        if self.use_cache:
            cache_filename = self.get_cache_filename(file_path, aug_idx)
            cache_file_path = os.path.join(self.cache_dir, cache_filename)
            
            # 优先读取新版路径无关缓存 (秒级读取)
            if os.path.exists(cache_file_path):
                try:
                    return torch.load(cache_file_path, weights_only=True)
                except Exception:
                    pass
            
            # 向前兼容：如果不存在新版缓存，则检查是否存在老版绝对路径缓存
            old_filename = self.get_old_cache_filename(file_path, aug_idx)
            old_file_path = os.path.join(self.cache_dir, old_filename)
            if os.path.exists(old_file_path):
                try:
                    tensor_out = torch.load(old_file_path, weights_only=True)
                    # 自动迁移并存为新格式缓存，方便以后复用
                    try:
                        torch.save(tensor_out, cache_file_path)
                    except Exception:
                        pass
                    return tensor_out
                except Exception:
                    pass
        
        # 1. 加载音频
        y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        
        # 2. 物理数据增强裂变处理 (如果 aug_idx > 0)
        if aug_idx > 0:
            # A. 随机音高偏移 (-2 到 2 个半音)
            n_steps = np.random.uniform(-2.0, 2.0)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            
            # B. 随机时间拉伸 (0.8x 到 1.2x)
            rate = np.random.uniform(0.8, 1.2)
            y = librosa.effects.time_stretch(y, rate=rate)
            
            # C. 随机高斯底噪干扰
            noise_amp = np.random.uniform(0.001, 0.005)
            noise = np.random.normal(0, noise_amp, y.shape)
            y = y + noise
        
        # 3. LUFS 响度归一化 (-23 LUFS)
        try:
            loudness = self.meter.integrated_loudness(y)
            if loudness > -100 and not np.isnan(loudness) and not np.isinf(loudness):
                y = pyln.normalize.loudness(y, loudness, -23.0)
        except Exception:
            pass
            
        # 4. 固定长度切片/填充 (确保时长一致)
        if len(y) < self.target_length:
            y = np.pad(y, (0, self.target_length - len(y)), mode='constant')
        else:
            y = y[:self.target_length]
            
        # 5. 提取 Mel 频谱图
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_fft=2048, hop_length=512, n_mels=self.n_mels
        )
        
        # 6. 转化为 Log-Mel 能量
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 7. 归一化到 [-1, 1] 像素区间以稳定 CNN 输入
        s_min = log_mel_spec.min()
        s_max = log_mel_spec.max()
        log_mel_spec = (log_mel_spec - s_min) / (s_max - s_min + 1e-6)
        log_mel_spec = (log_mel_spec * 2.0) - 1.0
        
        # 确保输出的 width 对齐 target_width
        if log_mel_spec.shape[1] < self.target_width:
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, self.target_width - log_mel_spec.shape[1])), mode='constant')
        elif log_mel_spec.shape[1] > self.target_width:
            log_mel_spec = log_mel_spec[:, :self.target_width]
            
        tensor_out = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0) # [1, H, W]
        
        # 写入物理变体缓存
        if self.use_cache and cache_file_path:
            try:
                torch.save(tensor_out, cache_file_path)
            except Exception:
                pass
                
        return tensor_out

    def __getitem__(self, idx):
        # 动态解析属于哪个音频文件的哪个变体
        file_idx = idx // self.fission_factor
        aug_idx = idx % self.fission_factor
        
        file_path = self.file_list[file_idx]
        label = self.labels[file_idx]
        
        try:
            spectrogram = self._process_audio(file_path, aug_idx)
            return spectrogram, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            return torch.zeros((1, self.n_mels, self.target_width)), torch.tensor(label, dtype=torch.long)
