import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from dataset import AudioSpectrogramDataset, get_file_lists

# ==========================================
# 纯 PyTorch 实现的标准 ResNet-18 结构 (免 torchvision 依赖)
# ==========================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # 首层接收 1 通道 Log-Mel
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def download_resnet_weights_china(log_callback=print):
    import urllib.request
    filename = "resnet18-f37072fd.pth"
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    dest_path = os.path.join(cache_dir, filename)
    
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 40000000:
        return
        
    mirrors = [
        f"https://mirror.sjtu.edu.cn/pytorch/models/{filename}",
        f"https://mirrors.ustc.edu.cn/pytorch/models/{filename}"
    ]
    
    log_callback("[INFO] 检测到本地预训练权重不存在，正在尝试高速下载...")
    for url in mirrors:
        try:
            log_callback(f"[INFO] 正在下载: {url}")
            urllib.request.urlretrieve(url, dest_path)
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 40000000:
                log_callback(f"[OK] 权重下载成功，已缓存至: {dest_path}")
                return
        except Exception as e:
            log_callback(f"[WARNING] 从 {url} 下载失败: {e}")
            if os.path.exists(dest_path):
                try: os.remove(dest_path)
                except: pass
    log_callback("[WARNING] 高速镜像源下载失败，将回退至 PyTorch 官方下载器。")

def build_model(pretrained=True, log_callback=print):
    # 使用纯 PyTorch 搭建的 ResNet-18 结构，杜绝 torchvision 的 C++ 算子冲突
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    
    if pretrained:
        try:
            download_resnet_weights_china(log_callback)
            filename = "resnet18-f37072fd.pth"
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
            weights_path = os.path.join(cache_dir, filename)
            
            # 如果本地没有，就用 torchvision 的官方链接 fallback 下载
            if not os.path.exists(weights_path) or os.path.getsize(weights_path) < 40000000:
                import urllib.request
                log_callback("[INFO] 正在从官方源下载预训练权重...")
                urllib.request.urlretrieve("https://download.pytorch.org/models/resnet18-f37072fd.pth", weights_path)
                
            state_dict = torch.load(weights_path, map_location="cpu")
            
            # 对齐首层 Conv1 的通道 (从 3 通道压缩至 1 通道平均值)
            orig_weight = state_dict['conv1.weight']
            new_weight = orig_weight.mean(dim=1, keepdim=True)
            state_dict['conv1.weight'] = new_weight
            
            # 去除全连接分类层，防止类别数不符报错
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)
            
            model.load_state_dict(state_dict, strict=False)
            log_callback("[OK] 预训练 ResNet18 权重已成功载入（包含单通道首层平滑对齐）。")
        except Exception as e:
            log_callback(f"[ERROR] 预训练模型权重下载/加载失败: {e}")
            log_callback("[HELP] 为了保证音频鉴伪的迁移学习效果，必须使用预训练权重。")
            log_callback("[HELP] 请使用迅雷、浏览器等工具手动下载权重文件，地址如下：")
            log_callback("       https://download.pytorch.org/models/resnet18-f37072fd.pth")
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
            log_callback(f"[HELP] 下载完成后，请将权重文件存放到以下目录：")
            log_callback(f"       {cache_dir}")
            log_callback("[HELP] 存放完成后重新点击训练即可。")
            raise RuntimeError("预训练权重加载失败，已终止训练。") from e
    else:
        log_callback("[OK] 已在无预训练权重模式下随机初始化模型权重（pretrained=False）。")
        
    return model

def train_and_export(ai_dir, human_dir, onnx_path="suno_detector_model.onnx", log_callback=print, epochs=25, pretrained=True, use_gpu=True):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    log_callback(f"Using device: {device}")
    
    log_callback(f"Loading dataset from AI: '{ai_dir}', Human: '{human_dir}'...")
    all_files, all_labels = get_file_lists(ai_dir, human_dir)
    
    total_samples = len(all_files)
    if total_samples == 0:
        log_callback("[ERROR] No valid audio samples (.mp3, .wav, .flac, .m4a) found in the directories.")
        return False
        
    log_callback(f"Total valid original samples found: {total_samples}")
    
    # 手动划分训练集与验证集 (物理隔离，严防变体泄露)
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    
    if train_size == 0 or val_size == 0:
        log_callback("[ERROR] Dataset size is too small to split into train and validation sets.")
        return False
        
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_files = [all_files[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    
    val_files = [all_files[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    # 实例化数据集：训练集开启 15 倍裂变，验证集保持 1 倍纯净
    fission_factor = 15
    log_callback(f"Applying lazy offline augmentation... Fission factor: {fission_factor}x for training set.")
    train_dataset = AudioSpectrogramDataset(train_files, train_labels, fission_factor=fission_factor)
    val_dataset = AudioSpectrogramDataset(val_files, val_labels, fission_factor=1)
    
    log_callback(f"Virtual Training Samples: {len(train_dataset)} | Virtual Validation Samples: {len(val_dataset)}")
    
    # 缓存自动核对与多线程预热补全机制
    log_callback("[INFO] 正在核对训练集和验证集的本地缓存完整性...")
    missing_tasks = []
    total_virtual_samples = len(train_dataset) + len(val_dataset)
    
    # 核对训练集缓存
    for idx in range(len(train_dataset)):
        file_idx = idx // train_dataset.fission_factor
        aug_idx = idx % train_dataset.fission_factor
        file_path = train_dataset.file_list[file_idx]
        cache_filename = train_dataset.get_cache_filename(file_path, aug_idx)
        cache_file_path = os.path.join(train_dataset.cache_dir, cache_filename)
        if not os.path.exists(cache_file_path):
            old_filename = train_dataset.get_old_cache_filename(file_path, aug_idx)
            old_file_path = os.path.join(train_dataset.cache_dir, old_filename)
            if not os.path.exists(old_file_path):
                missing_tasks.append((train_dataset, file_path, aug_idx))
            
    # 核对验证集缓存
    for idx in range(len(val_dataset)):
        file_idx = idx // val_dataset.fission_factor
        aug_idx = idx % val_dataset.fission_factor
        file_path = val_dataset.file_list[file_idx]
        cache_filename = val_dataset.get_cache_filename(file_path, aug_idx)
        cache_file_path = os.path.join(val_dataset.cache_dir, cache_filename)
        if not os.path.exists(cache_file_path):
            old_filename = val_dataset.get_old_cache_filename(file_path, aug_idx)
            old_file_path = os.path.join(val_dataset.cache_dir, old_filename)
            if not os.path.exists(old_file_path):
                missing_tasks.append((val_dataset, file_path, aug_idx))
            
    hit_count = total_virtual_samples - len(missing_tasks)
    log_callback(f"[INFO] 缓存核对结果：当前总虚拟样本数: {total_virtual_samples} | 已命中的预热缓存数: {hit_count} | 缺失缓存数: {len(missing_tasks)}")
    
    if len(missing_tasks) > 0:
        log_callback(f"[INFO] 检测到存在缺失缓存，正在开启多线程并行预热计算剩余的 {len(missing_tasks)} 个样本缓存...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_and_save(task):
            dataset_obj, file_path, aug_idx = task
            try:
                dataset_obj._process_audio(file_path, aug_idx)
                return True
            except Exception as e:
                err_msg = str(e) or type(e).__name__
                log_callback(f"[WARNING] 预热缓存失败 '{os.path.basename(file_path)}' (变体 {aug_idx}): {err_msg}")
                return False
                
        max_workers = max(1, os.cpu_count() - 1)
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_and_save, t): t for t in missing_tasks}
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 50 == 0 or completed == len(missing_tasks):
                    log_callback(f"[INFO] 缓存预热进度: [{completed}/{len(missing_tasks)}]")
        log_callback("[OK] 缺失缓存已全部多线程预热补全！")
    else:
        log_callback("[OK] 所有训练与验证样本已 100% 命中本地预热缓存，无需额外计算。")
        
    batch_size = min(16, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=(len(train_dataset) >= batch_size))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = build_model(pretrained=pretrained, log_callback=log_callback).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    import copy
    best_val_acc = -1.0
    best_loss = float('inf')
    best_model_weights = None
    
    log_callback("[INFO] Starting training loops...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            
        # 验证集评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / max(1, batch_count)
        val_acc = 100 * correct / max(1, total)
        log_callback(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        # 记录最优权重 (优先高准确率，同准确率下优先低 Loss)
        is_best = False
        if val_acc > best_val_acc:
            is_best = True
        elif abs(val_acc - best_val_acc) < 1e-5:
            if avg_loss < best_loss:
                is_best = True
                
        if is_best:
            best_val_acc = val_acc
            best_loss = avg_loss
            best_model_weights = copy.deepcopy(model.state_dict())

    # 导出为 ONNX 模型 (优先载入最佳权重，防止导出过拟合的最后一轮权重)
    if best_model_weights is not None:
        log_callback(f"[INFO] 正在载入最佳模型权重 (Best Val Acc: {best_val_acc:.2f}%, Loss: {best_loss:.4f})...")
        model.load_state_dict(best_model_weights)
        
    log_callback("[INFO] Exporting CNN model to ONNX format...")
    model.eval()
    
    # [Batch, Channel, Height, Width] -> (1, 1, 128, 130)
    dummy_input = torch.randn(1, 1, 128, 130).to(device)
    
    try:
        torch.onnx.export(
            model.to("cpu"), 
            dummy_input.to("cpu"), 
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['float_input'],
            output_names=['output'],
            dynamic_axes={'float_input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        log_callback(f"[OK] CNN ONNX model exported successfully as '{onnx_path}'")
        return True
    except Exception as e:
        log_callback(f"[ERROR] ONNX Export Error: {e}")
        return False

if __name__ == "__main__":
    train_and_export(ai_dir='./AI', human_dir='./人类')
