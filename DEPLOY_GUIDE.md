# RAICOM 2026 — 智海算法调优赛：完整部署与测试手册

> **适用平台**：Windows 10/11 · Ubuntu 20.04/22.04 · CentOS 8+
> **项目版本**：V1.0（2026-06-11）
> **技术栈**：PyTorch + EfficientNet-B3 + ONNX Runtime CPU

---

## 一、环境要求

### 1.1 硬件要求

| 场景 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **训练（GPU）** | NVIDIA GPU ≥ 4GB 显存 | RTX 3060 12GB / RTX 4070 |
| **训练（CPU）** | 8GB 内存，4 核 | 16GB+ 内存，8 核+ |
| **推理（CPU）** | 4GB 内存，2 核 | 8GB+ 内存 |
| **磁盘** | 10GB 可用空间 | 50GB+（含数据集） |

> GPU 非必须。本项目所有推理均基于 CPU（ONNX Runtime），训练阶段用 GPU 可大幅加速。
> 比赛评分使用 CPU 推理，因此训练和部署可完全分离。

### 1.2 软件要求

| 组件 | 版本要求 | 检查命令 |
|------|---------|---------|
| **Python** | 3.9 / 3.10 / 3.11 | `python --version` |
| **pip** | ≥ 22.0 | `pip --version` |
| **Git** | 任意版本 | `git --version` |
| **NVIDIA 驱动**（可选） | ≥ 525.x | `nvidia-smi` |
| **CUDA**（可选） | 11.8 / 12.1 | `nvcc --version` |

---

## 二、环境搭建

### 2.1 Windows 环境搭建

#### 方案 A：Conda（推荐）

```powershell
# 1. 安装 Miniconda（如未安装）
# 下载地址：https://docs.conda.io/en/latest/miniconda.html

# 2. 创建虚拟环境
conda create -n raicom python=3.10 -y

# 3. 激活环境
conda activate raicom

# 4. 确认 Python 版本
python --version
```

#### 方案 B：venv

```powershell
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活环境
.\venv\Scripts\activate

# 3. 确认 Python 版本
python --version
```

#### 安装依赖（Windows）

```powershell
# 进入项目目录
cd raicom-weather

# 安装所有依赖
pip install -r requirements.txt

# 如果网络慢，使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### GPU 版 PyTorch 安装（Windows）

```powershell
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 验证 GPU 是否可用
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

### 2.2 Linux 环境搭建

#### 方案 A：Conda（推荐）

```bash
# 1. 安装 Miniconda（如未安装）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/.bashrc

# 2. 创建虚拟环境
conda create -n raicom python=3.10 -y

# 3. 激活环境
conda activate raicom
```

#### 方案 B：venv

```bash
# 1. 创建虚拟环境
python3 -m venv venv

# 2. 激活环境
source venv/bin/activate
```

#### 系统依赖（Linux）

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# CentOS / RHEL
sudo yum install -y mesa-libGL glib2
```

#### 安装依赖（Linux）

```bash
# 进入项目目录
cd raicom-weather

# 安装所有依赖
pip install -r requirements.txt

# 国内镜像加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### GPU 版 PyTorch 安装（Linux）

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 验证
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## 三、数据准备

### 3.1 目录结构

```
raicom-weather/
└── data/
    ├── train/
    │   ├── cloudy/        ← 多云图片
    │   │   ├── img_001.jpg
    │   │   ├── img_002.jpg
    │   │   └── ...
    │   ├── rainy/         ← 雨天图片
    │   │   └── ...
    │   ├── snowy/         ← 雪天图片
    │   │   └── ...
    │   └── sunny/         ← 晴天图片
    │       └── ...
    ├── val/               ← (可选) 独立验证集，结构同 train/
    │   ├── cloudy/
    │   ├── rainy/
    │   ├── snowy/
    │   └── sunny/
    └── test/              ← (可选) 比赛测试集图片
        ├── test_001.jpg
        └── ...
```

### 3.2 创建目录

**Windows (PowerShell)**：
```powershell
mkdir -p data\train\cloudy, data\train\rainy, data\train\snowy, data\train\sunny
mkdir -p data\val\cloudy, data\val\rainy, data\val\snowy, data\val\sunny
mkdir -p data\test
mkdir -p outputs
```

**Linux (Bash)**：
```bash
mkdir -p data/train/{cloudy,rainy,snowy,sunny}
mkdir -p data/val/{cloudy,rainy,snowy,sunny}
mkdir -p data/test
mkdir -p outputs
```

### 3.3 数据集检查

### 3.3.1 训练期资源监控说明

训练脚本已直接内置监控能力，适合首次预训练定位问题：

- **CPU 使用率**：观察 DataLoader / 解码 / 预处理是否成为瓶颈
- **RAM 使用量**：观察内存是否持续上涨，排查缓存或加载策略问题
- **进程 RSS**：排查 Python 进程级内存泄漏
- **GPU 显存 / GPU 利用率**（安装 `pynvml` 时更完整）
- **epoch 级资源日志**：写入 `outputs/training_log.json -> resource`
- **独立资源日志**：
  - `outputs/train_resource_log.jsonl`
  - `outputs/train_resource_log.csv`

训练时你会看到类似输出：

```text
[BOOT] CPU 8% | RAM 12.5GB/44% | RSS 0.31GB | CUDA 512MB
Epoch  3/60 | LR 8.00e-04 | Train F1 0.9123 | Val F1 0.9056 | ... | CPU 36% | RAM 13.0GB/46% | RSS 1.85GB | CUDA 4210MB
```

如果要让 GPU 利用率/温度更完整，建议额外安装：

```bash
pip install psutil pynvml
```

## 3.4 数据集检查

放好图片后，运行以下命令确认各类别数量：

```bash
python -c "
import os
classes = ['cloudy', 'rainy', 'snowy', 'sunny']
total = 0
for cls in classes:
    d = os.path.join('data/train', cls)
    if os.path.isdir(d):
        n = len([f for f in os.listdir(d) if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))])
        print(f'  {cls:8s}: {n:>5d} 张')
        total += n
    else:
        print(f'  {cls:8s}: 目录不存在!')
print(f'  总计:    {total:>5d} 张')
"
```

### 3.4 CSV 标注模式（可选）

如果数据以 CSV 形式提供，按如下格式放置：

```
data/
├── train/
│   ├── cloudy/
│   ├── rainy/
│   ├── snowy/
│   └── sunny/
├── train.csv         ← 列：filename, label
└── test.csv          ← 列：filename
```

`train.csv` 示例：
```
filename,label
cloudy/img_001.jpg,cloudy
rainy/img_002.jpg,rainy
sunny/img_003.jpg,sunny
```

---

## 四、完整工作流

### 4.1 流程总览

```
① 训练 → ② 导出 ONNX → ③ INT8 量化 → ④ 推理测试 → ⑤ 提交
```

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  train.py    │────▶│ export_onnx  │────▶│ quantize     │────▶│  infer.py    │
│  训练模型     │     │ 导出 ONNX    │     │ INT8 量化     │     │ CPU 推理      │
│              │     │              │     │              │     │              │
│ 输出:        │     │ 输出:        │     │ 输出:        │     │ 输出:        │
│ best_model   │     │ model.onnx   │     │ model.int8   │     │ predictions  │
│ .pth         │     │ (FP32)       │     │ .onnx (INT8) │     │ .csv         │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

### 4.2 第一步：训练

```bash
# 默认配置训练（60 轮，EfficientNet-B3，300×300）
python train.py --data-root ./data

# 自定义参数
python train.py --data-root ./data --epochs 30 --batch-size 32 --lr 1e-3 --image-size 300

# 调整增强强度（0.0=轻，1.0=标准，2.0=极强）
python train.py --data-root ./data --aug-strength 1.0

# 从检查点恢复训练
python train.py --resume ./outputs/checkpoint.pth

# CPU 训练（无 GPU 时）
python train.py --data-root ./data --device cpu --epochs 10 --batch-size 8
```

**训练产出**：

| 文件 | 说明 |
|------|------|
| `outputs/best_model.pth` | 最佳验证 Macro F1 对应的模型权重 |
| `outputs/checkpoint.pth` | 最新一轮的检查点（可用于恢复训练） |
| `outputs/training_log.json` | 每轮训练指标日志（JSON 格式） |

**训练日志示例**：
```
Epoch  1/60 | LR 2.00e-04 | Train F1 0.7123 | Val F1 0.7456 | Val Acc 0.7512 | Best 0.7456 | 45s
Epoch  2/60 | LR 4.00e-04 | Train F1 0.8234 | Val F1 0.8345 | Val Acc 0.8401 | Best 0.8345 | 43s
...
Epoch 60/60 | LR 1.23e-06 | Train F1 0.9901 | Val F1 0.9567 | Val Acc 0.9602 | Best 0.9589 | 44s
```

**查看训练结果**：
```bash
python -c "
import json
with open('outputs/training_log.json') as f:
    log = json.load(f)
best = max(log, key=lambda x: x['val_f1'])
print(f'最佳 Epoch: {best[\"epoch\"]}')
print(f'最佳 Val F1: {best[\"val_f1\"]}')
print(f'最佳 Val Acc: {best[\"val_acc\"]}')
"
```

---

### 4.3 第二步：导出 ONNX

```bash
# 导出 FP32 ONNX 模型
python export_onnx.py --checkpoint ./outputs/best_model.pth

# 指定输入尺寸和 opset
python export_onnx.py --checkpoint ./outputs/best_model.pth --image-size 300 --opset 17

# 禁用动态批处理（固定 batch=1）
python export_onnx.py --checkpoint ./outputs/best_model.pth --no-dynamic-batch
```

**导出产出**：
- `outputs/model.onnx`（FP32 模型）

**导出会自动运行基准测试**，输出类似：
```
单张推理: 42.315 ms
等效 FPS: 23.6
70 分钟内可推理约 99,285 张图片
评分集（几千张）预计耗时约 211 秒
```

---

### 4.4 第三步：INT8 量化

#### 动态量化（快速测试，无需校准数据）

```bash
# Windows:
python quantize_onnx.py --onnx-path ./outputs/model.onnx --benchmark

# Linux:
python quantize_onnx.py --onnx-path ./outputs/model.onnx --benchmark
```

**输出**：`outputs/model.int8.dynamic.onnx`

#### 静态量化（推荐比赛最终版，需要校准图片）

```bash
# Windows:
python quantize_onnx.py --onnx-path ./outputs/model.onnx --mode static --calib-dir ./data/train --max-calib-images 256 --benchmark

# Linux:
python quantize_onnx.py --onnx-path ./outputs/model.onnx --mode static --calib-dir ./data/train --max-calib-images 256 --benchmark
```

**输出**：`outputs/model.int8.static.onnx`

**量化后基准测试输出示例**：
```
FP32: 42.315 ms / image | FPS 23.6
INT8: 18.520 ms / image | FPS 54.0
加速比: 2.28x
大小变化: 15.23 MB -> 4.12 MB (73.0% smaller)
```

---

### 4.5 第四步：CPU 推理

```bash
# 对测试集目录推理（批处理，快速）
python infer.py --onnx-path ./outputs/model.int8.static.onnx --data-root ./data/test

# 使用 TTA（5 次增强取平均，提分但变慢）
python infer.py --onnx-path ./outputs/model.int8.static.onnx --data-root ./data/test --tta

# 单张图片推理
python infer.py --onnx-path ./outputs/model.int8.static.onnx --image-path ./data/test/sample.jpg

# 自定义输出路径
python infer.py --onnx-path ./outputs/model.int8.static.onnx --data-root ./data/test --output ./results/predictions.csv
```

**推理产出**：`outputs/predictions.csv`

**CSV 格式示例**：
```csv
filename,pred_label,pred_class_id,prob_cloudy,prob_rainy,prob_snowy,prob_sunny
img_001.jpg,sunny,3,0.0123,0.0045,0.0012,0.9820
img_002.jpg,rainy,1,0.0234,0.9512,0.0123,0.0131
img_003.jpg,cloudy,0,0.9234,0.0312,0.0213,0.0241
```

---

## 五、模型对比策略

比赛规则同分时按推理时间排序。建议保留多个版本对比：

| 模型文件 | 类型 | 速度 | 精度 | 适用场景 |
|---------|------|------|------|---------|
| `model.onnx` | FP32 | 基准 | 最高 | 精度优先 |
| `model.int8.dynamic.onnx` | INT8 Dynamic | ~2x | 略降 | 快速测试 |
| `model.int8.static.onnx` | INT8 Static | ~2-3x | 略降 | **最终提交** |

**对比推理速度**：
```bash
# 对比 FP32 vs INT8
python -c "
import time, numpy as np, onnxruntime as ort

for name in ['outputs/model.onnx', 'outputs/model.int8.static.onnx']:
    sess = ort.InferenceSession(name, providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name
    x = np.random.randn(1, 3, 300, 300).astype(np.float32)
    for _ in range(10): sess.run([out], {inp: x})
    t0 = time.perf_counter()
    for _ in range(100): sess.run([out], {inp: x})
    ms = (time.perf_counter() - t0) / 100 * 1000
    print(f'{name:40s}: {ms:.2f} ms/image')
"
```

---

## 六、常见问题排查

### 6.1 依赖问题

| 错误 | 解决方案 |
|------|---------|
| `ModuleNotFoundError: No module named 'torch'` | `pip install torch torchvision` |
| `ModuleNotFoundError: No module named 'cv2'` | `pip install opencv-python` |
| `ModuleNotFoundError: No module named 'timm'` | `pip install timm` |
| `ModuleNotFoundError: No module named 'albumentations'` | `pip install albumentations` |
| `ModuleNotFoundError: No module named 'onnxruntime'` | `pip install onnx onnxruntime` |
| `No module named 'onnxruntime.quantization'` | 确认 `onnxruntime` 版本 ≥ 1.16.0 |
| `pip install` 超时 | 使用国内镜像：`pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple` |

### 6.2 训练问题

| 错误 | 解决方案 |
|------|---------|
| `CUDA out of memory` | 减小 `--batch-size`（如 32→16→8→4） |
| `CUDA is not available` | 检查 `nvidia-smi`；如无 GPU 加 `--device cpu` |
| `FileNotFoundError: 数据目录不存在` | 检查 `--data-root` 路径是否正确 |
| `训练集为 0 张` | 确认图片放在 `data/train/cloudy/` 等子目录下 |
| 训练 F1 一直不涨 | 尝试 `--lr 5e-4 --epochs 80 --aug-strength 1.5` |
| 验证 F1 远低于训练 F1（过拟合） | 调高 `--aug-strength 1.5`，或增加数据量 |

### 6.3 推理问题

| 错误 | 解决方案 |
|------|---------|
| `onnxruntime` 报 `protobuf` 错误 | `pip install protobuf==3.20.3` |
| `未找到图片文件` | 确认图片扩展名为 `.jpg/.png/.jpeg/.bmp` |
| INT8 模型精度下降明显 | 改用 `static` 量化，增加校准图片数量 |

---

## 七、比赛提交检查清单

提交前请逐项确认：

- [ ] **数据检查**：各类别图片数量是否均衡？是否有损坏/错标图片？
- [ ] **训练完成**：`outputs/best_model.pth` 已生成，验证 Macro F1 ≥ 0.90
- [ ] **ONNX 导出**：`outputs/model.onnx` 已生成，基准测试通过
- [ ] **量化完成**：`outputs/model.int8.static.onnx` 已生成，推理速度有提升
- [ ] **推理测试**：`outputs/predictions.csv` 已生成，预测分布合理
- [ ] **时间预算**：评分集推理总时间 < 70 分钟
- [ ] **代码规范**：代码注释完整，结构清晰

---

## 八、快速命令参考卡

```bash
# === 一键环境搭建 ===
conda create -n raicom python=3.10 -y && conda activate raicom
cd raicom-weather && pip install -r requirements.txt

# === 完整流程 ===
python train.py --data-root ./data --epochs 60 --batch-size 32
python export_onnx.py --checkpoint ./outputs/best_model.pth
python quantize_onnx.py --onnx-path ./outputs/model.onnx --mode static --calib-dir ./data/train --benchmark
python infer.py --onnx-path ./outputs/model.int8.static.onnx --data-root ./data/test

# === 速度测试 ===
python export_onnx.py --checkpoint ./outputs/best_model.pth
python quantize_onnx.py --onnx-path ./outputs/model.onnx --benchmark

# === 从检查点恢复训练 ===
python train.py --resume ./outputs/checkpoint.pth

# === 调参命令 ===
python train.py --data-root ./data --epochs 80 --lr 5e-4 --aug-strength 1.5 --batch-size 32
```

---

## 九、项目文件说明

| 文件 | 说明 |
|------|------|
| `config.py` | 全局配置（模型名、学习率、增强强度等） |
| `augmentations.py` | 数据增强模块（Albumentations） |
| `dataset.py` | 数据集加载（支持文件夹/CSV 两种模式） |
| `model.py` | EfficientNet-B3 + 自定义分类头 |
| `train_utils.py` | 训练工具（Macro F1、CosineWarmup LR、EarlyStopping） |
| `train.py` | 训练主脚本 |
| `export_onnx.py` | ONNX 导出 + CPU 基准测试 |
| `quantize_onnx.py` | ONNX INT8 量化（dynamic / static） |
| `infer.py` | CPU 推理脚本（支持批处理 / TTA / 单张） |
| `requirements.txt` | Python 依赖清单 |
| `references/` | 14 篇顶级文献 PDF + 阅读建议 |

---

## 十、联系方式

- **睿抗官网**：https://www.raicom.com.cn/
- **智海Mo平台**：https://momodel.cn/
- **QQ 交流群**：603641284（验证格式：学校+姓名）
- **报名截止**：2026年6月15日
- **预选赛时间**：2026年6月25日-27日（每天 9:00-17:30，线上）
