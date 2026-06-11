"""
RAICOM 2026 — 智海算法调优赛
全局配置
"""

from dataclasses import dataclass, field
from typing import Tuple, List
import os


@dataclass
class Config:
    # ─── 路径 ────────────────────────────────────────────
    data_root: str = "./data"               # 数据集根目录
    train_dir: str = ""                      # 训练图像目录（留空则自动拼接）
    val_dir: str = ""                        # 验证图像目录（留空则自动切分）
    output_dir: str = "./outputs"            # 模型保存 & 日志目录

    # ─── 类别 ────────────────────────────────────────────
    # 天气四分类（映射到智海赛题官方类别）
    classes: List[str] = field(default_factory=lambda: [
        "cloudy",    # 多云
        "rainy",     # 雨天
        "snowy",     # 雪天
        "sunny",     # 晴天
    ])
    num_classes: int = 4

    # ─── 模型 ────────────────────────────────────────────
    model_name: str = "efficientnet_b3"      # timm 模型名
    pretrained: bool = True                  # 使用 ImageNet 预训练权重
    in_channels: int = 3                     # 输入通道
    image_size: int = 300                    # 输入图像尺寸 (EfficientNet-B3 推荐 300)

    # ─── 训练 ────────────────────────────────────────────
    epochs: int = 60                         # 最大训练轮数
    batch_size: int = 64                     # 批大小（根据 GPU 显存调整）
    num_workers: int = 4                     # 数据加载线程数
    lr: float = 1e-3                        # 初始学习率
    weight_decay: float = 1e-4              # 权重衰减
    lr_min: float = 1e-6                    # 学习率下限（CosineAnnealing）
    warmup_epochs: int = 5                   # 预热轮数
    label_smoothing: float = 0.1            # 标签平滑
    early_stop_patience: int = 15           # 早停耐心值

    # ─── 数据增强强度 ────────────────────────────────────
    # 值越大增强越强（适合小数据集或过拟合时调高）
    aug_strength: float = 1.0               # 0.0 = 无增强, 1.0 = 标准, 2.0 = 极强

    # ─── 类别不衡处理 ──────────────────────────────────
    use_weighted_sampler: bool = True        # 是否使用加权采样器
    class_weights: List[float] = field(default_factory=list)  # 留空则自动从数据计算

    # ─── 验证 ────────────────────────────────────────────
    val_split: float = 0.2                   # 从训练集划分验证集比例（无独立 val 时）
    val_freq: int = 1                        # 每 N 轮验证一次

    # ─── 推理 (ONNX) ──────────────────────────────────────
    onnx_opset: int = 17                    # ONNX opset 版本
    cpu_inference_batch: int = 128          # CPU 推理批次

    # ─── 分布式 & 日志 ──────────────────────────────────
    device: str = "cuda"                     # cuda / cpu
    use_amp: bool = True                     # 混合精度训练
    log_freq: int = 50                       # 每 N 步打印日志
    seed: int = 42                           # 随机种子

    # ─── 自动派生字段（无需手动设置） ────────────────────
    def __post_init__(self):
        if not self.train_dir:
            self.train_dir = os.path.join(self.data_root, "train")
        if not self.val_dir:
            self.val_dir = os.path.join(self.data_root, "val")
        os.makedirs(self.output_dir, exist_ok=True)


# 全局单例
cfg = Config()
