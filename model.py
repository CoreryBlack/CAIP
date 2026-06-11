"""
RAICOM 2026 — 智海算法调优赛
模型定义 (EfficientNet-B3 + 自定义分类头)
"""

import timm
import torch
import torch.nn as nn
from typing import Optional


class WeatherClassifier(nn.Module):
    """
    天气分类模型

    主干: EfficientNet-B3 (timm)
    分类头: 自定义 FC 层 (支持 Dropout)
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name

        # 加载 EfficientNet-B3 主干
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # 去掉原始分类头，只取特征
            global_pool="avg",      # 平均池化
        )

        # 获取特征维度
        feat_dim = self.backbone.num_features

        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes),
        )

        # 初始化分类头权重
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)       # [B, feat_dim]
        logits = self.classifier(features)  # [B, num_classes]
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征向量（用于知识蒸馏或集成）"""
        return self.backbone(x)


def create_model(cfg) -> WeatherClassifier:
    """根据配置创建模型"""
    model = WeatherClassifier(
        num_classes=cfg.num_classes,
        model_name=cfg.model_name,
        pretrained=cfg.pretrained,
        dropout_rate=0.3,
    )
    return model


def load_model(checkpoint_path: str, cfg) -> WeatherClassifier:
    """从检查点加载模型权重"""
    model = create_model(cfg)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # 兼容 DDP 或完整 checkpoint
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # 去掉 module. 前缀（DDP 保存时加的）
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "")
        new_state_dict[key] = v

    model.load_state_dict(new_state_dict, strict=False)
    return model
