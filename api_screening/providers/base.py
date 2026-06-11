"""Provider 抽象基类"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class VisionProvider(ABC):
    """视觉 API provider 抽象"""

    @abstractmethod
    def screen_image(self, image_path: str) -> Dict[str, Any]:
        """对单张图片做混淆筛查，返回结构化结果"""
        ...
