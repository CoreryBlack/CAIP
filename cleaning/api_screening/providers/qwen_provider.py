"""
RAICOM 2026 — 千问 API provider

通过 DashScope OpenAI 兼容接口调用千问视觉模型。
环境变量：DASHSCOPE_API_KEY
"""

import os
import base64
import json
import time
import sys
from typing import Dict, Any, Optional

import requests

sys.stdout.reconfigure(encoding='utf-8')

from .base import VisionProvider


QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenVisionProvider(VisionProvider):
    def __init__(
        self,
        model: str = "qwen3-vl-flash",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 3.0,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.system_prompt = system_prompt

        if not self.api_key:
            raise ValueError(
                "缺少 DASHSCOPE_API_KEY。请先设置环境变量或传入 api_key。"
            )

        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_messages(self, image_path: str, user_prompt: str) -> list:
        b64 = self._encode_image(image_path)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": user_prompt},
                ],
            }
        )
        return messages

    def _call_api(self, messages: list) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 600,
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{QWEN_API_BASE}/chat/completions",
                    headers=self._headers,
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return self._parse_response(content)
            except Exception as e:
                print(f"      [WARN] API 调用失败 (第 {attempt+1}/{self.max_retries} 次): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return {
                        "status": "error",
                        "error": str(e),
                    }

    def _parse_response(self, content: str) -> Dict[str, Any]:
        try:
            # 去掉可能的 markdown 代码块标记
            text = content.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                start = 1 if lines[0].startswith("```") else 0
                end = -1 if lines[-1].strip() == "```" else len(lines)
                text = "\n".join(lines[start:end])
            return json.loads(text)
        except json.JSONDecodeError:
            # 如果模型没按 JSON 返回，就包一层
            return {
                "status": "parse_error",
                "raw_content": content,
            }

    def screen_image(
        self,
        image_path: str,
        user_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        if user_prompt is None:
            from ..prompts.confusion_prompt import SCREENING_USER_PROMPT
            user_prompt = SCREENING_USER_PROMPT

        messages = self._build_messages(image_path, user_prompt)
        result = self._call_api(messages)
        return result
