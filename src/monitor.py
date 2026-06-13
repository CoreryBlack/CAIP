"""
RAICOM 2026 — 训练期硬件/程序监控工具

目标：
- 低侵入接入训练流程
- 输出 CPU / RAM / 进程 RSS / GPU 显存 / GPU 利用率
- 便于初次预训练时快速发现：
  1) DataLoader 卡顿
  2) 显存持续上涨
  3) CPU / 内存瓶颈
  4) 进程内存泄漏

说明：
- CPU / RAM / 进程 RSS 依赖 psutil
- GPU 监控优先用 pynvml；若未安装则退化到 torch.cuda 显存统计
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import csv
import json

import psutil


@dataclass
class ResourceSnapshot:
    timestamp: float
    cpu_percent: float
    ram_used_gb: float
    ram_percent: float
    process_rss_gb: float
    process_threads: int
    gpu_name: str = ""
    gpu_util_percent: Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None
    gpu_mem_total_mb: Optional[float] = None
    gpu_temp_c: Optional[float] = None
    torch_allocated_mb: Optional[float] = None
    torch_reserved_mb: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class ResourceMonitor:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.proc = psutil.Process(os.getpid())
        self._nvml = None
        self._nvml_handle = None
        self._gpu_name = ""
        self._init_gpu_monitor()

    def _init_gpu_monitor(self):
        if self.device != "cuda":
            return
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._gpu_name = pynvml.nvmlDeviceGetName(self._nvml_handle)
            if isinstance(self._gpu_name, bytes):
                self._gpu_name = self._gpu_name.decode("utf-8", errors="ignore")
        except Exception:
            self._nvml = None
            self._nvml_handle = None
            self._gpu_name = ""

    def snapshot(self) -> ResourceSnapshot:
        vm = psutil.virtual_memory()
        mem_info = self.proc.memory_info()
        cpu_percent = psutil.cpu_percent(interval=None)

        gpu_util = None
        gpu_mem_used = None
        gpu_mem_total = None
        gpu_temp = None
        torch_alloc = None
        torch_reserved = None

        if self.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    torch_alloc = round(torch.cuda.memory_allocated() / 1024 / 1024, 2)
                    torch_reserved = round(torch.cuda.memory_reserved() / 1024 / 1024, 2)
            except Exception:
                pass

            if self._nvml and self._nvml_handle:
                try:
                    util = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    mem = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    gpu_util = float(util.gpu)
                    gpu_mem_used = round(mem.used / 1024 / 1024, 2)
                    gpu_mem_total = round(mem.total / 1024 / 1024, 2)
                    try:
                        gpu_temp = float(
                            self._nvml.nvmlDeviceGetTemperature(
                                self._nvml_handle,
                                self._nvml.NVML_TEMPERATURE_GPU,
                            )
                        )
                    except Exception:
                        gpu_temp = None
                except Exception:
                    pass

        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            ram_used_gb=round(vm.used / 1024 / 1024 / 1024, 2),
            ram_percent=float(vm.percent),
            process_rss_gb=round(mem_info.rss / 1024 / 1024 / 1024, 3),
            process_threads=self.proc.num_threads(),
            gpu_name=self._gpu_name,
            gpu_util_percent=gpu_util,
            gpu_mem_used_mb=gpu_mem_used,
            gpu_mem_total_mb=gpu_mem_total,
            gpu_temp_c=gpu_temp,
            torch_allocated_mb=torch_alloc,
            torch_reserved_mb=torch_reserved,
        )

    def compact(self, snap: Optional[ResourceSnapshot] = None) -> str:
        snap = snap or self.snapshot()
        parts = [
            f"CPU {snap.cpu_percent:.0f}%",
            f"RAM {snap.ram_used_gb:.1f}GB/{snap.ram_percent:.0f}%",
            f"RSS {snap.process_rss_gb:.2f}GB",
        ]
        if snap.torch_reserved_mb is not None:
            parts.append(f"CUDA {snap.torch_reserved_mb:.0f}MB")
        elif snap.gpu_mem_used_mb is not None and snap.gpu_mem_total_mb is not None:
            parts.append(f"GPU {snap.gpu_mem_used_mb:.0f}/{snap.gpu_mem_total_mb:.0f}MB")
        if snap.gpu_util_percent is not None:
            parts.append(f"GPUUtil {snap.gpu_util_percent:.0f}%")
        return " | ".join(parts)

    def print_snapshot(self, prefix: str = "[MONITOR]"):
        print(f"{prefix} {self.compact()}")

    def close(self):
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


class ResourceLogger:
    """将资源快照独立写入 JSONL 和 CSV，便于后续画图与定位瓶颈。"""

    def __init__(self, output_dir: str, run_name: str = "train"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.run_name = run_name
        self.jsonl_path = os.path.join(output_dir, f"{run_name}_resource_log.jsonl")
        self.csv_path = os.path.join(output_dir, f"{run_name}_resource_log.csv")
        self._csv_initialized = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0

    def log(self, snap: ResourceSnapshot, stage: str, step: Optional[int] = None, epoch: Optional[int] = None, extra: Optional[Dict] = None):
        payload = snap.to_dict()
        payload.update({
            "stage": stage,
            "step": step,
            "epoch": epoch,
        })
        if extra:
            payload.update(extra)

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        fieldnames = list(payload.keys())
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(payload)

    def paths(self) -> Dict[str, str]:
        return {
            "jsonl": self.jsonl_path,
            "csv": self.csv_path,
        }
