echo ============================================================
echo  RAICOM 2026 天气分类 — 远程服务器环境检查
echo ============================================================

echo.
echo [1/7] 操作系统
echo ------------------------------
ver
echo.

echo [2/7] Python
echo ------------------------------
python --version 2>nul || echo [FAIL] Python 未安装
pip --version 2>nul || echo [FAIL] pip 未安装
echo.

echo [3/7] NVIDIA GPU + CUDA
echo ------------------------------
nvidia-smi 2>nul || echo [FAIL] nvidia-smi 不可用（无 GPU 或无驱动）
echo.

echo [4/7] PyTorch + CUDA 可用性
echo ------------------------------
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" 2>nul || echo [FAIL] PyTorch 未安装
echo.

echo [5/7] 关键训练依赖
echo ------------------------------
python -c "import torchvision; print('torchvision:', torchvision.__version__)" 2>nul || echo [MISS] torchvision
python -c "import timm; print('timm:', timm.__version__)" 2>nul || echo [MISS] timm
python -c "import albumentations; print('albumentations:', albumentations.__version__)" 2>nul || echo [MISS] albumentations
python -c "import cv2; print('opencv:', cv2.__version__)" 2>nul || echo [MISS] opencv-python
python -c "import numpy; print('numpy:', numpy.__version__)" 2>nul || echo [MISS] numpy
python -c "import pandas; print('pandas:', pandas.__version__)" 2>nul || echo [MISS] pandas
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)" 2>nul || echo [MISS] scikit-learn
python -c "from tqdm import tqdm; print('tqdm: OK')" 2>nul || echo [MISS] tqdm
python -c "import psutil; print('psutil:', psutil.__version__)" 2>nul || echo [MISS] psutil
python -c "import pynvml; print('pynvml: OK')" 2>nul || echo [MISS] pynvml
echo.

echo [6/7] ONNX 推理依赖
echo ------------------------------
python -c "import onnx; print('onnx:', onnx.__version__)" 2>nul || echo [MISS] onnx
python -c "import onnxruntime; print('onnxruntime:', onnxruntime.__version__)" 2>nul || echo [MISS] onnxruntime
echo.

echo [7/7] 磁盘空间
echo ------------------------------
python -c "import shutil; t, u, f = shutil.disk_usage('.'); print(f'可用空间: {f//1024**3} GB / 总计 {t//1024**3} GB')" 2>nul || echo [FAIL]
echo.

echo ============================================================
echo  检查完成。根据上面的 [FAIL] 和 [MISS] 安装缺失依赖。
echo  一键安装命令:
echo    pip install torch torchvision timm albumentations opencv-python numpy pandas scikit-learn tqdm psutil pynvml pyyaml onnx onnxruntime
echo ============================================================
