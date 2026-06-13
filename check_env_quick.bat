python -c "
import shutil, sys, os

print('=== 环境报告 ===')
print('OS:', sys.platform)
print('Python:', sys.version.split()[0])
print('pip:', end=' ')
try:
    import importlib.metadata as im; print(im.version('pip'))
except: print('?')

print()

# 核心
for mod in ['torch','torchvision','timm','albumentations','cv2','onnx','onnxruntime',
            'numpy','pandas','sklearn','tqdm','psutil','pynvml','PIL']:
    try:
        m = __import__(mod)
        v = getattr(m, '__version__', 'OK')
        print(f'  {mod:16s}  {v}')
    except:
        print(f'  {mod:16s}  [MISS]')

# GPU
try:
    import torch
    print()
    print('CUDA:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1))
except: print('[MISS] torch')

# 磁盘
t, u, f = shutil.disk_usage('.')
print()
print(f'磁盘: 可用 {f//1024**3} GB / 总计 {t//1024**3} GB')
" 2>nul
