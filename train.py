#!/usr/bin/env python3
"""薄入口: python train.py -> 转到 src/train.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from train import main
if __name__ == '__main__':
    main()
