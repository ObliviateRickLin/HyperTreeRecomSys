#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    import torch
    print("PyTorch is available, version:", torch.__version__)
except ImportError:
    print("PyTorch is not available")

try:
    import numpy as np
    print("NumPy is available, version:", np.__version__)
except ImportError:
    print("NumPy is not available")

try:
    from tqdm import tqdm
    print("tqdm is available")
except ImportError:
    print("tqdm is not available") 