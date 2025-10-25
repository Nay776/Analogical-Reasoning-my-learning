# 创建脚本检查环境版本
import torch
print('Torch:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
