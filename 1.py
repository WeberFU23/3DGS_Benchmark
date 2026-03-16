import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"当前 GPU 设备: {torch.cuda.get_device_name(0)}")
else:
    print("当前使用的是 CPU，请检查安装命令是否正确。")