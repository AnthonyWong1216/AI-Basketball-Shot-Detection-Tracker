import torch
print(f"是否支援 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"目前使用的 GPU: {torch.cuda.get_device_name(0)}")
