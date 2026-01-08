import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA verfügbar: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Grafikkarte: {torch.cuda.get_device_name(0)}")
else:
    print("ACHTUNG: Du läufst auf CPU!")