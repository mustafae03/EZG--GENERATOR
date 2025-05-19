import torch

print("CUDA ", torch.cuda.is_available())

if torch.cuda.is_available():
    print("u GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU yok.")
