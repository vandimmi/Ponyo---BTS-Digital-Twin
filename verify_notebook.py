import os
import sys
import torch

print("--- Verifying Cell 1 ---")
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
    print("Device Name:", torch.cuda.get_device_name(0))

print("\n--- Verifying Cell 2 ---")
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
print("Set TORCH_CUDA_ARCH_LIST to", os.environ["TORCH_CUDA_ARCH_LIST"])

print("\n--- Verifying Cell 6 ---")
sys.path.append(r"d:\Ponyo---BTS-Digital-Twin\gaussian-splatting")
from scene.dataset_readers import readColmapSceneInfo
from scene.gaussian_model import GaussianModel
print("Successfully imported readColmapSceneInfo and GaussianModel!")
