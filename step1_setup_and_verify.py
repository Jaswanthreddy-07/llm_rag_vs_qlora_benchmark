import torch
import sys

def verify_environment():
    print("=" * 60)
    print("STEP 1: ENVIRONMENT & HARDWARE VERIFICATION")
    print("=" * 60)
    print("Python Version:", sys.version.split()[0])
    print("PyTorch Version:", torch.__version__)
    
    cuda_available = torch.cuda.is_available()
    print("CUDA (GPU) Available:", cuda_available)
    
    if cuda_available:
        print("GPU Model:", torch.cuda.get_device_name(0))
        print("Total VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")
    else:
        print("[NOTICE] Running on CPU mode. GPU recommended for lower latency and model evaluation.")
    print("=" * 60)

if __name__ == "__main__":
    verify_environment()
