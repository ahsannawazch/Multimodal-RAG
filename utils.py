import torch
import subprocess
import importlib.util
from importlib.metadata import PackageNotFoundError, version

# Function to get the optimal GPU configuration
def get_optimal_device_config():
    if not torch.cuda.is_available():
        return {"rag_device": "cpu", "vl_device": "cpu"}

    num_gpus = torch.cuda.device_count()
    total_memory = {i: torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)}

    if num_gpus >= 2:
        gpu0, gpu1 = sorted(total_memory.items(), key=lambda kv: kv[1], reverse=True)[:2]
        return {"rag_device": f"cuda:{gpu0[0]}", "vl_device": f"cuda:{gpu1[0]}"}
    return {"rag_device": "cuda:0", "vl_device": "cuda:0"}

def is_package_installed(package_name):
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except ModuleNotFoundError:
        return False

def install_flash_attention():
    try:
        import pip
        print("Installing flash-attn package...")
        pip.main(['install', 'flash-attn>=2.0.0'])
        return True
    except Exception as e:
        print(f"Failed to install flash-attn: {e}")
        return False

def get_flash_attention_version():
    if not torch.cuda.is_available():
        return None
    
    try:
        # Get GPU name
        gpu_name = torch.cuda.get_device_name()
        compute_capability = torch.cuda.get_device_capability()
        print(f'GPU Name: {gpu_name},  Compute Capability: {compute_capability}')

        # List of GPUs that support Flash Attention 2
        fa2_compatible_gpus = ['A100', 'H100', 'L4', 'A10G', 'A6000']
        
        # Check if current GPU supports Flash Attention 2
        supports_fa2 = any(gpu.lower() in gpu_name.lower() for gpu in fa2_compatible_gpus)
        
        if supports_fa2 and compute_capability[0] >= 8:
            # Try to install if not already installed
            if not is_package_installed("flash_attn"):
                if not install_flash_attention():
                    return None
            
            try:
                installed_version = version("flash-attn")
                major_version = int(installed_version.split('.')[0])
                if major_version >= 2:
                    return "flash_attention_2"
            except PackageNotFoundError:
                pass
    except Exception as e:
        print(f"Error checking Flash Attention compatibility: {e}")
        return None

def can_use_flash_attention():
    version = get_flash_attention_version()
    return version == "flash_attention_2"
