import torch
import subprocess

# Function to get the optimal GPU configuration
def get_optimal_device_config():
    if not torch.cuda.is_available():
        return {"rag_device": "cpu", "vl_device": "cpu"}

    num_gpus = torch.cuda.device_count()
    total_memory = {i: torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)}

    if num_gpus >= 2:
        # Use separate GPUs with the most available memory
        gpu0, gpu1 = sorted(total_memory.items(), key=lambda kv: kv[1], reverse=True)[:2]
        return {"rag_device": f"cuda:{gpu0[0]}", "vl_device": f"cuda:{gpu1[0]}"}
    return {"rag_device": "cuda:0", "vl_device": "cuda:0"}

def get_flash_attention_version():
    if not torch.cuda.is_available():
        return None

    compute_capability = torch.cuda.get_device_capability()
    if compute_capability[0] >= 8:
        return "flash_attention_2"
    elif compute_capability[0] >= 7:
        return "flash_attention_1"
    else:
        return None

def install_flash_attention(version):
    if version == "flash_attention_2":
        subprocess.check_call(["pip", "install", "flash-attn==2.x"])
    elif version == "flash_attention_1":
        subprocess.check_call(["pip", "install", "flash-attn==1.x"])

# Function to check if Flash Attention is supported
def can_use_flash_attention():
    version = get_flash_attention_version()
    if version:
        try:
            install_flash_attention(version)
            return True
        except subprocess.CalledProcessError:
            return False
    return False