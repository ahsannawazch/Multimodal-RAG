
import torch

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

# Function to check if Flash Attention is supported
def can_use_flash_attention():
    if not torch.cuda.is_available():
        return False
    compute_capability = torch.cuda.get_device_capability()
    return compute_capability[0] >= 8