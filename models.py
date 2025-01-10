from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from utils import get_optimal_device_config, get_flash_attention_version
import torch

# Function to initialize all resources
def initialize_resources():
    device_config = get_optimal_device_config()

    # Load RAG model
    rag_model = RAGMultiModalModel.from_pretrained(
        "vidore/colqwen2-v1.0", device=torch.device(device_config["rag_device"])
    )

    # Check if Flash Attention 2 is available and supported
    attn_implementation = get_flash_attention_version() or "sdpa"
    print(f"Using attention implementation: {attn_implementation}")


    # Load Qwen2VL model with correct attention implementation
    qwen2vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        attn_implementation=attn_implementation,
        device_map=device_config["vl_device"]
    ).eval()

    # Load processor
    min_pixels = 256 * 28 * 28
    max_pixels = 1024 * 28 * 28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    return device_config, rag_model, qwen2vl_model, processor