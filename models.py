
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from utils import get_optimal_device_config, can_use_flash_attention
import torch

# Function to initialize all resources
def initialize_resources():
    device_config = get_optimal_device_config()

    # Load RAG model
    rag_model = RAGMultiModalModel.from_pretrained(
        "vidore/colqwen2-v1.0", device=torch.device(device_config["rag_device"])
    )

    # Load Qwen2VL model with Flash Attention check
    qwen2vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2" if can_use_flash_attention() else "sdpa",
        device_map=device_config["vl_device"]
    ).eval()

    # Load processor
    min_pixels = 256 * 28 * 28
    max_pixels = 1024 * 28 * 28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    return device_config, rag_model, qwen2vl_model, processor