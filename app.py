import torch
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import chainlit as cl
from pdf2image import convert_from_path
import os
from io import BytesIO

# Initialize all resources in a single cached function
@cl.cache
def initialize_resources():
    # GPU Configuration
    def get_optimal_device_config():
        if not torch.cuda.is_available():
            return {"rag_device": "cpu", "vl_device": "cpu"}
        
        num_gpus = torch.cuda.device_count()
        total_memory = {i: torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)}
            
        if num_gpus >= 2:
            # Use separate GPUs - choose the ones with most available memory
            gpu0, gpu1 = sorted(total_memory.items(), key=lambda kv: kv[1], reverse=True)[:2]
            return {
                "rag_device": f"cuda:{gpu0[0]}", 
                "vl_device": f"cuda:{gpu1[0]}"
            }
        else:
            # Single GPU - use the same device for both
            return {
                "rag_device": "cuda:0",
                "vl_device": "cuda:0"
            }

    device_config = get_optimal_device_config()

    # Load RAG model
    rag_model = RAGMultiModalModel.from_pretrained(
        "vidore/colqwen2-v1.0",
        device=torch.device(device_config["rag_device"])
    )

    # Load Qwen2VL model
    def can_use_flash_attention():
        """Check if the GPU supports Flash Attention 2"""
        if not torch.cuda.is_available():
            return False
        
        # Get compute capability of the GPU
        compute_capability = torch.cuda.get_device_capability()
        # Flash Attention requires Ampere (compute capability >= 8.0) or newer
        return compute_capability[0] >= 8

    qwen2vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", 
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2" if can_use_flash_attention() else "sdpa",  # fallback to sdpa
        device_map=device_config["vl_device"]
    ).eval()

    # Load processor
    min_pixels = 256*28*28
    max_pixels = 1024*28*28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    return device_config, rag_model, qwen2vl_model, processor

# Chainlit App
@cl.on_chat_start
async def start():
    # Initialize all resources using the cached function
    device_config, rag_model, qwen2vl_model, processor = initialize_resources()
    cl.user_session.set("device_config", device_config)
    cl.user_session.set("rag_model", rag_model)
    cl.user_session.set("qwen2vl_model", qwen2vl_model)
    cl.user_session.set("processor", processor)

    # Ask the user to upload a PDF file
    files = await cl.AskFileMessage(
        content="Please upload a PDF file to begin!",
        accept=["application/pdf"],
        max_size_mb=50,  # Adjust the max file size as needed
        timeout=180,  # Allow 3 minutes for the user to upload
    ).send()

    if files:
        # Save the uploaded file
        pdf_file = files[0]
        pdf_path = pdf_file.path  # Use the path attribute to get the file path
        pdf_name = pdf_file.name

        # Read the file content in binary mode and save it
        with open(pdf_name, "wb") as f:
            with open(pdf_path, "rb") as uploaded_file:
                f.write(uploaded_file.read())

        # Index the PDF file using the RAG model
        rag_model.index(
            input_path=pdf_name, 
            index_name="uploaded_pdf_1", 
            store_collection_with_index=False,  # Store the base64 encoded documents
            overwrite=True  # Do not overwrite if the index already exists
        )

        # Convert the PDF to images
        images = convert_from_path(pdf_name)
        cl.user_session.set("images", images)
        cl.user_session.set("pdf_name", pdf_name)

        await cl.Message(content=f"PDF '{pdf_name}' uploaded and indexed successfully! You can now ask questions about it.").send()
    else:
        await cl.Message(content="No file uploaded. Please restart the app to upload a PDF.").send()

@cl.on_message
async def main(message: cl.Message):
    # Check if a PDF has been uploaded
    images = cl.user_session.get("images")
    if not images:
        await cl.Message(content="Please upload a PDF file first!").send()
        return

    # Get models and processor from the session
    rag_model = cl.user_session.get("rag_model")
    qwen2vl_model = cl.user_session.get("qwen2vl_model")
    processor = cl.user_session.get("processor")
    device_config = cl.user_session.get("device_config")

    query = message.content

    # Perform RAG search
    results = rag_model.search(query, k=2)
    pages = [result['page_num'] for result in results]

    # Get images for the pages
    image_index = [page - 1 for page in pages]

    # Display the relevant pages
    for idx in image_index:
        # Convert the PIL image to bytes
        img_byte_arr = BytesIO()
        images[idx].save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        # Send the image to the chat
        # await cl.Message(content=f"Relevant page {pages[image_index.index(idx)]}:").send()
        await cl.Image(name=f"Page {pages[image_index.index(idx)]}", display="inline", content=img_byte_arr).send(for_id=message.id)

    # Construct the messages list dynamically
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": images[index]} for index in image_index],  # Separate entries for each image
                {"type": "text", "text": query},  # Text query
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device_config["vl_device"]) for k, v in inputs.items()}

    # Inference: Generation of the output
    generated_ids = qwen2vl_model.generate(**inputs, max_new_tokens=128)  # Optimize the number of tokens

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Send the response back to the user
    await cl.Message(content=output_text[0]).send()

    # Clean up the uploaded PDF file after processing
    pdf_name = cl.user_session.get("pdf_name")
    if pdf_name and os.path.exists(pdf_name):
        os.remove(pdf_name)