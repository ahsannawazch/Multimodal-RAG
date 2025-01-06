import torch
from qwen_vl_utils import process_vision_info
import chainlit as cl
from pdf2image import convert_from_path
from io import BytesIO
import os
import asyncio

# Change from relative to absolute imports
from utils import get_optimal_device_config, can_use_flash_attention
from models import initialize_resources

# Chainlit App
@cl.on_chat_start
async def start():
    # Initialize resources using the cached function
    device_config, rag_model, qwen2vl_model, processor = initialize_resources()
    cl.user_session.set("device_config", device_config)
    cl.user_session.set("rag_model", rag_model)
    cl.user_session.set("qwen2vl_model", qwen2vl_model)
    cl.user_session.set("processor", processor)

    # Ask the user to upload a PDF file
    files = await cl.AskFileMessage(
        content="Please upload a PDF file to begin!",
        accept=["application/pdf"],
        max_size_mb=50,
        timeout=180,
    ).send()

    if files:
        pdf_file = files[0]
        pdf_path = pdf_file.path
        pdf_name = pdf_file.name

        # Show a loading message while indexing the PDF
        await cl.Message(content=f"Indexing PDF '{pdf_name}'. This may take a moment...").send()

        # Index the PDF asynchronously to avoid blocking the main thread

        await asyncio.to_thread(
            rag_model.index,
            input_path=pdf_path,
            index_name=pdf_name,
            store_collection_with_index=False,
            overwrite=True,
        )


        # Convert PDF to images asynchronously
        images = await asyncio.to_thread(convert_from_path, pdf_path)
        cl.user_session.set("images", images)
        cl.user_session.set("pdf_name", pdf_name)

        await cl.Message(content=f"PDF '{pdf_name}' uploaded and indexed successfully! You can now ask questions about it.").send()
    else:
        await cl.Message(content="No file uploaded. Please restart the app to upload a PDF.").send()

@cl.on_message
async def main(message: cl.Message):
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

    # Perform RAG search asynchronously
    results = await asyncio.to_thread(rag_model.search, query, k=2)
    pages = [result['page_num'] for result in results]

    # Get images for the pages
    image_index = [result['page_num'] - 1 for result in results]

    # Display the relevant pages
    for idx in image_index:
        img_byte_arr = BytesIO()
        images[idx].save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        await cl.Image(name=f"Page {pages[image_index.index(idx)]}", display="inline", content=img_byte_arr).send(for_id=message.id)

    # Construct the messages list dynamically
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": images[index]} for index in image_index],
                {"type": "text", "text": query},
            ],
        }
    ]

    # Prepare inputs for inference asynchronously
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device_config["vl_device"]) for k, v in inputs.items()}

    # Perform inference asynchronously
    generated_ids = await asyncio.to_thread(
        qwen2vl_model.generate,
        **inputs,
        max_new_tokens=128,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
    ]

    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Send the response back to the user
    await cl.Message(content=output_text[0]).send()

    # Clean up the uploaded PDF file after processing
    pdf_name = cl.user_session.get("pdf_name")
    if pdf_name and os.path.exists(pdf_name):
        os.remove(pdf_name)


