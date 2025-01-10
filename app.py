from byaldi import RAGMultiModalModel
from qwen_vl_utils import process_vision_info
import chainlit as cl
from pdf2image import convert_from_path
from io import BytesIO
import asyncio
from models import initialize_resources

# Remove the cache decorator
def load_resources():
    return initialize_resources()

# Cache PDF-to-image conversion
# @cl.cache
def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

# Chainlit App
@cl.on_chat_start
async def start():
    # Initialize resources using the cached function
    device_config, rag_model, qwen2vl_model, processor = load_resources()
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
        loading_message = cl.Message(content=f"Indexing PDF '{pdf_name}'. This may take a moment...")
        await loading_message.send()

        # Index the PDF asynchronously with proper error handling
        try:
            await asyncio.to_thread(
                rag_model.index,
                input_path=pdf_path,
                index_name=pdf_name,
                store_collection_with_index=False,
                overwrite=False
            )
            loading_message.content = f"PDF '{pdf_name}' indexed successfully!"
            await loading_message.update()
        except ValueError as e:
            if f"An index named {pdf_name} already exists" in str(e):
                # Instead of updating the loading message, send a new one
                await cl.Message(content="Index already exists. Loading the existing index...").send()

                # Load the existing index
                rag_model = RAGMultiModalModel.from_index(pdf_name)
                cl.user_session.set("rag_model", rag_model)

                # Confirm the existing index was loaded
                await cl.Message(content="Existing index loaded successfully!").send()
            else:
                # Handle other errors
                await cl.Message(content=f"Error during indexing: {str(e)}").send()
                return

        # Convert PDF to images asynchronously (cached)
        images = await asyncio.to_thread(convert_pdf_to_images, pdf_path)
        cl.user_session.set("images", images)
        cl.user_session.set("pdf_name", pdf_name)

        await cl.Message(content="You can now ask questions about the document.").send()
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

    # Prepare image elements
    image_elements = []
    for idx in image_index:
        img_byte_arr = BytesIO()
        images[idx].save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()
        image_element = cl.Image(
            name=f"Page {pages[image_index.index(idx)]}",
            content=img_byte_arr,
            display="inline",
            mime="image/jpeg",
            size="medium",
            url=None
        )
        image_elements.append(image_element)

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
        max_new_tokens=102,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
    ]

    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Send the response back to the user with images
    await cl.Message(
        content=output_text[0],
        elements=image_elements,
    ).send()

