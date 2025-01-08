🚀 Multimodal RAG App 🚀
Introducing the Multimodal RAG App—your ultimate solution for extracting and understanding information from complex PDF documents containing images, charts, tables, and graphs! 📄🔍

This application leverages ColQwen2-v1.0, a state-of-the-art visual retriever based on Qwen2-VL-2B-Instruct with ColBERT strategy. ColQwen2 processes entire document pages as images, generating ColBERT-style multi-vector representations that capture both textual and visual cues, preserving each page's structure and context.

To streamline interactions with ColQwen2, we utilize the Byaldi library designed to facilitate the use of late-interaction multi-modal models like ColPQwen with a familiar API, thereby enhancing the efficiency of document retrieval tasks.
 🛠️ 2 ✨ Features

    📄 PDF Upload: Upload your PDF documents directly to the app.

    🔍 Efficient Search: Perform searches within the document using advanced RAG models.

    🖼️ Image Handling: Extract and display images, charts, tables, and graphs from your documents.

    ⚡ Multi-GPU Support: Automatically detects and utilizes multiple GPUs for optimal performance.

    📑 Smart PDF Processing: Interactive PDF document upload, automatic indexing, and caching.

    🧠 Visual and Textual Context Understanding: Smart page selection based on query relevance.

🚀 Multi-GPU Support

    Automatically detects and utilizes multiple GPUs for optimal performance.

    Optimally distributes Vision-RAG (ColQwen2) and VL models across available GPUs.

    Falls back to single GPU or CPU when necessary.

⚡ Hardware Acceleration

    Automatic Flash Attention 2.0 support detection for compatible GPUs (Compute Capability ≥ 8.0).

    Falls back to SDPA for older GPUs.

📑 PDF Processing

    Interactive PDF document upload.

    Automatic PDF indexing and caching.

    Visual and textual context understanding.

    Smart page selection based on query relevance.

📋 Requirements

Before you begin, ensure you have the following installed:

    Python 3.10 or higher 🐍

    Poppler (used for PDF processing) 📄

📥 Installing Poppler
For Linux (Ubuntu) 🐧
bash
Copy

sudo apt-get install -y poppler-utils

For macOS 🍎
bash
Copy

brew install poppler

For Windows 🖥️

    Download Poppler for Windows from this source 10.

    Extract the archive and add the bin folder to your system's PATH variable.

🛠️ Installation

    Clone the repository:
    bash
    Copy

    git clone https://github.com/ahsannawazch/Multimodal-RAG.git
    cd Multimodal-RAG

    Install the required Python packages:
    bash
    Copy

    pip install -r requirements.txt

🚀 Usage

    Run the app:
    bash
    Copy

    chainlit run app.py

    Upload a PDF: When prompted, upload your PDF file to begin indexing and searching.

    Ask Questions: Once the PDF is uploaded and indexed, you can ask questions about the content, and the app will retrieve and display relevant information, including images and text.

Enjoy exploring your documents with the Multimodal RAG App! 🎉📚
