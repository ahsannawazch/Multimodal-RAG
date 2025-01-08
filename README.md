🚀 Multimodal RAG App 🚀

Introducing the Multimodal RAG App—your ultimate solution for extracting and understanding information from complex PDF documents containing images, charts, tables, and graphs! 📄🔍

This application leverages ColQwen2-v1.0, a state-of-the-art visual retriever based on Qwen2-VL-2B-Instruct with ColBERT strategy. ColQwen2 processes entire document pages as images, generating ColBERT-style multi-vector representations that capture both textual and visual cues, preserving each page's structure and context.

To streamline interactions with ColQwen2, we utilize the Byaldi library designed to facilitate the use of late-interaction multi-modal models like ColPQwen with a familiar API, thereby enhancing the efficiency of document retrieval tasks.

📑 **PDF Processing**

- Interactive PDF document upload.
- Automatic PDF indexing and caching.
- Visual and textual context understanding.
- Smart page selection based on query relevance.

🚀 **Multi-GPU Support**

- Automatically detects and utilizes multiple GPUs for optimal performance.
- Optimally distributes Visual Retriever (ColQwen2) and VL models across available GPUs.
- Falls back to single GPU or CPU when necessary.

⚡ **Hardware Acceleration**

- Automatic Flash Attention 2.0 support detection for compatible GPUs (Compute Capability ≥ 8.0).
- Falls back to SDPA for older GPUs.

📋 **Requirements**

Before you begin, ensure you have the following installed:

- Python 3.10 or higher 🐍
- Poppler (used for PDF processing) 📄

📥 **Installing Poppler**

For Linux (Ubuntu) 🐧
```bash
sudo apt-get install -y poppler-utils
```

For macOS 🍎
```bash
brew install poppler
```

For Windows 🖥️

1. Download Poppler for Windows from this [source](http://blog.alivate.com.au/poppler-windows/).
2. Extract the archive and add the bin folder to your system's PATH variable.

🛠️ **Installation**

1. Clone the repository:
    ```bash
    git clone https://github.com/ahsannawazch/Multimodal-RAG.git
    cd Multimodal-RAG
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

🚀 **Usage**

1. Run the app:
    ```bash
    chainlit run app.py
    ```

2. Upload a PDF: When prompted, upload your PDF file to begin indexing it on disk.

3. Ask Questions: Once the PDF is uploaded and indexed, you can ask questions about the content, and the app will retrieve and display relevant information, including images and text.

Enjoy exploring your documents with the Multimodal RAG App! 🎉📚
