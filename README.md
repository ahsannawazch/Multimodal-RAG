📚 Multimodal RAG App

Overview

Introducing Multimodal RAG App—your ultimate solution for extracting and understanding information from complex PDF documents containing images, charts, tables, and graphs.

This application utilizes ColQwen2-v1.0, a cutting-edge multimodal vector representation retriever that processes entire document pages as images. By generating multi-vector embeddings, it captures both textual and visual cues, preserving each page's structure and context. (huggingface.co)

To streamline interactions with ColQwen2, we're using Byaldi library. Byaldi serves as a user-friendly interface, simplifying the implementation of late-interaction multimodal models like ColPALI, thereby enhancing the efficiency of document retrieval tasks. (github.com)

Features

- 📄 **PDF Upload**: Upload your PDF documents directly to the app.

- 🔍 **Efficient Search**: Perform searches within the document using advanced RAG models.

- 🖼️ **Image Handling**: Extract and display images, charts, tables, and graphs from your documents.

🚀 Multi-GPU Support

Automatically detects and utilizes multiple GPUs

Optimally distributes Vision-RAG (ColQwen2) and VL models across available GPUs

Falls back to single GPU or CPU when necessary

⚡ Hardware Acceleration

Automatic Flash Attention 2.0 support detection

Enables Flash Attention on compatible GPUs (Compute Capability ≥ 8.0)

Falls back to SDPA for older GPUs

📑 PDF Processing

Interactive PDF document upload

Automatic PDF indexing and caching

Visual and textual context understanding

Smart page selection based on query relevance

Requirements

Before you begin, ensure you have the following installed:

Python 3.10 or higher

Poppler (used for PDF processing)

Installing Poppler

For Linux (Ubuntu)

sudo apt-get install -y poppler-utils

For macOS

brew install poppler

For Windows

Download Poppler for Windows from this source.

Extract the archive and add the bin folder to your system's PATH variable.

Installation

Clone the repository:

git clone https://github.com/ahsannawazch/Multimodal-RAG.git
cd Multimodal-RAG

Install the required Python packages:

pip install -r requirements.txt

Usage

Run the app:

chainlit run app.py

Upload a PDF: When prompted, upload your PDF file to begin indexing and searching.

Ask Questions: Once the PDF is uploaded and indexed, you can ask questions about the content, and the app will retrieve and display relevant information, including images and text.


