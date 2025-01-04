# 📚 Multimodal RAG App

## Overview

This app leverages the power of Multimodal Retrieval-Augmented Generation (RAG) to help you find and understand information from PDFs and documents that contain images, charts, tables, and graphs. Using the ColQwen retriever from the Byaldi library, this app can efficiently index and search through your documents.

## Features

- 📄 **PDF Upload**: Upload your PDF documents directly to the app.
- 🔍 **Efficient Search**: Perform searches within the document using advanced RAG models.
- 🖼️ **Image Handling**: Extract and display images, charts, tables, and graphs from your documents.
- 🤖 **AI-Powered**: Utilize state-of-the-art models for conditional generation and retrieval.

## Requirements

Before you begin, ensure you have the following installed:

- Python 3.10 or higher
- Poppler (used for PDF processing)

### Installing Poppler

#### For Linux (Ubuntu)
```bash
sudo apt-get install -y poppler-utils
```

#### For macOS
```bash
brew install poppler
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. **Install the required Python packages**:
```bash
pip install -r requirements.txt
```

## Usage

1. **Run the app**:
```bash
chainlit run app.py
```

2. **Upload a PDF**: When prompted, upload your PDF file to begin indexing and searching.

3. **Ask Questions**: Once the PDF is uploaded and indexed, you can ask questions about the content, and the app will retrieve and display relevant information, including images and text.

## Future Enhancements

- 📸 **Screenshots**: We will add screenshots of the app in action soon!

Enjoy using the Multimodal RAG App! 🚀