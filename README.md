# PDF Visual Question Answering App

An interactive application that allows users to ask questions about PDF documents using advanced multimodal AI models.

## Features

- 🚀 Multi-GPU Support
  - Automatically detects and utilizes multiple GPUs
  - Optimally distributes RAG and VL models across available GPUs
  - Falls back to single GPU or CPU when necessary

- ⚡ Hardware Acceleration
  - Automatic Flash Attention 2.0 support detection
  - Enables Flash Attention on compatible GPUs (Compute Capability ≥ 8.0)
  - Falls back to SDPA for older GPUs

- 📑 PDF Processing
  - Interactive PDF document upload
  - Automatic PDF indexing and caching
  - Visual and textual context understanding
  - Smart page selection based on query relevance

## System Requirements

- Python 3.8 or higher
- Poppler (required for PDF processing)
- For GPU acceleration:
  - CUDA-compatible GPU
  - For Flash Attention: NVIDIA GPU with Compute Capability ≥ 8.0
  - Minimum 8GB GPU RAM recommended
- RAM: 16GB minimum recommended

## Installation

### 1. Install Poppler

#### For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

#### For macOS:
```bash
brew install poppler
```

#### For Windows:
Download the latest binary from [poppler releases](http://blog.alivate.com.au/poppler-windows/), extract it, and add the `bin` directory to your PATH.

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application

```bash
chainlit run app.py
```

The app will be available at http://localhost:8000

## Models Used

- RAG: Vidore/colqwen2-v1.0
- VL: Qwen/Qwen2-VL-2B-Instruct

## Usage

1. Access the web interface at http://localhost:8000
2. Upload your PDF document
3. Wait for indexing to complete
4. Start asking questions about the document
5. View responses with relevant page extracts

## Future Enhancements

- 📸 **Screenshots**: We will add screenshots of the app in action soon!

Enjoy using the PDF Visual Question Answering App! 🚀