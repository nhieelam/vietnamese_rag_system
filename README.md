# 🇻🇳 Vietnamese RAG System

> A Retrieval-Augmented Generation (RAG) system built with Streamlit and LangChain, designed to answer questions from your Vietnamese documents.

---

## 📋 Prerequisites

-   **Python**: `3.10.3`
-   **Virtual Environment**: Recommended to avoid package conflicts.

---

## 🚀 Getting Started

Follow these steps to get the application up and running.

### 1. Set Up Environment

First, create and activate your virtual environment.

```powershell
# Create a virtual environment (if you haven't already)
python -m venv venv

# Activate it (on Windows)
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Run the Application

Launch the Streamlit web interface with the following command:

```bash
streamlit run app/main.py
```

---

## 📄 Document Processing & OCR

The system can extract text from various file formats, including:
-   PDFs (both text-based and scanned)
-   Images (`PNG`, `JPG`, `JPEG`)
-   Word Documents (`DOCX`)

For scanned documents and images, the application uses **Tesseract OCR** to recognize and extract text.

### Tesseract OCR Installation

To enable OCR functionality, you must install Google's Tesseract OCR engine on your system.

**On Windows:**
1.  Download the installer from the [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) page.
2.  Run the installer. **Important:** Make sure to add Tesseract to your system's `PATH` during installation so the application can find it.

**On macOS:**
```bash
brew install tesseract
```

**On Linux (Debian/Ubuntu):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

---

## ⚙️ LLM Configuration

This application supports multiple Large Language Model (LLM) providers.

### Using a Local LLM with Ollama

To run the system with a model on your local machine, follow these steps:

1.  **Install Ollama**:
    Download and install Ollama from the [official website](https://ollama.com/).

2.  **Download the Model**:
    Open your terminal and pull the recommended model. This project is configured to use `qwen2.5:7b`.
    ```bash
    ollama pull qwen2.5:7b
    ```

3.  **Run Ollama**:
    Ensure the Ollama application is running in the background **before** you start the Streamlit app. The system will connect to it automatically.