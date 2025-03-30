# LLMOCR

LLMOCR is a Python-based application for extracting and processing text from PDF documents using OCR (Optical Character Recognition) and LLM (Large Language Model) technologies. It combines the power of Tesseract, EasyOCR, and LangChain to provide accurate and structured text extraction.

## Features

- Extracts text from PDF documents using Tesseract and EasyOCR.
- Combines OCR results using a Large Language Model (LLM) for improved accuracy.
- Extracts structured information (e.g., invoice details) using LangChain and Pydantic.
- Supports image preprocessing for better OCR quality.
- Saves results in an Excel file for easy access.

## Requirements

- Python 3.8 or higher
- Tesseract OCR installed and configured
- OpenAI API key for LLM integration

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KaFaiLi/LLMOCR.git
   cd LLMOCR
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - Windows: Download and install from [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
   - Linux: Install via package manager (e.g., `sudo apt install tesseract-ocr`).

4. Update the `Config` class in `slow.py` with your OpenAI API key and Tesseract path if needed.

## Usage

1. Place your PDF files in the `input_pdfs` folder.
2. Run the application:
   ```bash
   python main.py
   ```
3. Processed images and results will be saved in the `output` folder.

## Folder Structure

- `input_pdfs`: Folder for input PDF files.
- `output/images`: Folder for extracted images from PDFs.
- `output/results`: Folder for OCR results and Excel file.

## Configuration

The `Config` class in `main.py` allows you to customize the following:

- Input and output folder paths.
- Tesseract command path.
- Image preprocessing settings (DPI, scale factor, denoise strength).
- LLM model name and API key.
- Fields to extract from the text.


## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.