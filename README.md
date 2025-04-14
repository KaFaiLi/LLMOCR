# LLMOCR

A Python package for OCR processing with LLM enhancement, combining Tesseract and EasyOCR with language model capabilities.

## Project Structure

```
LLMOCR/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   └── processor.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LLMOCR.git
cd LLMOCR
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
- On macOS: `brew install tesseract`
- On Ubuntu: `sudo apt-get install tesseract-ocr`
- On Windows: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

1. Place your PDF files in the `input_pdfs` directory
2. Run the main script:
```bash
python src/main.py
```

3. Results will be saved in the `output` directory:
- `output/images/`: Extracted and enhanced images
- `output/results/`: OCR results in Excel format

## Configuration

Edit `src/config.py` to customize:
- Input/output directories
- OCR settings
- LLM model and API key
- Extraction fields

## Features

- PDF to image conversion
- Image preprocessing and enhancement
- OCR using both Tesseract and EasyOCR
- LLM-based text correction and enhancement
- Structured information extraction
- Results export to Excel

## Requirements

- Python 3.8+
- Tesseract OCR
- CUDA (optional, for GPU acceleration with EasyOCR)

## License

MIT License