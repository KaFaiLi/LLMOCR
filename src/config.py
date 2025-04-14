import os
from typing import List

class Config:
    # Folders
    INPUT_FOLDER = 'input_pdfs'
    OUTPUT_FOLDER = 'output'
    IMAGES_FOLDER = os.path.join(OUTPUT_FOLDER, 'images')
    RESULTS_FOLDER = os.path.join(OUTPUT_FOLDER, 'results')
    MARKDOWN_FOLDER = os.path.join(OUTPUT_FOLDER, 'markdown')
    
    # OCR settings
    TESSERACT_CMD = r'tesseract'  # Update with your tesseract path if needed
    
    # Image preprocessing settings
    DPI = 300
    SCALE_FACTOR = 2.0  # For upscaling images
    DENOISE_STRENGTH = 10
    
    # LLM settings
    USE_LLM = True  # Set to False to disable LLM processing
    LLM_MODEL_NAME = "gpt-4-vision-preview"  # Using GPT-4 Vision for image processing
    LLM_API_KEY = "your-api-key"  # Replace with your actual API key
    LLM_API_BASE = "https://api.openai.com/v1"  # OpenAI API endpoint
    
    # Extraction fields (customize based on your needs)
    EXTRACT_FIELDS: List[str] = [
        "invoice_number", 
        "date", 
        "total_amount", 
        "vendor_name",
        "customer_name"
    ]
    
    # Image processing settings
    MAX_IMAGE_SIZE = 1024  # Maximum dimension for image processing
    IMAGE_QUALITY = 85  # JPEG quality for image compression 