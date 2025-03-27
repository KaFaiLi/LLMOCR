import os
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import pytesseract
import easyocr
import fitz  # PyMuPDF
from PIL import Image
import requests
from tqdm import tqdm
from typing import Optional

# LangChain imports for structured output
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ocr_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Folders
    INPUT_FOLDER = 'input_pdfs'
    OUTPUT_FOLDER = 'output'
    IMAGES_FOLDER = os.path.join(OUTPUT_FOLDER, 'images')
    RESULTS_FOLDER = os.path.join(OUTPUT_FOLDER, 'results')
    
    # OCR settings
    TESSERACT_CMD = r'tesseract'  # Update with your tesseract path if needed
    
    # Image preprocessing settings
    DPI = 300
    SCALE_FACTOR = 2.0  # For upscaling images
    DENOISE_STRENGTH = 10
    
    # LLM settings
    # In this updated version, we use ChatOpenAI from LangChain which takes an OpenAI API key
    LLM_MODEL_NAME = "gpt-4"
    LLM_API_KEY = "your-api-key"  # Replace with your actual API key
    
    # Extraction fields (customize based on your needs)
    EXTRACT_FIELDS = [
        "invoice_number", 
        "date", 
        "total_amount", 
        "vendor_name",
        "customer_name"
    ]

# Pydantic model for the extracted fields.
class ExtractionResult(BaseModel):
    invoice_number: Optional[str] = Field(None, description="Invoice number")
    date: Optional[str] = Field(None, description="Invoice date")
    total_amount: Optional[str] = Field(None, description="Total amount on the invoice")
    vendor_name: Optional[str] = Field(None, description="Name of the vendor")
    customer_name: Optional[str] = Field(None, description="Name of the customer")

class PDFProcessor:
    def __init__(self, config: Config):
        self.config = config
        # Initialize EasyOCR reader (do this once to avoid loading the model multiple times)
        self.reader = easyocr.Reader(['en'])
        # Set pytesseract command
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
        
        # Create output directories if they don't exist
        os.makedirs(config.INPUT_FOLDER, exist_ok=True)
        os.makedirs(config.IMAGES_FOLDER, exist_ok=True)
        os.makedirs(config.RESULTS_FOLDER, exist_ok=True)
        
        # Initialize results dataframe
        self.results_df = pd.DataFrame(columns=[
            'pdf_name', 
            'page_number', 
            'tesseract_text', 
            'easyocr_text', 
            'combined_text',
            *config.EXTRACT_FIELDS
        ])
    
    def process_all_pdfs(self):
        """Process all PDFs in the input folder"""
        start_time = time.time()
        logger.info(f"Starting to process PDFs in {self.config.INPUT_FOLDER}")
        
        pdf_files = [f for f in os.listdir(self.config.INPUT_FOLDER) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.config.INPUT_FOLDER}")
            return
        
        for pdf_file in pdf_files:
            self.process_pdf(pdf_file)
        
        # Save final results to Excel
        excel_path = os.path.join(self.config.RESULTS_FOLDER, 'ocr_results.xlsx')
        self.results_df.to_excel(excel_path, index=False)
        
        logger.info(f"Completed processing {len(pdf_files)} PDFs in {time.time() - start_time:.2f} seconds")
        logger.info(f"Results saved to {excel_path}")
    
    def process_pdf(self, pdf_filename):
        """Process a single PDF file"""
        pdf_path = os.path.join(self.config.INPUT_FOLDER, pdf_filename)
        logger.info(f"Processing PDF: {pdf_filename}")
        
        try:
            # Extract images from PDF
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                # Convert PDF page to image
                image_filename = f"{os.path.splitext(pdf_filename)[0]}_page_{page_num + 1}.png"
                image_path = os.path.join(self.config.IMAGES_FOLDER, image_filename)
                
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(self.config.SCALE_FACTOR, self.config.SCALE_FACTOR))
                
                # Save image
                self._save_pixmap(pix, image_path)
                
                # Process the extracted image
                self.process_image(pdf_filename, page_num + 1, image_path)
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_filename}: {str(e)}")
    
    def _save_pixmap(self, pix, image_path):
        """Save pixmap to file synchronously"""
        img_data = pix.tobytes("png")
        with open(image_path, 'wb') as f:
            f.write(img_data)
    
    def process_image(self, pdf_name, page_number, image_path):
        """Process a single image with OCR engines"""
        logger.info(f"Processing image: {image_path}")
        
        try:
            # Preprocess image to improve OCR quality
            enhanced_image = self.preprocess_image(image_path)
            
            # Run OCR with both engines sequentially
            tesseract_text = self.run_tesseract_ocr(enhanced_image)
            easyocr_text = self.run_easyocr(enhanced_image)
            
            # Use LLM to combine and clean up OCR results
            combined_text = self.enhance_with_llm(tesseract_text, easyocr_text)
            
            # Extract structured information using LangChain structured output
            extracted_info = self.extract_with_llm(combined_text)
            
            # Build result row
            result_row = {
                'pdf_name': pdf_name,
                'page_number': page_number,
                'tesseract_text': tesseract_text,
                'easyocr_text': easyocr_text,
                'combined_text': combined_text
            }
            
            # Add extracted fields from our structured extraction
            for field in self.config.EXTRACT_FIELDS:
                result_row[field] = extracted_info.get(field, '')
            
            # Append to results dataframe
            self.results_df = pd.concat([self.results_df, pd.DataFrame([result_row])], ignore_index=True)
            
            logger.info(f"Completed OCR processing for {pdf_name} page {page_number}")
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
    
    def preprocess_image(self, image_path):
        """Preprocess image to improve OCR quality"""
        return self._preprocess_image_sync(image_path)
    
    def _preprocess_image_sync(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        denoised = cv2.fastNlMeansDenoising(thresh, None, self.config.DENOISE_STRENGTH, 7, 21)
        if self.config.SCALE_FACTOR > 1:
            height, width = denoised.shape
            new_height, new_width = int(height * self.config.SCALE_FACTOR), int(width * self.config.SCALE_FACTOR)
            enhanced = cv2.resize(denoised, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        else:
            enhanced = denoised
        enhanced_path = image_path.replace('.png', '_enhanced.png')
        cv2.imwrite(enhanced_path, enhanced)
        return enhanced
    
    def run_tesseract_ocr(self, image):
        return self._run_tesseract_sync(image)
    
    def _run_tesseract_sync(self, image):
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        return text.strip()
    
    def run_easyocr(self, image):
        result = self.reader.readtext(image)
        text = ' '.join([item[1] for item in result])
        return text.strip()
    
    def enhance_with_llm(self, tesseract_text, easyocr_text):
        """Use LLM to enhance and combine OCR results"""
        prompt = f"""
        You are given two OCR results for the same document. Your task is to combine them into a single, 
        accurate, and well-formatted text. Fix any obvious OCR errors, correct spelling mistakes, and 
        ensure proper formatting.
        
        Tesseract OCR result:
        {tesseract_text}
        
        EasyOCR result:
        {easyocr_text}
        
        Combined and corrected result:
        """
        response = requests.post(
            "https://api.your-llm-provider.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.LLM_API_KEY}"
            },
            json={
                "model": self.config.LLM_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are an OCR correction expert."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
        )
        if response.status_code == 200:
            result = response.json()
            combined_text = result["choices"][0]["message"]["content"].strip()
            return combined_text
        else:
            logger.error(f"Error calling LLM API: {response.status_code} - {response.text}")
            return f"{tesseract_text}\n\n{easyocr_text}"
    
    def extract_with_llm(self, text):
        """
        Use LangChain’s structured output to extract information from the combined OCR text.
        This method builds a prompt with format instructions from the Pydantic model and then
        uses ChatOpenAI to invoke the model. The output is parsed using PydanticOutputParser.
        """
        # Set up the parser with our Pydantic model
        parser = PydanticOutputParser(pydantic_object=ExtractionResult)
        # Build a prompt that includes format instructions derived from our model's schema
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
             "Extract the following invoice details from the text. Output your answer as JSON matching the schema below:\n"
             f"{ExtractionResult.schema_json(indent=2)}"),
            ("user", "Text:\n{text}\n\nExtracted Information:")
        ])
        prompt = prompt_template.format_prompt(text=text).to_string()
        
        # Use ChatOpenAI from LangChain to invoke the model
        llm = ChatOpenAI(
            model_name=self.config.LLM_MODEL_NAME, 
            temperature=0.1,
            openai_api_key=self.config.LLM_API_KEY
        )
        response = llm.invoke(prompt)
        try:
            extraction = parser.parse(response)
        except Exception as e:
            logger.error(f"Error parsing LLM structured output: {str(e)}")
            extraction = ExtractionResult()
        return extraction.dict()

def main():
    """Main function to run the OCR application"""
    config = Config()
    processor = PDFProcessor(config)
    processor.process_all_pdfs()

if __name__ == "__main__":
    print("Starting OCR application...")
    print("Processing step by step...")
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"Error: {str(e)}")
