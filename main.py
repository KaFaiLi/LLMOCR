import os
import asyncio
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
import concurrent.futures
import requests
from tqdm import tqdm
import aiofiles
import aiohttp

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
    LLM_API_ENDPOINT = "https://api.your-llm-provider.com/v1/chat/completions"  # Replace with actual endpoint
    LLM_API_KEY = "your-api-key"  # Replace with actual API key
    
    # Extraction fields (customize based on your needs)
    EXTRACT_FIELDS = [
        "invoice_number", 
        "date", 
        "total_amount", 
        "vendor_name",
        "customer_name"
    ]


class PDFProcessor:
    def __init__(self, config):
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
    
    async def process_all_pdfs(self):
        """Process all PDFs in the input folder"""
        start_time = time.time()
        logger.info(f"Starting to process PDFs in {self.config.INPUT_FOLDER}")
        
        pdf_files = [f for f in os.listdir(self.config.INPUT_FOLDER) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.config.INPUT_FOLDER}")
            return
        
        # Process PDFs concurrently
        tasks = [self.process_pdf(pdf_file) for pdf_file in pdf_files]
        await asyncio.gather(*tasks)
        
        # Save final results to Excel
        excel_path = os.path.join(self.config.RESULTS_FOLDER, 'ocr_results.xlsx')
        self.results_df.to_excel(excel_path, index=False)
        
        logger.info(f"Completed processing {len(pdf_files)} PDFs in {time.time() - start_time:.2f} seconds")
        logger.info(f"Results saved to {excel_path}")
    
    async def process_pdf(self, pdf_filename):
        """Process a single PDF file"""
        pdf_path = os.path.join(self.config.INPUT_FOLDER, pdf_filename)
        logger.info(f"Processing PDF: {pdf_filename}")
        
        try:
            # Extract images from PDF
            pdf_document = fitz.open(pdf_path)
            tasks = []
            
            for page_num in range(len(pdf_document)):
                # Convert PDF page to image
                image_filename = f"{os.path.splitext(pdf_filename)[0]}_page_{page_num + 1}.png"
                image_path = os.path.join(self.config.IMAGES_FOLDER, image_filename)
                
                # Extract page as image
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(self.config.SCALE_FACTOR, self.config.SCALE_FACTOR))
                
                # Save image asynchronously
                await self._save_pixmap_async(pix, image_path)
                
                # Add task to process the extracted image
                tasks.append(self.process_image(pdf_filename, page_num + 1, image_path))
            
            # Process all pages concurrently
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_filename}: {str(e)}")
    
    async def _save_pixmap_async(self, pix, image_path):
        """Save pixmap to file asynchronously"""
        # Convert pixmap to PIL Image
        img_data = pix.tobytes("png")
        
        # Write image asynchronously
        async with aiofiles.open(image_path, 'wb') as f:
            await f.write(img_data)
    
    async def process_image(self, pdf_name, page_number, image_path):
        """Process a single image with OCR engines"""
        logger.info(f"Processing image: {image_path}")
        
        try:
            # Preprocess image to improve OCR quality
            enhanced_image = await self.preprocess_image(image_path)
            
            # Run OCR with both engines concurrently
            tesseract_text, easyocr_text = await asyncio.gather(
                self.run_tesseract_ocr(enhanced_image),
                self.run_easyocr(enhanced_image)
            )
            
            # Use LLM to combine and clean up OCR results
            combined_text = await self.enhance_with_llm(tesseract_text, easyocr_text)
            
            # Extract structured information using LLM
            extracted_info = await self.extract_with_llm(combined_text)
            
            # Add to results dataframe
            result_row = {
                'pdf_name': pdf_name,
                'page_number': page_number,
                'tesseract_text': tesseract_text,
                'easyocr_text': easyocr_text,
                'combined_text': combined_text
            }
            
            # Add extracted fields
            for field in self.config.EXTRACT_FIELDS:
                result_row[field] = extracted_info.get(field, '')
            
            # Append to results dataframe
            self.results_df = pd.concat([self.results_df, pd.DataFrame([result_row])], ignore_index=True)
            
            logger.info(f"Completed OCR processing for {pdf_name} page {page_number}")
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
    
    async def preprocess_image(self, image_path):
        """Preprocess image to improve OCR quality"""
        # This function runs CPU-intensive tasks, so we'll use an executor
        loop = asyncio.get_event_loop()
        enhanced_image = await loop.run_in_executor(None, self._preprocess_image_sync, image_path)
        return enhanced_image
    
    def _preprocess_image_sync(self, image_path):
        """Synchronous image preprocessing function (will be run in executor)"""
        # Load image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise (reduce noise)
        denoised = cv2.fastNlMeansDenoising(thresh, None, self.config.DENOISE_STRENGTH, 7, 21)
        
        # Upscale if needed
        if self.config.SCALE_FACTOR > 1:
            height, width = denoised.shape
            new_height, new_width = int(height * self.config.SCALE_FACTOR), int(width * self.config.SCALE_FACTOR)
            enhanced = cv2.resize(denoised, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        else:
            enhanced = denoised
        
        # Save enhanced image (optional)
        enhanced_path = image_path.replace('.png', '_enhanced.png')
        cv2.imwrite(enhanced_path, enhanced)
        
        return enhanced
    
    async def run_tesseract_ocr(self, image):
        """Run Tesseract OCR on an image"""
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, self._run_tesseract_sync, image)
        return text
    
    def _run_tesseract_sync(self, image):
        """Synchronous Tesseract OCR function (will be run in executor)"""
        # Run Tesseract OCR
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        return text.strip()
    
    async def run_easyocr(self, image):
        """Run EasyOCR on an image"""
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, self._run_easyocr_sync, image)
        return text
    
    def _run_easyocr_sync(self, image):
        """Synchronous EasyOCR function (will be run in executor)"""
        # Run EasyOCR
        result = self.reader.readtext(image)
        # Combine all text results
        text = ' '.join([item[1] for item in result])
        return text.strip()
    
    async def enhance_with_llm(self, tesseract_text, easyocr_text):
        """Use LLM to enhance and combine OCR results"""
        # Note: LLM calls are not using async as per your requirement
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
        
        # Call LLM API (synchronously as requested)
        response = requests.post(
            self.config.LLM_API_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.LLM_API_KEY}"
            },
            json={
                "model": "gpt-4",  # or your preferred model
                "messages": [
                    {"role": "system", "content": "You are an OCR correction expert."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
        )
        
        # Parse response
        if response.status_code == 200:
            result = response.json()
            combined_text = result["choices"][0]["message"]["content"].strip()
            return combined_text
        else:
            logger.error(f"Error calling LLM API: {response.status_code} - {response.text}")
            # If LLM fails, return a simple combination of both texts
            return f"{tesseract_text}\n\n{easyocr_text}"
    
    async def extract_with_llm(self, text):
        """Use LLM to extract structured information"""
        # Note: LLM calls are not using async as per your requirement
        fields_str = ", ".join(self.config.EXTRACT_FIELDS)
        
        prompt = f"""
        Extract the following information from the text below as JSON:
        Fields to extract: {fields_str}
        
        Text:
        {text}
        
        Return ONLY a valid JSON object with the extracted fields. If a field is not found, set its value to null.
        """
        
        # Call LLM API (synchronously as requested)
        response = requests.post(
            self.config.LLM_API_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.LLM_API_KEY}"
            },
            json={
                "model": "gpt-4",  # or your preferred model
                "messages": [
                    {"role": "system", "content": "You are a data extraction expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1
            }
        )
        
        # Parse response
        if response.status_code == 200:
            result = response.json()
            extracted_info = result["choices"][0]["message"]["content"]
            
            # Handle potential string JSON response
            if isinstance(extracted_info, str):
                try:
                    import json
                    extracted_info = json.loads(extracted_info)
                except:
                    logger.error("Failed to parse LLM JSON response")
                    extracted_info = {field: None for field in self.config.EXTRACT_FIELDS}
            
            return extracted_info
        else:
            logger.error(f"Error calling LLM API: {response.status_code} - {response.text}")
            # Return empty values if LLM fails
            return {field: None for field in self.config.EXTRACT_FIELDS}


async def main():
    """Main function to run the OCR application"""
    config = Config()
    processor = PDFProcessor(config)
    await processor.process_all_pdfs()


# Run the application
if __name__ == "__main__":
    print("Starting OCR application...")
    print("Press Ctrl+C to stop")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"Error: {str(e)}")
