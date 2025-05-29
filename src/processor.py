import os
import time
import logging
import cv2
import pytesseract
import easyocr
import fitz  # PyMuPDF
import pandas as pd
import requests
import base64
from typing import Dict, Any, Tuple, List
from PIL import Image
import io
from mimetypes import guess_type
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from .config import Config
from .models import ExtractionResult

def local_image_to_data_url(image_path: str) -> str:
    """Convert a local image to a data URL"""
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    return f"data:{mime_type};base64,{base64_encoded_data}"

class PDFProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.reader = easyocr.Reader(['en'])
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
        
        # Initialize LangChain chat model if LLM is enabled
        if config.USE_LLM:
            self.llm = ChatOpenAI(
                model_name=config.LLM_MODEL_NAME,
                openai_api_key=config.LLM_API_KEY,
                temperature=0.3
            )
        
        # Create output directories
        os.makedirs(config.INPUT_FOLDER, exist_ok=True)
        os.makedirs(config.IMAGES_FOLDER, exist_ok=True)
        os.makedirs(config.RESULTS_FOLDER, exist_ok=True)
        os.makedirs(config.MARKDOWN_FOLDER, exist_ok=True)
        
        # Initialize results dataframe
        self.results_df = pd.DataFrame(columns=[
            'pdf_name', 
            'page_number', 
            'tesseract_text', 
            'tesseract_confidence',
            'easyocr_text', 
            'easyocr_confidence',
            'combined_text',
            'llm_markdown',
            *config.EXTRACT_FIELDS
        ])
    
    def process_all_pdfs(self):
        """Process all PDFs in the input folder"""
        start_time = time.time()
        logging.info(f"Starting to process PDFs in {self.config.INPUT_FOLDER}")
        
        pdf_files = [f for f in os.listdir(self.config.INPUT_FOLDER) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logging.warning(f"No PDF files found in {self.config.INPUT_FOLDER}")
            return
        
        for pdf_file in pdf_files:
            self.process_pdf(pdf_file)
        
        # Save final results to Excel
        excel_path = os.path.join(self.config.RESULTS_FOLDER, 'ocr_results.xlsx')
        self.results_df.to_excel(excel_path, index=False)
        
        logging.info(f"Completed processing {len(pdf_files)} PDFs in {time.time() - start_time:.2f} seconds")
        logging.info(f"Results saved to {excel_path}")
    
    def process_pdf(self, pdf_filename: str):
        """Process a single PDF file"""
        pdf_path = os.path.join(self.config.INPUT_FOLDER, pdf_filename)
        logging.info(f"Processing PDF: {pdf_filename}")
        
        try:
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                image_filename = f"{os.path.splitext(pdf_filename)[0]}_page_{page_num + 1}.png"
                image_path = os.path.join(self.config.IMAGES_FOLDER, image_filename)
                
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(self.config.SCALE_FACTOR, self.config.SCALE_FACTOR))
                
                self._save_pixmap(pix, image_path)
                self.process_image(pdf_filename, page_num + 1, image_path)
            
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_filename}: {str(e)}")
    
    def _save_pixmap(self, pix, image_path: str):
        """Save pixmap to file synchronously"""
        img_data = pix.tobytes("png")
        with open(image_path, 'wb') as f:
            f.write(img_data)
    
    def process_image(self, pdf_name: str, page_number: int, image_path: str):
        """Process a single image with OCR engines and LLM"""
        logging.info(f"Processing image: {image_path}")
        
        try:
            # Preprocess image
            enhanced_image = self.preprocess_image(image_path)
            
            # Run traditional OCR
            tesseract_data, tesseract_text, tesseract_confidence = self.run_tesseract_ocr(enhanced_image)
            easyocr_data, easyocr_text, easyocr_confidence = self.run_easyocr(enhanced_image)
            
            # Initialize LLM markdown as empty string
            llm_markdown = ""
            
            # Run LLM-based image processing if enabled
            if self.config.USE_LLM:
                llm_markdown = self.process_image_with_llm(image_path)
                
                # Save markdown to file
                markdown_path = os.path.join(
                    self.config.MARKDOWN_FOLDER,
                    f"{os.path.splitext(pdf_name)[0]}_page_{page_number}.md"
                )
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(llm_markdown)
            
            # Combine OCR results with LLM markdown
            combined_text = self.enhance_with_llm(
                tesseract_text, easyocr_text, llm_markdown,
                tesseract_data=tesseract_data, easyocr_data=easyocr_data
            )
            extracted_info = self.extract_with_llm(combined_text)
            
            # Build result row
            result_row = {
                'pdf_name': pdf_name,
                'page_number': page_number,
                'tesseract_text': tesseract_text,
                'tesseract_confidence': tesseract_confidence,
                'easyocr_text': easyocr_text,
                'easyocr_confidence': easyocr_confidence,
                'combined_text': combined_text,
                'llm_markdown': llm_markdown
            }
            
            # Add extracted fields
            for field in self.config.EXTRACT_FIELDS:
                result_row[field] = extracted_info.get(field, '')
            
            # Append to results dataframe
            self.results_df = pd.concat([self.results_df, pd.DataFrame([result_row])], ignore_index=True)
            
            logging.info(f"Completed OCR processing for {pdf_name} page {page_number}")
            
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
    
    def preprocess_image(self, image_path: str) -> Any:
        """Preprocess image to improve OCR quality"""
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
    
    def run_tesseract_ocr(self, image: Any) -> Tuple[List[Dict[str, Any]], str, float]:
        """Run Tesseract OCR on the image and return structured data, plain text, and average confidence"""
        custom_config = r'--oem 3 --psm 6'
        structured_results = []
        plain_text = ""
        avg_confidence = 0.0

        try:
            data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DATAFRAME)
            # Filter out non-meaningful confidence values and empty text
            data = data[data.conf != -1]
            data.dropna(subset=['text'], inplace=True)
            data = data[data['text'].astype(str).str.strip() != '']

            if not data.empty:
                for _, row in data.iterrows():
                    structured_results.append({
                        'text': str(row['text']),
                        'left': int(row['left']),
                        'top': int(row['top']),
                        'width': int(row['width']),
                        'height': int(row['height']),
                        'conf': float(row['conf'] / 100.0)  # Normalize confidence to 0-1
                    })
                
                plain_text = " ".join(data['text'].astype(str).tolist())
                
                # Calculate average confidence from valid word confidences
                if not data['conf'].empty:
                    conf_mean = data['conf'].mean()
                    avg_confidence = 0.0 if pd.isna(conf_mean) else float(conf_mean / 100.0)
            
            return structured_results, plain_text.strip(), avg_confidence
        except Exception as e:
            logging.error(f"Error during Tesseract OCR: {str(e)}")
            return [], "", 0.0
    
    def run_easyocr(self, image: Any) -> Tuple[List[Dict[str, Any]], str, float]:
        """Run EasyOCR on the image and return structured_data, plain text, and average confidence"""
        structured_results = []
        all_texts = []
        all_confidences = []

        try:
            result = self.reader.readtext(image) # list of (bbox, text, confidence)
            if not result:
                return [], "", 0.0
            
            for (bbox, text, conf) in result:
                structured_results.append({
                    'text': text,
                    'bbox': bbox, 
                    'conf': float(conf) 
                })
                all_texts.append(text)
                all_confidences.append(float(conf))
            
            plain_text = ' '.join(all_texts)
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            
            return structured_results, plain_text.strip(), avg_confidence
        except Exception as e:
            logging.error(f"Error during EasyOCR: {str(e)}")
            return [], "", 0.0
    
    def process_image_with_llm(self, image_path: str) -> str:
        """Process image using GPT-4 Vision and return markdown formatted text"""
        # Convert image to data URL
        data_url = local_image_to_data_url(image_path)
        
        # Prepare the prompt
        prompt = """
        Analyze this image and extract all text content. Format the output in markdown with the following structure:
        
        1. Main heading with a descriptive title
        2. Sections for different types of content (e.g., header, body, footer)
        3. Tables if present
        4. Lists if present
        5. Important information highlighted
        
        Ensure proper formatting, including:
        - Headers with appropriate levels
        - Tables with proper alignment
        - Lists with proper indentation
        - Emphasis on important information
        - Code blocks for structured data
        """
        
        # Call GPT-4 Vision using LangChain
        messages = [
            SystemMessage(content="You are an expert at analyzing images and extracting text content in a structured format."),
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ])
        ]
        
        try:
            response = self.llm(messages)
            return response.content.strip()
        except Exception as e:
            logging.error(f"Error processing image with LLM: {str(e)}")
            return "Error processing image with LLM"
    
    def enhance_with_llm(self, tesseract_text: str, easyocr_text: str, llm_markdown: str,
                         tesseract_data: List[Dict[str, Any]] = None,
                         easyocr_data: List[Dict[str, Any]] = None) -> str:
        """Use LLM to enhance and combine OCR results with LLM markdown"""
        
        if not self.config.USE_LLM:
            # If LLM is not used, tesseract_data and easyocr_data are not used here.
            # We still rely on the plain text versions for the basic combination.
            return f"{tesseract_text}\n\n{easyocr_text}"

        # TODO: Update the prompt to utilize tesseract_data and easyocr_data for layout preservation
        # For now, the prompt uses the plain text versions as before.
        # The structured data is available for future enhancement of this prompt.
        prompt = f"""
        You are given three sources of text for the same document:
        1. Tesseract OCR result (plain text)
        2. EasyOCR result (plain text)
        3. LLM-generated markdown (from image analysis)
        
        Your task is to combine them into a single, accurate, and well-formatted text. 
        Use the LLM markdown as the primary structure and incorporate the OCR results to fill in any gaps or correct any errors.
        Consider the possibility that OCR results might contain more detailed or accurate text for certain parts.
        
        Tesseract OCR result:
        {tesseract_text}
        
        EasyOCR result:
        {easyocr_text}
        
        LLM Markdown:
        {llm_markdown}
        
        Combined and corrected result (aim for well-structured Markdown):
        """
        
        try:
            response = self.llm([
                SystemMessage(content="You are an expert at combining and correcting OCR results."),
                HumanMessage(content=prompt)
            ])
            return response.content.strip()
        except Exception as e:
            logging.error(f"Error enhancing text with LLM: {str(e)}")
            return f"{tesseract_text}\n\n{easyocr_text}\n\n{llm_markdown}"
    
    def extract_with_llm(self, text: str) -> Dict[str, str]:
        """Extract structured information using LLM"""
        if not self.config.USE_LLM:
            return {field: '' for field in self.config.EXTRACT_FIELDS}
        
        prompt = f"""
        Extract the following information from the text below:
        - Invoice number
        - Date
        - Total amount
        - Vendor name
        - Customer name
        
        Text:
        {text}
        
        Return the information in a structured format.
        """
        
        try:
            response = self.llm([
                SystemMessage(content="You are an expert at extracting structured information from text."),
                HumanMessage(content=prompt)
            ])
            result = response.content
            return ExtractionResult.parse_raw(result).dict()
        except Exception as e:
            logging.error(f"Error extracting information with LLM: {str(e)}")
            return {field: '' for field in self.config.EXTRACT_FIELDS} 