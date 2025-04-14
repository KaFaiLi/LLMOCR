import logging
from .config import Config
from .processor import PDFProcessor

def main():
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

    try:
        # Initialize configuration
        config = Config()
        
        # Initialize and run the PDF processor
        processor = PDFProcessor(config)
        processor.process_all_pdfs()
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 