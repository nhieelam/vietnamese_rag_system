import pytesseract
from app.utils.logger import logger


class TesseractOCRWrapper:
    def __init__(self, lang='vie', config='--psm 6'):
        self.lang = lang
        self.config = config
    
    def read_image(self, image):
        """
        Read text from image using Tesseract OCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Extracted text from image
            
        Raises:
            pytesseract.TesseractNotFoundError: If Tesseract is not installed
        """
        try:
            return pytesseract.image_to_string(
                image, 
                lang=self.lang, 
                config=self.config
            )
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract OCR not installed")
            raise
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            raise
