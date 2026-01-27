import pdfplumber
import pytesseract
from PIL import Image
from pathlib import Path
from app.utils.logger import logger


def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.type
    file_name = uploaded_file.name
    
    logger.info(f"Processing file: {file_name} ({file_type})")

    
    try:
        if file_type == "application/pdf":
            return _process_pdf(uploaded_file)
            
        elif file_type in ["image/jpeg", "image/png"]:
            return _process_image(uploaded_file)
            
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        logger.error(f"Error processing {file_name}: {e}")
        return "" 

def _process_pdf(file):

    text_content = ""
    
    file_to_open = file
    if hasattr(file, 'path'):
        file_to_open = file.path
    
    with pdfplumber.open(file_to_open) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"📄 Found {total_pages} pages in PDF.")
        
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            
            if page_text:
                text_content += page_text + "\n"
            else:
                logger.warning(f"⚠️ Page {i+1} appears to be empty or scanned.")

    if not text_content.strip():
        return "[ERROR] This PDF seems to be a scanned image. Please convert it to images or use OCR."
        
    return text_content

def _process_image(file):
    logger.info("Starting OCR process (this may take a moment)...")
    
    try:
        file_to_open = file
        if hasattr(file, 'path'):
            file_to_open = file.path
        
        image = Image.open(file_to_open)
        
        try:
            text = pytesseract.image_to_string(image, lang='vie', config='--psm 6')
            logger.info(f"OCR completed with Vietnamese language pack. Extracted {len(text)} characters.")
        except Exception as lang_error:
            logger.warning(f"Vietnamese language pack not found, trying English: {lang_error}")
            text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
            logger.info(f"OCR completed with English language pack. Extracted {len(text)} characters.")
        
        if not text.strip():
            return "[WARNING] No text detected in the image. The image might be empty or too low quality."
        
        return text
        
    except pytesseract.TesseractNotFoundError:
        error_msg = f"Tesseract not found at: {pytesseract.pytesseract.tesseract_cmd}"
        logger.error(error_msg)
        return f"[ERROR] {error_msg}\nPlease install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki"
    except Exception as e:
        logger.error(f"OCR Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"[ERROR] OCR failed: {str(e)}\nMake sure Tesseract is installed with Vietnamese language pack (vie.traineddata)"