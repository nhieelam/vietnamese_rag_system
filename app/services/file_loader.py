import pdfplumber
import pytesseract
from PIL import Image
from pathlib import Path
from typing import Dict, Any
from app.utils.logger import logger


def extract_text_from_file(uploaded_file) -> Dict[str, Any]:

    try:
        file_type = uploaded_file.type
        file_name = uploaded_file.name

        supported_types = ["application/pdf", "image/jpeg", "image/png"]
        if file_type not in supported_types:
            error_msg = f"Unsupported file type: {file_type}. Supported types: PDF, JPEG, PNG"
            logger.error(error_msg)
            return {
                'status_code': 400,
                'text': None,
                'message': error_msg,
                'metadata': {'file_name': file_name, 'file_type': file_type}
            }
        
        if file_type == "application/pdf":
            result = _process_pdf(uploaded_file)
            
        elif file_type in ["image/jpeg", "image/png"]:
            logger.info("Processing as image with OCR...")
            result = _process_image(uploaded_file)
        
        return result
        
    except AttributeError as e:
        error_msg = f"Invalid file object. Missing required attributes: {str(e)}"
        logger.error(error_msg)
        return {
            'status_code': 400,
            'text': None,
            'message': error_msg,
            'metadata': {}
        }
    except Exception as e:
        error_msg = f"Unexpected error processing file: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return {
            'status_code': 500,
            'text': None,
            'message': error_msg,
            'metadata': {}
        }


def get_file_info(uploaded_file) -> Dict[str, Any]:
    try:
        return {
            "name": uploaded_file.name if hasattr(uploaded_file, 'name') else "Unknown",
            "type": uploaded_file.type if hasattr(uploaded_file, 'type') else "Unknown",
            "size": uploaded_file.size if hasattr(uploaded_file, 'size') else 0
        }
    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        return {"name": "Unknown", "type": "Unknown", "size": 0}


def _process_pdf(file) -> Dict[str, Any]:
    text_content = ""
    empty_pages = []
    
    file_to_open = file
    if hasattr(file, 'path'):
        file_to_open = file.path
    
    try:
        with pdfplumber.open(file_to_open) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"PDF contains {total_pages} page(s)")
            
            if total_pages == 0:
                return {
                    'status_code': 400,
                    'text': None,
                    'message': "PDF file is empty (0 pages)",
                    'metadata': {'total_pages': 0}
                }
            
            # Extract text from each page
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    
                    if page_text and page_text.strip():
                        text_content += page_text + "\n"
                        logger.debug(f"Page {i+1}/{total_pages}: Extracted {len(page_text)} characters")
                    else:
                        empty_pages.append(i+1)
                        logger.debug(f"Page {i+1}/{total_pages}: No text found (empty or scanned)")
                        
                except Exception as page_error:
                    logger.warning(f"Error on page {i+1}: {str(page_error)}")
                    empty_pages.append(i+1)

        # Validate extracted content
        if not text_content.strip():
            if len(empty_pages) == total_pages:
                return {
                    'status_code': 422,
                    'text': None,
                    'message': f"No text extracted from PDF ({total_pages} page(s)). This appears to be a scanned document. Please use image format with OCR.",
                    'metadata': {
                        'total_pages': total_pages,
                        'empty_pages': empty_pages
                    }
                }
            else:
                return {
                    'status_code': 500,
                    'text': None,
                    'message': "PDF processing failed. No readable text found.",
                    'metadata': {'total_pages': total_pages}
                }
        
        # Log warnings about empty pages
        if empty_pages:
            logger.warning(f"Empty/scanned pages detected: {empty_pages}")
        
        extracted = text_content.strip()
        logger.info(f"PDF extraction complete: {len(extracted)} characters from {total_pages - len(empty_pages)}/{total_pages} pages")
        
        return {
            'status_code': 200,
            'text': extracted,
            'message': f"Successfully extracted text from {total_pages - len(empty_pages)}/{total_pages} pages",
            'metadata': {
                'total_pages': total_pages,
                'extracted_pages': total_pages - len(empty_pages),
                'empty_pages': empty_pages,
                'character_count': len(extracted)
            }
        }
        
    except FileNotFoundError:
        return {
            'status_code': 404,
            'text': None,
            'message': "PDF file not found. Please check the file path.",
            'metadata': {}
        }
    except Exception as e:
        error_msg = f"PDF extraction failed: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return {
            'status_code': 500,
            'text': None,
            'message': error_msg,
            'metadata': {}
        }


def _process_image(file) -> Dict[str, Any]:

    logger.info("Starting OCR process (this may take a moment)...")
    
    file_to_open = file
    if hasattr(file, 'path'):
        file_to_open = file.path
    
    try:
        image = Image.open(file_to_open)
        width, height = image.size
        logger.info(f"Image dimensions: {width}x{height} pixels")
        
        if width < 100 or height < 100:
            return {
                'status_code': 206,
                'text': None,
                'message': f"Image resolution is very low ({width}x{height}px). Text extraction may be inaccurate.",
                'metadata': {
                    'width': width,
                    'height': height,
                    'warning': 'low_resolution'
                }
            }
        
        text = None
        lang_used = None
        
        try:
            text = pytesseract.image_to_string(image, lang='vie', config='--psm 6')
            lang_used = "Vietnamese"
            logger.info(f"OCR completed with Vietnamese language pack")
            
        except Exception as vie_error:
            # Fallback to English if Vietnamese fails
            logger.warning(f"Vietnamese OCR failed: {str(vie_error)[:100]}")
            logger.info("Attempting OCR with English language pack...")
            
            try:
                text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
                lang_used = "English"
                logger.info(f"OCR completed with English language pack")
            except Exception as eng_error:
                logger.error(f"English OCR also failed: {str(eng_error)[:100]}")
                raise eng_error
        
        # Validate extracted text
        if not text or not text.strip():
            return {
                'status_code': 206,
                'text': None,
                'message': f"No text detected in image ({width}x{height}px). Possible causes: empty image, low quality, or image contains only graphics/photos.",
                'metadata': {
                    'width': width,
                    'height': height,
                    'language': lang_used,
                    'warning': 'no_text_detected'
                }
            }
        
        extracted = text.strip()
        
        word_count = len(extracted.split())
        logger.info(
            f"OCR extraction complete: {len(extracted)} characters, "
            f"~{word_count} words (Language: {lang_used})"
        )
        
        return {
            'status_code': 200,
            'text': extracted,
            'message': f"Successfully extracted text using {lang_used} OCR",
            'metadata': {
                'width': width,
                'height': height,
                'language': lang_used,
                'character_count': len(extracted),
                'word_count': word_count
            }
        }
        
    except pytesseract.TesseractNotFoundError:
        return {
            'status_code': 500,
            'text': None,
            'message': "Tesseract OCR engine not installed. Please install from: https://github.com/UB-Mannheim/tesseract/wiki",
            'metadata': {'error_type': 'tesseract_not_found'}
        }
    except FileNotFoundError:
        return {
            'status_code': 404,
            'text': None,
            'message': "Image file not found. Please check the file path.",
            'metadata': {}
        }
    except Exception as e:
        error_msg = f"Image OCR failed: {str(e)}. Ensure Tesseract is installed with language pack (vie.traineddata for Vietnamese)."
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return {
            'status_code': 500,
            'text': None,
            'message': error_msg,
            'metadata': {}
        }
