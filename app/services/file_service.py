import pdfplumber
from PIL import Image
from typing import Dict, Any
import tempfile
import os
from app.utils.logger import logger
from app.orc import TesseractOCRWrapper
from app.orc.pdf_convert import pdf_to_images
import docx


class FileService:
    SUPPORTED_TYPES = {
        "application/pdf": "pdf",
        "image/jpeg": "image",
        "image/png": "image",
        "image/jpg": "image",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    }

    @classmethod
    def extract(cls, uploaded_file) -> Dict[str, Any]:
        try:
            file_type = uploaded_file.type
            file_name = uploaded_file.name

            if file_type not in cls.SUPPORTED_TYPES:
                return cls._error(
                    400,
                    f"Unsupported file type: {file_type}. Supported types: PDF, JPEG, PNG, JPG, DOCX",
                    file_name=file_name,
                    file_type=file_type
                )

            if cls.SUPPORTED_TYPES[file_type] == "pdf":
                return cls._process_pdf(uploaded_file)
            
            if cls.SUPPORTED_TYPES[file_type] == "docx":
                return cls._process_doc(uploaded_file)

            logger.info("Processing as image with OCR...")
            return cls._process_image(uploaded_file)

        except AttributeError as e:
            logger.error(str(e))
            return cls._error(
                400,
                f"Invalid file object. Missing required attributes: {e}"
            )

        except Exception as e:
            logger.exception("Unexpected error processing file")
            return cls._error(500, f"Unexpected error processing file: {e}")

    @classmethod
    def get_file_info(cls, uploaded_file) -> Dict[str, Any]:
        try:
            return {
                "name": getattr(uploaded_file, "name", "Unknown"),
                "type": getattr(uploaded_file, "type", "Unknown"),
                "size": getattr(uploaded_file, "size", 0),
            }
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {"name": "Unknown", "type": "Unknown", "size": 0}

    @classmethod
    def _process_doc(cls, file) -> Dict[str, Any]:
        try:
            document = docx.Document(file)
            text_content = "\n".join([para.text for para in document.paragraphs])
            
            if not text_content.strip():
                logger.info("DOCX file is empty or contains no text.")
                return cls._error(422, "DOCX file is empty or contains no text.")

            extracted = text_content.strip()
            logger.info(f"DOCX extraction complete: {len(extracted)} chars")
            
            return cls._success(
                extracted,
                "Successfully extracted text from DOCX file.",
                character_count=len(extracted)
            )
        except Exception as e:
            logger.exception("DOCX extraction failed")
            return cls._error(500, f"DOCX extraction failed: {e}")

    @classmethod
    def _process_pdf(cls, file) -> Dict[str, Any]:
        text_content = ""
        empty_pages = []
        ocr_pages = []
        temp_file_path = None

        try:
            # Handle both file path and file-like objects (e.g., Streamlit UploadedFile)
            if hasattr(file, 'path') and os.path.isfile(file.path):
                file_to_open = file.path
            elif hasattr(file, 'read'):
                # File-like object: save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    tmp.write(file.read())
                    temp_file_path = tmp.name
                file_to_open = temp_file_path
                logger.debug(f"Created temporary PDF file: {temp_file_path}")
            else:
                file_to_open = str(file)

            with pdfplumber.open(file_to_open) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF contains {total_pages} page(s)")
                if total_pages == 0:
                    logger.info("PDF file is empty (0 pages)")
                    return cls._error(
                        400,
                        "PDF file is empty (0 pages)",
                        total_pages=0
                    )
                
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_content += page_text + "\n"
                        else:
                            empty_pages.append(i + 1)
                    except Exception:
                        logger.exception("Error extracting text from PDF page")
                        empty_pages.append(i + 1)

                # If no text extracted, try OCR on scanned pages
                if not text_content.strip() and len(empty_pages) > 0:
                    logger.info(f"No text extracted from PDF. Attempting OCR on {len(empty_pages)} page(s)")
                    text_content = cls._process_scanned_pdf(file_to_open, empty_pages)
                    if text_content.strip():
                        ocr_pages = empty_pages
                        empty_pages = []
                    else:
                        logger.warning("OCR also failed to extract text from PDF")

            if not text_content.strip():
                logger.info("No text extracted from PDF. This may be a scanned document.")
                return cls._error(
                    422,
                    "No text extracted from PDF. This may be a scanned document requiring OCR.",
                    total_pages=total_pages,
                    empty_pages=empty_pages
                )
            extracted = text_content.strip()
            extracted_pages = total_pages - len(empty_pages)
            
            logger.info(
                f"PDF extraction complete: {len(extracted)} chars "
                f"from {extracted_pages}/{total_pages} pages "
                f"({len(ocr_pages)} via OCR)"
            )
            return cls._success(
                extracted,
                f"Extracted text from {extracted_pages}/{total_pages} pages"
                + (f" ({len(ocr_pages)} via OCR)" if ocr_pages else ""),
                total_pages=total_pages,
                extracted_pages=extracted_pages,
                empty_pages=empty_pages,
                ocr_pages=ocr_pages,
                character_count=len(extracted)
            )
        except FileNotFoundError:
            return cls._error(404, "PDF file not found")
        except Exception as e:
            logger.exception("PDF extraction failed")
            return cls._error(500, f"PDF extraction failed: {e}")
        finally:
            # Clean up temporary file if created
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")

    @classmethod
    def _process_scanned_pdf(cls, pdf_path, page_numbers=None) -> str:
        """
        Process scanned PDF by converting pages to images and running OCR on them.
        
        Args:
            pdf_path: Path to PDF file
            page_numbers: List of specific page numbers to process (1-indexed).
                         If None, processes all pages.
        
        Returns:
            Extracted text from all processed pages
        """
        text_content = ""
        
        try:
            logger.info(f"Converting PDF to images for OCR...")
            images = pdf_to_images(pdf_path, dpi=200)
            
            if not images:
                logger.warning("Failed to convert PDF to images")
                return ""
            
            # Determine which pages to process
            if page_numbers:
                # page_numbers are 1-indexed
                images_to_process = [
                    (i, img) for i, img in enumerate(images, start=1)
                    if i in page_numbers
                ]
            else:
                images_to_process = list(enumerate(images, start=1))
            
            logger.info(f"Processing {len(images_to_process)} page(s) with OCR")
            
            for page_num, image in images_to_process:
                try:
                    logger.debug(f"Running OCR on page {page_num}")
                    ocr_vi = TesseractOCRWrapper(lang='vie')
                    page_text = ocr_vi.read_image(image)
                    
                    if page_text and page_text.strip():
                        text_content += page_text + "\n"
                        logger.debug(f"Page {page_num}: {len(page_text)} chars extracted")
                    else:
                        # Fallback to English
                        logger.debug(f"No Vietnamese text on page {page_num}, trying English")
                        ocr_en = TesseractOCRWrapper(lang='eng')
                        page_text = ocr_en.read_image(image)
                        if page_text and page_text.strip():
                            text_content += page_text + "\n"
                            logger.debug(f"Page {page_num}: {len(page_text)} chars extracted (English)")
                        
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num}: {e}")
                    continue
            
            logger.info(f"PDF OCR complete: {len(text_content)} chars extracted")
            return text_content
            
        except Exception as e:
            logger.exception(f"Failed to process scanned PDF: {e}")
            return ""


    @classmethod
    def _process_image(cls, file) -> Dict[str, Any]:
        file_to_open = getattr(file, "path", file)
        
        try:
            image = Image.open(file_to_open)
            width, height = image.size
            logger.info(f"Processing image with size: {width}x{height}")

            if width < 100 or height < 100:
                return cls._error(
                    206,
                    "Image resolution too low for OCR",
                    width=width,
                    height=height
                )

            text, lang_used = cls._run_ocr(image)

            if not text.strip():
                logger.info("No text detected in image")
                return cls._error(
                    206,
                    "No text detected in image",
                    width=width,
                    height=height,
                    language=lang_used
                )

            extracted = text.strip()
            word_count = len(extracted.split())
            logger.info(f"OCR successful ({lang_used}) , extracted text: {extracted}")

            return cls._success(
                extracted,
                f"OCR successful ({lang_used})",
                width=width,
                height=height,
                language=lang_used,
                character_count=len(extracted),
                word_count=word_count
            )

        except ImportError:
            return cls._error(
                500,
                "Tesseract OCR not installed",
                error_type="tesseract_not_found"
            )
        except FileNotFoundError:
            return cls._error(404, "Image file not found")
        except Exception as e:
            logger.exception("Image OCR failed")
            return cls._error(500, f"Image OCR failed: {e}")

    @classmethod
    def _run_ocr(cls, image):
        """
        Run OCR on image using Tesseract, with Vietnamese as primary language.
        Falls back to English if Vietnamese extraction fails.
        """
        try:
            ocr_vi = TesseractOCRWrapper(lang='vie')
            text = ocr_vi.read_image(image)
            if text.strip():
                return (text, "Vietnamese")
            else:
                logger.warning("No text extracted with Vietnamese, trying English")
                ocr_en = TesseractOCRWrapper(lang='eng')
                return (ocr_en.read_image(image), "English")
        except Exception:
            logger.warning("Vietnamese OCR failed, fallback to English")
            try:
                ocr_en = TesseractOCRWrapper(lang='eng')
                return (ocr_en.read_image(image), "English")
            except Exception as e:
                logger.exception("OCR failed for both Vietnamese and English")
                raise

    @staticmethod
    def _success(text: str, message: str, **metadata):
        return {
            "status_code": 200,
            "text": text,
            "message": message,
            "metadata": metadata,
        }

    @staticmethod
    def _error(status: int, message: str, **metadata):
        logger.error(message)
        return {
            "status_code": status,
            "text": None,
            "message": message,
            "metadata": metadata,
        }
