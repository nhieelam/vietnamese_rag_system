import pdfplumber
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
import tempfile
import os
from app.utils.logger import logger
from app.orc import TesseractOCRWrapper
from app.utils.pdf_convert import pdf_to_images
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
            try:
                if hasattr(file, "seek"):
                    file.seek(0)
            except Exception:
                pass
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
        empty_pages = []
        ocr_pages = []
        temp_file_path = None
        pdf_bytes: Optional[bytes] = None

        try:
            if hasattr(file, 'path') and os.path.isfile(file.path):
                file_to_open = file.path
                try:
                    with open(file_to_open, "rb") as fh:
                        pdf_bytes = fh.read()
                except Exception:
                    pdf_bytes = None
            elif hasattr(file, 'read'):
                # Streamlit UploadedFile: dùng getvalue() để lấy bytes đầy đủ
                # mà không phụ thuộc vị trí con trỏ, seek(0) trước đọc để an toàn.
                raw: Optional[bytes] = None
                if hasattr(file, "getvalue"):
                    try:
                        raw = file.getvalue()
                    except Exception:
                        raw = None
                if not raw:
                    try:
                        file.seek(0)
                    except Exception:
                        pass
                    raw = file.read() or b""
                try:
                    file.seek(0)
                except Exception:
                    pass

                if not raw:
                    return cls._error(422, "Uploaded file is empty or cannot be read")

                pdf_bytes = raw
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    tmp.write(raw)
                    temp_file_path = tmp.name
                file_to_open = temp_file_path
                logger.debug(f"Created temporary PDF file: {temp_file_path}")
            else:
                file_to_open = str(file)
                try:
                    with open(file_to_open, "rb") as fh:
                        pdf_bytes = fh.read()
                except Exception:
                    pdf_bytes = None

            page_ranges: list = []
            text_parts: list = []
            cursor = 0

            with pdfplumber.open(file_to_open) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF contains {total_pages} page(s)")
                if total_pages == 0:
                    return cls._error(400, "PDF file is empty (0 pages)", total_pages=0)

                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    try:
                        page_text = page.extract_text() or ""
                    except Exception:
                        logger.exception("Error extracting text from PDF page")
                        page_text = ""

                    if not page_text.strip():
                        empty_pages.append(page_num)
                        continue

                    piece = page_text + "\n"
                    start = cursor
                    end = cursor + len(piece)
                    page_ranges.append({
                        "page": page_num,
                        "start": start,
                        "end": end,
                    })
                    text_parts.append(piece)
                    cursor = end

                text_content = "".join(text_parts)

                if not text_content.strip() and len(empty_pages) > 0:
                    logger.info(f"No text extracted. Attempting OCR on {len(empty_pages)} page(s)")
                    ocr_pairs = cls._process_scanned_pdf_with_pages(file_to_open, empty_pages)
                    if ocr_pairs:
                        for page_num, page_text in ocr_pairs:
                            piece = page_text + "\n"
                            start = cursor
                            end = cursor + len(piece)
                            page_ranges.append({
                                "page": page_num,
                                "start": start,
                                "end": end,
                            })
                            text_parts.append(piece)
                            cursor = end
                        text_content = "".join(text_parts)
                        ocr_pages = [p for p, _ in ocr_pairs]
                        empty_pages = [p for p in empty_pages if p not in ocr_pages]
                    else:
                        logger.warning("OCR also failed to extract text from PDF")

            if not text_content.strip():
                return cls._error(
                    422,
                    "No text extracted from PDF. This may be a scanned document requiring OCR.",
                    total_pages=total_pages,
                    empty_pages=empty_pages
                )

            extracted = text_content
            extracted_pages = len(page_ranges)

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
                character_count=len(extracted),
                page_ranges=page_ranges,
                pdf_bytes=pdf_bytes,
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
    def _process_scanned_pdf_with_pages(
        cls, pdf_path, page_numbers=None
    ) -> List[Tuple[int, str]]:
        """OCR per page, trả về list (page_number, text) theo thứ tự trang."""
        results: List[Tuple[int, str]] = []
        try:
            images = pdf_to_images(pdf_path, dpi=200)
            if not images:
                return results

            if page_numbers:
                targets = [(i, img) for i, img in enumerate(images, start=1) if i in page_numbers]
            else:
                targets = list(enumerate(images, start=1))

            for page_num, image in targets:
                try:
                    ocr_vi = TesseractOCRWrapper(lang='vie')
                    page_text = ocr_vi.read_image(image) or ""
                    if not page_text.strip():
                        ocr_en = TesseractOCRWrapper(lang='eng')
                        page_text = ocr_en.read_image(image) or ""
                    if page_text.strip():
                        results.append((page_num, page_text))
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num}: {e}")
                    continue
            return results
        except Exception:
            logger.exception("Failed to OCR scanned PDF")
            return results

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
            try:
                if hasattr(file_to_open, "seek"):
                    file_to_open.seek(0)
            except Exception:
                pass
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
