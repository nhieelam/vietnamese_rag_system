import pdfplumber
import pytesseract
from PIL import Image
from typing import Dict, Any
from app.utils.logger import logger


class FileService:
    SUPPORTED_TYPES = {
        "application/pdf": "pdf",
        "image/jpeg": "image",
        "image/png": "image",
    }

    # ========= PUBLIC API =========
    @classmethod
    def extract(cls, uploaded_file) -> Dict[str, Any]:
        try:
            file_type = uploaded_file.type
            file_name = uploaded_file.name

            if file_type not in cls.SUPPORTED_TYPES:
                return cls._error(
                    400,
                    f"Unsupported file type: {file_type}. Supported types: PDF, JPEG, PNG",
                    file_name=file_name,
                    file_type=file_type
                )

            if cls.SUPPORTED_TYPES[file_type] == "pdf":
                return cls._process_pdf(uploaded_file)

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

    # ========= INTERNAL =========
    @classmethod
    def _process_pdf(cls, file) -> Dict[str, Any]:
        text_content = ""
        empty_pages = []

        file_to_open = getattr(file, "path", file)

        try:
            with pdfplumber.open(file_to_open) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF contains {total_pages} page(s)")

                if total_pages == 0:
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
                        empty_pages.append(i + 1)

            if not text_content.strip():
                return cls._error(
                    422,
                    "No text extracted from PDF. This may be a scanned document.",
                    total_pages=total_pages,
                    empty_pages=empty_pages
                )

            extracted = text_content.strip()
            logger.info(
                f"PDF extraction complete: {len(extracted)} chars "
                f"from {total_pages - len(empty_pages)}/{total_pages} pages"
            )

            return cls._success(
                extracted,
                f"Extracted text from {total_pages - len(empty_pages)}/{total_pages} pages",
                total_pages=total_pages,
                extracted_pages=total_pages - len(empty_pages),
                empty_pages=empty_pages,
                character_count=len(extracted)
            )

        except FileNotFoundError:
            return cls._error(404, "PDF file not found")
        except Exception as e:
            logger.exception("PDF extraction failed")
            return cls._error(500, f"PDF extraction failed: {e}")

    @classmethod
    def _process_image(cls, file) -> Dict[str, Any]:
        file_to_open = getattr(file, "path", file)

        try:
            image = Image.open(file_to_open)
            width, height = image.size
            logger.info(f"Image size: {width}x{height}")

            if width < 100 or height < 100:
                return cls._error(
                    206,
                    "Image resolution too low for OCR",
                    width=width,
                    height=height
                )

            text, lang_used = cls._run_ocr(image)

            if not text.strip():
                return cls._error(
                    206,
                    "No text detected in image",
                    width=width,
                    height=height,
                    language=lang_used
                )

            extracted = text.strip()
            word_count = len(extracted.split())

            return cls._success(
                extracted,
                f"OCR successful ({lang_used})",
                width=width,
                height=height,
                language=lang_used,
                character_count=len(extracted),
                word_count=word_count
            )

        except pytesseract.TesseractNotFoundError:
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
        try:
            return (
                pytesseract.image_to_string(image, lang="vie", config="--psm 6"),
                "Vietnamese",
            )
        except Exception:
            logger.warning("Vietnamese OCR failed, fallback to English")
            return (
                pytesseract.image_to_string(image, lang="eng", config="--psm 6"),
                "English",
            )

    # ========= RESPONSE HELPERS =========
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
