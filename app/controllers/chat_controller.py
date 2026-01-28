import logging
from typing import Dict, Any
from fastapi import HTTPException, UploadFile
from app.services.file_loader import extract_text_from_file
from app.services.ingest_service import ingest_file
from app.services.rag_service import get_rag_service
from app.utils.logger import logger as app_logger

logger = logging.getLogger("ChatController")


class ChatController:
    
    def __init__(self):
        logger.info("Initializing ChatController...")
        try:
            self.rag_service = get_rag_service()
            logger.info("ChatController initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatController: {e}")
            raise e

    async def handle_chat_with_file(self, uploaded_file: UploadFile, question: str) -> Dict[str, Any]:

        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if len(question) > 500:
            raise HTTPException(status_code=400, detail="Question is too long (max 500 chars)")
        
        if not uploaded_file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        try:
            logger.info(f"Processing file: {uploaded_file.filename} with question: {question[:50]}...")
            
            app_logger.info(f"Step 1/3: Extracting text from {uploaded_file.filename}...")
            extraction_result = extract_text_from_file(uploaded_file)
            
            if extraction_result['status_code'] != 200:
                if extraction_result['status_code'] == 206:
                    logger.warning(f"Warning during extraction: {extraction_result['message']}")
                    if not extraction_result['text']:
                        raise HTTPException(
                            status_code=400,
                            detail=extraction_result['message']
                        )
                else:
                    # Error status codes (400, 404, 422, 500)
                    raise HTTPException(
                        status_code=extraction_result['status_code'],
                        detail=extraction_result['message']
                    )
            
            extracted_text = extraction_result['text']
            logger.info(f"Step 1/3 Complete: Extracted {len(extracted_text)} characters")
            
            app_logger.info(f"Step 2/3: Ingesting text into vector database...")
            try:
                ingest_file(extracted_text)
                logger.info("Step 2/3 Complete: Text ingested successfully")
            except Exception as ingest_error:
                logger.error(f"Ingestion failed: {ingest_error}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to ingest document into vector database: {str(ingest_error)}"
                )
            
            rag_response = self.rag_service.get_answer(question)
            
            if rag_response['status_code'] != 200:
                raise HTTPException(
                    status_code=rag_response['status_code'],
                    detail=rag_response['message']
                )
            
            return {
                "answer": rag_response['answer'],
                "status": "success",
                "question": question,
                "file_info": {
                    "file_name": uploaded_file.filename,
                    "file_type": uploaded_file.content_type,
                    "text_length": len(extracted_text)
                },
                "extraction_metadata": extraction_result['metadata'],
                "rag_metadata": rag_response['metadata']
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in workflow: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Internal Server Error: {str(e)}"
            )

    async def handle_chat(self, question: str) -> Dict[str, Any]:

        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if len(question) > 500:
            raise HTTPException(status_code=400, detail="Question is too long (max 500 chars)")

        try:
            logger.info(f"Processing question (no file): {question[:50]}...")
            
            rag_response = self.rag_service.get_answer(question)
            
            if rag_response['status_code'] != 200:
                raise HTTPException(
                    status_code=rag_response['status_code'],
                    detail=rag_response['message']
                )
            
            logger.info("Answer generated successfully")
            
            return {
                "answer": rag_response['answer'],
                "status": "success",
                "question": question,
                "rag_metadata": rag_response['metadata']
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Internal Server Error: {str(e)}"
            )
