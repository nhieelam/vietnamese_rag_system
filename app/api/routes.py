from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Dict, Any
from app.controllers.chat_controller import get_chat_controller

router = APIRouter()


@router.post("/chat/upload", response_model=Dict[str, Any])
async def chat_with_file_upload(
    file: UploadFile = File(..., description="PDF or image file to process"),
    question: str = Form(..., description="Question about the uploaded document")
) -> Dict[str, Any]:
    controller = get_chat_controller()
    return await controller.handle_chat_with_file(file, question)


@router.post("/chat", response_model=Dict[str, Any])
async def chat_without_file(
    question: str = Form(..., description="Question to ask about existing documents")
) -> Dict[str, Any]:
    controller = get_chat_controller()
    return await controller.handle_chat(question)


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Vietnamese RAG API"}
