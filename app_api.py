
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.utils.logger import logger

# Create FastAPI app
app = FastAPI(
    title="Vietnamese RAG System API",
    description="Upload documents and ask questions using Retrieval-Augmented Generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["RAG"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Vietnamese RAG System API...")
    logger.info("API documentation available at: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Vietnamese RAG System API...")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Vietnamese RAG System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "upload_and_chat": "/api/v1/chat/upload",
            "chat": "/api/v1/chat",
            "health": "/api/v1/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
