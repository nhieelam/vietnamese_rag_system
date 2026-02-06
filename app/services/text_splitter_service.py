from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import AIConfig
from app.utils.logger import logger


class TextSplitterService:
    """
    Text Splitter Service
    Split raw text into chunks for embedding
    """

    @classmethod
    def split(cls, text: str) -> List[str]:
        if not text or not text.strip():
            raise ValueError("Text is empty, cannot split")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=AIConfig.CHUNK_SIZE,
            chunk_overlap=AIConfig.CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ],
        )

        chunks = splitter.split_text(text)

        logger.info(
            f"Split text into {len(chunks)} chunks "
            f"(chunk_size={AIConfig.CHUNK_SIZE}, overlap={AIConfig.CHUNK_OVERLAP})"
        )

        return chunks
