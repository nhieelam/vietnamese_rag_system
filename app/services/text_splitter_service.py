from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import AppConfig
from app.utils.logger import logger


class TextSplitterService:

    @classmethod
    def split(cls, text: str) -> List[str]:
        if not text or not text.strip():
            raise ValueError("Text is empty, cannot split")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=AppConfig.CHUNK_SIZE,
            chunk_overlap=AppConfig.CHUNK_OVERLAP,
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
            f"(chunk_size={AppConfig.CHUNK_SIZE}, overlap={AppConfig.CHUNK_OVERLAP})"
        )

        return chunks
