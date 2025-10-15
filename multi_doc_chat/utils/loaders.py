from typing import Iterable, List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,PyPDFLoader,Docx2txtLoader)
from multi_doc_chat.logger.logger import get_logger
from multi_doc_chat.exception.exception import CustomException
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

def load_documents(file_paths: List[str]) -> List[Document]:
    """
    Load documents from a list of file paths.

    Args:
        file_paths (List[str]): List of file paths to load documents from.

    Returns:
        List[Document]: List of loaded Document objects.
    """
    logger = get_logger(__file__)
    documents = []
    for file_path in file_paths:
        try:
            if file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_path.endswith('.pdf'):
                # Try normal PDF loader first, then fallback to OCR if no text
                docs = PyPDFLoader(file_path).load()
                if not docs or all(not d.page_content.strip() for d in docs):
                    texts = ocr_pdf(file_path)
                    for idx, text in enumerate(texts):
                        if text.strip():
                            documents.append(Document(page_content=text, metadata={"source": file_path, "page": idx+1}))
                else:
                    documents.extend(docs)
            elif file_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                text = ocr_image(file_path)
                if text.strip():
                    documents.append(Document(page_content=text, metadata={"source": file_path}))
                else:
                    logger.warning(f"No text found in image: {file_path}")
                continue
            else:
                logger.warning(f"Unsupported file format: {file_path}. Skipping.")
                continue
            
            logger.info(f"Loaded {len(docs)} documents from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load document from {file_path}: {e}")
            raise CustomException(f"Failed to load document from {file_path}", e)
    
    return documents

def ocr_image(file_path: str) -> str:
    """Extract text from an image file using OCR."""
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return text

def ocr_pdf(file_path: str) -> List[str]:
    """Extract text from each page of a PDF using OCR."""
    images = convert_from_path(file_path)
    texts = [pytesseract.image_to_string(img) for img in images]
    return texts
