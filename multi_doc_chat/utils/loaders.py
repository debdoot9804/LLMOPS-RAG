from typing import Iterable, List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,PyPDFLoader,Docx2txtLoader)
from multi_doc_chat.logger.logger import get_logger
from multi_doc_chat.exception.exception import CustomException

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
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_path}. Skipping.")
                continue
            
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} documents from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load document from {file_path}: {e}")
            raise CustomException(f"Failed to load document from {file_path}", e)
    
    return documents
