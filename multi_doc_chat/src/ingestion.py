import os
import logging
from typing import List, Optional, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Import custom modules
from multi_doc_chat.utils.loaders import load_documents
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from azure_clients import get_embedding_client

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into a FAISS vector store."""
    
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        vector_store_path: str = "vector_store",
        session_id: str = None,
    ):
        """
        Initialize the document ingestion pipeline.
        
        Args:
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between consecutive chunks
            vector_store_path: Path to save the vector store
            session_id: Unique identifier for the user session
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.session_id = session_id or self._generate_session_id()
        
        # Include session_id in vector store path if provided
        if session_id:
            self.vector_store_path = os.path.join(vector_store_path, f"session_{self.session_id}")
        else:
            self.vector_store_path = vector_store_path
            
        self.embedding_model = get_embedding_client()
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        logger.info(f"Initialized document ingestion pipeline with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, session_id={self.session_id}")
    
    def _generate_session_id(self) -> str:
        """
        Generate a unique session ID using timestamp and random characters.
        
        Returns:
            A unique session ID string
        """
        import uuid
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        return f"{timestamp}_{unique_id}"

    def process_documents(self, file_paths: List[str], metadata: Optional[Dict[str, Any]] = None) -> FAISS:
        """
        Process documents through the ingestion pipeline.
        
        Args:
            file_paths: List of paths to documents to ingest
            metadata: Optional additional metadata to add to all documents
            
        Returns:
            FAISS vector store containing the processed documents
        """
        logger.info(f"Starting document ingestion for {len(file_paths)} files with session_id={self.session_id}")
        
        # Step 1: Load documents
        try:
            documents = load_documents(file_paths)
            logger.info(f"Loaded {len(documents)} documents successfully")
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
        
        # Initialize metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add session ID to metadata
        metadata["session_id"] = self.session_id
        
        # Add additional metadata to documents
        for doc in documents:
            doc.metadata.update(metadata)
        
        # Step 2: Split documents into chunks
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split documents into {len(chunks)} chunks")
            
            # Log chunk statistics
            avg_chunk_length = sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0
            logger.info(f"Average chunk length: {avg_chunk_length:.2f} characters")
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise
        
        # Step 3: Create embeddings and store in vector store
        try:
            vector_store = FAISS.from_documents(chunks, self.embedding_model)
            logger.info(f"Created vector store with {len(chunks)} embedded chunks")
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
        
        # Step 4: Save the vector store if path is provided
        if self.vector_store_path:
            try:
                os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
                vector_store.save_local(self.vector_store_path)
                logger.info(f"Saved vector store to {self.vector_store_path}")
            except Exception as e:
                logger.error(f"Error saving vector store: {e}")
                # Continue even if saving fails
        
        return vector_store
    
    def load_vector_store(self) -> Optional[FAISS]:
        """
        Load an existing vector store.
        
        Returns:
            FAISS vector store if it exists, None otherwise
        """
        try:
            if os.path.exists(self.vector_store_path):
                vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embedding_model, 
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded vector store from {self.vector_store_path} for session_id={self.session_id}")
                return vector_store
            else:
                logger.warning(f"Vector store not found at {self.vector_store_path} for session_id={self.session_id}")
                return None
        except Exception as e:
            logger.error(f"Error loading vector store for session_id={self.session_id}: {e}")
            return None
    
    def add_documents(self, file_paths: List[str], metadata: Optional[Dict[str, Any]] = None) -> Optional[FAISS]:
        """
        Add documents to an existing vector store.
        
        Args:
            file_paths: List of paths to documents to add
            metadata: Optional additional metadata to add to all documents
            
        Returns:
            Updated FAISS vector store
        """
        # Load existing vector store
        vector_store = self.load_vector_store()
        if vector_store is None:
            logger.info("No existing vector store found, creating new one")
            return self.process_documents(file_paths, metadata)
        
        # Load and process new documents
        try:
            documents = load_documents(file_paths)
            logger.info(f"Loaded {len(documents)} new documents successfully for session_id={self.session_id}")
            
            # Initialize metadata if not provided
            if metadata is None:
                metadata = {}
            
            # Add session ID to metadata
            metadata["session_id"] = self.session_id
            
            # Add additional metadata to documents
            for doc in documents:
                doc.metadata.update(metadata)
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split new documents into {len(chunks)} chunks")
            
            # Add new chunks to vector store
            vector_store.add_documents(chunks)
            logger.info(f"Added {len(chunks)} new chunks to vector store")
            
            # Save updated vector store
            if self.vector_store_path:
                vector_store.save_local(self.vector_store_path)
                logger.info(f"Saved updated vector store to {self.vector_store_path}")
            
            return vector_store
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return vector_store  # Return original vector store if there was an error


# Example usage
if __name__ == "__main__":
    # Define paths to documents
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    file_paths = [
        os.path.join(data_dir, "xyz.pdf"),
        # Add more file paths as needed
    ]
    
    # Create a unique session ID for this example run
    # In a real application, you would get this from a user's session
    import uuid
    session_id = str(uuid.uuid4())[:12]
    print(f"Using session ID: {session_id}")
    
    # Create and use the ingestion pipeline with a session ID
    pipeline = DocumentIngestionPipeline(
        chunk_size=2000,
        chunk_overlap=400,
        vector_store_path=os.path.join(data_dir, "vector_store"),
        session_id=session_id
    )
    
    # Process documents and create vector store
    vector_store = pipeline.process_documents(
        file_paths=file_paths,
        metadata={"source": "initial_ingestion", "date": "2025-10-09"}
    )
    
    # Verify vector store creation
    if vector_store:
        print(f"Successfully created vector store with {len(vector_store.index_to_docstore_id)} vectors")
        print(f"Vector store saved to: {pipeline.vector_store_path}")
        
        # Test retrieval functionality
        test_query = "What is the compensation package?"
        docs = vector_store.similarity_search(test_query, k=3)
        
        print(f"\nTest query: '{test_query}'")
        print(f"Found {len(docs)} relevant documents:")
        
        for i, doc in enumerate(docs, 1):
            print(f"\nDocument {i}:")
            print("-" * 40)
            print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
            print(f"Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            print(f"Session ID: {doc.metadata.get('session_id', 'Not available')}")
            print("-" * 40)