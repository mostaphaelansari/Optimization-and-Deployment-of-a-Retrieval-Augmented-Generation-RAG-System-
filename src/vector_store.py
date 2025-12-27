"""Vector store management for the RAG system (Q1 - Storage)."""

from pathlib import Path
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from utils import get_config, get_logger


class VectorStore:
    """
    Manages vector storage using ChromaDB.
    
    Handles:
    - Creating and persisting vector stores
    - Adding documents with embeddings
    - Loading existing stores
    """
    
    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        config_path: str = "config.yaml"
    ):
        """
        Initialize the vector store.
        
        Args:
            embeddings: HuggingFace embeddings model
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.logger = get_logger(__name__)
        self.embeddings = embeddings
        
        # Load settings
        self.persist_dir = Path(self.config.get('paths.vector_store_dir', 'vector_store'))
        self.collection_name = self.config.get('vector_store.collection_name', 'rag_documents')
        
        self._vector_store: Optional[Chroma] = None
        self.logger.info(f"VectorStore initialized with collection: {self.collection_name}")
    
    def create_store(self, documents: List[Document]) -> Chroma:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            Chroma vector store instance
        """
        self.logger.info(f"Creating vector store with {len(documents)} documents")
        
        # Ensure persist directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self._vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_dir)
        )
        
        self.logger.info(f"Vector store created and persisted to {self.persist_dir}")
        return self._vector_store
    
    def load_store(self) -> Chroma:
        """
        Load an existing vector store from disk.
        
        Returns:
            Chroma vector store instance
        """
        if not self.persist_dir.exists():
            raise FileNotFoundError(f"Vector store not found at {self.persist_dir}")
        
        self.logger.info(f"Loading vector store from {self.persist_dir}")
        
        self._vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir)
        )
        
        return self._vector_store
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to an existing vector store.
        
        Args:
            documents: List of Document objects to add
        """
        if self._vector_store is None:
            raise ValueError("Vector store not initialized. Call create_store or load_store first.")
        
        self.logger.info(f"Adding {len(documents)} documents to vector store")
        self._vector_store.add_documents(documents)
    
    def get_store(self) -> Chroma:
        """
        Get the vector store instance, loading if necessary.
        
        Returns:
            Chroma vector store instance
        """
        if self._vector_store is None:
            self._vector_store = self.load_store()
        return self._vector_store
    
    def similarity_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        Perform similarity search without scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        store = self.get_store()
        return store.similarity_search(query, k=k)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        store = self.get_store()
        return store.similarity_search_with_score(query, k=k)
    
    def delete_collection(self) -> None:
        """Delete the current collection."""
        if self._vector_store is not None:
            self._vector_store.delete_collection()
            self._vector_store = None
            self.logger.info(f"Deleted collection: {self.collection_name}")
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        store = self.get_store()
        collection = store._collection
        
        return {
            'name': self.collection_name,
            'count': collection.count(),
            'persist_directory': str(self.persist_dir)
        }
    
    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a retriever interface for the vector store.
        
        Args:
            search_kwargs: Optional search parameters
            
        Returns:
            Retriever instance
        """
        store = self.get_store()
        k = self.config.get('retrieval.top_k', 5)
        
        kwargs = search_kwargs or {'k': k}
        return store.as_retriever(search_kwargs=kwargs)