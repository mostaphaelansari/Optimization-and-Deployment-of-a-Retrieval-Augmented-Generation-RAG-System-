"""Document indexing module for the RAG system (Q1)."""
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    CharacterTextSplitter,
)

from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
)
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings

from utils import get_config, get_logger

class DocumentIndexer:
    """
    Handles document loading, splitting, and embedding generation.
    
    Pipeline:
    1. Loading - Load documents from files
    2. Splitting - Divide into chunks with metadata
    3. Embedding - Generate embeddings via HuggingFace model
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the document indexer with configuration."""
        self.config = get_config(config_path)
        self.logger = get_logger(__name__)
        
        # Load settings from config
        self.chunk_size = self.config.get('document_processing.chunk_size', 512)
        self.chunk_overlap = self.config.get('document_processing.chunk_overlap', 50)
        self.split_method = self.config.get('document_processing.split_method', 'recursive')
        
        # Initialize embeddings model
        self.embeddings = self._init_embeddings()
        self.text_splitter = self._init_text_splitter()
        
        self.logger.info(f"DocumentIndexer initialized with {self.split_method} splitter")
    
    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize the HuggingFace embeddings model."""
        model_name = self.config.get(
            'embeddings.model_name',
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        device = self.config.get('embeddings.device', 'cpu')
        
        self.logger.info(f"Loading embeddings model: {model_name}")
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _init_text_splitter(self):
        """Initialize the text splitter based on configuration."""
        splitter_map = {
            'markdown': MarkdownTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            ),
            'recursive': RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            ),
            'character': CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        }
        return splitter_map.get(self.split_method, splitter_map['recursive'])
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document from file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects with metadata
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        self.logger.info(f"Loading document: {file_path}")
        
        # Select loader based on file extension
        extension = path.suffix.lower()
        if extension == '.pdf':
            loader = PyPDFLoader(str(path))
        elif extension in ['.txt', '.md']:
            loader = TextLoader(str(path), encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
        documents = loader.load()
        
        # Enrich metadata
        for doc in documents:
            doc.metadata['source'] = str(path.name)
            doc.metadata['file_path'] = str(path)
            doc.metadata['file_type'] = extension
        
        self.logger.info(f"Loaded {len(documents)} pages from {file_path}")
        return documents
    
    def load_directory(self, directory_path: str, glob_pattern: str = "**/*.pdf") -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            directory_path: Path to the directory
            glob_pattern: Pattern to match files
            
        Returns:
            List of all Document objects
        """
        path = Path(directory_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        self.logger.info(f"Loading documents from directory: {directory_path}")
        
        loader = DirectoryLoader(
            str(path),
            glob=glob_pattern,
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        documents = loader.load()
        self.logger.info(f"Loaded {len(documents)} documents from directory")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks while preserving metadata.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        self.logger.info(f"Splitting {len(documents)} documents into chunks")
        
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        self.logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        return self.embeddings.embed_documents(texts)
    
    def process_documents(
        self,
        source: str,
        is_directory: bool = False,
        glob_pattern: str = "**/*.pdf"
    ) -> List[Document]:
        """
        Full pipeline: load, split, and prepare documents for indexing.
        
        Args:
            source: File path or directory path
            is_directory: Whether source is a directory
            glob_pattern: Pattern for directory loading
            
        Returns:
            List of processed Document chunks
        """
        # Load documents
        if is_directory:
            documents = self.load_directory(source, glob_pattern)
        else:
            documents = self.load_document(source)
        
        # Split into chunks
        chunks = self.split_documents(documents)
        
        self.logger.info(f"Document processing complete: {len(chunks)} chunks ready")
        return chunks
    
    def get_embeddings_model(self) -> HuggingFaceEmbeddings:
        """Return the embeddings model for use with vector store."""
        return self.embeddings
