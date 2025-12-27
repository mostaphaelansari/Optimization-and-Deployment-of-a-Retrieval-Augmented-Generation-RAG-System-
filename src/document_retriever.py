"""Document retrieval module for the RAG system (Q2)."""

from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document

from vector_store import VectorStore
from document_indexer import DocumentIndexer
from utils import get_config, get_logger


class DocumentRetriever:
    """
    Handles document retrieval from the vector store.
    
    Provides:
    - Query-based document retrieval
    - Relevance scoring
    - Result formatting
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        config_path: str = "config.yaml"
    ):
        """
        Initialize the document retriever.
        
        Args:
            vector_store: Optional VectorStore instance
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.logger = get_logger(__name__)
        
        # Retrieval settings
        self.top_k = self.config.get('retrieval.top_k', 5)
        self.score_threshold = self.config.get('retrieval.score_threshold', 0.3)
        
        # Initialize vector store if not provided
        if vector_store is None:
            indexer = DocumentIndexer(config_path)
            self.vector_store = VectorStore(
                embeddings=indexer.get_embeddings_model(),
                config_path=config_path
            )
        else:
            self.vector_store = vector_store
        
        self.logger.info(f"DocumentRetriever initialized (top_k={self.top_k})")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        with_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            top_k: Number of documents to retrieve (overrides config)
            with_scores: Whether to include similarity scores
            
        Returns:
            List of documents with metadata and optional scores
        """
        k = top_k or self.top_k
        self.logger.info(f"Retrieving top {k} documents for query: '{query[:50]}...'")
        
        if with_scores:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return self._format_results_with_scores(results)
        else:
            results = self.vector_store.similarity_search(query, k=k)
            return self._format_results(results)
    
    def _format_results_with_scores(
        self,
        results: List[Tuple[Document, float]]
    ) -> List[Dict[str, Any]]:
        """Format search results with scores."""
        formatted = []
        
        for rank, (doc, score) in enumerate(results, 1):
            # Convert distance to similarity score (ChromaDB returns distance)
            # Lower distance = higher similarity
            similarity_score = 1 / (1 + score)
            
            formatted.append({
                'rank': rank,
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': round(similarity_score, 4),
                'distance': round(score, 4)
            })
        
        return formatted
    
    def _format_results(self, results: List[Document]) -> List[Dict[str, Any]]:
        """Format search results without scores."""
        return [
            {
                'rank': rank,
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            for rank, doc in enumerate(results, 1)
        ]
    
    def retrieve_with_threshold(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents above a similarity threshold.
        
        Args:
            query: User query string
            top_k: Maximum number of documents
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of documents meeting the threshold
        """
        k = top_k or self.top_k
        min_score = threshold or self.score_threshold
        
        results = self.retrieve(query, top_k=k, with_scores=True)
        
        filtered = [r for r in results if r['score'] >= min_score]
        self.logger.info(
            f"Retrieved {len(filtered)}/{len(results)} documents above threshold {min_score}"
        )
        
        return filtered
    
    def get_context_for_llm(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> str:
        """
        Get formatted context string for LLM input.
        
        Args:
            query: User query string
            top_k: Number of documents to include
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k=top_k, with_scores=True)
        
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for r in results:
            source = r['metadata'].get('source', 'Unknown')
            page = r['metadata'].get('page', 'N/A')
            context_parts.append(
                f"[Source: {source}, Page: {page}]\n{r['content']}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a LangChain retriever interface.
        
        Args:
            search_kwargs: Optional search parameters
            
        Returns:
            LangChain Retriever instance
        """
        return self.vector_store.as_retriever(search_kwargs)
    
    def print_results(self, results: List[Dict[str, Any]]) -> None:
        """Pretty print retrieval results to console."""
        print("\n" + "=" * 60)
        print(f"Retrieved {len(results)} documents")
        print("=" * 60)
        
        for r in results:
            print(f"\n[Rank {r['rank']}]", end="")
            if 'score' in r:
                print(f" - Score: {r['score']:.4f}", end="")
            print()
            
            source = r['metadata'].get('source', 'Unknown')
            page = r['metadata'].get('page', 'N/A')
            print(f"Source: {source} | Page: {page}")
            print("-" * 40)
            
            # Truncate long content for display
            content = r['content']
            if len(content) > 300:
                content = content[:300] + "..."
            print(content)
        
        print("\n" + "=" * 60)
