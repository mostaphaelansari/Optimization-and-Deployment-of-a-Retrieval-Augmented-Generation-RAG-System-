"""LLM-based Question-Answering system for RAG (Q3) - Using Ollama."""

import sys
from typing import Dict, Any
from pathlib import Path

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from template import RAG_PROMPT_TEMPLATE

from document_retriever import DocumentRetriever
from document_indexer import DocumentIndexer
from vector_store import VectorStore
from utils import get_config, get_logger


class LLMQASystem:
    """
    Question-Answering system using Ollama LLM with RAG.
    
    Supports:
    - Ollama (local LLM server)
    - Document retrieval from vector store
    - Prompt template management
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the QA system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.logger = get_logger(__name__)
        
        # LLM settings
        self.model_name = self.config.get('llm.model_name', 'phi3.5')
        self.base_url = self.config.get('llm.base_url', 'http://localhost:11434')
        self.temperature = self.config.get('llm.temperature', 0.7)
        
        # Initialize components (lazy loading)
        self._llm = None
        self._retriever = None
        self._chain = None
        
        self.logger.info(f"LLMQASystem initialized with Ollama")
        self.logger.info(f"Model: {self.model_name}")
    
    def _init_llm(self) -> OllamaLLM:
        """Initialize the Ollama LLM."""
        self.logger.info(f"Connecting to Ollama at {self.base_url}")
        self.logger.info(f"Model: {self.model_name}")
        
        llm = OllamaLLM(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
        )
        
        # Test connection
        try:
            test_response = llm.invoke("Say OK")
            self.logger.info(f"Ollama connection successful!")
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            self.logger.error("Make sure Ollama is running: ollama serve")
            self.logger.error(f"And model is pulled: ollama pull {self.model_name}")
            raise ConnectionError(f"Cannot connect to Ollama: {e}")
        
        return llm
    
    def _init_retriever(self) -> DocumentRetriever:
        """Initialize the document retriever."""
        indexer = DocumentIndexer()
        vector_store = VectorStore(embeddings=indexer.get_embeddings_model())
        return DocumentRetriever(vector_store=vector_store)
    
    @property
    def llm(self) -> OllamaLLM:
        """Lazy loading of LLM."""
        if self._llm is None:
            self._llm = self._init_llm()
        return self._llm
    
    @property
    def retriever(self) -> DocumentRetriever:
        """Lazy loading of retriever."""
        if self._retriever is None:
            self._retriever = self._init_retriever()
        return self._retriever
    
    def _build_chain(self):
        """Build the RAG chain using LangChain."""
        prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain = (
            {
                "context": self.retriever.get_retriever() | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    @property
    def chain(self):
        """Lazy loading of RAG chain."""
        if self._chain is None:
            self._chain = self._build_chain()
        return self._chain
    
    def answer(
        self,
        question: str,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate an answer for a question using RAG.
        
        Args:
            question: User question
            return_sources: Whether to include source documents
            
        Returns:
            Dictionary with answer and optional sources
        """
        self.logger.info(f"Processing question: '{question[:50]}...'")
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question, with_scores=True)
        
        # Build context from retrieved documents
        context = self.retriever.get_context_for_llm(question)
        
        # Generate answer using prompt template
        prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        formatted_prompt = prompt.format(context=context, question=question)
        
        # Generate response via Ollama
        self.logger.info("Generating answer with Ollama...")
        answer = self.llm.invoke(formatted_prompt)
        
        # Clean up the answer
        if isinstance(answer, str):
            answer = answer.strip()
        
        result = {
            'question': question,
            'answer': answer,
            'context': context
        }
        
        if return_sources:
            result['sources'] = [
                {
                    'source': doc['metadata'].get('source', 'Unknown'),
                    'page': doc['metadata'].get('page', 'N/A'),
                    'score': doc.get('score', 0)
                }
                for doc in retrieved_docs
            ]
        
        self.logger.info("Answer generated successfully")
        return result
    
    def answer_with_chain(self, question: str) -> str:
        """
        Generate an answer using the pre-built chain.
        
        Args:
            question: User question
            
        Returns:
            Generated answer string
        """
        return self.chain.invoke(question)
    
    def set_custom_prompt(self, template: str) -> None:
        """
        Set a custom prompt template.
        
        Args:
            template: Prompt template string with {context} and {question}
        """
        global RAG_PROMPT_TEMPLATE
        RAG_PROMPT_TEMPLATE = template
        self._chain = None
        self.logger.info("Custom prompt template set")
    
    def list_available_models(self) -> list:
        """List models available in Ollama."""
        import requests
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
        return []