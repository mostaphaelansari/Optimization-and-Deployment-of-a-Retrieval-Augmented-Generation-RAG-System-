"""RAG System - Core modules and utilities."""

# Utility imports (re-exported from utils)
from .utils import ConfigLoader, get_config, RAGLogger, get_logger

# Core modules
from .document_indexer import DocumentIndexer
from .vector_store import VectorStore
from .document_retriever import DocumentRetriever
from .llm_qa_system import LLMQASystem
from .evaluator import RAGEvaluator, EvaluationSample, EvaluationResult
from .chatbot import RAGChatbot

# New experimentation and quality modules
from .experimenter import RAGExperimenter, ExperimentConfig, ExperimentResult
from .quality_evaluator import QualityEvaluator, QualityMetrics
from .query_rewriter import QueryRewriter, RewriteResult, RewriteStrategy

__all__ = [
    # Utilities
    'ConfigLoader',
    'get_config',
    'RAGLogger', 
    'get_logger',
    # Core modules
    'DocumentIndexer',
    'VectorStore',
    'DocumentRetriever',
    'LLMQASystem',
    'RAGEvaluator',
    'EvaluationSample',
    'EvaluationResult',
    'RAGChatbot',
    # Experimentation
    'RAGExperimenter',
    'ExperimentConfig',
    'ExperimentResult',
    # Quality Evaluation
    'QualityEvaluator',
    'QualityMetrics',
    # Query Rewriting
    'QueryRewriter',
    'RewriteResult',
    'RewriteStrategy',
]

__version__ = "2.0.0"

