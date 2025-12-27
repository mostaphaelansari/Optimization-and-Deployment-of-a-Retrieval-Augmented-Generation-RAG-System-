"""RAG Experimentation Module - Test different configurations systematically."""

import json
import time
import shutil
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from document_indexer import DocumentIndexer
from vector_store import VectorStore
from document_retriever import DocumentRetriever
from llm_qa_system import LLMQASystem
from evaluator import RAGEvaluator, EvaluationSample
from utils import get_config, get_logger


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    similarity_threshold: float
    top_k: int = 5
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def __str__(self) -> str:
        return f"chunk={self.chunk_size}, embed={self.embedding_model.split('/')[-1]}, threshold={self.similarity_threshold}"


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    retrieval_metrics: Dict[str, float]
    answer_metrics: Dict[str, float]
    avg_latency_ms: float
    num_samples: int
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            'config': self.config.to_dict(),
            'retrieval_metrics': self.retrieval_metrics,
            'answer_metrics': self.answer_metrics,
            'avg_latency_ms': self.avg_latency_ms,
            'num_samples': self.num_samples,
            'error': self.error,
            'timestamp': self.timestamp
        }


class RAGExperimenter:
    """
    Runs systematic experiments with different RAG configurations.
    
    Tests combinations of:
    - Chunk sizes (document splitting granularity)
    - Embedding models (semantic representation quality)
    - Similarity thresholds (retrieval precision vs recall)
    """
    
    # Default experiment configurations
    DEFAULT_CHUNK_SIZES = [256, 512, 1000, 2000]
    DEFAULT_EMBEDDING_MODELS = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    ]
    DEFAULT_THRESHOLDS = [0.2, 0.3, 0.4, 0.5]
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the experimenter."""
        self.config = get_config(config_path)
        self.logger = get_logger(__name__)
        self.config_path = config_path
        
        # Load experiment settings from config or use defaults
        exp_config = self.config.get('experimentation', {})
        self.chunk_sizes = exp_config.get('chunk_sizes', self.DEFAULT_CHUNK_SIZES)
        self.embedding_models = exp_config.get('embedding_models', self.DEFAULT_EMBEDDING_MODELS)
        self.thresholds = exp_config.get('similarity_thresholds', self.DEFAULT_THRESHOLDS)
        self.results_dir = Path(exp_config.get('results_dir', 'experiments'))
        
        # Base configuration values
        self.base_chunk_overlap = self.config.get('document_processing.chunk_overlap', 200)
        self.base_top_k = self.config.get('retrieval.top_k', 5)
        self.device = self.config.get('embeddings.device', 'cuda')
        self.data_dir = self.config.get('paths.data_dir', 'data')
        self.vector_store_dir = self.config.get('paths.vector_store_dir', 'vector_store')
        
        # Results storage
        self.results: List[ExperimentResult] = []
        
        self.logger.info("RAGExperimenter initialized")
        self.logger.info(f"Chunk sizes: {self.chunk_sizes}")
        self.logger.info(f"Embedding models: {[m.split('/')[-1] for m in self.embedding_models]}")
        self.logger.info(f"Thresholds: {self.thresholds}")
    
    def _get_device(self) -> str:
        """Get the appropriate device (cuda if available, else cpu)."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    
    def _create_embeddings_model(self, model_name: str) -> HuggingFaceEmbeddings:
        """Create an embeddings model instance."""
        device = self._get_device()
        self.logger.info(f"Loading embedding model: {model_name} on {device}")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _create_indexer(self, chunk_size: int, chunk_overlap: int, embedding_model: str) -> DocumentIndexer:
        """Create a document indexer with specific configuration."""
        # Create a temporary indexer with custom settings
        indexer = DocumentIndexer(self.config_path)
        
        # Override settings
        indexer.chunk_size = chunk_size
        indexer.chunk_overlap = chunk_overlap
        indexer.embeddings = self._create_embeddings_model(embedding_model)
        indexer.text_splitter = indexer._init_text_splitter()
        
        return indexer
    
    def _clear_vector_store(self) -> None:
        """Clear the existing vector store for fresh indexing."""
        import gc
        
        vector_store_path = Path(self.vector_store_dir)
        if vector_store_path.exists():
            # Try multiple times with delay for Windows file locking
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Force garbage collection to release file handles
                    gc.collect()
                    time.sleep(0.5)  # Give time for handles to be released
                    
                    shutil.rmtree(vector_store_path)
                    self.logger.info("Cleared existing vector store")
                    return
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Retry {attempt + 1}/{max_retries}: {e}")
                        time.sleep(1)  # Wait before retry
                    else:
                        # Last resort: try to delete files individually
                        self.logger.warning("Could not delete vector store, creating new collection name")
                        # Change collection name to avoid conflicts
                        self.collection_override = f"rag_exp_{int(time.time())}"
    
    def _index_documents(self, indexer: DocumentIndexer, collection_name: str = None) -> VectorStore:
        """Index documents and return the vector store."""
        # Process documents
        chunks = indexer.process_documents(self.data_dir, is_directory=True)
        self.logger.info(f"Processed {len(chunks)} chunks")
        
        # Create and populate vector store using create_store
        vector_store = VectorStore(
            embeddings=indexer.get_embeddings_model(),
            config_path=self.config_path
        )
        
        # Override collection name if needed (to avoid conflicts)
        if hasattr(self, 'collection_override') and self.collection_override:
            vector_store.collection_name = self.collection_override
            self.collection_override = None  # Reset for next experiment
        
        vector_store.create_store(chunks)
        
        return vector_store
    
    def run_single_experiment(
        self,
        config: ExperimentConfig,
        evaluation_samples: List[EvaluationSample]
    ) -> ExperimentResult:
        """
        Run a single experiment with the given configuration.
        
        Args:
            config: Experiment configuration
            evaluation_samples: Test samples for evaluation
            
        Returns:
            ExperimentResult with metrics
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Running experiment: {config}")
        self.logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Clear and recreate vector store
            self._clear_vector_store()
            
            # Create indexer and index documents
            indexer = self._create_indexer(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                embedding_model=config.embedding_model
            )
            vector_store = self._index_documents(indexer)
            
            # Create retriever with threshold
            retriever = DocumentRetriever(
                vector_store=vector_store,
                config_path=self.config_path
            )
            retriever.score_threshold = config.similarity_threshold
            retriever.top_k = config.top_k
            
            # Create QA system
            qa_system = LLMQASystem(self.config_path)
            qa_system._retriever = retriever
            
            # Run evaluation
            evaluator = RAGEvaluator(qa_system=qa_system, config_path=self.config_path)
            eval_results = evaluator.evaluate_dataset(evaluation_samples)
            
            # Aggregate metrics
            aggregated = evaluator._aggregate_results()
            
            total_time = (time.time() - start_time) * 1000
            
            result = ExperimentResult(
                config=config,
                retrieval_metrics=aggregated.get('retrieval_metrics', {}),
                answer_metrics=aggregated.get('answer_metrics', {}),
                avg_latency_ms=aggregated.get('avg_latency_ms', total_time / len(evaluation_samples)),
                num_samples=len(evaluation_samples)
            )
            
            self.logger.info(f"Experiment completed in {total_time/1000:.2f}s")
            self.logger.info(f"Retrieval P@5: {result.retrieval_metrics.get('precision@5', 0):.4f}")
            self.logger.info(f"Answer Relevance: {result.answer_metrics.get('answer_relevance', 0):.4f}")
            
            return result
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.logger.error(f"Experiment failed: {error_msg}")
            self.logger.error(traceback.format_exc())
            print(f"\n[ERROR] Experiment failed: {error_msg}")
            print(traceback.format_exc())
            # Return result with error message
            return ExperimentResult(
                config=config,
                retrieval_metrics={},
                answer_metrics={},
                avg_latency_ms=0,
                num_samples=0,
                error=error_msg
            )
    
    def generate_experiment_configs(
        self,
        chunk_sizes: Optional[List[int]] = None,
        embedding_models: Optional[List[str]] = None,
        thresholds: Optional[List[float]] = None
    ) -> List[ExperimentConfig]:
        """Generate all experiment configurations."""
        chunk_sizes = chunk_sizes or self.chunk_sizes
        embedding_models = embedding_models or self.embedding_models
        thresholds = thresholds or self.thresholds
        
        configs = []
        for chunk_size, model, threshold in product(chunk_sizes, embedding_models, thresholds):
            # Chunk overlap is typically 20% of chunk size
            chunk_overlap = min(int(chunk_size * 0.2), 200)
            
            configs.append(ExperimentConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model=model,
                similarity_threshold=threshold,
                top_k=self.base_top_k
            ))
        
        return configs
    
    def run_all_experiments(
        self,
        evaluation_samples: List[EvaluationSample],
        chunk_sizes: Optional[List[int]] = None,
        embedding_models: Optional[List[str]] = None,
        thresholds: Optional[List[float]] = None,
        save_intermediate: bool = True
    ) -> List[ExperimentResult]:
        """
        Run experiments across all configuration combinations.
        
        Args:
            evaluation_samples: Test samples for evaluation
            chunk_sizes: Override default chunk sizes
            embedding_models: Override default embedding models
            thresholds: Override default thresholds
            save_intermediate: Save results after each experiment
            
        Returns:
            List of all experiment results
        """
        configs = self.generate_experiment_configs(
            chunk_sizes=chunk_sizes,
            embedding_models=embedding_models,
            thresholds=thresholds
        )
        
        self.logger.info(f"\nRunning {len(configs)} experiment configurations")
        self.logger.info(f"Using {len(evaluation_samples)} evaluation samples")
        
        self.results = []
        
        for i, config in enumerate(configs, 1):
            self.logger.info(f"\n[{i}/{len(configs)}] Starting experiment...")
            
            result = self.run_single_experiment(config, evaluation_samples)
            self.results.append(result)
            
            if save_intermediate:
                self._save_intermediate_results()
        
        self.logger.info(f"\nAll {len(configs)} experiments completed!")
        return self.results
    
    def _save_intermediate_results(self) -> None:
        """Save intermediate results to file."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.results_dir / "intermediate_results.json"
        
        with open(filepath, 'w') as f:
            json.dump(
                {'results': [r.to_dict() for r in self.results]},
                f,
                indent=2
            )
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive experiment report.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Report dictionary
        """
        if not self.results:
            self.logger.warning("No results to generate report from")
            return {}
        
        # Sort results by answer relevance
        sorted_by_relevance = sorted(
            self.results,
            key=lambda r: r.answer_metrics.get('answer_relevance', 0),
            reverse=True
        )
        
        # Sort results by retrieval precision
        sorted_by_precision = sorted(
            self.results,
            key=lambda r: r.retrieval_metrics.get('precision@5', 0),
            reverse=True
        )
        
        # Find best configurations
        best_overall = sorted_by_relevance[0] if sorted_by_relevance else None
        best_retrieval = sorted_by_precision[0] if sorted_by_precision else None
        
        # Analyze by parameter
        chunk_size_analysis = self._analyze_by_parameter('chunk_size')
        model_analysis = self._analyze_by_parameter('embedding_model')
        threshold_analysis = self._analyze_by_parameter('similarity_threshold')
        
        report = {
            'summary': {
                'total_experiments': len(self.results),
                'timestamp': datetime.now().isoformat(),
                'best_config_overall': best_overall.config.to_dict() if best_overall else None,
                'best_config_retrieval': best_retrieval.config.to_dict() if best_retrieval else None,
            },
            'best_metrics': {
                'highest_answer_relevance': best_overall.answer_metrics.get('answer_relevance', 0) if best_overall else 0,
                'highest_precision': best_retrieval.retrieval_metrics.get('precision@5', 0) if best_retrieval else 0,
            },
            'parameter_analysis': {
                'chunk_size': chunk_size_analysis,
                'embedding_model': model_analysis,
                'similarity_threshold': threshold_analysis,
            },
            'all_results': [r.to_dict() for r in self.results],
            'rankings': {
                'by_answer_relevance': [
                    {'rank': i+1, 'config': str(r.config), 'score': r.answer_metrics.get('answer_relevance', 0)}
                    for i, r in enumerate(sorted_by_relevance[:10])
                ],
                'by_retrieval_precision': [
                    {'rank': i+1, 'config': str(r.config), 'score': r.retrieval_metrics.get('precision@5', 0)}
                    for i, r in enumerate(sorted_by_precision[:10])
                ]
            }
        }
        
        # Save report
        if output_path:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Report saved to {output_path}")
        
        return report
    
    def _analyze_by_parameter(self, param_name: str) -> Dict[str, Any]:
        """Analyze results grouped by a specific parameter."""
        from collections import defaultdict
        
        grouped = defaultdict(list)
        
        for result in self.results:
            if param_name == 'chunk_size':
                key = result.config.chunk_size
            elif param_name == 'embedding_model':
                key = result.config.embedding_model.split('/')[-1]
            elif param_name == 'similarity_threshold':
                key = result.config.similarity_threshold
            else:
                continue
                
            grouped[key].append(result)
        
        analysis = {}
        for key, results in grouped.items():
            relevance_scores = [r.answer_metrics.get('answer_relevance', 0) for r in results]
            precision_scores = [r.retrieval_metrics.get('precision@5', 0) for r in results]
            
            analysis[str(key)] = {
                'count': len(results),
                'avg_answer_relevance': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
                'avg_precision': sum(precision_scores) / len(precision_scores) if precision_scores else 0,
                'max_answer_relevance': max(relevance_scores) if relevance_scores else 0,
                'max_precision': max(precision_scores) if precision_scores else 0,
            }
        
        return analysis
    
    def print_summary(self) -> None:
        """Print a formatted summary of experiment results."""
        if not self.results:
            print("No experiment results available.")
            return
        
        print("\n" + "=" * 80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 80)
        
        # Best configurations
        sorted_results = sorted(
            self.results,
            key=lambda r: r.answer_metrics.get('answer_relevance', 0),
            reverse=True
        )
        
        print("\n📊 TOP 5 CONFIGURATIONS (by Answer Relevance):")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"\n{i}. {result.config}")
            print(f"   Answer Relevance: {result.answer_metrics.get('answer_relevance', 0):.4f}")
            print(f"   Faithfulness: {result.answer_metrics.get('faithfulness', 0):.4f}")
            print(f"   Retrieval P@5: {result.retrieval_metrics.get('precision@5', 0):.4f}")
            print(f"   Avg Latency: {result.avg_latency_ms:.2f}ms")
        
        # Parameter analysis
        print("\n\n📈 PARAMETER IMPACT ANALYSIS:")
        print("-" * 80)
        
        report = self.generate_report()
        
        for param, analysis in report.get('parameter_analysis', {}).items():
            print(f"\n{param.upper().replace('_', ' ')}:")
            for value, metrics in analysis.items():
                print(f"  {value}: Avg Relevance={metrics['avg_answer_relevance']:.4f}, "
                      f"Avg P@5={metrics['avg_precision']:.4f}")
        
        print("\n" + "=" * 80)
