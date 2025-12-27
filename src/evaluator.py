"""RAG System Evaluator (Q4) - Comprehensive evaluation module."""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from utils import get_config, get_logger
from utils.metrics import RetrievalMetrics, calculate_f1_score


@dataclass
class EvaluationSample:
    """Single evaluation sample with question, ground truth, and expected sources."""
    question: str
    ground_truth: str
    expected_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete result of evaluating a single sample."""
    question: str
    generated_answer: str
    ground_truth: str
    retrieved_context: str
    retrieval_metrics: Dict[str, float]
    answer_metrics: Dict[str, float]
    sources_retrieved: List[str]
    latency_ms: float = 0.0


class AnswerEvaluator:
    """
    Evaluates answer quality using multiple metrics.
    
    Metrics:
    - Lexical: Word overlap, BLEU-like scores
    - Semantic: Embedding similarity (if available)
    - Faithfulness: Whether answer is grounded in context
    """
    
    def __init__(self, embeddings=None):
        """
        Initialize answer evaluator.
        
        Args:
            embeddings: Optional embeddings model for semantic similarity
        """
        self.embeddings = embeddings
    
    def word_overlap_score(
        self,
        generated: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Calculate word overlap metrics between generated and reference.
        
        Args:
            generated: Generated answer
            reference: Reference/ground truth answer
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        # Normalize and tokenize
        gen_words = set(self._normalize_text(generated).split())
        ref_words = set(self._normalize_text(reference).split())
        
        if not gen_words or not ref_words:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        overlap = len(gen_words & ref_words)
        precision = overlap / len(gen_words) if gen_words else 0.0
        recall = overlap / len(ref_words) if ref_words else 0.0
        f1 = calculate_f1_score(precision, recall)
        
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text
    
    def ngram_overlap(
        self,
        generated: str,
        reference: str,
        n: int = 2
    ) -> float:
        """
        Calculate n-gram overlap (BLEU-like metric).
        
        Args:
            generated: Generated answer
            reference: Reference answer
            n: N-gram size
            
        Returns:
            N-gram overlap score
        """
        def get_ngrams(text: str, n: int) -> List[Tuple[str, ...]]:
            words = self._normalize_text(text).split()
            return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        gen_ngrams = get_ngrams(generated, n)
        ref_ngrams = get_ngrams(reference, n)
        
        if not gen_ngrams or not ref_ngrams:
            return 0.0
        
        gen_set = set(gen_ngrams)
        ref_set = set(ref_ngrams)
        overlap = len(gen_set & ref_set)
        
        return round(overlap / len(gen_set), 4) if gen_set else 0.0
    
    def faithfulness_score(
        self,
        answer: str,
        context: str
    ) -> float:
        """
        Estimate faithfulness - how well answer is grounded in context.
        Uses word overlap between answer and context as proxy.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Faithfulness score (0-1)
        """
        answer_words = set(self._normalize_text(answer).split())
        context_words = set(self._normalize_text(context).split())
        
        if not answer_words:
            return 0.0
        
        # Remove common stopwords for better signal
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'between', 'under', 'again', 'further',
                     'then', 'once', 'here', 'there', 'when', 'where', 'why',
                     'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                     'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                     'because', 'until', 'while', 'this', 'that', 'these', 'those'}
        
        answer_content = answer_words - stopwords
        context_content = context_words - stopwords
        
        if not answer_content:
            return 1.0  # Empty answer after stopword removal
        
        grounded = len(answer_content & context_content)
        return round(grounded / len(answer_content), 4)
    
    def answer_relevance_score(
        self,
        answer: str,
        question: str
    ) -> float:
        """
        Estimate answer relevance to the question.
        
        Args:
            answer: Generated answer
            question: Original question
            
        Returns:
            Relevance score (0-1)
        """
        # Extract key terms from question (nouns, verbs - simplified)
        question_words = set(self._normalize_text(question).split())
        answer_words = set(self._normalize_text(answer).split())
        
        # Question words that appear in answer (excluding stopwords)
        stopwords = {'what', 'who', 'where', 'when', 'why', 'how', 'which',
                     'is', 'are', 'was', 'were', 'do', 'does', 'did', 'can',
                     'could', 'would', 'should', 'the', 'a', 'an'}
        
        question_keywords = question_words - stopwords
        
        if not question_keywords:
            return 1.0
        
        covered = len(question_keywords & answer_words)
        return round(covered / len(question_keywords), 4)
    
    def length_score(
        self,
        generated: str,
        reference: str
    ) -> float:
        """
        Score based on length similarity.
        
        Args:
            generated: Generated answer
            reference: Reference answer
            
        Returns:
            Length similarity score (0-1)
        """
        gen_len = len(generated.split())
        ref_len = len(reference.split())
        
        if gen_len == 0 and ref_len == 0:
            return 1.0
        if gen_len == 0 or ref_len == 0:
            return 0.0
        
        return round(min(gen_len, ref_len) / max(gen_len, ref_len), 4)
    
    def evaluate(
        self,
        generated: str,
        reference: str,
        context: str,
        question: str
    ) -> Dict[str, float]:
        """
        Run all answer evaluation metrics.
        
        Args:
            generated: Generated answer
            reference: Ground truth answer
            context: Retrieved context
            question: Original question
            
        Returns:
            Dictionary of all metric scores
        """
        word_overlap = self.word_overlap_score(generated, reference)
        
        return {
            'word_overlap_precision': word_overlap['precision'],
            'word_overlap_recall': word_overlap['recall'],
            'word_overlap_f1': word_overlap['f1'],
            'bigram_overlap': self.ngram_overlap(generated, reference, n=2),
            'trigram_overlap': self.ngram_overlap(generated, reference, n=3),
            'faithfulness': self.faithfulness_score(generated, context),
            'answer_relevance': self.answer_relevance_score(generated, question),
            'length_similarity': self.length_score(generated, reference)
        }


class RAGEvaluator:
    """
    Comprehensive RAG system evaluator.
    
    Evaluates:
    - Retrieval quality (Precision@K, Recall@K, MRR, nDCG)
    - Answer quality (Faithfulness, Relevance, Word Overlap)
    - End-to-end performance
    """
    
    def __init__(
        self,
        qa_system=None,
        config_path: str = "config.yaml"
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            qa_system: LLMQASystem instance to evaluate
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.logger = get_logger(__name__)
        
        self.qa_system = qa_system
        self.answer_evaluator = AnswerEvaluator()
        
        self.results: List[EvaluationResult] = []
        self.evaluation_config = {
            'top_k': self.config.get('retrieval.top_k', 5),
            'metrics': self.config.get('evaluation.metrics', [])
        }
        
        self.logger.info("RAGEvaluator initialized")
    
    def set_qa_system(self, qa_system) -> None:
        """Set or update the QA system to evaluate."""
        self.qa_system = qa_system
        self.logger.info("QA system updated")
    
    def create_evaluation_dataset(
        self,
        questions: List[str],
        ground_truths: List[str],
        expected_sources: Optional[List[List[str]]] = None
    ) -> List[EvaluationSample]:
        """
        Create an evaluation dataset from lists.
        
        Args:
            questions: List of test questions
            ground_truths: List of expected answers
            expected_sources: Optional list of expected sources per question
            
        Returns:
            List of EvaluationSample objects
        """
        if len(questions) != len(ground_truths):
            raise ValueError("Questions and ground truths must have same length")
        
        if expected_sources is None:
            expected_sources = [[] for _ in questions]
        
        samples = [
            EvaluationSample(
                question=q,
                ground_truth=gt,
                expected_sources=es
            )
            for q, gt, es in zip(questions, ground_truths, expected_sources)
        ]
        
        self.logger.info(f"Created evaluation dataset with {len(samples)} samples")
        return samples
    
    def load_evaluation_dataset(self, filepath: str) -> List[EvaluationSample]:
        """
        Load evaluation dataset from JSON file.
        
        Expected format:
        [
            {
                "question": "...",
                "ground_truth": "...",
                "expected_sources": ["doc1.pdf", ...]
            },
            ...
        ]
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of EvaluationSample objects
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = [
            EvaluationSample(
                question=item['question'],
                ground_truth=item['ground_truth'],
                expected_sources=item.get('expected_sources', []),
                metadata=item.get('metadata', {})
            )
            for item in data
        ]
        
        self.logger.info(f"Loaded {len(samples)} samples from {filepath}")
        return samples
    
    def evaluate_retrieval(
        self,
        question: str,
        expected_sources: List[str],
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval quality for a single question.
        
        Args:
            question: Test question
            expected_sources: List of expected relevant source names
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with metrics and retrieved documents
        """
        if self.qa_system is None:
            raise ValueError("QA system not set")
        
        # Retrieve documents
        retrieved = self.qa_system.retriever.retrieve(question, top_k=k)
        retrieved_sources = [
            r['metadata'].get('source', '') for r in retrieved
        ]
        
        # Normalize source names for comparison (handle path differences)
        def normalize_source(s):
            """Extract just the filename from a path."""
            return Path(s).name.lower() if s else ""
        
        normalized_expected = [normalize_source(s) for s in expected_sources]
        normalized_retrieved = [normalize_source(s) for s in retrieved_sources]
        
        # Calculate retrieval metrics
        metrics = RetrievalMetrics.evaluate(
            relevant_docs=normalized_expected,
            retrieved_docs=normalized_retrieved,
            k=k
        )
        
        # Get context string
        context = self.qa_system.retriever.get_context_for_llm(question, top_k=k)
        
        return {
            'metrics': metrics,
            'retrieved_sources': retrieved_sources,
            'context': context,
            'retrieved_docs': retrieved
        }
    
    def evaluate_sample(
        self,
        sample: EvaluationSample,
        k: int = 5
    ) -> EvaluationResult:
        """
        Evaluate a single sample end-to-end.
        
        Args:
            sample: EvaluationSample to evaluate
            k: Number of documents to retrieve
            
        Returns:
            Complete EvaluationResult
        """
        import time
        
        self.logger.info(f"Evaluating: '{sample.question[:50]}...'")
        start_time = time.time()
        
        # Step 1: Evaluate retrieval
        retrieval_result = self.evaluate_retrieval(
            sample.question,
            sample.expected_sources,
            k=k
        )
        
        # Step 2: Generate answer
        qa_result = self.qa_system.answer(sample.question, return_sources=True)
        generated_answer = qa_result['answer']
        
        # Step 3: Evaluate answer quality
        answer_metrics = self.answer_evaluator.evaluate(
            generated=generated_answer,
            reference=sample.ground_truth,
            context=retrieval_result['context'],
            question=sample.question
        )
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        return EvaluationResult(
            question=sample.question,
            generated_answer=generated_answer,
            ground_truth=sample.ground_truth,
            retrieved_context=retrieval_result['context'],
            retrieval_metrics=retrieval_result['metrics'],
            answer_metrics=answer_metrics,
            sources_retrieved=retrieval_result['retrieved_sources'],
            latency_ms=round(latency, 2)
        )
    
    def evaluate_dataset(
        self,
        dataset: List[EvaluationSample],
        k: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate entire dataset.
        
        Args:
            dataset: List of EvaluationSample objects
            k: Number of documents to retrieve
            verbose: Whether to print progress
            
        Returns:
            Aggregated evaluation results
        """
        self.results = []
        
        for i, sample in enumerate(dataset):
            if verbose:
                print(f"Evaluating sample {i+1}/{len(dataset)}...")
            
            result = self.evaluate_sample(sample, k)
            self.results.append(result)
        
        aggregated = self._aggregate_results()
        self.logger.info(f"Evaluation complete: {len(self.results)} samples")
        
        return aggregated
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all evaluated samples."""
        if not self.results:
            return {'error': 'No results to aggregate'}
        
        n = len(self.results)
        
        # Aggregate retrieval metrics
        retrieval_keys = list(self.results[0].retrieval_metrics.keys())
        avg_retrieval = {}
        for key in retrieval_keys:
            values = [r.retrieval_metrics.get(key, 0) for r in self.results]
            avg_retrieval[key] = round(sum(values) / n, 4)
        
        # Aggregate answer metrics
        answer_keys = list(self.results[0].answer_metrics.keys())
        avg_answer = {}
        for key in answer_keys:
            values = [r.answer_metrics.get(key, 0) for r in self.results]
            avg_answer[key] = round(sum(values) / n, 4)
        
        # Latency stats
        latencies = [r.latency_ms for r in self.results]
        avg_latency = sum(latencies) / n
        
        return {
            'summary': {
                'num_samples': n,
                'avg_latency_ms': round(avg_latency, 2),
                'evaluation_date': datetime.now().isoformat()
            },
            'retrieval_metrics': avg_retrieval,
            'answer_metrics': avg_answer,
            'individual_results': [
                {
                    'question': r.question,
                    'generated_answer': r.generated_answer,
                    'ground_truth': r.ground_truth,
                    'sources': r.sources_retrieved,
                    'retrieval_metrics': r.retrieval_metrics,
                    'answer_metrics': r.answer_metrics,
                    'latency_ms': r.latency_ms
                }
                for r in self.results
            ]
        }
    
    def save_results(self, filepath: str) -> None:
        """Save evaluation results to JSON file."""
        results = self._aggregate_results()
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def print_summary(self) -> None:
        """Print formatted evaluation summary to console."""
        results = self._aggregate_results()
        
        print("\n" + "=" * 70)
        print("                    RAG EVALUATION SUMMARY")
        print("=" * 70)
        
        summary = results.get('summary', {})
        print(f"\nSamples Evaluated: {summary.get('num_samples', 0)}")
        print(f"Average Latency: {summary.get('avg_latency_ms', 0):.2f} ms")
        print(f"Evaluation Date: {summary.get('evaluation_date', 'N/A')}")
        
        print("\n" + "-" * 70)
        print("RETRIEVAL METRICS")
        print("-" * 70)
        for key, value in results.get('retrieval_metrics', {}).items():
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            print(f"  {key:20} {bar} {value:.4f}")
        
        print("\n" + "-" * 70)
        print("ANSWER QUALITY METRICS")
        print("-" * 70)
        for key, value in results.get('answer_metrics', {}).items():
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            print(f"  {key:20} {bar} {value:.4f}")
        
        print("\n" + "=" * 70)
    
    def print_detailed_results(self) -> None:
        """Print detailed results for each sample."""
        if not self.results:
            print("No results available.")
            return
        
        for i, r in enumerate(self.results, 1):
            print(f"\n{'='*70}")
            print(f"SAMPLE {i}")
            print(f"{'='*70}")
            print(f"\nQuestion: {r.question}")
            print(f"\nGround Truth: {r.ground_truth}")
            print(f"\nGenerated Answer: {r.generated_answer}")
            print(f"\nSources Retrieved: {', '.join(r.sources_retrieved)}")
            print(f"\nRetrieval Metrics: {r.retrieval_metrics}")
            print(f"Answer Metrics: {r.answer_metrics}")
            print(f"Latency: {r.latency_ms:.2f} ms")
