"""Enhanced Quality Evaluation for RAG System - Factuality, Coherence, Precision."""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

from langchain_huggingface import HuggingFaceEmbeddings

from utils import get_config, get_logger


@dataclass
class QualityMetrics:
    """Complete quality metrics for a single evaluation."""
    # Factuality metrics
    claim_coverage: float
    hallucination_score: float
    source_grounding: float
    
    # Coherence metrics
    sentence_connectivity: float
    topic_consistency: float
    structure_score: float
    
    # Precision metrics
    semantic_similarity: float
    key_entity_coverage: float
    answer_completeness: float
    
    def to_dict(self) -> dict:
        return {
            'factuality': {
                'claim_coverage': round(self.claim_coverage, 4),
                'hallucination_score': round(self.hallucination_score, 4),
                'source_grounding': round(self.source_grounding, 4),
                'factuality_avg': round((self.claim_coverage + self.hallucination_score + self.source_grounding) / 3, 4)
            },
            'coherence': {
                'sentence_connectivity': round(self.sentence_connectivity, 4),
                'topic_consistency': round(self.topic_consistency, 4),
                'structure_score': round(self.structure_score, 4),
                'coherence_avg': round((self.sentence_connectivity + self.topic_consistency + self.structure_score) / 3, 4)
            },
            'precision': {
                'semantic_similarity': round(self.semantic_similarity, 4),
                'key_entity_coverage': round(self.key_entity_coverage, 4),
                'answer_completeness': round(self.answer_completeness, 4),
                'precision_avg': round((self.semantic_similarity + self.key_entity_coverage + self.answer_completeness) / 3, 4)
            }
        }


class QualityEvaluator:
    """
    Evaluates quality of RAG responses with detailed metrics.
    
    Metrics Categories:
    1. Factuality - How well grounded in source documents
    2. Coherence - Logical flow and structure
    3. Precision - Accuracy and completeness
    """
    
    def __init__(
        self,
        embeddings: Optional[HuggingFaceEmbeddings] = None,
        config_path: str = "config.yaml"
    ):
        """
        Initialize the quality evaluator.
        
        Args:
            embeddings: Optional embeddings model for semantic similarity
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.logger = get_logger(__name__)
        
        # Initialize or load embeddings
        if embeddings is None:
            model_name = self.config.get(
                'embeddings.model_name',
                'sentence-transformers/all-MiniLM-L6-v2'
            )
            device = self.config.get('embeddings.device', 'cpu')
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            self.embeddings = embeddings
        
        # Common stop words for text processing
        self.stop_words = set([
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'and', 'but', 'if', 'or', 'because', 'until', 'while', 'although',
            'though', 'after', 'before', 'this', 'that', 'these', 'those', 'it',
            'its', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their'
        ])
        
        self.logger.info("QualityEvaluator initialized")
    
    # =========================================================================
    # FACTUALITY METRICS
    # =========================================================================
    
    def evaluate_factuality(
        self,
        answer: str,
        context: str,
        question: str
    ) -> Dict[str, float]:
        """
        Measure how well the answer is grounded in source documents.
        
        Args:
            answer: Generated answer
            context: Retrieved context from documents
            question: Original question
            
        Returns:
            Dictionary with factuality metrics
        """
        # Extract claims from answer (simple sentence-based extraction)
        claims = self._extract_claims(answer)
        
        if not claims:
            return {
                'claim_coverage': 0.0,
                'hallucination_score': 1.0,  # 1.0 = no hallucinations, 0.0 = all hallucinations
                'source_grounding': 0.0
            }
        
        # Check each claim against context
        supported_claims = 0
        for claim in claims:
            if self._is_claim_supported(claim, context):
                supported_claims += 1
        
        claim_coverage = supported_claims / len(claims)
        
        # Hallucination score (inverse of unsupported claims)
        hallucination_score = claim_coverage  # Higher = fewer hallucinations
        
        # Source grounding (word overlap with context)
        source_grounding = self._calculate_source_grounding(answer, context)
        
        return {
            'claim_coverage': claim_coverage,
            'hallucination_score': hallucination_score,
            'source_grounding': source_grounding
        }
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract individual claims from text (sentence-based)."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Filter meaningful sentences (at least 3 words)
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) >= 3:
                claims.append(sent)
        
        return claims
    
    def _is_claim_supported(self, claim: str, context: str, threshold: float = 0.3) -> bool:
        """Check if a claim is supported by the context using word overlap."""
        claim_words = self._get_content_words(claim)
        context_words = self._get_content_words(context)
        
        if not claim_words:
            return True  # Empty claim is trivially supported
        
        # Calculate overlap
        overlap = len(claim_words & context_words) / len(claim_words)
        return overlap >= threshold
    
    def _get_content_words(self, text: str) -> set:
        """Extract content words (non-stopwords) from text."""
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        return set(w for w in words if w not in self.stop_words)
    
    def _calculate_source_grounding(self, answer: str, context: str) -> float:
        """Calculate how much of the answer is grounded in the source."""
        answer_words = self._get_content_words(answer)
        context_words = self._get_content_words(context)
        
        if not answer_words:
            return 0.0
        
        grounded_words = answer_words & context_words
        return len(grounded_words) / len(answer_words)
    
    # =========================================================================
    # COHERENCE METRICS
    # =========================================================================
    
    def evaluate_coherence(
        self,
        answer: str,
        question: str
    ) -> Dict[str, float]:
        """
        Measure logical flow and structure of the answer.
        
        Args:
            answer: Generated answer
            question: Original question
            
        Returns:
            Dictionary with coherence metrics
        """
        # Sentence connectivity
        sentence_connectivity = self._calculate_sentence_connectivity(answer)
        
        # Topic consistency
        topic_consistency = self._calculate_topic_consistency(answer, question)
        
        # Structure score
        structure_score = self._calculate_structure_score(answer)
        
        return {
            'sentence_connectivity': sentence_connectivity,
            'topic_consistency': topic_consistency,
            'structure_score': structure_score
        }
    
    def _calculate_sentence_connectivity(self, text: str) -> float:
        """
        Calculate how well sentences connect to each other.
        Uses embedding similarity between consecutive sentences.
        """
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if len(sentences) <= 1:
            return 1.0  # Single sentence is perfectly connected
        
        # Get embeddings for all sentences
        try:
            embeddings = self.embeddings.embed_documents(sentences)
            
            # Calculate similarity between consecutive sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                similarities.append(sim)
            
            return sum(similarities) / len(similarities) if similarities else 1.0
        except Exception as e:
            self.logger.warning(f"Error calculating sentence connectivity: {e}")
            return 0.5  # Default middle score on error
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _calculate_topic_consistency(self, answer: str, question: str) -> float:
        """
        Calculate how well the answer stays on topic with the question.
        """
        try:
            answer_embedding = self.embeddings.embed_query(answer)
            question_embedding = self.embeddings.embed_query(question)
            
            return self._cosine_similarity(answer_embedding, question_embedding)
        except Exception as e:
            self.logger.warning(f"Error calculating topic consistency: {e}")
            return 0.5
    
    def _calculate_structure_score(self, text: str) -> float:
        """
        Evaluate the structural quality of the answer.
        Considers: sentence length variation, paragraph structure, list usage.
        """
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if not sentences:
            return 0.0
        
        scores = []
        
        # 1. Sentence length appropriateness (not too long, not too short)
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        # Ideal range: 10-25 words per sentence
        if 10 <= avg_length <= 25:
            length_score = 1.0
        elif 5 <= avg_length <= 35:
            length_score = 0.7
        else:
            length_score = 0.4
        scores.append(length_score)
        
        # 2. Sentence length variation (some variation is good)
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences]
            std_dev = np.std(lengths)
            # Moderate variation is good (std between 3-10)
            if 3 <= std_dev <= 10:
                variation_score = 1.0
            elif 1 <= std_dev <= 15:
                variation_score = 0.7
            else:
                variation_score = 0.5
            scores.append(variation_score)
        
        # 3. Proper capitalization at sentence starts
        cap_score = sum(1 for s in sentences if s and s[0].isupper()) / len(sentences)
        scores.append(cap_score)
        
        # 4. Not too repetitive (check for repeated n-grams)
        words = text.lower().split()
        if len(words) >= 10:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            unique_ratio = len(set(trigrams)) / len(trigrams) if trigrams else 1.0
            scores.append(unique_ratio)
        
        return sum(scores) / len(scores)
    
    # =========================================================================
    # PRECISION METRICS
    # =========================================================================
    
    def evaluate_precision(
        self,
        answer: str,
        ground_truth: str,
        question: str
    ) -> Dict[str, float]:
        """
        Measure answer accuracy and completeness.
        
        Args:
            answer: Generated answer
            ground_truth: Reference/expected answer
            question: Original question
            
        Returns:
            Dictionary with precision metrics
        """
        # Semantic similarity to ground truth
        semantic_similarity = self._calculate_semantic_similarity(answer, ground_truth)
        
        # Key entity coverage
        key_entity_coverage = self._calculate_entity_coverage(answer, ground_truth)
        
        # Answer completeness
        answer_completeness = self._calculate_completeness(answer, ground_truth, question)
        
        return {
            'semantic_similarity': semantic_similarity,
            'key_entity_coverage': key_entity_coverage,
            'answer_completeness': answer_completeness
        }
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using embeddings."""
        try:
            emb1 = self.embeddings.embed_query(text1)
            emb2 = self.embeddings.embed_query(text2)
            return self._cosine_similarity(emb1, emb2)
        except Exception as e:
            self.logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _calculate_entity_coverage(self, answer: str, ground_truth: str) -> float:
        """Calculate coverage of key entities from ground truth in answer."""
        # Extract potential entities (capitalized words, numbers, quoted terms)
        gt_entities = self._extract_entities(ground_truth)
        answer_lower = answer.lower()
        
        if not gt_entities:
            return 1.0  # No entities to cover
        
        covered = sum(1 for entity in gt_entities if entity.lower() in answer_lower)
        return covered / len(gt_entities)
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text."""
        entities = []
        
        # Capitalized words (potential named entities)
        cap_words = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
        entities.extend(cap_words)
        
        # Numbers and percentages
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        entities.extend(numbers)
        
        # Quoted terms
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted)
        
        # Technical terms (multi-word capitalized)
        tech_terms = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', text)
        entities.extend(tech_terms)
        
        return list(set(entities))
    
    def _calculate_completeness(
        self,
        answer: str,
        ground_truth: str,
        question: str
    ) -> float:
        """
        Calculate how complete the answer is relative to ground truth.
        """
        # Get content words from ground truth
        gt_words = self._get_content_words(ground_truth)
        answer_words = self._get_content_words(answer)
        
        if not gt_words:
            return 1.0
        
        # What fraction of ground truth content is in the answer
        covered = gt_words & answer_words
        coverage = len(covered) / len(gt_words)
        
        # Check if question keywords are addressed
        question_words = self._get_content_words(question)
        if question_words:
            q_coverage = len(question_words & answer_words) / len(question_words)
            # Average of both measures
            return (coverage + q_coverage) / 2
        
        return coverage
    
    # =========================================================================
    # COMBINED EVALUATION
    # =========================================================================
    
    def evaluate(
        self,
        answer: str,
        ground_truth: str,
        context: str,
        question: str
    ) -> QualityMetrics:
        """
        Run complete quality evaluation.
        
        Args:
            answer: Generated answer
            ground_truth: Reference/expected answer
            context: Retrieved context from documents
            question: Original question
            
        Returns:
            QualityMetrics with all evaluation scores
        """
        self.logger.debug(f"Evaluating quality for question: '{question[:50]}...'")
        
        # Factuality
        factuality = self.evaluate_factuality(answer, context, question)
        
        # Coherence
        coherence = self.evaluate_coherence(answer, question)
        
        # Precision
        precision = self.evaluate_precision(answer, ground_truth, question)
        
        return QualityMetrics(
            # Factuality
            claim_coverage=factuality['claim_coverage'],
            hallucination_score=factuality['hallucination_score'],
            source_grounding=factuality['source_grounding'],
            # Coherence
            sentence_connectivity=coherence['sentence_connectivity'],
            topic_consistency=coherence['topic_consistency'],
            structure_score=coherence['structure_score'],
            # Precision
            semantic_similarity=precision['semantic_similarity'],
            key_entity_coverage=precision['key_entity_coverage'],
            answer_completeness=precision['answer_completeness']
        )
    
    def evaluate_batch(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of results and return aggregated metrics.
        
        Args:
            results: List of result dictionaries with 'answer', 'ground_truth', 
                    'context', and 'question' keys
                    
        Returns:
            Aggregated metrics across all samples
        """
        all_metrics = []
        
        for result in results:
            metrics = self.evaluate(
                answer=result.get('answer', ''),
                ground_truth=result.get('ground_truth', ''),
                context=result.get('context', ''),
                question=result.get('question', '')
            )
            all_metrics.append(metrics)
        
        # Aggregate
        if not all_metrics:
            return {}
        
        aggregated = {
            'factuality': {
                'claim_coverage': np.mean([m.claim_coverage for m in all_metrics]),
                'hallucination_score': np.mean([m.hallucination_score for m in all_metrics]),
                'source_grounding': np.mean([m.source_grounding for m in all_metrics]),
            },
            'coherence': {
                'sentence_connectivity': np.mean([m.sentence_connectivity for m in all_metrics]),
                'topic_consistency': np.mean([m.topic_consistency for m in all_metrics]),
                'structure_score': np.mean([m.structure_score for m in all_metrics]),
            },
            'precision': {
                'semantic_similarity': np.mean([m.semantic_similarity for m in all_metrics]),
                'key_entity_coverage': np.mean([m.key_entity_coverage for m in all_metrics]),
                'answer_completeness': np.mean([m.answer_completeness for m in all_metrics]),
            },
            'overall': {
                'factuality_avg': np.mean([
                    (m.claim_coverage + m.hallucination_score + m.source_grounding) / 3 
                    for m in all_metrics
                ]),
                'coherence_avg': np.mean([
                    (m.sentence_connectivity + m.topic_consistency + m.structure_score) / 3 
                    for m in all_metrics
                ]),
                'precision_avg': np.mean([
                    (m.semantic_similarity + m.key_entity_coverage + m.answer_completeness) / 3 
                    for m in all_metrics
                ]),
            }
        }
        
        # Add overall average
        aggregated['overall']['quality_score'] = np.mean([
            aggregated['overall']['factuality_avg'],
            aggregated['overall']['coherence_avg'],
            aggregated['overall']['precision_avg']
        ])
        
        # Round all values
        for category in aggregated:
            for metric in aggregated[category]:
                aggregated[category][metric] = round(aggregated[category][metric], 4)
        
        return aggregated
    
    def print_metrics(self, metrics: QualityMetrics) -> None:
        """Print formatted quality metrics."""
        print("\n" + "=" * 60)
        print("QUALITY METRICS")
        print("=" * 60)
        
        metrics_dict = metrics.to_dict()
        
        for category, values in metrics_dict.items():
            print(f"\n📊 {category.upper()}")
            print("-" * 40)
            for metric, score in values.items():
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"  {metric:25s} {bar} {score:.4f}")
        
        print("\n" + "=" * 60)
