"""Evaluation metrics for the RAG system (Q4)."""

from typing import List, Dict, Any, Optional
import math


def calculate_precision_at_k(
    relevant_docs: List[str],
    retrieved_docs: List[str],
    k: int
) -> float:
    """
    Calculate Precision@K for retrieval evaluation.
    
    Precision@K = (# of relevant docs in top K) / K
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        k: Number of top documents to consider
        
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k <= 0 or not retrieved_docs:
        return 0.0
    
    top_k = retrieved_docs[:k]
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
    
    return relevant_in_top_k / k


def calculate_recall_at_k(
    relevant_docs: List[str],
    retrieved_docs: List[str],
    k: int
) -> float:
    """
    Calculate Recall@K for retrieval evaluation.
    
    Recall@K = (# of relevant docs in top K) / (total # of relevant docs)
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        k: Number of top documents to consider
        
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not relevant_docs:
        return 0.0
    
    top_k = retrieved_docs[:k]
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
    
    return relevant_in_top_k / len(relevant_docs)


def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        precision: Precision score
        recall: Recall score
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_mrr(
    relevant_docs: List[str],
    retrieved_docs: List[str]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR = 1 / (rank of first relevant document)
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    for rank, doc in enumerate(retrieved_docs, 1):
        if doc in relevant_docs:
            return 1.0 / rank
    return 0.0


def calculate_average_precision(
    relevant_docs: List[str],
    retrieved_docs: List[str]
) -> float:
    """
    Calculate Average Precision (AP).
    
    AP = sum(P@k * rel(k)) / |relevant docs|
    where P@k is precision at k and rel(k) is 1 if doc at k is relevant
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        
    Returns:
        Average Precision score (0.0 to 1.0)
    """
    if not relevant_docs:
        return 0.0
    
    relevant_set = set(relevant_docs)
    precisions = []
    relevant_count = 0
    
    for k, doc in enumerate(retrieved_docs, 1):
        if doc in relevant_set:
            relevant_count += 1
            precision_at_k = relevant_count / k
            precisions.append(precision_at_k)
    
    if not precisions:
        return 0.0
    
    return sum(precisions) / len(relevant_docs)


def calculate_dcg_at_k(
    relevance_scores: List[float],
    k: int
) -> float:
    """
    Calculate Discounted Cumulative Gain at K.
    
    DCG@K = sum(rel_i / log2(i + 1)) for i in 1..K
    
    Args:
        relevance_scores: Relevance scores for retrieved documents (in order)
        k: Number of top documents to consider
        
    Returns:
        DCG@K score
    """
    if not relevance_scores or k <= 0:
        return 0.0
    
    scores = relevance_scores[:k]
    dcg = scores[0] if scores else 0.0
    
    for i, score in enumerate(scores[1:], 2):
        dcg += score / math.log2(i + 1)
    
    return dcg


def calculate_ndcg_at_k(
    relevance_scores: List[float],
    k: int
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    
    nDCG@K = DCG@K / IDCG@K
    where IDCG is the ideal DCG (with perfectly sorted relevance)
    
    Args:
        relevance_scores: Relevance scores for retrieved documents (in order)
        k: Number of top documents to consider
        
    Returns:
        nDCG@K score (0.0 to 1.0)
    """
    if not relevance_scores or k <= 0:
        return 0.0
    
    # Calculate actual DCG
    dcg = calculate_dcg_at_k(relevance_scores, k)
    
    # Calculate ideal DCG (with sorted relevance scores)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg_at_k(ideal_scores, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_hit_rate_at_k(
    relevant_docs: List[str],
    retrieved_docs: List[str],
    k: int
) -> float:
    """
    Calculate Hit Rate at K (whether any relevant doc is in top K).
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs
        k: Number of top documents to consider
        
    Returns:
        1.0 if any relevant doc in top K, else 0.0
    """
    if not relevant_docs or not retrieved_docs:
        return 0.0
    
    top_k = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    
    return 1.0 if top_k & relevant_set else 0.0


class RetrievalMetrics:
    """
    Collection of retrieval evaluation metrics.
    
    Provides unified interface for computing all retrieval metrics.
    """
    
    @staticmethod
    def evaluate(
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int = 5,
        relevance_scores: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Calculate all retrieval metrics.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs (ordered by relevance)
            k: Number of top documents for @K metrics
            relevance_scores: Optional relevance scores for nDCG calculation
            
        Returns:
            Dictionary of all metric scores
        """
        precision = calculate_precision_at_k(relevant_docs, retrieved_docs, k)
        recall = calculate_recall_at_k(relevant_docs, retrieved_docs, k)
        f1 = calculate_f1_score(precision, recall)
        mrr = calculate_mrr(relevant_docs, retrieved_docs)
        ap = calculate_average_precision(relevant_docs, retrieved_docs)
        hit_rate = calculate_hit_rate_at_k(relevant_docs, retrieved_docs, k)
        
        metrics = {
            f'precision@{k}': round(precision, 4),
            f'recall@{k}': round(recall, 4),
            f'f1@{k}': round(f1, 4),
            'mrr': round(mrr, 4),
            'map': round(ap, 4),
            f'hit_rate@{k}': round(hit_rate, 4)
        }
        
        # Add nDCG if relevance scores provided
        if relevance_scores:
            ndcg = calculate_ndcg_at_k(relevance_scores, k)
            metrics[f'ndcg@{k}'] = round(ndcg, 4)
        
        return metrics
    
    @staticmethod
    def evaluate_batch(
        queries_relevant: List[List[str]],
        queries_retrieved: List[List[str]],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Calculate average metrics across multiple queries.
        
        Args:
            queries_relevant: List of relevant docs per query
            queries_retrieved: List of retrieved docs per query
            k: Number of top documents for @K metrics
            
        Returns:
            Dictionary of averaged metric scores
        """
        if len(queries_relevant) != len(queries_retrieved):
            raise ValueError("Number of queries must match")
        
        all_metrics = []
        for relevant, retrieved in zip(queries_relevant, queries_retrieved):
            metrics = RetrievalMetrics.evaluate(relevant, retrieved, k)
            all_metrics.append(metrics)
        
        # Average all metrics
        if not all_metrics:
            return {}
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = round(sum(values) / len(values), 4)
        
        return avg_metrics
