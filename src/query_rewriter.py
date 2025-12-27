"""Query Rewriting Module for RAG - Improve retrieval through query reformulation."""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from utils import get_config, get_logger


class RewriteStrategy(Enum):
    """Available query rewriting strategies."""
    EXPAND = "expand"
    DECOMPOSE = "decompose"
    HYDE = "hyde"
    STEPBACK = "stepback"
    AUTO = "auto"


@dataclass
class RewriteResult:
    """Result of a query rewriting operation."""
    original_query: str
    rewritten_queries: List[str]
    strategy_used: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> dict:
        return {
            'original_query': self.original_query,
            'rewritten_queries': self.rewritten_queries,
            'strategy_used': self.strategy_used,
            'metadata': self.metadata or {}
        }


# Prompt templates for different strategies
EXPANSION_PROMPT = """Given a search query, expand it by adding synonyms, related terms, and alternative phrasings to improve search results.

Original Query: {query}

Generate an expanded version of the query that includes:
1. The original query
2. Key synonyms for important terms
3. Related technical terms or concepts
4. Alternative phrasings

Provide ONLY the expanded query, no explanations. Keep it concise (max 50 words).

Expanded Query:"""


DECOMPOSITION_PROMPT = """Break down this complex query into simpler, focused sub-queries that can be answered individually.

Original Query: {query}

Generate 2-4 simpler sub-queries that together address the original question. Each sub-query should:
1. Be self-contained and answerable independently
2. Focus on one specific aspect
3. Be clear and concise

Format: One query per line, numbered (1., 2., etc.)

Sub-queries:"""


HYDE_PROMPT = """Generate a hypothetical passage that would perfectly answer this question. 
Write as if you're quoting from an authoritative source document.

Question: {query}

Write a detailed, factual passage (2-3 sentences) that directly answers this question. 
Include specific technical details and terminology that would appear in a real document.
DO NOT include phrases like "I think" or "might be" - write definitively.

Hypothetical Passage:"""


STEPBACK_PROMPT = """Given a specific question, generate a more general, abstract version of the question.
This broader question can help retrieve more comprehensive context.

Specific Question: {query}

Generate a broader, more abstract question that:
1. Covers the general topic/concept behind the specific question
2. Would retrieve background information helpful for answering the specific question
3. Uses more general terminology

Provide ONLY the step-back question, no explanations.

Step-back Question:"""


class QueryRewriter:
    """
    Rewrites queries to improve retrieval performance.
    
    Strategies:
    - EXPAND: Add synonyms and related terms
    - DECOMPOSE: Break complex queries into simpler sub-queries
    - HYDE: Hypothetical Document Embedding - generate hypothetical answer
    - STEPBACK: Abstract to broader context first
    - AUTO: Automatically select best strategy
    """
    
    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        config_path: str = "config.yaml"
    ):
        """
        Initialize the query rewriter.
        
        Args:
            llm: Optional LLM instance (uses Ollama by default)
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.logger = get_logger(__name__)
        
        # LLM for rewriting
        self._llm = llm
        
        # Load configuration
        qr_config = self.config.get('query_rewriting', {})
        self.enabled = qr_config.get('enabled', True)
        self.default_strategy = qr_config.get('default_strategy', 'auto')
        self.hyde_max_tokens = qr_config.get('hyde', {}).get('max_tokens', 256)
        self.hyde_temperature = qr_config.get('hyde', {}).get('temperature', 0.3)
        self.expansion_max_terms = qr_config.get('expansion', {}).get('max_terms', 5)
        
        # Initialize prompt templates
        self.prompts = {
            'expand': PromptTemplate(template=EXPANSION_PROMPT, input_variables=['query']),
            'decompose': PromptTemplate(template=DECOMPOSITION_PROMPT, input_variables=['query']),
            'hyde': PromptTemplate(template=HYDE_PROMPT, input_variables=['query']),
            'stepback': PromptTemplate(template=STEPBACK_PROMPT, input_variables=['query'])
        }
        
        self.logger.info("QueryRewriter initialized")
    
    @property
    def llm(self) -> OllamaLLM:
        """Lazy loading of LLM."""
        if self._llm is None:
            model_name = self.config.get('llm.model_name', 'qwen2.5:1.5b')
            base_url = self.config.get('llm.base_url', 'http://localhost:11434')
            
            self._llm = OllamaLLM(
                model=model_name,
                base_url=base_url,
                temperature=0.3,  # Lower temperature for more focused rewriting
            )
            self.logger.info(f"LLM loaded: {model_name}")
        
        return self._llm
    
    def expand_query(self, query: str) -> RewriteResult:
        """
        Expand query with synonyms and related terms.
        
        Args:
            query: Original search query
            
        Returns:
            RewriteResult with expanded query
        """
        self.logger.info(f"Expanding query: '{query[:50]}...'")
        
        try:
            prompt = self.prompts['expand'].format(query=query)
            expanded = self.llm.invoke(prompt).strip()
            
            # Clean up the response
            expanded = self._clean_response(expanded)
            
            return RewriteResult(
                original_query=query,
                rewritten_queries=[expanded],
                strategy_used='expand',
                metadata={'expansion_source': 'llm'}
            )
        except Exception as e:
            self.logger.error(f"Expansion failed: {e}")
            # Fallback: return original query
            return RewriteResult(
                original_query=query,
                rewritten_queries=[query],
                strategy_used='expand',
                metadata={'error': str(e), 'fallback': True}
            )
    
    def decompose_query(self, query: str) -> RewriteResult:
        """
        Break complex query into simpler sub-queries.
        
        Args:
            query: Complex query to decompose
            
        Returns:
            RewriteResult with list of sub-queries
        """
        self.logger.info(f"Decomposing query: '{query[:50]}...'")
        
        try:
            prompt = self.prompts['decompose'].format(query=query)
            response = self.llm.invoke(prompt).strip()
            
            # Parse numbered sub-queries
            sub_queries = self._parse_numbered_list(response)
            
            if not sub_queries:
                sub_queries = [query]  # Fallback to original
            
            return RewriteResult(
                original_query=query,
                rewritten_queries=sub_queries,
                strategy_used='decompose',
                metadata={'num_sub_queries': len(sub_queries)}
            )
        except Exception as e:
            self.logger.error(f"Decomposition failed: {e}")
            return RewriteResult(
                original_query=query,
                rewritten_queries=[query],
                strategy_used='decompose',
                metadata={'error': str(e), 'fallback': True}
            )
    
    def hyde_rewrite(self, query: str) -> RewriteResult:
        """
        Hypothetical Document Embedding (HyDE).
        Generate a hypothetical answer to use for retrieval.
        
        This technique generates a hypothetical passage that would answer the query,
        then uses that passage for similarity search instead of the original query.
        
        Args:
            query: Original question
            
        Returns:
            RewriteResult with hypothetical document
        """
        self.logger.info(f"Generating HyDE for query: '{query[:50]}...'")
        
        try:
            prompt = self.prompts['hyde'].format(query=query)
            hypothetical = self.llm.invoke(prompt).strip()
            
            # Clean up the response
            hypothetical = self._clean_response(hypothetical)
            
            return RewriteResult(
                original_query=query,
                rewritten_queries=[hypothetical],
                strategy_used='hyde',
                metadata={
                    'hypothetical_doc_length': len(hypothetical),
                    'original_query_preserved': True
                }
            )
        except Exception as e:
            self.logger.error(f"HyDE generation failed: {e}")
            return RewriteResult(
                original_query=query,
                rewritten_queries=[query],
                strategy_used='hyde',
                metadata={'error': str(e), 'fallback': True}
            )
    
    def stepback_rewrite(self, query: str) -> RewriteResult:
        """
        Step-back prompting for more abstract queries.
        
        Generates a broader question that can retrieve more comprehensive context.
        
        Args:
            query: Original specific question
            
        Returns:
            RewriteResult with step-back query
        """
        self.logger.info(f"Generating step-back query: '{query[:50]}...'")
        
        try:
            prompt = self.prompts['stepback'].format(query=query)
            stepback = self.llm.invoke(prompt).strip()
            
            # Clean up the response
            stepback = self._clean_response(stepback)
            
            # Return both step-back and original for multi-query retrieval
            return RewriteResult(
                original_query=query,
                rewritten_queries=[stepback, query],  # Step-back first, then original
                strategy_used='stepback',
                metadata={'includes_original': True}
            )
        except Exception as e:
            self.logger.error(f"Step-back generation failed: {e}")
            return RewriteResult(
                original_query=query,
                rewritten_queries=[query],
                strategy_used='stepback',
                metadata={'error': str(e), 'fallback': True}
            )
    
    def auto_select_strategy(self, query: str) -> str:
        """
        Automatically select the best rewriting strategy based on query characteristics.
        
        Args:
            query: The query to analyze
            
        Returns:
            Strategy name (expand, decompose, hyde, stepback)
        """
        words = query.split()
        query_lower = query.lower()
        
        # Check for comparison/multi-part questions -> decompose
        comparison_keywords = ['compare', 'versus', 'vs', 'difference', 'between', 'and']
        if any(kw in query_lower for kw in comparison_keywords):
            return 'decompose'
        
        # Check for very specific/detailed questions -> stepback
        specific_keywords = ['exactly', 'specifically', 'precise', 'particular', 'exact', 
                           'how many', 'what is the', 'what percentage']
        if any(kw in query_lower for kw in specific_keywords):
            return 'stepback'
        
        # Check for factual questions that need context -> hyde
        factual_patterns = ['what is', 'what are', 'explain', 'describe', 'define', 'how does']
        if any(query_lower.startswith(pat) for pat in factual_patterns):
            return 'hyde'
        
        # Short queries benefit from expansion
        if len(words) <= 5:
            return 'expand'
        
        # Default to hyde for general questions
        return 'hyde'
    
    def rewrite(
        self,
        query: str,
        strategy: str = "auto"
    ) -> RewriteResult:
        """
        Main entry point for query rewriting.
        
        Args:
            query: Original user query
            strategy: One of "expand", "decompose", "hyde", "stepback", or "auto"
            
        Returns:
            RewriteResult with rewritten query/queries
        """
        if not self.enabled:
            self.logger.info("Query rewriting is disabled")
            return RewriteResult(
                original_query=query,
                rewritten_queries=[query],
                strategy_used='none',
                metadata={'rewriting_disabled': True}
            )
        
        # Determine strategy
        if strategy == 'auto':
            strategy = self.auto_select_strategy(query)
            self.logger.info(f"Auto-selected strategy: {strategy}")
        
        # Apply the selected strategy
        strategy_map = {
            'expand': self.expand_query,
            'decompose': self.decompose_query,
            'hyde': self.hyde_rewrite,
            'stepback': self.stepback_rewrite
        }
        
        rewrite_func = strategy_map.get(strategy, self.expand_query)
        return rewrite_func(query)
    
    def rewrite_and_merge(
        self,
        query: str,
        strategies: List[str] = None
    ) -> RewriteResult:
        """
        Apply multiple strategies and merge results.
        
        Args:
            query: Original query
            strategies: List of strategies to apply (default: all)
            
        Returns:
            RewriteResult with merged queries from all strategies
        """
        if strategies is None:
            strategies = ['expand', 'stepback']  # Reasonable default combination
        
        all_queries = [query]  # Always include original
        all_metadata = {'strategies_applied': strategies}
        
        for strategy in strategies:
            result = self.rewrite(query, strategy=strategy)
            all_queries.extend(result.rewritten_queries)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in all_queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)
        
        return RewriteResult(
            original_query=query,
            rewritten_queries=unique_queries,
            strategy_used='merged',
            metadata=all_metadata
        )
    
    def _clean_response(self, text: str) -> str:
        """Clean up LLM response."""
        # Remove common prefixes
        prefixes = [
            'expanded query:', 'here is', 'the expanded', 'step-back question:',
            'hypothetical passage:', 'answer:', 'response:'
        ]
        text_lower = text.lower()
        for prefix in prefixes:
            if text_lower.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove quotes if present
        text = text.strip('"\'')
        
        # Remove trailing punctuation if it's just explanation
        lines = text.split('\n')
        if lines:
            text = lines[0].strip()  # Take first line only
        
        return text
    
    def _parse_numbered_list(self, text: str) -> List[str]:
        """Parse numbered list from LLM response."""
        queries = []
        
        # Try to parse numbered format (1., 2., etc.)
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Remove numbering
            match = re.match(r'^[\d]+[.):]\s*(.+)', line)
            if match:
                queries.append(match.group(1).strip())
            elif line and not line.startswith('#'):
                # Include non-empty, non-header lines
                queries.append(line)
        
        return queries
    
    def print_result(self, result: RewriteResult) -> None:
        """Pretty print rewrite result."""
        print("\n" + "=" * 60)
        print("QUERY REWRITE RESULT")
        print("=" * 60)
        print(f"\n📝 Original Query:")
        print(f"   {result.original_query}")
        print(f"\n🔄 Strategy: {result.strategy_used.upper()}")
        print(f"\n✨ Rewritten Queries:")
        for i, query in enumerate(result.rewritten_queries, 1):
            print(f"   {i}. {query}")
        if result.metadata:
            print(f"\n📊 Metadata: {result.metadata}")
        print("\n" + "=" * 60)
