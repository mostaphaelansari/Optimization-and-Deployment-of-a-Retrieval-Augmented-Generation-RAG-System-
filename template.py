"""Prompt templates for the RAG system (Q3)."""

# Basic RAG prompt template - Optimized for better answers
RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions based on the provided context documents.

CONTEXT FROM DOCUMENTS:
{context}

---

QUESTION: {question}

INSTRUCTIONS:
1. Answer the question based on the context provided above.
2. Be concise and direct - aim for 2-4 sentences unless more detail is needed.
3. If you find relevant information, provide the answer confidently.
4. Only say "I don't have enough information" if the context truly contains nothing relevant.
5. Do NOT repeat the sources in your answer - they will be shown separately.

ANSWER:"""


# RAG prompt with chat history for chatbot (Q5)
RAG_CHAT_PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions based on documents and conversation history.

CONVERSATION HISTORY:
{chat_history}

---

CONTEXT FROM DOCUMENTS:
{context}

---

CURRENT QUESTION: {question}

INSTRUCTIONS:
1. Consider the conversation history for context.
2. Answer based on the document context provided.
3. Be concise - 2-4 sentences unless more detail is needed.
4. Maintain consistency with previous responses.

ANSWER:"""


# System message for chatbot
SYSTEM_MESSAGE = """You are an intelligent assistant specialized in answering questions about AI research papers, specifically about Transformers, BERT, and GPT-3. You provide accurate, concise answers based on the document content."""


# Query reformulation prompt (for better retrieval)
QUERY_REFORMULATION_TEMPLATE = """Given the conversation history and the latest user question, reformulate the question to be standalone and clear.

CONVERSATION HISTORY:
{chat_history}

LATEST QUESTION: {question}

Reformulated standalone question:"""


# Evaluation prompt for answer assessment
EVALUATION_PROMPT_TEMPLATE = """Evaluate the following answer based on the provided context and question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

Rate the answer on these criteria (1-5 scale):
1. FAITHFULNESS: Does the answer only contain information from the context?
2. RELEVANCE: Does the answer address the question asked?
3. COMPLETENESS: Does the answer cover all relevant information from the context?
4. CLARITY: Is the answer clear and well-structured?

Provide your evaluation as JSON:
{{"faithfulness": X, "relevance": X, "completeness": X, "clarity": X, "explanation": "..."}}"""