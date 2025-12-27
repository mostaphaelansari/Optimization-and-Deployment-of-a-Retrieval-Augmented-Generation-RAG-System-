#!/usr/bin/env python3
"""
RAG Chatbot - Streamlit Web Interface
Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

# Page config
st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def load_chatbot():
    """Load chatbot (cached)."""
    from chatbot import Chatbot
    return Chatbot()


@st.cache_resource
def load_qa_system():
    """Load QA system (cached)."""
    from llm_qa_system import LLMQASystem
    return LLMQASystem()


def main():
    # Header
    st.title("🤖 RAG System")
    st.markdown("**Ask questions about AI research papers:** Transformers, BERT, and GPT-3")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        mode = st.radio("Mode", ["💬 Chat", "❓ Single Q&A", "🔍 Search"])
        
        st.divider()
        
        st.markdown("**Configuration:**")
        st.markdown("- LLM: Qwen 2.5 (1.5B)")
        st.markdown("- Embeddings: MiniLM-L6")
        st.markdown("- Vector Store: ChromaDB")
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat Mode
    if mode == "💬 Chat":
        st.subheader("💬 Chat with AI")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about Transformers, BERT, or GPT-3..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chatbot = load_chatbot()
                        result = chatbot.chat(prompt)
                        response = result['response']
                    except Exception as e:
                        response = f"Error: {str(e)}"
                
                st.markdown(response)
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Single Q&A Mode
    elif mode == "❓ Single Q&A":
        st.subheader("❓ Ask a Question")
        
        question = st.text_area("Your Question", placeholder="What is the Transformer architecture?")
        show_sources = st.checkbox("Show Sources", value=True)
        
        if st.button("🔍 Get Answer", type="primary"):
            if question.strip():
                with st.spinner("Generating answer..."):
                    try:
                        qa_system = load_qa_system()
                        result = qa_system.answer(question, return_sources=True)
                        
                        st.success("Answer:")
                        st.markdown(result['answer'])
                        
                        if show_sources and 'sources' in result:
                            st.divider()
                            st.markdown("**📚 Sources:**")
                            for src in result['sources']:
                                st.markdown(f"- `{src['source']}` (page {src['page']}) - score: {src['score']:.4f}")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a question.")
    
    # Search Mode
    elif mode == "🔍 Search":
        st.subheader("🔍 Semantic Search")
        
        query = st.text_input("Search Query", placeholder="attention mechanism")
        top_k = st.slider("Number of Results", 1, 10, 5)
        
        if st.button("🔍 Search", type="primary"):
            if query.strip():
                with st.spinner("Searching..."):
                    try:
                        qa_system = load_qa_system()
                        results = qa_system.retriever.retrieve(query, top_k=top_k, with_scores=True)
                        
                        st.success(f"Found {len(results)} results:")
                        
                        for r in results:
                            source = r['metadata'].get('source', 'Unknown')
                            page = r['metadata'].get('page', 'N/A')
                            score = r.get('score', 0)
                            content = r['content'][:300] + "..." if len(r['content']) > 300 else r['content']
                            
                            with st.expander(f"**[{score:.3f}]** {source} (page {page})"):
                                st.markdown(content)
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a search query.")


if __name__ == "__main__":
    main()
