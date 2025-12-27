"""Chatbot with conversation history for RAG system (Q5 - Bonus)."""

import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

sys.path.insert(0, str(Path(__file__).parent.parent))
from template import RAG_CHAT_PROMPT_TEMPLATE, QUERY_REFORMULATION_TEMPLATE

from llm_qa_system import LLMQASystem
from document_retriever import DocumentRetriever
from utils import get_config, get_logger


@dataclass
class Message:
    """Single chat message."""
    role: str  # 'user' or 'assistant'
    content: str


@dataclass
class ConversationHistory:
    """Manages conversation history."""
    messages: List[Message] = field(default_factory=list)
    max_length: int = 10
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to history."""
        self.messages.append(Message(role=role, content=content))
        # Trim if exceeds max length
        if len(self.messages) > self.max_length * 2:  # *2 for user+assistant pairs
            self.messages = self.messages[-self.max_length * 2:]
    
    def get_formatted_history(self) -> str:
        """Get history as formatted string."""
        if not self.messages:
            return "No previous conversation."
        
        formatted = []
        for msg in self.messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            formatted.append(f"{role_label}: {msg.content}")
        
        return "\n".join(formatted)
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []
    
    def get_last_n(self, n: int) -> List[Message]:
        """Get last n messages."""
        return self.messages[-n:] if self.messages else []


class Chatbot:
    """
    Conversational RAG chatbot with history management.
    
    Features:
    - Maintains conversation history
    - Reformulates queries using context
    - Provides contextual answers based on documents
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the chatbot.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.logger = get_logger(__name__)
        
        # Settings
        max_history = self.config.get('chatbot.max_history_length', 10)
        self.system_message = self.config.get(
            'chatbot.system_message',
            "You are a helpful assistant."
        )
        
        # Initialize components
        self.qa_system = LLMQASystem(config_path)
        self.history = ConversationHistory(max_length=max_history)
        
        self.logger.info("Chatbot initialized")
    
    def _reformulate_query(self, question: str) -> str:
        """
        Reformulate question using conversation history for better retrieval.
        
        Args:
            question: Current user question
            
        Returns:
            Reformulated standalone question
        """
        if not self.history.messages:
            return question
        
        # Use LLM to reformulate
        prompt = PromptTemplate(
            template=QUERY_REFORMULATION_TEMPLATE,
            input_variables=["chat_history", "question"]
        )
        
        formatted = prompt.format(
            chat_history=self.history.get_formatted_history(),
            question=question
        )
        
        try:
            reformulated = self.qa_system.llm.invoke(formatted)
            self.logger.debug(f"Query reformulated: '{question}' -> '{reformulated}'")
            return reformulated.strip()
        except Exception as e:
            self.logger.warning(f"Query reformulation failed: {e}")
            return question
    
    def chat(
        self,
        message: str,
        use_reformulation: bool = True
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate response.
        
        Args:
            message: User message
            use_reformulation: Whether to reformulate query using history
            
        Returns:
            Response dictionary with answer and metadata
        """
        self.logger.info(f"User message: '{message[:50]}...'")
        
        # Add user message to history
        self.history.add_message("user", message)
        
        # Optionally reformulate for better retrieval
        query = self._reformulate_query(message) if use_reformulation else message
        
        # Get context from retriever
        context = self.qa_system.retriever.get_context_for_llm(query)
        
        # Build prompt with history
        prompt = PromptTemplate(
            template=RAG_CHAT_PROMPT_TEMPLATE,
            input_variables=["chat_history", "context", "question"]
        )
        
        formatted_prompt = prompt.format(
            chat_history=self.history.get_formatted_history(),
            context=context,
            question=message
        )
        
        # Generate response
        response = self.qa_system.llm.invoke(formatted_prompt)
        
        # Add assistant response to history
        self.history.add_message("assistant", response)
        
        return {
            'response': response,
            'query_used': query,
            'context': context,
            'history_length': len(self.history.messages)
        }
    
    def get_response(self, message: str) -> str:
        """Simple interface - just get the response text."""
        result = self.chat(message)
        return result['response']
    
    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self.history.clear()
        self.logger.info("Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history as list of dicts."""
        return [
            {'role': msg.role, 'content': msg.content}
            for msg in self.history.messages
        ]
    
    def interactive_mode(self) -> None:
        """Run chatbot in interactive terminal mode."""
        print("\n" + "=" * 60)
        print("RAG CHATBOT - Interactive Mode")
        print("=" * 60)
        print("Commands: 'quit' to exit, 'clear' to reset history")
        print("=" * 60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.reset_conversation()
                    print("Conversation cleared.\n")
                    continue
                
                response = self.get_response(user_input)
                print(f"\nAssistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                self.logger.error(f"Error: {e}")
                print(f"Error occurred: {e}\n")
