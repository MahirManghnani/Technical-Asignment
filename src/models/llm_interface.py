import os
from typing import Dict, Optional, List
import google.generativeai as genai
from dataclasses import dataclass
from utils.env import load_environment
from ..config.prompts import SYSTEM_PROMPT, INITIAL_PROMPT, format_question_prompt


@dataclass
class LLMConfig:
    """Configuration for the LLM model."""
    temperature: float = 0
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192
    response_mime_type: str = "text/plain"

@dataclass
class QuestionContext:
    """Context for a question."""
    pre_text: str
    post_text: str
    table: str
    questions: List[str]
    entry_id: str  # To identify questions from the same entry

class LLMInterface:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LLM interface."""
        # Load environment variables
        load_environment()
        
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Set up default configuration
        self.config = LLMConfig()
        
        # Initialize the model
        self.model = self._setup_model()
        
        # Current chat session and entry tracking
        self.current_chat = None
        self.current_entry_id = None
    
    def _setup_model(self):
        """Set up the Gemini model with configuration."""
        generation_config = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "max_output_tokens": self.config.max_output_tokens,
            "response_mime_type": self.config.response_mime_type,
        }
        
        return genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
            system_instruction=SYSTEM_PROMPT
        )
    
    def _create_new_chat_session(self):
        """Create a new chat session with initial prompt."""
        return self.model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [INITIAL_PROMPT],
                }
            ]
        )
    
    def _should_create_new_chat(self, entry_id: str) -> bool:
        """Determine if we should create a new chat session."""
        return self.current_chat is None or self.current_entry_id != entry_id

    async def get_answers(self, context: QuestionContext) -> List[str]:
        """
        Get answers for one or more questions from the same entry.
        
        Args:
            context: QuestionContext containing entry information and questions
            
        Returns:
            List of JSON strings containing formulas and formatting instructions
        """
        try:
            answers = []
            
            # Check if we need a new chat session
            if self._should_create_new_chat(context.entry_id):
                self.current_chat = self._create_new_chat_session()
                self.current_entry_id = context.entry_id
            
            # Process each question
            for i, question in enumerate(context.questions):
                if i == 0:
                    # First question: send full context
                    prompt = format_question_prompt({
                        "pre_text": context.pre_text,
                        "post_text": context.post_text,
                        "table": context.table,
                        "question": question
                    })
                else:
                    # Subsequent questions: send only the question
                    prompt = f"question: {question}"
                
                response = self.current_chat.send_message(prompt)
                answers.append(response.text)
            
            return answers
            
        except Exception as e:
            raise Exception(f"Error getting answer from LLM: {str(e)}")