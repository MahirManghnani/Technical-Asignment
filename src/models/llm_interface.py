import os
from typing import Dict, Optional, List
import google.generativeai as genai
from dataclasses import dataclass
from utils.env import load_environment
from config.prompts import SYSTEM_PROMPT, INITIAL_PROMPT, format_question_prompt
import asyncio
import time
from datetime import date


@dataclass
class LLMConfig:
    """Configuration for the LLM model."""
    temperature: float = 0
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192
    response_mime_type: str = "text/plain"
    rpm: int = 10  # Rate limit in requests per minute
    daily_limit: int = 1500  # Daily request limit
    max_retries: int = 3  # Maximum number of retries for rate limit errors
    retry_delay: int = 60  # Base delay between retries in seconds


@dataclass
class QuestionContext:
    """Context for a question."""
    pre_text: str
    post_text: str
    table: str
    questions: List[str]
    entry_id: str  # To identify questions from the same entry


class LLMInterface:
    def __init__(self, api_key: Optional[str] = None, config: Optional[LLMConfig] = None):
        """Initialize the LLM interface."""
        # Load environment variables
        load_environment()
        
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Set up configuration
        self.config = config or LLMConfig()
        
        # Initialize the model
        self.model = self._setup_model()
        
        # Current chat session and entry tracking
        self.current_chat = None
        self.current_entry_id = None
        
        # Rate limiting state
        self.requests = []  # Timestamp of recent requests
        self.daily_requests = 0  # Count of requests today
        self.last_reset_date = date.today()
    
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
    
    async def _check_rate_limits(self):
        """Check and enforce rate limits."""
        current_time = time.time()
        current_date = date.today()
        
        # Reset daily counter if it's a new day
        if current_date != self.last_reset_date:
            self.daily_requests = 0
            self.last_reset_date = current_date
        
        # Check daily limit
        if self.daily_requests >= self.config.daily_limit:
            raise Exception(f"Daily request limit of {self.config.daily_limit} reached")
        
        # Remove requests older than 1 minute
        self.requests = [t for t in self.requests if current_time - t < 60]
        
        # If we've hit the rate limit, wait
        if len(self.requests) >= self.config.rpm:
            wait_time = 60 - (current_time - self.requests[0])
            if wait_time > 0:
                print(f"\nRate limit reached, waiting {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
        
        # Add current request
        self.requests.append(current_time)
        self.daily_requests += 1

    async def _send_message_with_retry(self, prompt: str, retry_count: int = 0) -> str:
        """Send message to LLM with retry logic for rate limit errors."""
        try:
            # Check rate limits before sending
            await self._check_rate_limits()
            
            # Send the message
            response = self.current_chat.send_message(prompt)
            return response.text
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a rate limit error (429)
            if "429" in error_str or "resource exhausted" in error_str:
                if retry_count < self.config.max_retries:
                    # Calculate delay with exponential backoff
                    delay = self.config.retry_delay * (2 ** retry_count)
                    print(f"\nRate limit exceeded (429). Retrying in {delay} seconds... (Attempt {retry_count + 1}/{self.config.max_retries})")
                    
                    # Wait before retrying
                    await asyncio.sleep(delay)
                    
                    # Clear the recent requests to reset rate limiting
                    self.requests = []
                    
                    # Retry with incremented count
                    return await self._send_message_with_retry(prompt, retry_count + 1)
                else:
                    raise Exception(f"Maximum retries ({self.config.max_retries}) exceeded for rate limit error")
            else:
                # Re-raise other errors
                raise

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
                
                # Send message with retry logic
                response_text = await self._send_message_with_retry(prompt)
                answers.append(response_text)
            
            return answers
            
        except Exception as e:
            raise Exception(f"Error getting answer from LLM: {str(e)}")
    
    def get_remaining_quota(self) -> Dict[str, int]:
        """Get remaining request quotas."""
        current_date = date.today()
        if current_date != self.last_reset_date:
            return {
                "daily_remaining": self.config.daily_limit,
                "minute_remaining": self.config.rpm
            }
        
        # Clean up old requests
        current_time = time.time()
        self.requests = [t for t in self.requests if current_time - t < 60]
        
        return {
            "daily_remaining": self.config.daily_limit - self.daily_requests,
            "minute_remaining": self.config.rpm - len(self.requests)
        }