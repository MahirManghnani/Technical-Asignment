import json
import re
from .expression_evaluator import ExpressionEvaluator
from .answer_formatter import AnswerFormatter

class AnswerProcessor:
    def __init__(self):
        self.evaluator = ExpressionEvaluator()
        self.formatter = AnswerFormatter()
    
    def _clean_llm_response(self, llm_response: str) -> str:
        """
        Clean the LLM response by removing markdown formatting and newlines.
        
        Args:
            llm_response: Raw response from LLM with markdown formatting
            
        Returns:
            Clean JSON string
        """
        # Remove markdown code block formatting
        cleaned = re.sub(r'```json\s*', '', llm_response)
        cleaned = re.sub(r'\s*```', '', cleaned)
        
        # Remove any extra newlines and whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def process_answer(self, llm_response: str) -> str:
        """
        Process the LLM JSON response through evaluation and formatting.
        
        Args:
            llm_response: JSON string from LLM containing formula and formatting instructions
            
        Returns:
            Formatted answer string
        """
        try:
            # Clean the response
            cleaned_response = self._clean_llm_response(llm_response)
            
            # Parse the JSON response
            response_dict = json.loads(cleaned_response)
            
            # Extract formula and formatting instructions
            formula = response_dict.get('formula')
            format_dict = response_dict.get('formatting_instructions')
            
            if not formula or not format_dict:
                raise ValueError("Missing required fields in LLM response")
            
            # Evaluate the mathematical expression
            result = self.evaluator.evaluate(formula)
            
            # Parse formatting instructions and format the result
            formatting = self.formatter.parse_instructions(format_dict)
            formatted_result = self.formatter.format_number(result, formatting)
            
            return formatted_result
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in LLM response: {str(e)}\nResponse was: {llm_response}")
        except KeyError as e:
            raise ValueError(f"Missing required field in LLM response: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing answer: {str(e)}")