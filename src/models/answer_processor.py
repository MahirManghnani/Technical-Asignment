import json
from .expression_evaluator import ExpressionEvaluator
from .answer_formatter import AnswerFormatter

class AnswerProcessor:
    def __init__(self):
        self.evaluator = ExpressionEvaluator()
        self.formatter = AnswerFormatter()
    
    def process_answer(self, llm_response: str) -> str:
        """
        Process the LLM JSON response through evaluation and formatting.
        
        Args:
            llm_response: JSON string from LLM containing formula and formatting instructions
        
        Returns:
            Formatted answer string
        """
        try:
            # Parse the JSON response
            response_dict = json.loads(llm_response)
            
            # Extract formula and formatting instructions
            formula = response_dict['formula']
            format_dict = response_dict['formatting_instructions']
            
            # Evaluate the mathematical expression
            result = self.evaluator.evaluate(formula)
            
            # Parse formatting instructions and format the result
            formatting = self.formatter.parse_instructions(format_dict)
            formatted_result = self.formatter.format_number(result, formatting)
            
            return formatted_result
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in LLM response: {str(e)}")
        except KeyError as e:
            raise ValueError(f"Missing required field in LLM response: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing answer: {str(e)}")