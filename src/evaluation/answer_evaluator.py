from dataclasses import dataclass
import re
from typing import Tuple, Dict, List

@dataclass
class ParsedAnswer:
    """Structured representation of a formatted number."""
    raw_string: str
    number: float
    prefix: str
    suffix: str
    decimal_places: int

class AnswerEvaluator:
    def __init__(self):
        """Initialize the evaluator."""
        pass

    def _calculate_smape(self, generated: float, expected: float) -> float:
        """
        Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) between two numbers.
        
        Args:
            generated: Generated number
            expected: Expected number
            
        Returns:
            SMAPE value as a percentage
        """
        if abs(generated) == 0 and abs(expected) == 0:
            return 0.0
            
        return 200 * abs(generated - expected) / (abs(generated) + abs(expected))
    
    def parse_answer(self, answer: str) -> ParsedAnswer:
        """
        Parse a formatted answer string into its components.
        
        Args:
            answer: String like "$123.45%" or "1.23"
            
        Returns:
            ParsedAnswer object containing components
        """
        # Remove any whitespace
        answer = answer.strip()
        
        # Extract prefix (any non-digit characters at start)
        prefix_match = re.match(r'^[^\d.-]*', answer)
        prefix = prefix_match.group(0) if prefix_match else ""
        
        # Extract suffix (any non-digit characters at end)
        suffix_match = re.search(r'[^\d.]*$', answer)
        suffix = suffix_match.group(0) if suffix_match else ""
        
        # Extract the number
        number_str = answer[len(prefix):len(answer)-len(suffix)]
        try:
            number = float(number_str)
        except ValueError:
            raise ValueError(f"Could not parse number from {answer}")
        
        # Count decimal places
        decimal_places = 0
        if '.' in number_str:
            decimal_places = len(number_str.split('.')[1])
        
        return ParsedAnswer(
            raw_string=answer,
            number=number,
            prefix=prefix,
            suffix=suffix,
            decimal_places=decimal_places
        )
    
    def compare_answers(self, generated: str, expected: str) -> Dict[str, float]:
        """
        Compare generated answer with expected answer.
        
        Args:
            generated: Generated answer string
            expected: Expected answer string
            
        Returns:
            Dictionary with SMAPE and format matching results
        """
        try:
            gen_parsed = self.parse_answer(generated)
            exp_parsed = self.parse_answer(expected)
            
            # Calculate SMAPE for the numbers
            smape = self._calculate_smape(gen_parsed.number, exp_parsed.number)
            
            # Check format matching
            results = {
                'smape': smape,
                'prefix_match': gen_parsed.prefix == exp_parsed.prefix,
                'suffix_match': gen_parsed.suffix == exp_parsed.suffix,
                'decimal_places_match': gen_parsed.decimal_places == exp_parsed.decimal_places
            }
            
            return results
            
        except ValueError as e:
            raise ValueError(f"Error comparing answers: {str(e)}")
    
    def evaluate_batch(self, pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Evaluate a batch of generated-expected answer pairs.
        
        Args:
            pairs: List of (generated, expected) answer pairs
            
        Returns:
            Dictionary of metrics including mean SMAPE and format matching rates
        """
        total = len(pairs)
        if total == 0:
            return {}
        
        format_counts = {
            'prefix_match': 0,
            'suffix_match': 0,
            'decimal_places_match': 0
        }
        
        total_smape = 0.0
        valid_comparisons = 0
        
        for generated, expected in pairs:
            try:
                results = self.compare_answers(generated, expected)
                
                # Track SMAPE
                total_smape += results['smape']
                valid_comparisons += 1
                
                # Track format matching
                for key in format_counts:
                    if results.get(key, False):
                        format_counts[key] += 1
                    
            except ValueError:
                continue
        
        # Calculate metrics
        metrics = {
            f"{key}_rate": count/total 
            for key, count in format_counts.items()
        }
        
        # Add mean SMAPE
        if valid_comparisons > 0:
            metrics['mean_smape'] = total_smape / valid_comparisons
        
        return metrics