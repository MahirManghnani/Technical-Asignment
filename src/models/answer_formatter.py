from dataclasses import dataclass
from typing import Dict

@dataclass
class FormattingInstructions:
    prefix: str
    suffix: str
    rounding: int
    multiplier: float

class AnswerFormatter:
    def parse_instructions(self, format_dict: Dict) -> FormattingInstructions:
        """Parse formatting instructions from dictionary into a structured object."""
        return FormattingInstructions(
            prefix=format_dict.get('prefix', ''),
            suffix=format_dict.get('suffix', ''),
            rounding=format_dict.get('rounding', 2),
            multiplier=format_dict.get('multiplier', 1.0)
        )
    
    def format_number(self, number: float, instructions: FormattingInstructions) -> str:
        """Format a number according to the given instructions."""
        # Apply multiplier
        value = number * instructions.multiplier
        
        # Round to specified decimal places
        rounded_value = round(value, instructions.rounding)
        
        # Convert to string with proper decimal places
        number_str = f"{{:.{instructions.rounding}f}}".format(rounded_value)
        
        # Combine with prefix and suffix
        return f"{instructions.prefix}{number_str}{instructions.suffix}"