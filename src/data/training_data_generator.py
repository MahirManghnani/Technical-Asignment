# src/data/training_data_generator.py
import json
import random
from pathlib import Path
from typing import List, Dict
from .data_loader import DataLoader
from ..models.answer_processor import AnswerProcessor
from ..config.prompts import format_question_prompt

class TrainingDataGenerator:
    def __init__(self, train_split: float = 0.8, seed: int = 42):
        self.train_split = train_split
        random.seed(seed)
        self.data_loader = DataLoader()
        
    def generate_answer_json(self, question: str, answer: str) -> Dict:
        """Generate the answer JSON based on the question and answer."""
        # Common patterns to detect in questions
        is_percentage = 'percentage' in question.lower() or '%' in question
        is_currency = '$' in answer or 'dollars' in question.lower()
        
        # Extract numbers from answer
        numbers = [float(s) for s in answer.replace('$', '').replace('%', '').replace(',', '').split() if s.replace('.', '').isdigit()]
        if not numbers:
            return None
            
        result = numbers[0]
        
        # Determine formatting
        formatting = {
            "prefix": "$" if is_currency else "",
            "suffix": "%" if is_percentage else "",
            "rounding": 2 if (is_percentage or is_currency) else 0,
            "multiplier": 1
        }
        
        # Generate formula based on answer pattern
        formula = str(result)  # Simple case, can be enhanced
        
        return {
            "formula": formula,
            "formatting": formatting
        }
    
    def generate_training_data(self) -> tuple[List[Dict], List[Dict]]:
        """Generate training and testing datasets."""
        entries = self.data_loader.load_data()
        random.shuffle(entries)
        
        split_idx = int(len(entries) * self.train_split)
        train_entries = entries[:split_idx]
        test_entries = entries[split_idx:]
        
        train_data = []
        test_data = []
        
        for entry in train_entries:
            for qa in entry['qa_pairs']:
                text_input = format_question_prompt({
                    'pre_text': entry['pre_text'],
                    'post_text': entry['post_text'],
                    'table': entry['table'],
                    'question': qa['question']
                })
                
                answer_json = self.generate_answer_json(qa['question'], qa['answer'])
                if answer_json:
                    train_data.append({
                        "text_input": text_input,
                        "output": json.dumps(answer_json)
                    })
        
        for entry in test_entries:
            for qa in entry['qa_pairs']:
                test_data.append({
                    'entry_id': entry['entry_id'],
                    'text_input': format_question_prompt({
                        'pre_text': entry['pre_text'],
                        'post_text': entry['post_text'],
                        'table': entry['table'],
                        'question': qa['question']
                    }),
                    'expected_answer': qa['answer']
                })
        
        return train_data, test_data

if __name__ == "__main__":
    generator = TrainingDataGenerator()
    train_data, test_data = generator.generate_training_data()
    
    # Save the data
    output_dir = Path(__file__).parent.parent.parent / "data"
    
    with open(output_dir / "train_data.json", "w") as f:
        json.dump(train_data, f, indent=2)
        
    with open(output_dir / "test_data.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Generated {len(train_data)} training examples and {len(test_data)} test examples")