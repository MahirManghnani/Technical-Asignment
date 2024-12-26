from pathlib import Path
import json
from typing import Dict, List, Union
import random

class DataLoader:
    def __init__(self, split: str = None, seed: int = 42):
        """
        Initialize DataLoader with optional split type.
        
        Args:
            split: Either 'train', 'test', or None for all data
            seed: Random seed for consistent splitting
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.raw_data_path = self.project_root / "data" / "train.json"
        self.split = split
        self.seed = seed
    
    def split_data(self, data: List[Dict], train_ratio: float = 0.8) -> tuple[List[Dict], List[Dict]]:
        """Split data into training and test sets."""
        random.seed(self.seed)
        data_copy = data.copy()
        random.shuffle(data_copy)
        
        split_idx = int(len(data_copy) * train_ratio)
        return data_copy[:split_idx], data_copy[split_idx:]
    
    def load_data(self) -> List[Dict[str, Union[str, List[Dict[str, str]], str]]]:
        """
        Load text and tabular data with question-answer pairs from JSON file.
        Adds an entry_id to each item for tracking related questions.
        """
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.raw_data_path}")
            
        with open(self.raw_data_path, 'r') as f:
            data = json.load(f)
        
        entries = []
        for idx, item in enumerate(data):
            entry_id = f"entry_{idx:04d}"  # Creates IDs like entry_000, entry_001, etc.
            
            # Collect all QA pairs for this entry
            qa_list = []
            
            # Check for single QA pair
            if 'qa' in item:
                qa_list.append({
                    'question': item['qa']['question'], 
                    'answer': item['qa']['answer'],
                    'formula': item['qa']['program_re']
                })
            
            # Check for multiple QA pairs (qa_0, qa_1)
            for i in range(2):  # Assuming maximum of qa_0 and qa_1
                qa_key = f'qa_{i}'
                if qa_key in item:
                    qa_list.append({
                        'question': item[qa_key]['question'],
                        'answer': item[qa_key]['answer'],
                        'formula': item[qa_key]['program_re'].replace('const_', '')
                    })
            
            entries.append({
                'entry_id': entry_id,
                'pre_text': item['pre_text'],
                'post_text': item['post_text'],
                'table': item['table'],
                'qa_pairs': qa_list
            })

        if self.split:
            train_data, test_data = self.split_data(entries)
            return train_data if self.split == 'train' else test_data
        
        return entries