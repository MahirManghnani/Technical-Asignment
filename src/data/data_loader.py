from pathlib import Path
import json
from typing import Dict, List, Union

class DataLoader:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.raw_data_path = self.project_root / "data" / "train.json"
    
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
                    'answer': item['qa']['answer']
                })
            
            # Check for multiple QA pairs (qa_0, qa_1)
            for i in range(2):  # Assuming maximum of qa_0 and qa_1
                qa_key = f'qa_{i}'
                if qa_key in item:
                    qa_list.append({
                        'question': item[qa_key]['question'],
                        'answer': item[qa_key]['answer']
                    })
            
            entries.append({
                'entry_id': entry_id,
                'pre_text': item['pre_text'],
                'post_text': item['post_text'],
                'table': item['table'],
                'qa_pairs': qa_list
            })
        
        return entries