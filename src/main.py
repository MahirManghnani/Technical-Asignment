import asyncio
from pathlib import Path
from typing import List, Dict
import json
from tqdm import tqdm
from datetime import datetime

from data.data_loader import DataLoader
from models.llm_interface import LLMInterface, QuestionContext
from models.answer_processor import AnswerProcessor
from evaluation.answer_evaluator import AnswerEvaluator

async def process_dataset():
    """Main function to process the dataset and evaluate results."""
    
    # Initialize components
    data_loader = DataLoader(split='test', seed=42)
    llm = LLMInterface()  # Rate limiting is now handled internally
    processor = AnswerProcessor()
    evaluator = AnswerEvaluator()
    
    # Load data
    print("Loading dataset...")
    entries = data_loader.load_data()
    
    # Calculate total requests needed
    total_questions = sum(len(entry['qa_pairs']) for entry in entries)
    print(f"Dataset contains {len(entries)} entries with {total_questions} total questions")
    
    # Check against daily limit
    quota = llm.get_remaining_quota()
    if total_questions > quota['daily_remaining']:
        print(f"\nWARNING: Total questions ({total_questions}) exceed daily remaining quota ({quota['daily_remaining']})")
        proceed = input("Do you want to proceed with partial processing? (y/n): ")
        if proceed.lower() != 'y':
            return
    
    # Store results
    all_results = []
    processed_questions = 0
    
    # Process each entry
    print("\nProcessing entries...")
    try:
        for entry in tqdm(entries):
            try:
                # Check if we would exceed daily limit
                questions_in_entry = len(entry['qa_pairs'])
                quota = llm.get_remaining_quota()
                if questions_in_entry > quota['daily_remaining']:
                    print("\nDaily limit would be exceeded. Stopping processing.")
                    break
                
                # Create context for LLM
                context = QuestionContext(
                    pre_text=entry['pre_text'],
                    post_text=entry['post_text'],
                    table=entry['table'],
                    questions=[qa['question'] for qa in entry['qa_pairs']],
                    entry_id=entry['entry_id']
                )
                
                llm_responses = await llm.get_answers(context)
                
                # Process each response
                processed_answers = []
                for response in llm_responses:
                    try:
                        processed_answer = processor.process_answer(response)
                        processed_answers.append(processed_answer)
                    except Exception as e:
                        print(f"\nError processing answer: {str(e)}")
                        processed_answers.append(None)
                
                # Store results
                entry_results = {
                    'entry_id': entry['entry_id'],
                    'questions': [qa['question'] for qa in entry['qa_pairs']],
                    'expected_answers': [qa['answer'] for qa in entry['qa_pairs']],
                    'llm_responses': llm_responses,
                    'processed_answers': processed_answers
                }
                all_results.append(entry_results)
                
                processed_questions += questions_in_entry
                
            except Exception as e:
                print(f"\nError processing entry {entry.get('entry_id', 'unknown')}: {str(e)}")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Saving partial results...")
    
    # Evaluate results
    print(f"\nEvaluating results for {processed_questions} processed questions...")
    evaluation_pairs = []
    for result in all_results:
        for expected, generated in zip(result['expected_answers'], result['processed_answers']):
            if generated is not None:
                evaluation_pairs.append((generated, expected))
    
    accuracies = evaluator.evaluate_batch(evaluation_pairs)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"Processed {processed_questions}/{total_questions} questions")
    for metric, value in accuracies.items():
        if metric == 'mean_smape':
            print(f"{metric}: {value:.2f}%")
        else:
            print(f"{metric}: {value:.2%}")
    
    # Save results
    save_results(all_results, accuracies, processed_questions, total_questions)

def save_results(results: List[Dict], accuracies: Dict, processed: int, total: int):
    """Save results to files."""
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"detailed_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save accuracies with metadata
    final_results = {
        'accuracies': accuracies,
        'metadata': {
            'processed_questions': processed,
            'total_questions': total,
            'completion_percentage': (processed / total * 100) if total > 0 else 0,
            'timestamp': timestamp
        }
    }
    
    with open(output_dir / f"accuracies_{timestamp}.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    # Create a summary file
    with open(output_dir / f"summary_{timestamp}.txt", "w") as f:
        f.write("Evaluation Results\n")
        f.write("=================\n\n")
        f.write(f"Processed {processed}/{total} questions ")
        f.write(f"({processed/total*100:.1f}% complete)\n\n")
        for metric, value in accuracies.items():
            if metric == 'mean_smape':  # Changed from average_percentage_diff
                f.write(f"{metric}: {value:.2f}%\n")
            else:
                f.write(f"{metric}: {value:.2%}\n")

if __name__ == "__main__":
    # Run the main process
    asyncio.run(process_dataset())