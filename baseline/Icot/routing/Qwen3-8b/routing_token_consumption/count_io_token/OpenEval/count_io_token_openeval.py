import json
from transformers import AutoTokenizer  
from typing import List, Dict, Any

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data from file into a list of dictionaries."""
    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
    return entries

def count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    """Count tokens using Qwen's tokenizer."""
    return len(tokenizer.encode(text))

def process_entry(entry: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """Process a single entry with Qwen's tokenizer."""
    system_prompt = "You are an expert in analyzing code task difficulty."
    user_prompt = f"""You will see a Python function definition (including comments and input parameters).
    Please classify its task complexity into one of three categories:
    - Easy
    - Medium
    - Hard

    Consider data structures, algorithmic complexity, edge cases, implementation steps, recursion, etc.

    Output in the following format:
    Medium: because it requires recursive logic and careful base case handling.

    The function task is as follows:

The function task is as follows:
{entry['prompt']}"""
    
    input_tokens = count_tokens(system_prompt, tokenizer) + count_tokens(user_prompt, tokenizer)
    output_tokens = count_tokens(entry['routing_reason'], tokenizer)
    
    return {
        "task_id": f"HumanEval/{entry['task_id']}",
        "approach": "Qwen3-8B-routing",
        "input_token": input_tokens,
        "output_token": output_tokens
    }

def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save results to a JSONL file."""
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

def main():
    """Main execution function."""

    tokenizer = AutoTokenizer.from_pretrained("")  
    
    # Load data
    dataset_path = ""
    output_path = ""
    
    dataset_entries = load_json_data(dataset_path)
    
    # Process each entry
    results = []
    for entry in dataset_entries:
        results.append(process_entry(entry, tokenizer))
    
    # Save results
    save_results(results, output_path)
    print(f"Processing complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()