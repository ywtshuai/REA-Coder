import json
import tiktoken
from typing import List, Dict, Any

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data from file into a list of dictionaries.
    
    Args:
        file_path: Path to the JSONL file.
    
    Returns:
        List of dictionary entries from the JSONL file.
    """
    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
    return entries

def count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    """Count tokens in a given text using the specified encoder.
    
    Args:
        text: Input text to count tokens for.
        encoder: tiktoken encoder instance.
    
    Returns:
        Number of tokens in the text.
    """
    return len(encoder.encode(text))

def process_entry(entry: Dict[str, Any], encoder: tiktoken.Encoding) -> Dict[str, Any]:
    """Process a single dataset entry to calculate token counts.
    
    Args:
        entry: Dictionary containing task data.
        encoder: tiktoken encoder instance.
    
    Returns:
        Dictionary with token count information.
    """
    # Construct the full input prompt
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
    
    # Count tokens for input (system + user prompts)
    input_tokens = count_tokens(system_prompt, encoder) + count_tokens(user_prompt, encoder)
    
    # Count tokens for output (routing_reason)
    output_tokens = count_tokens(entry['routing_reason'], encoder)
    
    return {
        "task_id": f"HumanEval/{entry['task_id']}",
        "approach": "gpt-routing",
        "input_token": input_tokens,
        "output_token": output_tokens
    }

def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save results to a JSONL file.
    
    Args:
        results: List of result dictionaries.
        output_path: Path to save the results.
    """
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

def main():
    """Main execution function."""
    # Initialize tokenizer
    enc = tiktoken.encoding_for_model("gpt-4o")
    
    # Load data
    dataset_path = ""
    output_path = ""
    
    dataset_entries = load_json_data(dataset_path)
    
    # Process each entry
    results = []
    for entry in dataset_entries:
        results.append(process_entry(entry, enc))
    
    # Save results
    
    save_results(results, output_path)
    
    print(f"Processing complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()