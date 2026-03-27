import json
import os
from pathlib import Path
from typing import Dict, List

def process_json_file(file_path: Path) -> Dict[str, int]:
    """Process a single JSON file to calculate token sums.
    
    Args:
        file_path: Path object pointing to the JSON file.
    
    Returns:
        Dictionary with 'input_token' and 'output_token' sums.
    """
    input_sum = 0
    output_sum = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                input_sum += data.get('input_token', 0)
                output_sum += data.get('output_token', 0)
            except json.JSONDecodeError:
                continue
    
    return {
        'input_token': input_sum,
        'output_token': output_sum
    }

def scan_directory(root_dir: str) -> List[Dict[str, any]]:
    """Recursively scan directory for JSON files and process them.
    
    Args:
        root_dir: Root directory to start scanning from.
    
    Returns:
        List of dictionaries containing:
        - 'file': Full path to the JSON file (relative to root_dir)
        - 'input_token': Sum of input tokens in the file
        - 'output_token': Sum of output tokens in the file
    """
    results = []
    root_path = Path(root_dir)
    
    # Recursively find all JSON files
    for json_file in root_path.rglob('*.jsonl'):
        if json_file.is_file():
            token_sums = process_json_file(json_file)
            
            # Get relative path from root directory
            relative_path = json_file.relative_to(root_path)
            
            results.append({
                'file': str(relative_path),  # Full relative path including filename
                'input_token': token_sums['input_token'],
                'output_token': token_sums['output_token']
            })
    
    return results

def save_results(results: List[Dict[str, any]], output_file: str) -> None:
    """Save aggregated results to a JSON file.
    
    Args:
        results: List of result dictionaries.
        output_file: Path to the output JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

def main():
    """Main execution function."""
    # Configuration
    root_directory = ""
    output_file = ""
    
    # Process files
    aggregated_results = scan_directory(root_directory)
    
    # Save results
    save_results(aggregated_results, output_file)
    
    print(f"Processed {len(aggregated_results)} JSON files. Results saved to {output_file}")

if __name__ == "__main__":
    main()