import json
from pathlib import Path
from typing import Dict, List, Union
from transformers import AutoTokenizer
import torch
import yaml
import os
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print(f"Using GPU: {torch.cuda.get_device_name(2)}")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")
class TokenCounter:
    def __init__(self, tokenizer_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
    def count_tokens(self, text: str) -> int:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    def count_tokens_in_messages(self, messages: List[Dict[str, str]]) -> int:
        total_tokens = 0
        for msg in messages:
            total_tokens += self.count_tokens(msg["content"])
        return total_tokens
def load_config(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)  
def load_json_data(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
def ensure_file_path(file_path: str) -> None:
    dir_path = os.path.dirname(file_path)
    if dir_path:  
        os.makedirs(dir_path, exist_ok=True)  
    if os.path.exists(file_path):  
        with open(file_path, "w", encoding="utf-8") as f:
            f.truncate()
def process_task(
    task_id: str,
    prompt: str,
    test_case: str,
    xcot_entries: List[Dict],
    code_entries: List[Dict],
    token_counter: TokenCounter,
    config: Dict,
) -> Dict[str, Union[str, int]]:
    xcot_messages = [
        {
            "role": "system",
            "content": config["messages"]["xcot_system"],
        }
    ]
    for example in config["messages"]["xcot_examples"]:
        xcot_messages.append({"role": "user", "content": example["user"]})
        xcot_messages.append({"role": "assistant", "content": example["assistant"]})
    xcot_user_content = config["messages"]["xcot_user"].format(prompt=prompt,test_case=test_case)
    xcot_messages.append({"role": "user", "content": xcot_user_content})
    xcot_input_tokens = token_counter.count_tokens_in_messages(xcot_messages)
    code_input_tokens = 0
    code_output_tokens = 0
    for xcot_entry in xcot_entries:
        if xcot_entry["task_id"] != task_id:
            continue
        code_messages = [
            {
                "role": "system",
                "content": config["messages"]["code_system"],
            }
        ]
        for example in config["messages"]["code_examples"]:
            code_messages.append({"role": "user", "content": example["user"]})
            code_messages.append({"role": "assistant", "content": example["assistant"]})
        code_user_content = config["messages"]["code_user"].format(prompt=prompt,xcot=xcot_entry['xcot'],test_case=test_case)
        code_messages.append({"role": "user", "content": code_user_content})
        code_input_tokens += token_counter.count_tokens_in_messages(code_messages)
    for xcot_entry in xcot_entries:
        if xcot_entry["task_id"] == task_id:
            code_output_tokens += token_counter.count_tokens(xcot_entry["xcot"])
    for code_entry in code_entries:
        if code_entry["task_id"] == task_id:
            code_output_tokens += token_counter.count_tokens(code_entry["completion"])
    return {
        "task_id": task_id,
        "approach": "icot",
        "input_token": xcot_input_tokens + code_input_tokens,
        "output_token": code_output_tokens,
    }
def main():
    tokenizer_dir = ""  
    token_counter = TokenCounter(tokenizer_dir)
    config = load_config("")  
    xcot_entries = load_json_data("")
    code_entries = load_json_data("")
    dataset_entries = load_json_data("")
    result_file = ""
    ensure_file_path(result_file)  
    results = []
    for task_entry in dataset_entries:
        task_id = task_entry["task_id"]
        prompt = task_entry["prompt"]
        test_case = task_entry["test_list"]
        task_xcots = [x for x in xcot_entries if x["task_id"] == task_id]
        task_codes = [c for c in code_entries if c["task_id"] == task_id]
        if not task_xcots or not task_codes:
            continue 
        result = process_task(task_id, prompt,test_case, task_xcots, task_codes, token_counter, config)
        results.append(result)
    with open(result_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print("")
if __name__ == "__main__":
    main()