import json
import os
import pandas as pd
from typing import List


def summarize_token_counts_from_dir(root_dir: str, output_excel_path: str):
    summary = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".jsonl"):
                jsonl_path = os.path.join(dirpath, filename)
                input_sum = 0
                output_sum = 0

                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        obj = json.loads(line.strip())
                        input_sum += obj.get("input_token", 0)
                        output_sum += obj.get("output_token", 0)


                parts = os.path.normpath(jsonl_path).split(os.sep)
                file_name = parts[-1]
                eval_name = parts[-2] if len(parts) >= 2 else ""
                model_name = parts[-3] if len(parts) >= 3 else ""

                summary.append({
                    "model_name": model_name,
                    "eval_name": eval_name,
                    "file_name": file_name,
                    "input_token": input_sum,
                    "output_token": output_sum
                })


    df = pd.DataFrame(summary)
    df.to_excel(output_excel_path, index=False)



root_dir = ""
output_excel = ""

summarize_token_counts_from_dir(root_dir, output_excel)