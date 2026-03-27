

import json
from collections import Counter


def count_classifications(input_path: str, output_path: str):

    label_counter = Counter()


    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            label = item.get("routing_label", "Unknown")
            label_counter[label] += 1


    total = sum(label_counter.values())


    result = {
        "label_counts": dict(label_counter),
        "total_items": total
    }


    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f" Statistics saved to {output_path}")


if __name__ == "__main__":
    input_file = ""
    output_file = ""
    count_classifications(input_file, output_file)
