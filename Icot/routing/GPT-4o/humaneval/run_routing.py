
import json
import yaml
import time
import re
from tqdm import tqdm
from openai import OpenAI


def load_config(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def classify_with_gpt(client, model: str, system_prompt: str, user_prompt_template: str, prompt_text: str, temperature: float, max_tokens: int) -> tuple[str, str]:
    user_prompt = user_prompt_template.format(prompt_text=prompt_text)

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content.strip()


        match = re.match(r"(Easy|Medium|Hard)\s*[:：]\s*(.+)", content, re.IGNORECASE)
        if match:
            label = match.group(1).capitalize()
            reason = match.group(2).strip()
            return label, reason
        else:
            return "Unknown", content

    except Exception as e:
        print(" Error:", e)
        return "Error", str(e)


def process_dataset(config_path: str):
    config = load_config(config_path)
    openai_cfg = config["openai"]
    prompt_cfg = config["prompts"]
    proc_cfg = config["processing"]

    client = OpenAI(
        api_key=openai_cfg["api_key"],
        base_url=openai_cfg.get("base_url", "")
    )

    with open(proc_cfg["input_path"], "r", encoding="utf-8") as f:
        lines = f.readlines()

    output = []
    for line in tqdm(lines[:proc_cfg.get("max_samples")]):
        item = json.loads(line)
        prompt_text = item.get("prompt", "")

        label, reason = classify_with_gpt(
            client=client,
            model=openai_cfg["model"],
            system_prompt=prompt_cfg["system"],
            user_prompt_template=prompt_cfg["user"],
            prompt_text=prompt_text,
            temperature=openai_cfg.get("temperature", 0),
            max_tokens=openai_cfg.get("max_tokens", 100)
        )

        item["routing_label"] = label
        item["routing_reason"] = reason
        output.append(item)
        time.sleep(proc_cfg.get("sleep_seconds", 1.0))

    with open(proc_cfg["output_path"], "w", encoding="utf-8") as f:
        for item in output:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f" Done. Processed {len(output)} items. Results saved to {proc_cfg['output_path']}.")

if __name__ == "__main__":
    process_dataset("")
