import os
import json
import yaml
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_messages(prompt_cfg: dict, prompt_text: str, code_text: str) -> list:
    return [
        {"role": "system", "content": prompt_cfg["system"]},
        {"role": "user", "content": prompt_cfg["user"].format(prompt=prompt_text, code=code_text)}
    ]

def build_prompt_string(
    messages: list,
    tokenizer,
    enable_thinking: bool = True
) -> str:

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking 
    )
    return text

def classify_with_transformers(
    prompt_text: str,
    code_text: str,
    model,
    tokenizer,
    prompt_cfg: dict,
    enable_thinking: bool = True,
    max_new_tokens: int = 128
) -> tuple[str, str, str]:
    messages = build_messages(prompt_cfg, prompt_text, code_text)
    prompt = build_prompt_string(messages, tokenizer, enable_thinking)


    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


    if enable_thinking:
        think_end = response.find("</think>")
        if think_end != -1:
            thinking_content = response[:think_end].strip()
            final_answer = response[think_end + len("</think>"):].strip()
        else:
            thinking_content = ""
            final_answer = response
    else:
        thinking_content = ""
        final_answer = response


    match = re.match(r"(Easy|Medium|Hard)\s*[:：]\s*(.+)", final_answer, re.IGNORECASE)
    if match:
        label = match.group(1).capitalize()
        reason = match.group(2).strip()
    else:
        label, reason = "Unknown", final_answer

    return label, reason, thinking_content if enable_thinking else ""

def process_dataset(config_path: str, enable_thinking: bool = False):
    config = load_config(config_path)
    model_cfg = config["model"]
    prompt_cfg = config["prompts"]
    proc_cfg = config["processing"]


    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    print(" Loading model with Transformers (8-GPU parallel)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_path"],
        torch_dtype=torch.bfloat16,  
        device_map="auto",          
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_path"],
        trust_remote_code=True
    )

    with open(proc_cfg["input_path"], "r", encoding="utf-8") as fin:
        lines = fin.readlines()
    max_samples = proc_cfg.get("max_samples", None)
    lines_to_process = lines if max_samples is None else lines[:max_samples]

    with open(proc_cfg["output_path"], "w", encoding="utf-8") as fout:
        for line in tqdm(lines_to_process):
            item = json.loads(line)
            prompt_text = item.get("prompt", "")
            code_text = item.get("canonical_solution", "")

            label, reason, thinking_content = classify_with_transformers(
                prompt_text=prompt_text,
                code_text=code_text,
                model=model,
                tokenizer=tokenizer,
                prompt_cfg=prompt_cfg,
                enable_thinking=enable_thinking,
                max_new_tokens=proc_cfg.get("max_tokens", 128)
            )

            item["routing_label"] = label
            item["routing_reason"] = reason
            if enable_thinking:
                item["thinking_content"] = thinking_content
            json.dump(item, fout, ensure_ascii=False)
            fout.write("\n")
            fout.flush()

    print(f"Routing completed. Output saved to {proc_cfg['output_path']}")

if __name__ == "__main__":
    process_dataset(
        config_path="",
        enable_thinking=False  
    )