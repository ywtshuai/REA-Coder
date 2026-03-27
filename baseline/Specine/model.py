import json
import time
from typing import Dict, Any, List

import requests
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


def _format_messages_plain(messages: List[Dict[str, str]]) -> str:
    """
    将多轮 messages 合并成一个文本块，喂给 Gemini 的 contents.parts[0].text
    """
    chunks: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "user").strip()
        content = msg.get("content") or ""
        # 保守做法：保留 role 以免丢信息
        chunks.append(f"{role}:\n{content}".strip())
    return "\n\n".join(chunks).strip()


def _wrap_messages_for_gemini(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    按示例的“Gemini 3 Pro 请求体结构”，将其 JSON 字符串塞进 chat.completions 的 user.content
    """
    merged_text = _format_messages_plain(messages)

    generation_config: Dict[str, Any] = {
        "thinkingConfig": {
            "thinkingBudget": 0
        },
        "thinking_level": "minimal"
    }

    request_body: Dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {"text": merged_text}
                ]
            }
        ],
        "generationConfig": generation_config
    }
    return [
        {
            "role": "user",
            "content": json.dumps(request_body, ensure_ascii=False)
        }
    ]


def load_model(model_name):
    if model_name in ['Qwen2.5-Coder-7B-Instruct', 'deepseek-coder-7b-instruct-v1.5']:
        model_name = f"./LLMs/{model_name}"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        exit()

    return model, tokenizer


def generate_code(args, prompt, model, tokenizer, max_new_tokens=60000):
    if args.debug:
        print(prompt)
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    if args.model_name == 'Qwen2.5-Coder-1.5B-Instruct':
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.8
        )
    else:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.8
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    code = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if args.debug:
        print(code)

    return code


def generate_code_api(args, prompt, max_tokens=8192):
    if args.debug:
        print(prompt)
    if args.model_name == "gpt-4o-mini-2024-07-18":
        client = OpenAI(
            base_url='Base Url',
            api_key='Your API KEY'
        )
    elif args.model_name == "gemini-1.5-flash-002":
        client = OpenAI(
            base_url='Base Url',
            api_key='Your API KEY'
        )
    elif args.model_name == "deepseek-chat":
        client = OpenAI(
            base_url='Base Url',
            api_key='Your API KEY'
        )
    elif args.model_name == "qwen3-coder-30b-a3b-instruct":
        client = OpenAI(
            base_url='Base Url',
            api_key='Your API KEY'
        )
    elif args.model_name == "gpt-5-mini-2025-08-07":
        client = OpenAI(
            base_url='Base Url',
            api_key='Your API KEY'
        )
    elif args.model_name == "gemini-3-flash-preview-minimal":
        client = OpenAI(
            base_url='Base Url',
            api_key='Your API KEY'
        )
    elif args.model_name == "gemini-3-flash-preview":
        client = OpenAI(
            base_url='Base Url',
            api_key='Your API KEY'
        )
    else:
        raise ValueError("Unsupported model for API generation.")
    messages = [
        {"role": "user", "content": prompt}
    ]

    code = ""
    retry = 6
    while retry:
        try:
            retry -= 1
            if args.model_name == "gpt-5-mini-2025-08-07":
                response = client.responses.create(
                    model=args.model_name,
                    input=messages,
                    max_output_tokens=max_tokens,
                    reasoning={"effort": "minimal"}
                )

                code = response.output_text
            elif args.model_name == "gemini-3-flash-preview":
                chunks: List[str] = []
                for msg in messages:
                    role = (msg.get("role") or "user").strip()
                    content = msg.get("content") or ""
                    # 简单拼接：role:\ncontent
                    chunks.append(f"{role}:\n{content}".strip())
                merged_text = "\n\n".join(chunks).strip()

                generation_config = {
                    "thinkingConfig": {"thinkingBudget": 0},
                    "thinking_level": "minimal"
                }

                # 构建 Gemini 原生请求体
                gemini_request_body = {
                    "contents": [
                        {
                            "parts": [
                                {"text": merged_text}
                            ]
                        }
                    ],
                    "generationConfig": generation_config
                }

                # 将原生请求体打包成 JSON 字符串，放入 Standard Chat API 的 content 字段
                wrapped_messages = [
                    {
                        "role": "user",
                        "content": json.dumps(gemini_request_body, ensure_ascii=False)
                    }
                ]

                # --- 4. 构建最终 HTTP 请求 (原 _build_request) ---
                url = str(client.base_url).rstrip("/") + "/chat/completions"
                api_key = client.api_key

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }

                payload = {
                    "model": args.model_name,
                    "messages": wrapped_messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.8,
                }
                response = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False),
                    timeout=240,
                )

                data = response.json()

                # 按照 OpenAI 兼容格式提取内容
                code = data["choices"][0]["message"]["content"]
            else:
                completion = client.chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    temperature=0.8,
                    max_tokens=max_tokens
                )
                code = completion.choices[0].message.content
            break
        except Exception as e:
            code = None
            time.sleep(20)
            print(e, 'sleep and retry!')
            continue

    if args.debug:
        print(code)
    return code
