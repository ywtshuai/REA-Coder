"""
muFiX stdio 模式：支持 CodeContest、apps、apps_eval、xCodeEval 数据集
使用 apps_eval 进行 stdin/stdout 评测，支持 qwen 和 deepseek-chat API。

目录结构:
  mufix_outputs_{dataset}_{model}/
    run_20260221_123456/     # 每次运行新建
      generation_checkpoint.json
      {task_id}-7-all
      ...

用法:
  python generate_stdio.py --model qwen --dataset code_contests --prompt mufix --temperature 0.7
  python generate_stdio.py --model qwen --dataset xCodeEval --prompt mufix --temperature 0.7
  python generate_stdio.py --model qwen --dataset livecodebench --prompt mufix --temperature 0.7
  python generate_stdio.py --model qwen --dataset code_contests --mode evaluate --resume-from mufix_outputs_code_contests_qwen/run_20260221_123456
"""

import os
import re
import json
import time
import argparse
import platform
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from apps_eval.executor import eval_single_code_stdio

def get_code(input_string, split_word):
    """从 markdown 中提取 ```python ... ``` 代码块"""
    if input_string.find(split_word[0]) == -1 or input_string.find(split_word[1]) == -1:
        output_string = input_string
    else:
        pattern = re.compile(fr'{re.escape(split_word[0])}(.*?){re.escape(split_word[1])}', re.DOTALL)
        matches = re.findall(pattern, input_string)
        output_string = ''.join(matches)

    code_1 = []
    for i in output_string.split('\n'):
        if (i[:7] != 'assert ' if len(i) >= 7 else True) and (i[:1] != '#' if len(i) >= 1 else True):
            code_1.append(i)
    return '\n'.join(code_1).strip()


# ============ API 调用 ============
# 用于并发生成时累计 token 使用量（参考 scot_baseline）
_token_usage_lock = threading.Lock()
_total_tokens = 0


def _reset_token_counter():
    """每次生成流程开始前重置 token 计数"""
    global _total_tokens
    with _token_usage_lock:
        _total_tokens = 0


def _add_token_usage(tokens: int):
    """线程安全地累加 token 使用量"""
    global _total_tokens
    if tokens > 0:
        with _token_usage_lock:
            _total_tokens += tokens


def _get_total_tokens() -> int:
    """获取累计 token 使用量"""
    with _token_usage_lock:
        return _total_tokens


def _call_api_chat(messages: list, model_name: str, temperature: float, max_tokens: int = 8192, api_key_override: str = None) -> str:
    """调用 OpenAI 兼容的 Chat API，带有自动截断续写机制"""
    import requests
    
    # 模型配置
    if model_name == "qwen" or "qwen" in model_name.lower():
        base_url = os.environ.get("MODEL_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        api_key = api_key_override or os.environ.get("DASHSCOPE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        model = os.environ.get("MODEL_C", "qwen3-coder-30b-a3b-instruct")
    elif model_name == "deepseek-chat" or "deepseek" in model_name.lower():
        base_url = os.environ.get("MODEL_API_BASE_URL", "https://api.deepseek.com/v1")
        api_key = api_key_override or os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        model = os.environ.get("MODEL_C", "deepseek-chat")
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    if not api_key:
        raise RuntimeError("请设置 API Key：PowerShell 用 $env:DASHSCOPE_API_KEY='sk-xxx'，或使用 --api_key 参数")
    
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    max_continuations = 3  # 最大允许的续写次数
    full_content = ""
    current_messages = list(messages)  # 浅拷贝，避免污染外部原始的 messages
    
    for cont_idx in range(max_continuations + 1):
        payload = {
            "model": model,
            "messages": current_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        success = False
        current_response_text = ""
        finish_reason = ""
        
        # 内层循环：处理网络错误和重试
        for attempt in range(5):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=120)
                if r.status_code != 200:
                    raise RuntimeError(f"API error {r.status_code}: {r.text[:500]}")
                data = r.json()
                
                choice = data["choices"][0]
                current_response_text = choice["message"]["content"]
                finish_reason = choice.get("finish_reason", "")
                
                usage = data.get("usage", {})
                tokens = usage.get("total_tokens") or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
                _add_token_usage(tokens)
                
                full_content += current_response_text
                success = True
                break  # 网络请求成功，跳出内层循环
            except Exception as e:
                print(f"API 调用失败 (attempt {attempt+1}): {e}")
                time.sleep(min(2 ** attempt, 20))
                
        if not success:
            raise RuntimeError("API 调用失败，已达最大重试次数")
            
        # --- 截断检测逻辑 ---
        is_truncated = False
        
        # 1. 官方标记检测 (DeepSeek/Qwen 均支持 finish_reason="length")
        if finish_reason in ["length", "max_tokens"]:
            is_truncated = True
            
        # 2. 启发式检测 (防丢失：代码块开始了但没闭合)
        python_starts = full_content.count("```python")
        code_ends = full_content.count("```") - python_starts
        if python_starts > code_ends:
            is_truncated = True
            
        # 如果没有截断，说明生成完毕，跳出外层续写循环
        if not is_truncated:
            break
            
        # 检查是否达最大续写次数
        if cont_idx == max_continuations:
            print(f"⚠️ 达到最大续写次数 ({max_continuations})，停止续写。")
            break
            
        # 触发续写：将当前的半截回答追加到历史，并给出继续指令
        print(f"⚠️ 响应过长发生截断 (正在进行第 {cont_idx + 1} 次自动续写)...")
        current_messages.append({"role": "assistant", "content": current_response_text})
        current_messages.append({
            "role": "user", 
            "content": "你的输出被截断了。请**严格从你刚才中断的最后一个字符开始**继续输出，不要输出任何开场白、解释或重复前面的内容。如果在写代码，请不要重新加上 ```python 标记。"
        })
        
    return full_content

def call_llms(args, content, title=('Prompt', 'Reply')):
    if args.model in ('qwen', 'deepseek-chat'):
        reply = _call_api_chat(
            content,
            args.model,
            args.temperature,
            max_tokens=8192,
            api_key_override=getattr(args, 'api_key', None)
        )
        return reply
    elif args.model == 'gpt-5-mini-2025-08-07':
        from openai import OpenAI
        api_key = getattr(args, 'api_key', None) or os.environ.get("GPT5_MINI_API_KEY", "***")
        base_url = os.environ.get("MODEL_API_BASE_URL", "http://api.yesapikey.com/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)

        max_continuations = 3
        full_content = ""
        current_messages = list(content)

        for cont_idx in range(max_continuations + 1):
            success = False
            for attempt in range(5):
                try:
                    response = client.responses.create(
                        model=args.model,
                        input=current_messages,
                        reasoning={"effort": "minimal"}
                    )
                    
                    if hasattr(response, 'usage') and response.usage:
                        tokens = getattr(response.usage, 'total_tokens', 0)
                        _add_token_usage(tokens)

                    current_response_text = response.output_text
                    full_content += current_response_text
                    success = True
                    break
                except Exception as e:
                    wait = min(2 ** attempt, 30)
                    print(f"[WARN] API 调用失败 (attempt {attempt+1}/5): {e}，{wait}s 后重试...")
                    time.sleep(wait)
            if not success:
                raise RuntimeError("API 调用失败，已达最大重试次数")
            
            is_truncated = False
            python_starts = full_content.count("```python")
            code_ends = full_content.count("```") - python_starts
            if python_starts > code_ends:
                is_truncated = True
                
            if not is_truncated:
                break
                
            if cont_idx == max_continuations:
                print(f"⚠️ 达到最大续写次数 ({max_continuations})，停止续写。")
                break
                
            print(f"⚠️ 响应过长发生截断 (正在进行第 {cont_idx + 1} 次自动续写)...")
            current_messages.append({"role": "assistant", "content": current_response_text})
            current_messages.append({
                "role": "user", 
                "content": "你的输出被截断了。请**严格从你刚才中断的最后一个字符开始**继续输出，不要输出任何开场白、解释或重复前面的内容。如果在写代码，请不要重新加上 ```python 标记。"
            })
            
        return full_content

    elif args.model == 'gemini-3-flash-preview':
        from openai import OpenAI
        import json
        api_key = getattr(args, 'api_key', None) or os.environ.get("GEMINI_FLASH_API_KEY", "***")
        base_url = os.environ.get("MODEL_API_BASE_URL", "http://api.yesapikey.com/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)

        max_continuations = 3
        full_content = ""
        current_messages = list(content)

        for cont_idx in range(max_continuations + 1):
            success = False
            for attempt in range(5):
                try:
                    problem_text = "\n\n".join([f"{m['role']}:\n{m['content']}" for m in current_messages])
                    request_body = {
                        "contents": [
                            {
                                "parts": [
                                    {"text": problem_text}
                                ]
                            }
                        ],
                        "generationConfig": {
                            "thinkingConfig": {
                                "thinkingBudget": 0
                            },
                            "thinking_level": "minimal"
                        }
                    }
                    response = client.chat.completions.create(
                        model="gemini-3-flash-preview",
                        messages=[
                            {
                                "role": "user",
                                "content": json.dumps(request_body)
                            }
                        ],
                        max_tokens=60000,
                        temperature=args.temperature
                    )
                    
                    if hasattr(response, 'usage') and response.usage:
                        tokens = getattr(response.usage, 'total_tokens', 0)
                        _add_token_usage(tokens)
                    
                    current_response_text = response.choices[0].message.content
                    full_content += current_response_text
                    success = True
                    break
                except Exception as e:
                    wait = min(2 ** attempt, 30)
                    print(f"[WARN] API 调用失败 (attempt {attempt+1}/5): {e}，{wait}s 后重试...")
                    time.sleep(wait)
            if not success:
                raise RuntimeError("API 调用失败，已达最大重试次数")
            
            is_truncated = False
            python_starts = full_content.count("```python")
            code_ends = full_content.count("```") - python_starts
            if python_starts > code_ends:
                is_truncated = True
                
            if not is_truncated:
                break
                
            if cont_idx == max_continuations:
                print(f"⚠️ 达到最大续写次数 ({max_continuations})，停止续写。")
                break
                
            print(f"⚠️ 响应过长发生截断 (正在进行第 {cont_idx + 1} 次自动续写)...")
            current_messages.append({"role": "assistant", "content": current_response_text})
            current_messages.append({
                "role": "user", 
                "content": "你的输出被截断了。请**严格从你刚才中断的最后一个字符开始**继续输出，不要输出任何开场白、解释或重复前面的内容。如果在写代码，请不要重新加上 ```python 标记。"
            })
            
        return full_content

    elif args.model == 'chatgpt':
        import openai
        chat = openai.ChatCompletion.create(
            model=args.openai_model,
            messages=content,
            temperature=args.temperature
        )
        usage = getattr(chat, 'usage', None)
        if usage:
            tokens = getattr(usage, 'total_tokens', None) or (getattr(usage, 'prompt_tokens', 0) + getattr(usage, 'completion_tokens', 0))
            _add_token_usage(tokens)
        return chat.choices[0].message.content
    elif args.model == 'deepseek-coder':
        reply = args.local_model.codegen2(content, do_sample=True, num_samples=1)[0].split('```')[0].strip()
        return reply
    else:
        raise ValueError(f"Unsupported model: {args.model}")


# ============ stdio 格式解析 ============

def _parse_stdio_prompt(data: str) -> tuple:
    """
    解析 stdio 格式的 prompt，提取 problem 和 test cases 文本。
    Returns: (original_prompt_without_testcases, testcases_text, testcases_list)
    """
    # 查找 "# Test Cases" 或 "(1) Input:" 开始的位置
    idx = data.find('\n(1) Input:')
    if idx == -1:
        idx = data.find('\n# Test Cases')
    if idx == -1:
        return data.strip(), "", []
    
    original = data[:idx].strip()
    testcases_block = data[idx:].strip()
    
    # 解析为 [{"input": "...", "output": "..."}, ...]
    testcases_list = []
    pattern = re.compile(r'\((\d+)\)\s+Input:\s*\n(.*?)(?=Expected output:)\s*Expected output:\s*\n(.*?)(?=\n\(\d+\) Input:|\Z)', re.DOTALL)
    for m in pattern.finditer(testcases_block):
        inp = m.group(2).strip()
        out = m.group(3).strip()
        testcases_list.append({"input": inp, "output": out})
    
    # 若正则未匹配，尝试简单按 "(n) Input:" 分割
    if not testcases_list and "(1) Input:" in testcases_block:
        parts = re.split(r'\(\d+\)\s+Input:\s*\n', testcases_block)[1:]
        for p in parts:
            if 'Expected output:' in p:
                inp, rest = p.split('Expected output:', 1)
                out = rest.strip().split('\n\n')[0].strip()
                testcases_list.append({"input": inp.strip(), "output": out})
    
    return original, testcases_block, testcases_list


# ============ thought_eliciting_phase (stdio) ============

def thought_eliciting_phase_stdio(args, path: str, task_id: str) -> str:
    prompt_path = os.path.join(path, f"{task_id}-prompt")
    meta_path = os.path.join(path, f"{task_id}-meta.json")
    
    if not os.path.exists(prompt_path):
        return -1
    
    data = open(prompt_path, 'r', encoding='utf-8').read()
    if data.strip() == '':
        return -1
    
    original, testcases_block, testcases_list = _parse_stdio_prompt(data)
    if not testcases_list:
        # 无测试用例，直接生成代码
        content = [{"role": "user", "content": f"{data.strip()}\n\n# Instruction: Please complete the code in a markdown style code block. Use sys.stdin for input and print() for output:\n\n```python\n\n```\n"}]
        reply = call_llms(args, content, ('Prompt', 'Code'))
        time.sleep(1)
        return reply
    
    # 构建分析 prompt（stdio 格式）
    analysis_prompt = f"# Problem:\n{original}\n\n# Test Cases:\n{testcases_block}\n\n"
    analysis_prompt += "# Instruction: For each test case, analyze step by step. You MUST follow the format:\n"
    analysis_prompt += '"(<n>) Input: <describe input>. Expected output: <describe output>.\nAnalysis: <your analysis>.\nTherefore, the expected output is <value>."\n\n'
    analysis_prompt += "Provide your analysis for each test case:"
    
    content = [{"role": "user", "content": analysis_prompt}]
    analysis_1 = call_llms(args, content, ('Analysis-1', 'Reply'))
    time.sleep(1)
    
    # 简化：直接使用 analysis_1 生成代码（跳过 analysis-2/3 以节省调用）
    code_prompt = f"# Problem:\n{original}\n\n# Analysis of Test Cases:\n{analysis_1}\n\n"
    code_prompt += "# Instruction: Please complete the Python code in a markdown style code block. Use sys.stdin.read() for input and print() for output:\n\n```python\n\n```\n"
    
    content = [{"role": "user", "content": code_prompt}]
    reply = call_llms(args, content, ('Code', 'Reply'))
    time.sleep(1)
    
    return reply


# ============ feedback_phase (stdio) ============

def feedback_phase_stdio(args, path: str, task_id: str, code_1: str) -> str:
    meta_path = os.path.join(path, f"{task_id}-meta.json")
    prompt_path = os.path.join(path, f"{task_id}-prompt")
    
    if not os.path.exists(meta_path):
        return code_1
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    # CodeContests: feedback 阶段仅用 public 用例，避免 private 泄漏导致评估偏高
    # test_cases=all 仅用于最终评估
    test_cases = meta.get("public_test_cases") or meta.get("test_cases", [])
    if not test_cases:
        return code_1

    code_clean = get_code(code_1, ["```python", "```"])
    if not code_clean.strip():
        code_clean = code_1
    
    all_passed, feedback = eval_single_code_stdio(code_clean, test_cases, timeout=10.0)
    
    if all_passed:
        return code_clean
    
    if feedback == -1:
        # 执行出错，重新生成
        content = [{"role": "user", "content": f"# Instruction: The code has execution errors. Please regenerate correct code:\n\n```python\n\n```\n"}]
        reply = call_llms(args, content, ('Prompt', 'Code'))
        return reply
    
    # 有失败用例，带 feedback 重新生成
    data = open(prompt_path, 'r', encoding='utf-8').read()
    original, _, _ = _parse_stdio_prompt(data)
    
    prompt = f"{feedback}\n\n# Problem:\n{original}\n\n"
    prompt += "# Instruction: The code failed some test cases. Please fix and regenerate the code in a markdown style code block:\n\n```python\n\n```\n"
    content = [
        {"role": "user", "content": f"# Problem:\n{original}\n\n# Instruction: Complete the code:\n\n```python\n\n```\n"},
        {"role": "assistant", "content": code_clean},
        {"role": "user", "content": prompt}
    ]
    reply = call_llms(args, content, ('Feedback', 'Code'))
    return reply


# ============ 并行处理单任务 ============

def process_single_task(args_tuple):
    """
    处理单个任务的代码生成（供并行调用）
    Args: (args, prompts_path, task_id, run_dir, temp_key, idx, total)
    Returns: 成功时返回 {"task_id", "instance_id", "code"}，跳过时返回 None
    """
    args, prompts_path, task_id, run_dir, temp_key, idx, total = args_tuple
    try:
        reply = thought_eliciting_phase_stdio(args, prompts_path, task_id)
        if reply == -1:
            return None

        all1_path = run_dir / f"{task_id}-{temp_key}-all-1"
        if not all1_path.exists():
            all1_path.write_text(reply, encoding='utf-8')

        if args.prompt == 'mufix':
            reply = feedback_phase_stdio(args, prompts_path, task_id, reply)

        if reply == -1:
            return None

        (run_dir / f"{task_id}-{temp_key}-all").write_text(reply, encoding='utf-8')

        meta_path = os.path.join(prompts_path, f"{task_id}-meta.json")
        instance_id = task_id
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                instance_id = json.load(f).get("instance_id", task_id)

        code_clean = get_code(reply, ["```python", "```"])
        if not code_clean.strip():
            code_clean = reply

        return {
            "task_id": task_id,
            "instance_id": instance_id,
            "code": code_clean,
        }
    except Exception as e:
        print(f"\n[{idx}/{total}] {task_id} 异常: {e}")
        return None


def _extract_index(instance_id: str) -> int:
    """从 instance_id 提取题目索引。
    支持两种格式:
      - 纯数字字符串: '42' -> 42  (apps/apps_eval 实际格式)
      - 带前缀格式:   'apps_42' -> 42, 'apps_eval_123' -> 123
    """
    s = str(instance_id)
    if s.isdigit():
        return int(s)
    m = re.search(r'_(\d+)$', s)
    return int(m.group(1)) if m else -1


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str,
                        help="qwen | deepseek-chat | chatgpt | deepseek-coder | gpt-5-mini-2025-08-07 | gemini-3-flash-preview")
    parser.add_argument("--prompt", default="mufix", type=str)
    parser.add_argument("--dataset", required=True, type=str,
                        help="code_contests | code_contests_raw | apps | apps_eval | xCodeEval | livecodebench")
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--limit", default=None, type=int, help="闄愬埗澶勭悊鏁伴噺")
    parser.add_argument("--instance-ids", default=None, type=str,
                        help="浠呭鐞嗘寚瀹氶鍙凤紝閫楀彿鍒嗛殧锛屽: --instance-ids 3708,3677,3832")
    parser.add_argument("--prompts_dir", default="./dataset/prompts", help="prompt 数据目录")
    parser.add_argument("--output_base", default="./mufix_outputs", help="输出根目录")
    parser.add_argument("--api_key", default=None, help="API Key")
    parser.add_argument("--mode", default="all", choices=["all", "generate", "evaluate"],
                        help="all=生成+评估, generate=仅生成, evaluate=仅评估")
    parser.add_argument("--resume-from", default=None,
                        help="evaluate 模式时指定 run 目录路径")
    parser.add_argument("--eval-dataset", default=None,
                        help="evaluate 模式：覆盖评估数据集。如 apps checkpoint 用 apps_eval 评估（更多测试用例）")
    parser.add_argument("--eval-output", default=None,
                        help="evaluate 模式：评估结果输出目录。若指定，会在该目录下创建 run_{timestamp} 存放结果，不覆盖原 run 目录")
    parser.add_argument("--workers", default=16, type=int,
                        help="生成阶段并行线程数（默认: 16）")
    args = parser.parse_args()
    
    selected_instance_ids = None
    if args.instance_ids:
        selected_instance_ids = {
            item.strip() for item in args.instance_ids.split(',') if item.strip()
        }

    if args.model == 'chatgpt':
        import openai
        args.openai_model = 'gpt-3.5-turbo-0613'
        args.local_model = None
    elif args.model == 'deepseek-coder':
        from model import make_model
        args.local_model = make_model(name='deepseek-coder-6.7b-instruct', batch_size=1, temperature=args.temperature)
    
    # prompt 数据路径（只读）
    prompts_path = os.path.join(args.prompts_dir, args.dataset)
    if not os.path.isdir(prompts_path):
        print(f"错误: prompt 目录不存在: {prompts_path}")
        print("请先运行: python prepare_stdio_data.py --dataset", args.dataset)
        return 1
    
    # 输出目录：mufix_outputs_{dataset}_{model}/run_{timestamp}
    output_dir = f"{args.output_base}_{args.dataset}_{args.model}"
    
    if args.mode == 'evaluate' and args.resume_from:
        run_dir = Path(args.resume_from)
        if not run_dir.exists():
            print(f"错误: 目录不存在: {run_dir}")
            return 1
        if run_dir.name.startswith('run_'):
            pass
        else:
            runs = sorted(run_dir.glob('run_*'), key=lambda x: x.name, reverse=True)
            if not runs:
                print(f"错误: 在 {run_dir} 中未找到 run_* 目录")
                return 1
            run_dir = runs[0]
        print(f"评估模式: 使用运行目录 {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(output_dir) / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"运行目录: {run_dir}")
    
    task_ids_path = os.path.join(prompts_path, "task_ids.json")
    if os.path.exists(task_ids_path):
        with open(task_ids_path, 'r', encoding='utf-8') as f:
            all_task_id = json.load(f)
    else:
        all_task_id = []
        for f in os.listdir(prompts_path):
            if f.endswith('-prompt') and not f.endswith('-prompt-1'):
                all_task_id.append(f.replace('-prompt', ''))
        all_task_id.sort()
    
    if selected_instance_ids:
        all_task_id = [task_id for task_id in all_task_id if task_id in selected_instance_ids]

    if args.limit:
        all_task_id = all_task_id[:args.limit]
    
    print(f"模型: {args.model}, 数据集: {args.dataset}, 任务数: {len(all_task_id)}, 并行数: {args.workers}")
    
    temp_key = int(args.temperature * 10)
    all_results = []
    
    ckpt_dataset_name = args.dataset  # 默认与 --dataset 一致
    if args.mode == 'evaluate':
        # 从 checkpoint 加载
        ckpt = run_dir / "generation_checkpoint.json"
        if not ckpt.exists():
            print(f"错误: 找不到 {ckpt}，请先运行生成模式")
            return 1
        with open(ckpt, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_results = data['results']
        if selected_instance_ids:
            all_results = [r for r in all_results if r["instance_id"] in selected_instance_ids]
        ckpt_dataset_name = data.get('dataset_name', args.dataset)
        generation_time = data.get('generation_time', 0.0)
        token_usage = data.get('token_usage', {})
        total_tokens = token_usage.get('total', 0) if isinstance(token_usage, dict) else (token_usage or 0)
        print(f"已加载 {len(all_results)} 条生成结果")
        if generation_time > 0:
            print(f"   生成耗时: {generation_time:.2f} 秒")
        if total_tokens > 0:
            print(f"   总 Token: {total_tokens:,}")
    else:
        # 并行生成代码（参考 scot_baseline）
        _reset_token_counter()
        gen_start = time.time()
        args_list = [
            (args, prompts_path, task_id, run_dir, temp_key, idx + 1, len(all_task_id))
            for idx, task_id in enumerate(all_task_id)
        ]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_single_task, a) for a in args_list]
            for future in tqdm(as_completed(futures), total=len(futures), desc="生成代码"):
                result = future.result()
                if result is not None:
                    all_results.append(result)

        generation_time = time.time() - gen_start
        total_tokens = _get_total_tokens()

        # 按 task_id 原始顺序排序
        task_id_order = {tid: i for i, tid in enumerate(all_task_id)}
        all_results.sort(key=lambda r: task_id_order.get(r["task_id"], 999999))

        # 保存 checkpoint（包含生成时间和 token 使用量，供 report 使用）
        ckpt_data = {
            "results": all_results,
            "dataset_name": args.dataset,
            "model_name": args.model,
            "temperature": args.temperature,
            "generation_time": generation_time,
            "token_usage": {
                "total": total_tokens,
                "average_per_problem": total_tokens / len(all_results) if all_results else 0,
            },
            "timestamp": datetime.now().isoformat(),
        }
        (run_dir / "generation_checkpoint.json").write_text(
            json.dumps(ckpt_data, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        print(f"生成结果已保存: {run_dir / 'generation_checkpoint.json'}")
        
        if args.mode == 'generate':
            print(f"\n下一步评估: python eval_stdio.py --run-dir \"{run_dir}\"")
            return 0
    
    # 评估
    from apps_eval.data import get_data
    from apps_eval.parallel_runner import eval_code

    eval_dataset_name = args.eval_dataset or ckpt_dataset_name
    if args.eval_dataset:
        print(f"评估数据集覆盖: {ckpt_dataset_name} -> {eval_dataset_name}")

    base = Path(__file__).resolve().parent.parent.parent
    datasets_dir = str(base / "Self-collaboration-Code-Generation-main" / "Datasets")
    dataset = get_data(eval_dataset_name, datasets_dir)
    if selected_instance_ids:
        dataset = [inst for inst in dataset if inst.instance_id in selected_instance_ids]

    result_by_id = {r["instance_id"]: r["code"] for r in all_results}
    match_by_index = False
    if eval_dataset_name != ckpt_dataset_name and eval_dataset_name in ('apps_eval', 'apps') and ckpt_dataset_name in ('apps_eval', 'apps'):
        match_by_index = True
        result_by_index = {_extract_index(r["instance_id"]): r for r in all_results}
        eval_dataset = []
        eval_solutions = []
        for inst in dataset:
            idx = _extract_index(inst.instance_id)
            if idx in result_by_index:
                eval_dataset.append(inst)
                eval_solutions.append(result_by_index[idx]["code"])
    else:
        eval_dataset = []
        eval_solutions = []
        for inst in dataset:
            if inst.instance_id in result_by_id:
                eval_dataset.append(inst)
                eval_solutions.append(result_by_id[inst.instance_id])
    
    if not eval_dataset:
        print("错误: 无匹配的评估数据")
        return 1

    # 决定评估输出目录：--eval-output 指定时在该目录下创建 run_{timestamp}，否则写入源 run_dir
    if getattr(args, 'eval_output', None):
        eval_out_base = Path(args.eval_output)
        eval_out_base.mkdir(parents=True, exist_ok=True)
        ts_eval = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_run_dir = eval_out_base / f"run_{ts_eval}"
        out_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"评估结果将保存到: {out_run_dir}")
    else:
        out_run_dir = run_dir

    # 平台感知的评估并行数（参考 scot_baseline）
    if platform.system() == 'Windows':
        eval_workers = min(args.workers * 2, 16)
    else:
        eval_workers = min(args.workers * 4, 60)
    print(f"\n评估 {len(eval_dataset)} 道题目 (评估数据集: {eval_dataset_name}, workers={eval_workers})...")
    eval_results = eval_code(eval_dataset, eval_solutions, timeout=10.0, workers=eval_workers, show_progress=True)
    
    pass_count = sum(1 for acc, _ in eval_results if acc == 1.0)
    total = len(eval_results)
    pass_at_1 = pass_count / total if total > 0 else 0
    avg_pass_ratio = sum(acc for acc, _ in eval_results) / total if total > 0 else 0
    
    print("\n" + "=" * 50)
    print(f"Pass@1: {pass_count}/{total} ({pass_at_1*100:.2f}%)")
    print(f"AvgPassRatio: {avg_pass_ratio:.4f}")
    print("=" * 50)
    
    # 构建详细结果（参考 scot_baseline）
    if match_by_index:
        result_by_index_full = {_extract_index(r["instance_id"]): r for r in all_results}
    else:
        result_by_id_full = {r["instance_id"]: r for r in all_results}
    detailed_results = []
    for inst, (acc_rate, eval_result_list) in zip(eval_dataset, eval_results):
        if match_by_index:
            r = result_by_index_full.get(_extract_index(inst.instance_id), {})
        else:
            r = result_by_id_full.get(inst.instance_id, {})
        detailed_results.append({
            "instance_id": inst.instance_id,
            "code": r.get("code", ""),
            "accuracy": acc_rate,
            "passed": acc_rate == 1.0,
            "test_results": [
                {
                    "status": er.status,
                    "time_cost": er.time_cost,
                    "stdin": (str(er.stdin)[:200] + "..." if len(str(er.stdin or "")) > 200 else str(er.stdin or "")),
                    "stdout": (str(er.stdout)[:200] + "..." if len(str(er.stdout or "")) > 200 else str(er.stdout or "")),
                    "expected": (str(er.expected)[:200] + "..." if len(str(er.expected or "")) > 200 else str(er.expected or "")),
                    "stderr": (str(er.stderr)[:100] + "..." if len(str(er.stderr or "")) > 100 else str(er.stderr or "")),
                }
                for er in eval_result_list
            ],
        })
    
    summary = {
        "summary": {
            "method": "muFiX (stdio)",
            "dataset": ckpt_dataset_name,
            "eval_dataset": eval_dataset_name,
            "model": args.model,
            "temperature": args.temperature,
            "pass_at_1": pass_at_1,
            "passed": pass_count,
            "total": total,
            "avg_pass_ratio": avg_pass_ratio,
            "generation_time": generation_time,
            "token_usage": {
                "total": total_tokens,
                "average_per_problem": total_tokens / total if (total > 0 and total_tokens > 0) else 0,
            },
            "config": {"prompt": args.prompt},
            "timestamp": datetime.now().isoformat(),
        },
        "results": detailed_results,
    }
    
    out_json = out_run_dir / "eval_results.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"结果已保存: {out_json}")
    
    # 生成 REPORT.txt
    report_file = out_run_dir / "REPORT.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("muFiX stdio 运行报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"方法: muFiX (stdio)\n")
        if getattr(args, 'eval_output', None):
            f.write(f"源 run 目录: {run_dir}\n")
        f.write(f"生成数据集: {ckpt_dataset_name.upper()}\n")
        if eval_dataset_name != ckpt_dataset_name:
            f.write(f"评估数据集: {eval_dataset_name.upper()} (更多测试用例)\n")
        f.write(f"模型: {args.model}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"题目数量: {total}\n\n")
        f.write(f"结果统计:\n")
        f.write(f"  - Pass@1: {pass_at_1*100:.2f}% ({pass_count}/{total})\n")
        f.write(f"  - AvgPassRatio: {avg_pass_ratio:.4f}\n")
        f.write(f"  - 生成总耗时: {generation_time:.2f} 秒\n")
        if total_tokens > 0:
            f.write(f"  - 总 Token: {total_tokens:,}\n")
            f.write(f"  - 平均每题 Token: {total_tokens/total:.0f}\n")
        else:
            f.write(f"  - 总 Token: N/A\n")
        f.write("\n")
        f.write(f"详细结果:\n")
        for dr in detailed_results:
            status = "✅ PASS" if dr["passed"] else "❌ FAIL"
            f.write(f"  {status} {dr['instance_id']} (准确率: {dr['accuracy']*100:.0f}%)\n")
            for i, tr in enumerate(dr["test_results"], 1):
                f.write(f"      用例{i}: {tr['status']} (耗时: {tr['time_cost']:.2f}s)\n")
    print(f"报告已保存: {report_file}")
    
    return 0


if __name__ == '__main__':
    exit(main())
