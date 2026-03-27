"""
SCoT (Structured Chain-of-Thought) Baseline
使用 3-Shot Few-Shot 策略，提供算法示例来教导模型处理不同的逻辑结构
"""

import os
import sys
import re
import json
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入依赖
from core.generate_code import build_llm, LLMConfig
from apps_eval.data import get_data, InstanceData
from apps_eval.parallel_runner import eval_code

# ============================================================
# 数据集与模型选项（与 ldb_baseline 对齐，便于统一调用）
# ============================================================
DATASET_OPTIONS = ["code_contests_raw", "code_contests", "apps", "apps_eval", "xCodeEval", "livecodebench"]
# 常用模型：deepseek-chat, qwen3-coder-30b-a3b-instruct, gpt-5-mini-2025-08-07（可传任意模型名，输出目录按后缀区分）
MODEL_SUFFIX_MAP = {
    "gpt-5-mini-2025-08-07": "gpt5mini",
    "gemini-3-flash-preview": "gemini3flash",
}
def _model_to_suffix(model_name: str) -> str:
    if model_name in MODEL_SUFFIX_MAP:
        return MODEL_SUFFIX_MAP[model_name]
    if "qwen" in (model_name or "").lower():
        return "qwen"
    return "deepseek"

def _setup_model_env(model_name: str, api_key: str = None) -> None:
    """根据模型名设置 API base_url、api_key_env 与 MODEL_C；可选传入 --api-key。"""
    if model_name == "gpt-5-mini-2025-08-07":
        os.environ["MODEL_API_BASE_URL"] = "http://api.yesapikey.com/v1"
        os.environ["MODEL_API_KEY_ENV"] = "GPT5_MINI_API_KEY"
        os.environ["MODEL_C"] = model_name
        if api_key:
            os.environ["GPT5_MINI_API_KEY"] = api_key
        else:
            os.environ.setdefault("GPT5_MINI_API_KEY", "***")
    elif model_name == "gemini-3-flash-preview":
        # Gemini 3 Flash 通过 yesapikey 的 OpenAI 兼容 Chat Completions 接口调用
        os.environ["MODEL_API_BASE_URL"] = "http://api.yesapikey.com/v1"
        os.environ["MODEL_API_KEY_ENV"] = "GEMINI_FLASH_API_KEY"
        os.environ["MODEL_C"] = model_name
        if api_key:
            os.environ["GEMINI_FLASH_API_KEY"] = api_key
    elif "qwen" in (model_name or "").lower():
        os.environ["MODEL_API_BASE_URL"] = os.environ.get("MODEL_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        os.environ["MODEL_API_KEY_ENV"] = "DASHSCOPE_API_KEY"
        os.environ["MODEL_C"] = model_name
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key
    else:
        os.environ["MODEL_API_BASE_URL"] = os.environ.get("MODEL_API_BASE_URL", "https://api.deepseek.com/v1")
        os.environ["MODEL_API_KEY_ENV"] = os.environ.get("MODEL_API_KEY_ENV", "DEEPSEEK_API_KEY")
        os.environ["MODEL_C"] = model_name
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key

# ============================================================
# 全局配置：数据集默认值（未传 --dataset 时使用，建议用命令行 --dataset 覆盖）
# ============================================================
# Options: 'code_contests', 'code_contests_raw', 'apps', 'apps_eval', 'xCodeEval', 'livecodebench'
DATASET_NAME = 'xCodeEval'

# ============================================================
# SCoT System Prompt (3-Shot)
# ============================================================

# CodeContests 专用 Prompt (Standard I/O 风格)
SCOT_SYSTEM_PROMPT_CODE_CONTESTS = """You are an expert programmer.
You are required to generate a Structured Chain-of-Thought (SCoT) before writing the code.
The SCoT must describe the logical steps using three specific structures: "Sequence", "Branch", and "Loop".

You must follow this format:
1. **Input/Output Analysis**: Define input format and expected output.
2. **Structured Plan**: Describe the algorithm using "Sequence", "Branch", and "Loop".
3. **Code**: Write the full Python script using `sys.stdin`.

Here are 3 examples of the required format:

--- EXAMPLE 1 ---
Problem: Find two numbers in `nums` that add up to `target`.

SCoT:
1. Input/Output Analysis:
   - Input: Array `nums`, Integer `target`.
   - Output: Indices of the two numbers.
2. Structured Plan:
   - Sequence: Initialize an empty dictionary `num_map`.
   - Loop: Iterate through `nums` with index `i` and value `num`:
     - Sequence: Calculate `complement = target - num`.
     - Branch: If `complement` is in `num_map`:
       - Sequence: Return `[num_map[complement], i]`.
     - Sequence: Store `num_map[num] = i`.
   - Sequence: Return empty list if no solution.

3. Code:
```python
import sys

def two_sum():
    lines = sys.stdin.read().strip().split('\\n')
    nums = list(map(int, lines[0].split()))
    target = int(lines[1])
    
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            print(f"{num_map[complement]} {i}")
            return
        num_map[num] = i
    print("")

if __name__ == "__main__":
    two_sum()
```

--- EXAMPLE 2 ---
Problem: Check if the input string containing brackets is valid.

SCoT:
1. Input/Output Analysis:
   - Input: String s.
   - Output: Boolean (True/False).
2. Structured Plan:
   - Sequence: Initialize an empty stack and a mapping of closing to opening brackets.
   - Loop: Iterate through each character char in s:
     - Branch: If char is a closing bracket:
       - Branch: If stack is empty or top element doesn't match:
         - Sequence: Return False.
       - Sequence: Pop from stack.
     - Branch: Else (opening bracket):
       - Sequence: Push char onto stack.
   - Sequence: Return True if stack is empty, else False.

3. Code:
```python
import sys

def is_valid():
    s = sys.stdin.read().strip()
    
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack[-1] != mapping[char]:
                print("False")
                return
            stack.pop()
        else:
            stack.append(char)
    
    print("True" if not stack else "False")

if __name__ == "__main__":
    is_valid()
```

--- EXAMPLE 3 ---
Problem: Merge all overlapping intervals.

SCoT:
1. Input/Output Analysis:
   - Input: List of intervals.
   - Output: List of merged intervals.
2. Structured Plan:
   - Sequence: Sort intervals by start time.
   - Sequence: Initialize merged list with the first interval.
   - Loop: Iterate through remaining intervals:
     - Sequence: Let last be the last interval in merged, curr be current interval.
     - Branch: If curr.start <= last.end (Overlap):
       - Sequence: Update last.end to max(last.end, curr.end).
     - Branch: Else (No overlap):
       - Sequence: Append curr to merged.
   - Sequence: Return merged.

3. Code:
```python
import sys

def merge_intervals():
    lines = sys.stdin.read().strip().split('\\n')
    n = int(lines[0])
    intervals = []
    for i in range(1, n + 1):
        start, end = map(int, lines[i].split())
        intervals.append([start, end])
    
    if not intervals:
        print("")
        return
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for curr in intervals[1:]:
        last = merged[-1]
        if curr[0] <= last[1]:
            last[1] = max(last[1], curr[1])
        else:
            merged.append(curr)
    
    for interval in merged:
        print(f"{interval[0]} {interval[1]}")

if __name__ == "__main__":
    merge_intervals()
```

--- END OF EXAMPLES ---


Now, solve the following problem using the same format.

IMPORTANT:
- Focus your SCoT on algorithmic design, state transitions, and complexity. Avoid endless manual tracing of large test cases.
- Do NOT output multiple draft code blocks. Output exactly ONE final, optimized code block in step 3.
- Use sys.stdin.read() for input.
- Output to stdout.
- Wrap code in ```python ... ```.

"""





# ============================================================
# SCoT Agent
# ============================================================

class SCoTAgent:
    """SCoT (Structured Chain-of-Thought) Agent"""
    
    def __init__(self, model_name: str = "deepseek-chat", temperature: float = 0.0):
        """
        初始化 SCoT Agent
        
        Args:
            model_name: 模型名称
            temperature: 温度参数
        """
        self.model_name = model_name
        # 设置模型环境变量（如果还没有设置）
        if 'MODEL_C' not in os.environ:
            os.environ['MODEL_C'] = model_name
        elif model_name != os.environ.get('MODEL_C'):
            # 如果传入的模型名和环境变量不同，更新环境变量
            os.environ['MODEL_C'] = model_name
        
        # 使用 build_llm 函数构建 LLM 客户端
        # 增加 max_tokens 以避免复杂问题的代码被截断
        # gpt-5-mini-2025-08-07: 使用 max_tokens=60000 与 reasoning={"effort": "minimal"}
        if model_name == "gpt-5-mini-2025-08-07":
            self.llm = build_llm(
                model_env='MODEL_C',
                temperature=temperature,
                max_tokens=60000,
                reasoning={"effort": "minimal"},
            )
        elif model_name == "gemini-3-flash-preview":
            # Gemini 3 Flash：通过 Chat Completions API，使用更大的 max_tokens
            self.llm = build_llm(
                model_env='MODEL_C',
                temperature=temperature,
                max_tokens=60000,
            )
        else:
            self.llm = build_llm(
                model_env='MODEL_C',
                temperature=temperature,
                max_tokens=8192  
            )
        self.temperature = temperature
        self.dataset_name = None  # 由 main 设置为当前 --dataset

    def generate(self, problem_desc: str) -> str:
        """
        生成代码

        Args:
            problem_desc: 问题描述

        Returns:
            生成的 Python 代码
        """
        response = self.generate_with_response(problem_desc)
        code = self._extract_code(response)
        return code
    
    def generate_with_response(self, problem_desc: str) -> str:
        """
        生成代码并返回完整响应（包含增强版自动截断续写逻辑）
        
        Args:
            problem_desc: 问题描述
            
        Returns:
            完整的 LLM 响应（包含 SCoT 和代码）
        """
        # 根据 dataset 类型选择 System Prompt（与 code_contests/apps/xCodeEval/livecodebench 等一致均为 stdio）
        ds = getattr(self, 'dataset_name', None) or DATASET_NAME
        if ds in ('code_contests', 'code_contests_raw', 'apps', 'apps_eval', 'xCodeEval', 'livecodebench'):
            system_prompt = SCOT_SYSTEM_PROMPT_CODE_CONTESTS
            user_content = f"Problem:\n{problem_desc}\n\nSCoT:"
        else:
            raise ValueError(f"Unsupported dataset: {ds}")
        
        # 对不同模型构造不同的消息格式
        if self.model_name == "gemini-3-flash-preview":
            # 将 SCoT 的系统 + 用户提示拼接成一个 problem 文本，包在 Gemini 原生 request_body 中
            problem = f"{system_prompt}\n\n{user_content}"
            request_body = {
                "contents": [
                    {
                        "parts": [
                            {"text": problem}
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
            messages = [
                {
                    "role": "user",
                    "content": json.dumps(request_body, ensure_ascii=False)
                }
            ]
        else:
            # 标准 OpenAI 兼容 messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        
        try:
            full_response = ""
            max_continuations = 3  # 最大允许的续写次数（初始 1 次 + 续写 3 次 = 最多 4 次 API 调用）
            
            for i in range(max_continuations + 1):
                # 调用 LLM
                current_response = self.llm.chat(messages, temperature=self.temperature)
                full_response += current_response
                
                # --- 增强版截断检测逻辑 ---
                is_truncated = False
                prompt_for_continue = ""
                
                # 统计代码块标记数量
                python_starts = full_response.count("```python")
                code_ends = full_response.count("```") - python_starts
                
                # 情况 1：代码块开始了，但没闭合（在写代码时截断）
                if python_starts > code_ends:
                    is_truncated = True
                    prompt_for_continue = (
                        "你的输出在代码生成阶段被截断了。请**严格从你刚才中断的最后一个字符开始**"
                        "继续输出代码，不要输出任何开场白、解释或重复前面的代码，也不要重新加上 ```python 标记。"
                    )
                    
                # 情况 2：连 ```python 都还没出现，说明在 SCoT 思考阶段就截断了
                # 这里设置一个合理的长度阈值（如单次返回超过 2000 字符），防止把模型简短的“拒绝回答”误判为截断
                elif python_starts == 0 and len(current_response) > 2000:
                    is_truncated = True
                    prompt_for_continue = (
                        "你的输出在思考计划(SCoT)阶段被截断了。请**严格从你刚才中断的最后一个字符开始**"
                        "继续输出剩下的分析，并确保最终输出包含 ```python 包裹的完整代码。"
                    )
                
                # 如果没有检测到截断，说明生成完毕，跳出循环
                if not is_truncated:
                    break
                
                # 检查是否达到最大续写次数限制
                if i == max_continuations:
                    print(f"⚠️ 达到最大续写次数 ({max_continuations})，停止续写。")
                    break
                    
                # 触发续写，将当前的回复和继续的指令追加到上下文中
                print(f"⚠️ 题目响应过长发生截断 (正在进行第 {i+1} 次自动续写)...")
                messages.append({"role": "assistant", "content": current_response})
                messages.append({"role": "user", "content": prompt_for_continue})
                
            return full_response
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return f"# Generation failed: {e}"
    
    def _extract_code(self, response: str) -> str:
        """
        从响应中提取代码（改进版，支持截断的代码）
        
        Args:
            response: LLM 响应
            
        Returns:
            提取的代码
        """
        # 方法1: 尝试提取完整的 ```python ... ``` 代码块
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            # 返回第一个匹配的代码块
            code = matches[0].strip()
            return code
        
        # 方法2: 如果没有完整代码块，尝试提取从 ```python 开始的代码（即使被截断）
        pattern_start = r"```python(.*)$"
        matches_start = re.findall(pattern_start, response, re.DOTALL)
        
        if matches_start:
            print("⚠️  警告: 代码可能被截断（没有结束标记），尝试提取")
            code = matches_start[0].strip()
            # 移除可能的 SCoT 分析部分（以数字+点开头的行，如 "1. **Input/Output Analysis**"）
            lines = code.split('\n')
            code_lines = []
            in_code = False
            for line in lines:
                # 检测是否开始真正的代码（import 或 def 语句）
                if line.strip().startswith(('import ', 'from ', 'def ', 'class ')):
                    in_code = True
                # 跳过 SCoT 分析行
                if not in_code and re.match(r'^\d+\.\s+\*\*', line.strip()):
                    continue
                if in_code or line.strip().startswith(('import ', 'from ', 'def ', 'class ', '#', 'if ', 'while ', 'for ')):
                    code_lines.append(line)
            
            if code_lines:
                return '\n'.join(code_lines).strip()
            else:
                return code.strip()
        
        # 方法3: 寻找代码块（查找以 import/def 开头的部分）
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # 检测代码开始
            if line.strip().startswith(('import ', 'from ', 'def ', 'class ')):
                in_code = True
            
            # 如果在代码区域内，收集所有行
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            print("⚠️  警告: 未找到标准代码块标记，尝试提取代码部分")
            return '\n'.join(code_lines).strip()
        
        # 方法4: 如果都失败了，返回整个响应
        print("❌ 错误: 无法提取代码，返回整个响应")
        return response.strip()


# ============================================================
# 详细日志系统
# ============================================================

class DetailedLogger:
    """为每个题目创建详细的日志记录"""
    
    def __init__(self, output_dir: str = "scot_baseline_outputs_deepseek"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
    def create_problem_dir(self, instance_id: str) -> Path:
        """为单个问题创建目录"""
        problem_dir = self.run_dir / instance_id
        problem_dir.mkdir(exist_ok=True)
        return problem_dir
    
    def save_problem_info(self, problem_dir: Path, instance: InstanceData):
        """保存问题描述"""
        with open(problem_dir / "problem_statement.txt", "w", encoding="utf-8") as f:
            f.write(instance.problem_statement)
    
    def save_generation_info(self, problem_dir: Path, code: str, response: str):
        """保存生成信息"""
        # 保存最终代码（确保是纯代码）
        with open(problem_dir / "generated_code.py", "w", encoding="utf-8") as f:
            # 确保保存的是提取出的代码，而不是整个响应
            f.write(code)
        
        # 保存完整响应（包含 SCoT）
        with open(problem_dir / "full_response.txt", "w", encoding="utf-8") as f:
            f.write(response)
        
        # 如果代码看起来被截断或包含非代码内容，记录警告
        if len(response) > len(code) * 2 or "**Input/Output Analysis**" in code:
            with open(problem_dir / "extraction_warning.txt", "w", encoding="utf-8") as f:
                f.write("警告: 代码提取可能不完整或包含非代码内容\n")
                f.write(f"响应长度: {len(response)} 字符\n")
                f.write(f"提取代码长度: {len(code)} 字符\n")
    
    def save_summary(self, summary: Dict):
        """保存总体摘要"""
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


# ============================================================
# 主流程
# ============================================================

def _run_reeval_raw_summary(
    raw_summary_path: str,
    max_workers: int = 16,
    model_name: str = 'deepseek-chat',
    reeval_dataset: str = 'code_contests'
):
    """
    从 raw_summary.json 读取已生成的代码，使用指定数据集的 all_test_cases 重评估。
    支持 code_contests (Datasets/code_contests.jsonl) 或 new/code_contests (Datasets/new/code_contests.jsonl)。
    """
    import platform
    
    if not raw_summary_path:
        print("❌ 错误: reeval_raw_summary 模式需要指定 --raw-summary 参数")
        print("   示例: python scot_baseline.py --mode reeval_raw_summary --raw-summary scot_baseline_outputs/run_20260210_201740/raw_summary.json")
        sys.exit(1)
    
    raw_path = Path(raw_summary_path)
    if not raw_path.exists():
        print(f"❌ 错误: raw_summary 文件不存在: {raw_summary_path}")
        sys.exit(1)
    
    print(f"\n📂 加载 raw_summary: {raw_path}")
    with open(raw_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    raw_results = raw_data.get('results', [])
    if not raw_results:
        print("❌ 错误: raw_summary 中无 results 数据")
        sys.exit(1)
    
    print(f"✅ 从 raw_summary 读取 {len(raw_results)} 个问题的代码")
    
    # 加载指定数据集（支持 code_contests 或 new/code_contests）
    dataset_label = "new/code_contests" if reeval_dataset == "new/code_contests" else "code_contests"
    print(f"\n📂 加载 {dataset_label} 数据集（all_test_cases）...")
    dataset = get_data(reeval_dataset)
    dataset_by_id = {inst.instance_id: inst for inst in dataset}
    print(f"✅ 数据集共 {len(dataset)} 个问题")
    
    # 按 instance_id 匹配，构建评估数据
    eval_dataset = []
    eval_solutions = []
    all_results = []
    skipped = 0
    for r in raw_results:
        instance_id = r.get('instance_id')
        code = r.get('code', '')
        if instance_id not in dataset_by_id:
            skipped += 1
            continue
        inst = dataset_by_id[instance_id]
        eval_dataset.append(inst)
        eval_solutions.append(code)
        all_results.append({
            'instance_id': instance_id,
            'code': code,
            'test_cases': inst.test_cases,
            'generation_time': r.get('generation_time', 0.0),
            'problem_dir': r.get('problem_dir', ''),
            'response': r.get('response', '')
        })
    
    if skipped > 0:
        print(f"⚠️  跳过 {skipped} 个在数据集中不存在的 instance_id")
    
    if not eval_dataset:
        print("❌ 错误: 没有可匹配的题目")
        sys.exit(1)
    
    print(f"✅ 将评估 {len(eval_dataset)} 个问题（使用 {dataset_label} all_test_cases）")
    
    # 创建输出目录（根据数据集区分）
    output_dir = Path('scot_baseline_reeval_code_contests_new' if reeval_dataset == 'new/code_contests' else 'scot_baseline_reeval_code_contests')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ 输出目录: {run_dir}")
    
    # 运行评估
    eval_workers = min(max_workers * 2, 16) if platform.system() == 'Windows' else min(max_workers * 4, 60)
    print(f"\n🔍 评估中... ({eval_workers} 进程)")
    eval_start = time.time()
    eval_results = eval_code(eval_dataset, eval_solutions, timeout=10.0, workers=eval_workers, show_progress=True)
    eval_time = time.time() - eval_start
    
    # 统计
    total = len(eval_results)
    passed = sum(1 for acc, _ in eval_results if acc == 1.0)
    pass_at_1 = (passed / total * 100) if total > 0 else 0.0
    
    # 保存
    detailed = []
    for i, (result, (acc_rate, eval_list)) in enumerate(zip(all_results, eval_results)):
        detailed.append({
            'instance_id': result['instance_id'],
            'problem_dir': result.get('problem_dir', ''),
            'code': result['code'],
            'accuracy': acc_rate,
            'passed': acc_rate == 1.0,
            'generation_time': result.get('generation_time', 0.0),
            'test_results': [
                {'status': r.status, 'time_cost': r.time_cost, 'stdin': str(r.stdin)[:100] if r.stdin else '',
                 'stdout': str(r.stdout)[:100] if r.stdout else '', 'expected': str(r.expected)[:100] if r.expected else ''}
                for r in eval_list
            ],
            'response': result.get('response', '')
        })
    
    summary = {
        'summary': {
            'method': 'SCoT Reeval (raw_summary)',
            'dataset': reeval_dataset,
            'source': str(raw_path),
            'model': raw_data.get('summary', {}).get('model', model_name),
            'pass_at_1': pass_at_1,
            'passed': passed,
            'total': total,
            'time_cost': {'evaluation': eval_time},
            'config': {'eval_workers': eval_workers},
            'timestamp': datetime.now().isoformat()
        },
        'results': detailed
    }
    
    with open(run_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    results_filename = "scot_baseline_reeval_code_contests_new_results.json" if reeval_dataset == 'new/code_contests' else "scot_baseline_reeval_code_contests_results.json"
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 报告
    with open(run_dir / "REPORT.txt", 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\nSCoT Raw Summary 重评估报告\n" + "=" * 80 + "\n\n")
        f.write(f"来源: {raw_path}\n")
        f.write(f"数据集: {reeval_dataset}\n")
        f.write(f"题目数: {total}\n")
        f.write(f"Pass@1: {pass_at_1:.2f}% ({passed}/{total})\n")
        f.write(f"评估耗时: {eval_time:.2f}s\n\n")
        for d in detailed:
            status = "✅ PASS" if d['passed'] else "❌ FAIL"
            f.write(f"  {status} {d['instance_id']} ({d['accuracy']*100:.0f}%)\n")
    
    print("\n" + "=" * 80)
    print(f"✅ Pass@1: {pass_at_1:.2f}% ({passed}/{total})")
    print(f"⏱️  评估耗时: {eval_time:.2f}s")
    print(f"📁 结果: {run_dir}")
    print(f"📁 摘要: {results_filename}")
    print("=" * 80)


def process_single_instance(args: tuple) -> Dict[str, Any]:
    """
    处理单个实例
    
    Args:
        args: (model_name, temperature, dataset_name, instance, idx, total, logger) 元组
        
    Returns:
        结果字典
    """
    model_name, temperature, dataset_name, instance, idx, total, logger = args
    
    try:
        print(f"[{idx}/{total}] 开始处理: {instance.instance_id}")
        
        agent = SCoTAgent(model_name=model_name, temperature=temperature)
        agent.dataset_name = dataset_name

        # 创建问题目录
        problem_dir = logger.create_problem_dir(instance.instance_id)
        logger.save_problem_info(problem_dir, instance)

        start_time = time.time()

        tokens_before = getattr(agent.llm, 'total_tokens', 0)

        # 生成代码（修改 generate 方法以返回完整响应）
        response = agent.generate_with_response(instance.problem_statement)
        code = agent._extract_code(response)
        tokens_after = getattr(agent.llm, 'total_tokens', 0)
        tokens_used = max(tokens_after - tokens_before, 0)
        
        generation_time = time.time() - start_time
        
        # 保存生成信息
        logger.save_generation_info(problem_dir, code, response)
        
        if code and not code.startswith("# Generation failed"):
            print(f"[{idx}/{total}] ✅ {instance.instance_id} 生成成功 (耗时: {generation_time:.2f}s)")
        else:
            print(f"[{idx}/{total}] ❌ {instance.instance_id} 生成失败")
        
        return {
            'instance_id': instance.instance_id,
            'code': code,
            'test_cases': instance.test_cases,
            'generation_time': generation_time,
            'problem_dir': str(problem_dir),
            'response': response,
            'tokens_used': tokens_used
        }
        
    except Exception as e:
        print(f"[{idx}/{total}] ❌ {instance.instance_id} 异常: {e}")
        return {
            'instance_id': instance.instance_id,
            'code': f"# Exception: {e}",
            'test_cases': instance.test_cases,
            'generation_time': 0.0,
            'error': str(e),
            'problem_dir': '',
            'tokens_used': 0
        }


def main(
    model_name: str = None,
    dataset_name: str = None,
    temperature: float = 0.0,
    max_workers: int = 16,
    output_dir: str = None,
    limit: int = None,
    instance_ids: str = None,
    mode: str = 'all',
    resume_from: str = None,
    raw_summary_path: str = None,
    reeval_dataset: str = 'code_contests',
    api_key: str = None,
    eval_dataset: str = None,
    eval_output: str = None,
):
    """
    主函数
    
    Args:
        model_name: 模型名称（默认从环境变量 MODEL_C 读取）
        temperature: 温度参数
        max_workers: 并行线程数
        output_dir: 输出目录（None 则根据模型自动选择）
        limit: 限制处理的问题数量（用于测试）
        mode: 运行模式 ('all': 生成+评估, 'generate': 仅生成, 'evaluate': 仅评估, 'reeval_raw_summary': 从 raw_summary 重评估)
        resume_from: 评估模式时，指定包含 generation_checkpoint.json 的 run 目录路径
        raw_summary_path: reeval_raw_summary 模式时，指定 raw_summary.json 路径
        reeval_dataset: reeval_raw_summary 模式时，指定评估数据集 ('code_contests' 或 'new/code_contests')
    """
    print("=" * 80)
    print("SCoT (Structured Chain-of-Thought) Baseline")
    if mode == 'generate':
        print("模式: 仅生成代码")
    elif mode == 'evaluate':
        print("模式: 仅评估代码")
    elif mode == 'reeval_raw_summary':
        print("模式: 从 raw_summary.json 重评估（使用新 all_test_cases）")
    else:
        print("模式: 完整流程（生成+评估）")
    print("=" * 80)

    if dataset_name is None:
        dataset_name = DATASET_NAME
    selected_instance_ids = None
    if instance_ids:
        selected_instance_ids = {
            item.strip() for item in instance_ids.split(',') if item.strip()
        }
    # evaluate 模式：若未显式传 --model，则以 checkpoint 中的 model_name 为准（避免误用环境变量导致报告错配）
    if mode != 'evaluate' and model_name is None:
        model_name = os.environ.get('MODEL_C', 'deepseek-chat')

    if mode == 'reeval_raw_summary':
        _run_reeval_raw_summary(
            raw_summary_path=raw_summary_path,
            max_workers=max_workers,
            model_name=model_name or os.environ.get('MODEL_C', 'deepseek-chat'),
            reeval_dataset=reeval_dataset
        )
        return

    # 评估模式必须指定 resume_from
    if mode == 'evaluate' and not resume_from:
        print("❌ 评估模式需要指定 --resume-from（包含 generation_checkpoint.json 的 run 目录）")
        sys.exit(1)

    generation_dataset_name = dataset_name  # 生成阶段数据集（evaluate 时会从 checkpoint 覆盖）

    # 非 evaluate：此时 model_name 已确定，设置环境并确定输出目录
    if mode != 'evaluate':
        _setup_model_env(model_name, api_key)

        # 根据数据集 + 模型自动选择输出目录
        if output_dir is None:
            suffix = _model_to_suffix(model_name)
            run_name = f"{dataset_name}_{suffix}"
            output_dir = os.path.join("scot_baseline_outputs", run_name)
    
    print(f"\n⚙️  配置信息:")
    print(f"  - 运行模式: {mode.upper()}")
    if mode != 'evaluate':
        print(f"  - 数据集: {dataset_name.upper()}")
        print(f"  - 模型: {model_name}")
        print(f"  - Temperature: {temperature}")
        print(f"  - 并行数: {max_workers} (生成)")
        print(f"  - 输出目录: {output_dir}")
    else:
        print(f"  - resume_from: {resume_from}")
        print(f"  - Temperature: {temperature}")
        print(f"  - 并行数: {max_workers} (评估)")
    
    # generate/all：提前加载数据集；evaluate：等读取 checkpoint 后再加载 eval_dataset
    if mode != 'evaluate':
        # 加载数据集（根据 dataset_name）
        print(f"\n[步骤 1/5] 加载 {dataset_name.upper()} 数据集...")

        if dataset_name == 'code_contests_raw':
            try:
                dataset = get_data('code_contests_raw')
            except FileNotFoundError:
                print("❌ code_contests 数据集文件未找到！")
                sys.exit(1)
        elif dataset_name == 'code_contests':
            dataset = get_data('code_contests')
        elif dataset_name == 'apps':
            # 尝试加载 APPS 数据集，如果不存在则提示下载
            try:
                dataset = get_data('apps')
            except FileNotFoundError:
                print("❌ APPS 数据集文件未找到！")
                print("💡 请先运行以下命令下载数据集:")
                print("   python prepare_apps_data.py")
                sys.exit(1)
        elif dataset_name == 'apps_eval':
            # 尝试加载 APPS-Eval（competition 难度）数据集，如果不存在则提示下载
            try:
                dataset = get_data('apps_eval')
            except FileNotFoundError:
                print("❌ APPS-Eval 数据集文件未找到！")
                print("💡 请先运行以下命令下载数据集:")
                print("   python prepare_apps_eval_data.py")
                sys.exit(1)
        elif dataset_name == 'xCodeEval':
            try:
                dataset = get_data('xCodeEval')
            except FileNotFoundError:
                print("❌ xCodeEval 数据集文件未找到！")
                print("💡 请确保 Datasets/xCodeEval.jsonl 存在")
                sys.exit(1)
        elif dataset_name == 'livecodebench':
            try:
                dataset = get_data('livecodebench')
            except FileNotFoundError:
                print("❌ livecodebench 数据集文件未找到！")
                print("💡 请确保 Datasets/livecodebench.jsonl 存在")
                sys.exit(1)
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

        if selected_instance_ids:
            dataset = [inst for inst in dataset if inst.instance_id in selected_instance_ids]
            print(f"按题号过滤后，共 {len(dataset)} 个问题")

        # 如果指定了限制，只取前N个问题
        if limit is not None and limit > 0:
            dataset = dataset[:limit]
            print(f"✅ 加载完成，共 {len(dataset)} 个问题（限制为前 {limit} 个）")
        else:
            print(f"✅ 加载完成，共 {len(dataset)} 个问题")
    
    # 创建日志记录器
    # 评估模式：如果指定了 resume_from，直接使用该目录
    if mode == 'evaluate' and resume_from:
        # 使用指定的 run 目录
        resume_path = Path(resume_from)
        if not resume_path.exists():
            print(f"❌ 错误: 指定的目录不存在: {resume_from}")
            sys.exit(1)
        
        # 检查是否是 run 目录（包含时间戳）
        if resume_path.name.startswith('run_'):
            # 直接使用该目录
            logger = DetailedLogger.__new__(DetailedLogger)
            logger.output_dir = resume_path.parent
            logger.run_dir = resume_path
            print(f"✅ 使用已有运行目录: {logger.run_dir}")
        else:
            # 可能是 output_dir，查找最新的 run 目录
            run_dirs = sorted(resume_path.glob('run_*'), key=lambda x: x.name, reverse=True)
            if not run_dirs:
                print(f"❌ 错误: 在 {resume_from} 中未找到 run_* 目录")
                sys.exit(1)
            logger = DetailedLogger.__new__(DetailedLogger)
            logger.output_dir = resume_path
            logger.run_dir = run_dirs[0]
            print(f"✅ 使用最新运行目录: {logger.run_dir}")
    else:
        # 正常创建新的 run 目录
        logger = DetailedLogger(output_dir)
        print(f"✅ 运行目录: {logger.run_dir}")
    
    # === 评估模式：从已有结果文件加载 ===
    if mode == 'evaluate':
        print(f"\n🔍 评估模式：从运行目录加载生成结果...")

        # 读取已保存的生成结果
        checkpoint_file = logger.run_dir / "generation_checkpoint.json"
        if not checkpoint_file.exists():
            print(f"❌ 错误: 找不到生成结果文件: {checkpoint_file}")
            print(f"💡 请先运行生成模式: python scot_baseline.py --mode generate --dataset <name> --model <name>")
            sys.exit(1)

        print(f"📂 加载生成结果: {checkpoint_file}")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)

        all_results = checkpoint_data['results']
        if selected_instance_ids:
            all_results = [r for r in all_results if r['instance_id'] in selected_instance_ids]
        generation_time = checkpoint_data.get('generation_time', 0.0)
        generation_dataset_name = checkpoint_data.get('dataset_name', generation_dataset_name)
        # evaluate 模式的模型：若未显式传 --model，以 checkpoint 的 model_name 为准
        if model_name is None:
            model_name = checkpoint_data.get('model_name', None) or os.environ.get('MODEL_C', 'deepseek-chat')

        # 设置模型环境（用于后续 token 统计/兼容逻辑等）
        _setup_model_env(model_name, api_key)

        # 评估时使用的数据集：--eval-dataset 覆盖 checkpoint 中的 dataset_name
        eval_dataset_name = eval_dataset or generation_dataset_name
        # 后续报告与 summary 统一使用“评估数据集”
        dataset_name = eval_dataset_name

        print(f"✅ 加载完成，共 {len(all_results)} 个问题的生成结果")
        print(f"   - generation_dataset: {generation_dataset_name}")
        print(f"   - eval_dataset: {eval_dataset_name}")
        print(f"   - model: {model_name}")
        print(f"   生成耗时: {generation_time:.2f} 秒（之前运行）")
        print(f"\n[步骤 1/5] 加载 {eval_dataset_name.upper()} 评估数据集...")
        dataset = get_data(eval_dataset_name)
        if selected_instance_ids:
            dataset = [inst for inst in dataset if inst.instance_id in selected_instance_ids]
        start_time = time.time()
        
    # === 生成模式或完整模式 ===
    else:
        print(f"\n[步骤 2/5] 初始化 SCoT Agent 配置...")
        print(f"✅ Agent 配置完成")
        
        # 并发生成代码
        print(f"\n[步骤 3/5] 开始生成代码...")
        print("=" * 80)
        
        all_results = []
        start_time = time.time()
        
        # 准备参数
        args_list = [
            (model_name, temperature, dataset_name, instance, idx + 1, len(dataset), logger)
            for idx, instance in enumerate(dataset)
        ]
        
        # 使用 ThreadPoolExecutor 并发生成
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_instance, args) for args in args_list]
            
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
        
        # 按照原始顺序排序（根据 instance_id）
        instance_id_order = {inst.instance_id: idx for idx, inst in enumerate(dataset)}
        all_results.sort(key=lambda x: instance_id_order.get(x['instance_id'], 999999))
        
        generation_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print(f"✅ 代码生成完成！")
        print(f"⏱️  生成耗时: {generation_time:.2f} 秒")
        print(f"📈 平均每题: {generation_time / len(dataset):.2f} 秒")
        print("=" * 80)
        
        # 💾 保存生成结果（用于评估模式）
        print(f"\n💾 保存生成结果...")
        generation_checkpoint = {
            'results': all_results,
            'generation_time': generation_time,
            'dataset_name': dataset_name,
            'model_name': model_name,
            'temperature': temperature,
            'token_usage': {
                'total': sum(result.get('tokens_used', 0) for result in all_results),
                'per_problem_details': [result.get('tokens_used', 0) for result in all_results]
            },
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = logger.run_dir / "generation_checkpoint.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(generation_checkpoint, f, indent=2, ensure_ascii=False)
        print(f"✅ 生成结果已保存到: {checkpoint_file}")
        
        # 如果是仅生成模式，到此结束
        if mode == 'generate':
            print("\n" + "=" * 80)
            print("🎉 生成阶段完成！")
            print("=" * 80)
            print(f"✅ 详细结果已保存到: {logger.run_dir}")
            print(f"\n💡 下一步: 运行评估")
            print(f"   python scot_baseline.py --mode evaluate --resume-from \"{logger.run_dir}\"")
            return
    
    # 评估代码
    print(f"\n[步骤 4/5] 评估生成的代码...")
    print("=" * 80)
    
    eval_start_time = time.time()
    
    # 准备评估数据
    eval_dataset = []
    eval_solutions = []
    
    for result in all_results:
        instance = next((inst for inst in dataset if inst.instance_id == result['instance_id']), None)
        if instance:
            eval_dataset.append(instance)
            eval_solutions.append(result['code'])
    
    # 确定评估的并行进程数（Windows 限制）
    import platform
    if platform.system() == 'Windows':
        eval_workers = max_workers # Windows: 最多 8 个进程
    else:
        eval_workers = max_workers # Linux/Mac: 最多 60 个进程
    
    print(f"🔍 使用 {eval_workers} 个进程并行评估... (平台: {platform.system()})")
    print(f"💡 提示: 安装 tqdm 可显示进度条 (pip install tqdm)\n")
    
    try:
        eval_results = eval_code(eval_dataset, eval_solutions, timeout=10.0, workers=eval_workers, show_progress=True)
        eval_time = time.time() - eval_start_time
        
        print(f"\n✅ 评估完成！")
        print(f"⏱️  评估耗时: {eval_time:.2f} 秒")
        print(f"📈 平均每题: {eval_time / len(dataset):.2f} 秒")
        
    except Exception as e:
        print(f"\n❌ 评估过程出错: {e}")
        print(f"⚠️  已生成的代码已保存在: {logger.run_dir}")
        print(f"💡 提示: 可以使用恢复脚本完成评估")
        raise
    
    # 💾 立即保存评估结果（防止后续统计阶段出错导致丢失）
    print(f"\n💾 保存评估中间结果...")
    try:
        eval_checkpoint = {
            'eval_results': [
                {
                    'instance_id': result['instance_id'],
                    'accuracy': acc_rate,
                    'passed': acc_rate == 1.0,
                    'test_count': len(eval_result_list),
                    'passed_tests': sum(1 for r in eval_result_list if r.status == 'AC'),
                    'test_statuses': [r.status for r in eval_result_list]
                }
                for result, (acc_rate, eval_result_list) in zip(all_results, eval_results)
            ],
            'eval_time': eval_time,
            'eval_workers': eval_workers,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = logger.run_dir / "eval_checkpoint.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(eval_checkpoint, f, indent=2, ensure_ascii=False)
        print(f"✅ 评估结果已保存到: {checkpoint_file}")
        print(f"   即使后续步骤出错，评估数据也不会丢失")
    except Exception as e:
        print(f"⚠️  警告: 保存评估结果失败: {e}")
        print(f"   将继续执行，但建议检查")
    
    # 统计结果
    print(f"\n[步骤 5/5] 统计结果...")
    print("=" * 80)
    
    total_problems = len(eval_results)
    passed = sum(1 for acc_rate, _ in eval_results if acc_rate == 1.0)
    pass_at_1 = (passed / total_problems * 100) if total_problems > 0 else 0.0
    
    total_time = time.time() - start_time
    
    total_tokens = sum(result.get('tokens_used', 0) for result in all_results)
    if total_tokens == 0:
        print("⚠️  注意: 未能统计到 Token 使用量")
    else:
        print("✅ 已汇总每题 Token 使用量")
    
    # 打印最终结果
    print(f"\n📊 最终结果")
    print("=" * 80)
    print(f"✅ Pass@1: {pass_at_1:.2f}% ({passed}/{total_problems})")
    print(f"⏱️  总耗时: {total_time:.2f} 秒")
    
    # 根据模式显示不同的时间分解
    if mode == 'evaluate':
        print(f"   - 代码评估: {eval_time:.2f} 秒")
        if 'generation_time' in locals():
            print(f"   (代码生成: {generation_time:.2f} 秒，之前运行)")
    else:
        print(f"   - 代码生成: {generation_time:.2f} 秒 ({generation_time/total_time*100:.1f}%)")
        if 'eval_time' in locals():
            print(f"   - 代码评估: {eval_time:.2f} 秒 ({eval_time/total_time*100:.1f}%)")
    
    if total_tokens > 0:
        print(f"🔢 总 Token 使用量: {total_tokens:,}")
        print(f"📈 平均每题 Token: {total_tokens/total_problems:.0f}")
        print(f"💰 估算成本 (按 $0.27/1M tokens): ${total_tokens * 0.27 / 1_000_000:.4f}")
    else:
        print(f"🔢 总 Token 使用量: N/A {'(仅评估模式)' if mode == 'evaluate' else '(并行模式下未统计)'}")
    
    print("=" * 80)
    
    # 保存结果
    print(f"\n保存结果...")
    print("=" * 80)
    
    # 整理详细结果
    detailed_results = []
    for i, (result, (acc_rate, eval_result_list)) in enumerate(zip(all_results, eval_results)):
        detailed_results.append({
            'instance_id': result['instance_id'],
            'problem_dir': result.get('problem_dir', ''),
            'code': result['code'],
            'accuracy': acc_rate,
            'passed': acc_rate == 1.0,
            'generation_time': result.get('generation_time', 0.0),
            'tokens_used': result.get('tokens_used', 0),
            'test_results': [
                {
                    'status': r.status,
                    'time_cost': r.time_cost,
                    'stdin': str(r.stdin)[:100] if r.stdin else '',
                    'stdout': str(r.stdout)[:100] if r.stdout else '',
                    'expected': str(r.expected)[:100] if r.expected else ''
                }
                for r in eval_result_list
            ],
            'response': result.get('response', '')
        })
    
    # 保存到 run 目录
    summary = {
        'summary': {
            'method': 'SCoT (Structured Chain-of-Thought)',
            'dataset': dataset_name,  # eval_dataset
            'generation_dataset': generation_dataset_name,
            'model': model_name,
            'temperature': temperature,
            'pass_at_1': pass_at_1,
            'passed': passed,
            'total': total_problems,
            'time_cost': {
                'total': total_time,
                'generation': generation_time,
                'evaluation': eval_time
            },
            'token_usage': {
                'total': total_tokens if total_tokens > 0 else 'N/A',
                'average_per_problem': total_tokens / total_problems if (total_problems > 0 and total_tokens > 0) else 'N/A',
                'per_problem_details': [result.get('tokens_used', 0) for result in all_results]
            },
            'config': {
                'max_workers': max_workers,
                'eval_workers': eval_workers,
                'few_shot': '3-Shot (Two Sum, Valid Parentheses, Merge Intervals)'
            },
            'timestamp': datetime.now().isoformat()
        },
        'results': detailed_results
    }
    
    logger.save_summary(summary)
    
    # 也保存到根目录（兼容旧版）- 根据 dataset + model 后缀命名
    suffix = _model_to_suffix(model_name)
    output_file = f"scot_baseline_results_{dataset_name}_{suffix}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 详细结果已保存到: {logger.run_dir}")
    print(f"✅ 摘要已保存到: {output_file}")
    print(f"✅ 每个问题的详细日志: {logger.run_dir}/<problem_id>/")
    
    # 生成可读的摘要报告
    report_file = logger.run_dir / "REPORT.txt"
    
    # 根据数据集确定 Few-Shot 说明
    if dataset_name in ('apps', 'apps_eval'):
        few_shot_desc = "3-Shot LeetCode Style (Two Sum, Valid Parentheses, Merge Intervals)"
        prompt_style = "Function-based"
    elif dataset_name in ('code_contests', 'code_contests_raw', 'xCodeEval'):
        few_shot_desc = "3-Shot StdIO Style (Two Sum, Valid Parentheses, Merge Intervals)"
        prompt_style = "Standard I/O"
    else:
        few_shot_desc = "3-Shot StdIO Style (Two Sum, Valid Parentheses, Merge Intervals)"
        prompt_style = "Standard I/O"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SCoT Baseline 运行报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"方法: SCoT (Structured Chain-of-Thought)\n")
        f.write(f"数据集(评估): {dataset_name.upper()}\n")
        if generation_dataset_name and generation_dataset_name != dataset_name:
            f.write(f"数据集(生成): {generation_dataset_name.upper()}\n")
        f.write(f"模型: {model_name}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Few-Shot: {few_shot_desc}\n")
        f.write(f"Prompt 风格: {prompt_style}\n")
        f.write(f"题目数量: {total_problems}\n\n")
        f.write(f"配置信息:\n")
        f.write(f"  - 并行线程数: {max_workers}\n")
        f.write(f"  - 评估进程数: {eval_workers}\n\n")
        f.write(f"结果统计:\n")
        f.write(f"  - Pass@1: {pass_at_1:.2f}% ({passed}/{total_problems})\n")
        f.write(f"  - 总耗时: {total_time:.2f} 秒\n")
        f.write(f"  - 生成耗时: {generation_time:.2f} 秒\n")
        f.write(f"  - 评估耗时: {eval_time:.2f} 秒\n")
        if total_tokens > 0:
            f.write(f"  - 总 Token: {total_tokens:,}\n")
            f.write(f"  - 平均每题 Token: {total_tokens/total_problems:.0f}\n\n")
        else:
            f.write(f"  - 总 Token: N/A\n\n")
        f.write(f"详细结果:\n")
        for result in detailed_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            f.write(f"  {status} {result['instance_id']} (准确率: {result['accuracy']*100:.0f}%)\n")
    
    print(f"✅ 可读报告已保存到: {report_file}")

    # 评估模式且指定了 --eval-output 时，额外写入到该目录下的 run_{timestamp}
    if mode == 'evaluate' and eval_output:
        out_base = Path(eval_output)
        out_base.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_run_dir = out_base / f"run_{ts}"
        out_run_dir.mkdir(parents=True, exist_ok=True)
        (out_run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        report_out = out_run_dir / "REPORT.txt"
        with open(report_out, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SCoT Baseline 运行报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"方法: SCoT (Structured Chain-of-Thought)\n")
            f.write(f"数据集(评估): {dataset_name.upper()}\n")
            if generation_dataset_name and generation_dataset_name != dataset_name:
                f.write(f"数据集(生成): {generation_dataset_name.upper()}\n")
            f.write(f"模型: {model_name}\n")
            f.write(f"题目数量: {total_problems}\n\n")
            f.write(f"结果统计: Pass@1 {pass_at_1:.2f}% ({passed}/{total_problems})\n")
            f.write(f"详细结果:\n")
            for result in detailed_results:
                status = "✅ PASS" if result['passed'] else "❌ FAIL"
                f.write(f"  {status} {result['instance_id']} (准确率: {result['accuracy']*100:.0f}%)\n")
        print(f"✅ 评估结果已写入: {out_run_dir}")
    
    print("\n" + "=" * 80)
    print("🎉 所有任务完成！")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SCoT Baseline Runner')
    parser.add_argument('--model', type=str, default=None,
                       help='模型名称（默认: deepseek-chat，可选 qwen3-coder-30b-a3b-instruct、gpt-5-mini-2025-08-07 等）')
    parser.add_argument('--dataset', type=str, default=None,
                       choices=DATASET_OPTIONS,
                       help=f'数据集（默认: xCodeEval）。可选: {", ".join(DATASET_OPTIONS)}')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API Key（也可用环境变量 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / GPT5_MINI_API_KEY）')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature 参数（默认: 0.0）')
    parser.add_argument('--workers', type=int, default=16,
                       help='并行数（生成模式：线程数，评估模式：进程数，默认: 16）')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录（默认: 根据数据集自动选择）')
    parser.add_argument('--limit', type=int, default=None,
                       help='限制处理的问题数量（用于测试，如: --limit 5）')
    parser.add_argument('--instance-ids', type=str, default=None,
                       help='仅处理指定题号，逗号分隔，如: --instance-ids 3708,3677,3832')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'generate', 'evaluate', 'reeval_raw_summary'],
                       help='运行模式: all=生成+评估, generate=仅生成, evaluate=仅评估, reeval_raw_summary=从 raw_summary 重评估（默认: all）')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='评估模式：指定包含 generation_checkpoint.json 的 run 目录（或上级目录，将自动选最新 run_*）')
    parser.add_argument('--eval-dataset', type=str, default=None,
                       help='评估模式：覆盖评估数据集（如从 apps 生成 → 用 apps_eval 评估）')
    parser.add_argument('--eval-output', type=str, default=None,
                       help='评估模式：评估结果输出目录，会在此下创建 run_{timestamp}')
    parser.add_argument('--raw-summary', type=str, default=None,
                       help='reeval_raw_summary 模式：指定 raw_summary.json 路径（如 scot_baseline_outputs/run_20260210_201740/raw_summary.json）')
    parser.add_argument('--reeval-dataset', type=str, default='code_contests',
                       choices=['code_contests', 'new/code_contests'],
                       help='reeval_raw_summary 模式：指定评估数据集。code_contests=Datasets/code_contests.jsonl, new/code_contests=Datasets/new/code_contests.jsonl（默认: code_contests）')
    
    args = parser.parse_args()
    
    # 打印使用提示
    if args.mode != 'all':
        print("\n" + "=" * 80)
        if args.mode == 'generate':
            print("💡 提示: 生成模式允许使用更高的并行数（如 --workers 64）")
            print("   生成完成后，使用以下命令评估:")
            print(f"   python scot_baseline.py --mode evaluate --workers 64 --resume-from <run_dir>")
        elif args.mode == 'evaluate':
            print("💡 提示: 评估模式允许使用更高的并行数（如 --workers 64）")
            if not args.resume_from:
                print("   建议使用 --resume-from 指定生成结果所在的 run 目录")
                print("   示例: --resume-from scot_baseline_outputs_apps/run_20260214_160524")
        elif args.mode == 'reeval_raw_summary':
            print("💡 提示: 使用 --raw-summary 指定 raw_summary.json 路径")
            print("   示例: --raw-summary scot_baseline_outputs/run_20260210_201740/raw_summary.json")
            print("   使用 --reeval-dataset 选择数据集: code_contests 或 new/code_contests")
        print("=" * 80 + "\n")
    
    main(
        model_name=args.model,
        dataset_name=args.dataset,
        temperature=args.temperature,
        max_workers=args.workers,
        output_dir=args.output_dir,
        limit=args.limit,
        instance_ids=args.instance_ids,
        mode=args.mode,
        resume_from=args.resume_from,
        raw_summary_path=args.raw_summary,
        reeval_dataset=args.reeval_dataset,
        api_key=args.api_key,
        eval_dataset=args.eval_dataset,
        eval_output=args.eval_output,
    )
