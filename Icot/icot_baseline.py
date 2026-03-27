"""
ICoT (Implicit Chain-of-Thought) Baseline
基于 RoutingGen ICoT 方法，使用 API 模型（qwen3-coder-30b-a3b-instruct / deepseek-chat）
在 CodeContest-raw, CodeContests, apps, apps_eval, xCodeEval 数据集上测试
使用 RoutingGen-main/apps_eval 作为评估程序

用法:
  python icot_baseline.py --model deepseek-chat --dataset xCodeEval --limit 5
  python icot_baseline.py --mode evaluate --resume-from icot_baseline_outputs/run_20260301_123456
  python icot_baseline.py --mode generate --resume-from icot_baseline_outputs  # 续写
"""

import os
import sys
import re
import json
import time
import platform
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ============================================================
# 路径配置
# ============================================================
_BASE = Path(__file__).resolve().parent
_WORKSPACE = _BASE.parent
_DATASETS = _WORKSPACE / "Self-collaboration-Code-Generation-main" / "Datasets"
sys.path.insert(0, str(_WORKSPACE / "Self-collaboration-Code-Generation-main"))
sys.path.insert(0, str(_BASE))
os.environ.setdefault("DATASETS_DIR", str(_DATASETS))

# 导入依赖
from core.generate_code import build_llm
from apps_eval.data import get_data, InstanceData
from apps_eval.parallel_runner import eval_code

DATASET_OPTIONS = ["code_contests_raw", "code_contests", "apps", "apps_eval", "xCodeEval", "livecodebench"]
MODEL_OPTIONS = [
    "deepseek-chat",
    "qwen3-coder-30b-a3b-instruct",
    "gpt-5-mini-2025-08-07",
    "gemini-3-flash-preview",
]


def _setup_model_env(model_name: str) -> None:
    """统一设置模型 API 环境变量。MODEL_API_KEY_ENV 必须是环境变量名（如 DASHSCOPE_API_KEY），不是 key 值"""
    if model_name == "gpt-5-mini-2025-08-07":
        os.environ["MODEL_API_BASE_URL"] = "http://api.yesapikey.com/v1"
        os.environ["MODEL_API_KEY_ENV"] = "GPT5_MINI_API_KEY"
        os.environ["MODEL_C"] = model_name
        os.environ.setdefault("GPT5_MINI_API_KEY", "***")
    elif model_name == "gemini-3-flash-preview":
        os.environ["MODEL_API_BASE_URL"] = "http://api.yesapikey.com/v1"
        os.environ["MODEL_API_KEY_ENV"] = "GEMINI_FLASH_API_KEY"
        os.environ["MODEL_C"] = model_name
        if not os.environ.get("GEMINI_FLASH_API_KEY"):
            print("Warn: 使用 gemini-3-flash-preview 需设置 GEMINI_FLASH_API_KEY 环境变量或 --api-key")
    elif "qwen" in model_name.lower():
        os.environ["MODEL_API_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        os.environ["MODEL_API_KEY_ENV"] = "DASHSCOPE_API_KEY"
        os.environ["MODEL_C"] = model_name
        if not os.environ.get("DASHSCOPE_API_KEY"):
            print("Warn: 使用 Qwen 需设置 DASHSCOPE_API_KEY 环境变量或 --api-key")
    else:
        os.environ.setdefault("MODEL_API_BASE_URL", "https://api.deepseek.com/v1")
        os.environ.setdefault("MODEL_API_KEY_ENV", "DEEPSEEK_API_KEY")
        os.environ["MODEL_C"] = model_name
        key_env = os.environ["MODEL_API_KEY_ENV"]
        if not os.environ.get(key_env):
            print(f"Warn: 使用 DeepSeek 需设置 {key_env} 环境变量或 --api-key")


def _sanitize_dirname(name: str) -> str:
    """将 instance_id 转为 Windows 合法目录名（替换 \\ / : * ? \" < > |）"""
    invalid_chars = r'<>:"/\|?*'
    for c in invalid_chars:
        name = name.replace(c, "_")
    # 去除首尾空格和点（Windows 不允许）
    name = name.strip(" .")
    return name or "unknown"


def _to_json_serializable(obj):
    """将 bytes 等不可序列化类型转为 JSON 可序列化形式"""
    if obj is None:
        return None
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return obj.decode("latin-1", errors="replace")
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    return str(obj)

# ============================================================
# Routing Prompt（动态路由，Simple->Few-shot, Complex->ICoT）
# ============================================================
ROUTING_SYSTEM = "You are an expert in analyzing programming task difficulty for competitive programming (Standard I/O style)."
ROUTING_USER = """Classify the following problem into one of two categories:
- Simple: Straightforward implementation, basic I/O, simple loops/conditions, no complex algorithms.
- Complex: Requires non-trivial algorithms, data structures, recursion, or careful reasoning.

Output exactly one line: Simple or Complex

Problem:
{problem}"""

# ============================================================
# ICoT Prompts（参照 deepseek-v3_RG.yaml，适配 StdIO）
# ============================================================
XCOT_SYSTEM_STDIO = """You are an expert Python programmer who provides structured problem-solving processes.
Your task is to analyze a given programming problem (using standard input/output) and generate a structured solution breakdown.

Please provide your analysis in the following format:
1: Specification
   - Define the input format and output format concisely.
2: Idea: Summarize the core logic and optimal algorithm with time complexity."""

# XCOT Few-shot 示例（竞赛题 StdIO 风格，参照 deepseek-v3_RG xcot_examples）
XCOT_EXAMPLES_STDIO = [
    {
        "user": """Read two integers n and k. Apply: if n%10!=0 then n-=1 else n//=10, k times. Print result.

Please analyze the requirement and provide a structured solving process with the following format:
1: Specification
   - Define the input and output format concisely.
2: Idea: Summarize the core logic and optimal algorithm with time complexity.""",
        "assistant": """1: Specification:
   - Input: Two integers n and k on one line (space-separated).
   - Output: One integer, the result after applying the operation k times.
2: Idea:
   - Simulate the operation k times in a loop. Each step: if n%10!=0 then n-=1 else n//=10.
   - Time complexity O(k)."""
    },
    {
        "user": """Read list of integers and target. Find two indices (1-based) that sum to target. Print indices or impossible.

Please analyze the requirement and provide a structured solving process with the following format:
1: Specification
   - Define the input and output format concisely.
2: Idea: Summarize the core logic and optimal algorithm with time complexity.""",
        "assistant": """1: Specification:
   - Input: First line: space-separated integers. Second line: target value.
   - Output: Two 1-based indices whose values sum to target, or "impossible".
2: Idea:
   - Use a hash map to store value->index. For each element, check if (target - value) exists.
   - Time complexity O(n)."""
    },
    {
        "user": """Given positive integer x (1 ≤ x ≤ 10^18). Find the largest positive integer ≤ x with maximum digit sum. If tie, output the largest.

Please analyze the requirement and provide a structured solving process with the following format:
1: Specification
   - Define the input and output format concisely.
2: Idea: Summarize the core logic and optimal algorithm with time complexity.""",
        "assistant": """1: Specification:
   - Input: One positive integer x.
   - Output: The largest integer ≤ x with maximum digit sum (no leading zeros).
2: Idea:
   - Greedy: try decreasing each digit and filling right with 9's. Compare digit sums.
   - For each position, consider (prefix-1) + '9'*remaining. Pick best sum, then largest value on tie.
   - Time complexity O(d^2) where d is number of digits."""
    },
]

XCOT_USER_STDIO = """{problem}

Please analyze the requirement and provide a structured solving process with the following format:
1: Specification
   - Define the input and output format concisely.
2: Idea: Summarize the core logic and optimal algorithm with time complexity."""

# Few-shot 直接生成（Simple 题目用）
FEWSHOT_CODE_SYSTEM = "You are an expert Python programmer. Given a problem, provide complete implementation using sys.stdin for input and print() for output. Include `import sys`. Output exactly ONE ```python ... ``` block."
FEWSHOT_CODE_EXAMPLES = [
    {"user": "Read two integers n and k. Apply: if n%10!=0 then n-=1 else n//=10, k times. Print result.", "assistant": "```python\nimport sys\ndata = sys.stdin.read().split()\nn, k = int(data[0]), int(data[1])\nfor _ in range(k):\n    n = n - 1 if n % 10 else n // 10\nprint(n)\n```"},
    {"user": "Read list of integers and target. Find two indices (1-based) that sum to target. Print indices or impossible.", "assistant": "```python\nimport sys\nlines = sys.stdin.read().strip().split('\\n')\nnums = list(map(int, lines[0].split()))\ntarget = int(lines[1])\nseen = {}\nfor i, v in enumerate(nums):\n    if target - v in seen:\n        print(seen[target-v]+1, i+1)\n        exit(0)\n    seen[v] = i\nprint('impossible')\n```"},
]
FEWSHOT_CODE_USER = "{problem}"

# ICoT 代码生成
CODE_SYSTEM_STDIO = """You are an expert Python programmer. Given a problem and structured solving steps (xcot), complete the implementation.
MUST include `import sys` and use sys.stdin.read() or sys.stdin.readline() for input. Output exactly ONE ```python ... ``` block."""

CODE_USER_STDIO = """{problem}

Structured solving process (xcot):
{xcot}

Please check the above solving process and write a complete Python script. Include `import sys`. Use sys.stdin for input, print() for output. Wrap in ```python ... ```. The solving process may contain errors - use your judgment."""


# ============================================================
# ICoT Agent（含 Dynamic Routing + 分阶段 Token 统计）
# ============================================================

def _extract_code_robust(response: str) -> str:
    """竞赛题目代码提取：优先最长 ```python 块，确保包含 sys 等 I/O 模块"""
    # 1. 匹配完整 ```python ... ```
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        # 优先选择包含 sys 的最长块，否则选最长
        with_sys = [m for m in matches if "sys" in m]
        cand = with_sys if with_sys else matches
        return max(cand, key=len).strip()

    # 2. 截断的 ```python ...（无结束标记）
    pattern_start = r"```python\s*(.*)$"
    matches_start = re.findall(pattern_start, response, re.DOTALL)
    if matches_start:
        return matches_start[0].strip()

    # 3. 从 import/from 开始收集到末尾
    lines = response.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith(('import ', 'from ', 'def ', 'class ')):
            in_code = True
        if in_code:
            code_lines.append(line)
    if code_lines:
        code = '\n'.join(code_lines).strip()
        if 'import sys' not in code and 'from sys' not in code:
            code = "import sys\n" + code
        return code
    return response.strip()


class ICoTAgent:
    """ICoT Agent：Dynamic Routing + Few-shot/ICoT 双路径 + 分阶段 Token 统计"""

    def __init__(self, model_name: str = "deepseek-chat", temperature: float = 0.0, use_routing: bool = True):
        if 'MODEL_C' not in os.environ or model_name != os.environ.get('MODEL_C'):
            os.environ['MODEL_C'] = model_name
        if model_name == "gpt-5-mini-2025-08-07":
            self.llm = build_llm(
                model_env='MODEL_C',
                temperature=temperature,
                max_tokens=60000,
                reasoning={"effort": "minimal"},
            )
        elif model_name == "gemini-3-flash-preview":
            self.llm = build_llm(
                model_env='MODEL_C',
                temperature=temperature,
                max_tokens=60000,
            )
        else:
            self.llm = build_llm(model_env='MODEL_C', temperature=temperature, max_tokens=8192)
        self.model_name = model_name
        self.use_routing = use_routing

    def _get_token_delta(self) -> int:
        return getattr(self.llm, "total_tokens", 0) or 0

    def route(self, problem_desc: str) -> tuple:
        """路由：返回 (label, routing_tokens)。label: 'Simple' | 'Complex'"""
        if not self.use_routing:
            return "Complex", 0
        t0 = self._get_token_delta()
        messages = [
            {"role": "system", "content": ROUTING_SYSTEM},
            {"role": "user", "content": ROUTING_USER.format(problem=problem_desc[:2000])}
        ]
        try:
            out = self._chat(messages, temperature=0.0, max_tokens=50)
            routing_tokens = self._get_token_delta() - t0
            label = "Simple" if "simple" in (out or "").lower().strip() else "Complex"
            return label, routing_tokens
        except Exception as e:
            print(f"[ERR] 路由失败: {e}")
            return "Complex", 0

    def generate_fewshot(self, problem_desc: str) -> tuple:
        """Few-shot 直接生成（Simple 路径）。返回 (code, response, tokens)"""
        t0 = self._get_token_delta()
        messages = [{"role": "system", "content": FEWSHOT_CODE_SYSTEM}]
        for ex in FEWSHOT_CODE_EXAMPLES:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": FEWSHOT_CODE_USER.format(problem=problem_desc)})
        try:
            response = self._chat(messages, temperature=0.8)
            tokens = self._get_token_delta() - t0
            return _extract_code_robust(response), response, tokens
        except Exception as e:
            print(f"[ERR] Few-shot 生成失败: {e}")
            return "", f"# {e}", 0

    def _build_gemini_messages(self, messages: list) -> list:
        """将普通 messages 包装成 Gemini 原生 request_body 格式"""
        problem_text = "\n\n".join([f"{m['role']}:\n{m['content']}" for m in messages])
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
        return [{"role": "user", "content": json.dumps(request_body, ensure_ascii=False)}]

    def _chat(self, messages: list, temperature: float, max_tokens: Optional[int] = None) -> str:
        """按模型类型分发 chat 调用，带自动重试（最多5次，指数退避）。"""
        if self.model_name == "gemini-3-flash-preview":
            wrapped = self._build_gemini_messages(messages)
            actual_messages = wrapped
        else:
            actual_messages = messages

        max_retries = 5
        last_err: Exception = RuntimeError("_chat: no attempts made")
        for attempt in range(max_retries):
            try:
                return self.llm.chat(actual_messages, temperature=temperature, max_tokens=max_tokens)
            except Exception as e:
                last_err = e
                wait = min(2 ** attempt, 30)
                if attempt < max_retries - 1:
                    print(f"[WARN] _chat 调用失败 (attempt {attempt + 1}/{max_retries}): {e}，{wait}s 后重试...")
                    time.sleep(wait)
                else:
                    print(f"[ERR]  _chat 调用失败，已达最大重试次数: {e}")
        raise last_err



    def generate_xcot(self, problem_desc: str) -> str:
        """生成 xcot。论文：temperature=0.8 以增加多样性。含 Few-shot 示例"""
        messages = [{"role": "system", "content": XCOT_SYSTEM_STDIO}]
        for ex in XCOT_EXAMPLES_STDIO:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": XCOT_USER_STDIO.format(problem=problem_desc)})
        try:
            return self._chat(messages, temperature=0.8)
        except Exception as e:
            print(f"[ERR] xcot 生成失败: {e}")
            return ""

    def generate_code(self, problem_desc: str, xcot: str) -> str:
        """基于 xcot 生成代码。论文：temperature=0 保证稳定性"""
        messages = [
            {"role": "system", "content": CODE_SYSTEM_STDIO},
            {"role": "user", "content": CODE_USER_STDIO.format(problem=problem_desc, xcot=xcot)}
        ]
        try:
            return self._chat(messages, temperature=0.0)
        except Exception as e:
            print(f"[ERR] 代码生成失败: {e}")
            return f"# Generation failed: {e}"

    def generate_with_response(self, problem_desc: str) -> tuple:
        """返回 (code, full_response, token_stats)。token_stats: {routing, intention, code, total}"""
        stats = {"routing": 0, "intention": 0, "code": 0, "total": 0}
        t0 = self._get_token_delta()

        # 1. Routing
        label, stats["routing"] = self.route(problem_desc)

        if label == "Simple":
            code, resp, stats["code"] = self.generate_fewshot(problem_desc)
            full_response = f"[route=Simple]\n[code]\n{resp}"
        else:
            # 2. ICoT: xcot (intention)
            t1 = self._get_token_delta()
            xcot = self.generate_xcot(problem_desc)
            stats["intention"] = self._get_token_delta() - t1

            # 3. ICoT: code
            t2 = self._get_token_delta()
            resp = self.generate_code(problem_desc, xcot)
            stats["code"] = self._get_token_delta() - t2

            code = _extract_code_robust(resp)
            full_response = f"[route=Complex]\n[xcot]\n{xcot}\n\n[code]\n{resp}"

        stats["total"] = self._get_token_delta() - t0
        return code, full_response, stats


# ============================================================
# 详细日志
# ============================================================

class DetailedLogger:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)

    def create_problem_dir(self, instance_id: str) -> Path:
        safe_name = _sanitize_dirname(instance_id)
        problem_dir = self.run_dir / safe_name
        problem_dir.mkdir(exist_ok=True)
        return problem_dir

    def save_problem_info(self, problem_dir: Path, instance: InstanceData):
        with open(problem_dir / "problem_statement.txt", "w", encoding="utf-8") as f:
            f.write(instance.problem_statement)

    def save_generation_info(self, problem_dir: Path, code: str, response: str):
        with open(problem_dir / "generated_code.py", "w", encoding="utf-8") as f:
            f.write(code)
        with open(problem_dir / "full_response.txt", "w", encoding="utf-8") as f:
            f.write(response)

    def save_summary(self, summary: Dict):
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


# ============================================================
# 单实例处理
# ============================================================

def process_single_instance(args: tuple) -> Dict[str, Any]:
    """每个线程创建独立 Agent"""
    model_name, temperature, instance, idx, total, logger, use_routing = args
    try:
        print(f"[{idx}/{total}] 开始处理: {instance.instance_id}")
        problem_dir = logger.create_problem_dir(instance.instance_id)
        logger.save_problem_info(problem_dir, instance)

        agent = ICoTAgent(model_name=model_name, temperature=temperature, use_routing=use_routing)
        start_time = time.time()
        code, response, token_stats = agent.generate_with_response(instance.problem_statement)
        generation_time = time.time() - start_time

        logger.save_generation_info(problem_dir, code, response)

        tokens_used = token_stats.get("total", 0)
        if code and not code.startswith("# Generation failed"):
            print(f"[{idx}/{total}] [OK] {instance.instance_id} 生成成功 (耗时: {generation_time:.2f}s, tokens: {tokens_used})")
        else:
            print(f"[{idx}/{total}] [X] {instance.instance_id} 生成失败")

        return {
            'instance_id': instance.instance_id,
            'code': code,
            'test_cases': instance.test_cases,
            'generation_time': generation_time,
            'token_nums': tokens_used,
            'token_breakdown': token_stats,
            'problem_dir': str(problem_dir),
            'response': response
        }
    except Exception as e:
        print(f"[{idx}/{total}] [X] {instance.instance_id} 异常: {e}")
        return {
            'instance_id': instance.instance_id,
            'code': f"# Exception: {e}",
            'test_cases': instance.test_cases,
            'generation_time': 0.0,
            'token_nums': 0,
            'token_breakdown': {},
            'error': str(e),
            'problem_dir': ''
        }


# ============================================================
# 评估与报告
# ============================================================

def _run_evaluation(run_dir: Path, dataset_name: str, all_results: list, workers: int) -> dict:
    """从 results 执行评估，返回统计与详细结果"""
    dataset = get_data(dataset_name)
    result_by_id = {r["instance_id"]: r for r in all_results}

    eval_dataset = []
    eval_solutions = []
    for inst in dataset:
        if inst.instance_id in result_by_id:
            eval_dataset.append(inst)
            eval_solutions.append(result_by_id[inst.instance_id]["code"])

    if not eval_dataset:
        return {"pass_count": 0, "total": 0, "pass_at_1": 0.0, "avg_pass_ratio": 0.0, "detailed_results": [], "eval_results": []}

    eval_workers = workers if platform.system() == "Windows" else min(workers * 4, 60)
    print(f"\n评估 {len(eval_dataset)} 道题目 (workers={eval_workers})...")
    eval_results = eval_code(eval_dataset, eval_solutions, timeout=10.0, workers=eval_workers, show_progress=True)

    pass_count = sum(1 for acc, _ in eval_results if acc == 1.0)
    total = len(eval_results)
    pass_at_1 = pass_count / total if total > 0 else 0.0
    avg_pass_ratio = sum(acc for acc, _ in eval_results) / total if total > 0 else 0.0

    detailed_results = []
    for inst, (acc_rate, eval_list) in zip(eval_dataset, eval_results):
        r = result_by_id.get(inst.instance_id, {})
        detailed_results.append({
            "instance_id": inst.instance_id,
            "code": r.get("code", ""),
            "accuracy": acc_rate,
            "passed": acc_rate == 1.0,
            "generation_time": r.get("generation_time", 0.0),
            "token_nums": r.get("token_nums", 0),
            "test_results": [
                {
                    "status": er.status,
                    "time_cost": er.time_cost,
                    "input": _to_json_serializable(getattr(er, "stdin", None)),
                    "output": _to_json_serializable(getattr(er, "stdout", None)),
                    "expected": _to_json_serializable(getattr(er, "expected", None)),
                }
                for er in eval_list
            ],
        })

    return {
        "pass_count": pass_count,
        "total": total,
        "pass_at_1": pass_at_1,
        "avg_pass_ratio": avg_pass_ratio,
        "detailed_results": detailed_results,
        "eval_results": eval_results,
    }


def _write_report_and_summary(
    run_dir: Path,
    summary: dict,
    eval_output: dict,
    generation_time: float = 0.0,
    total_tokens: int = 0,
    token_breakdown: dict | None = None,
) -> None:
    """写入 REPORT.txt 和 summary.json"""
    pass_count = eval_output["pass_count"]
    total = eval_output["total"]
    pass_at_1 = eval_output["pass_at_1"]
    avg_pass_ratio = eval_output["avg_pass_ratio"]
    detailed_results = eval_output["detailed_results"]

    summary_data = {
        "summary": {
            "method": "ICoT (Implicit Chain-of-Thought)",
            "dataset": summary.get("dataset_name", ""),
            "model": summary.get("model_name", ""),
            "pass_at_1": pass_at_1,
            "passed": pass_count,
            "total": total,
            "avg_pass_ratio": avg_pass_ratio,
            "generation_time": generation_time,
            "token_usage": {
                "total": total_tokens,
                "average_per_problem": total_tokens / total if total and total_tokens else 0,
                "breakdown": token_breakdown or {},
            },
            "timestamp": datetime.now().isoformat(),
        },
        "results": detailed_results,
    }

    (run_dir / "summary.json").write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")

    report_file = run_dir / "REPORT.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ICoT Baseline 运行报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"方法: ICoT (Implicit Chain-of-Thought)\n")
        f.write(f"数据集: {summary.get('dataset_name', '').upper()}\n")
        f.write(f"模型: {summary.get('model_name', '')}\n")
        f.write(f"题目数量: {total}\n\n")
        f.write("结果统计:\n")
        f.write(f"  - Pass@1: {pass_at_1*100:.2f}% ({pass_count}/{total})\n")
        f.write(f"  - AvgPassRatio: {avg_pass_ratio:.4f}\n")
        f.write(f"  - 生成总耗时: {generation_time:.2f} 秒\n")
        if total_tokens > 0:
            f.write(f"  - 总 Token: {total_tokens:,}\n")
            f.write(f"  - 平均每题 Token: {total_tokens/total:.0f}\n")
            if token_breakdown:
                f.write(f"  - Token 分阶段: Routing={token_breakdown.get('routing', 0):,}, "
                        f"Intention={token_breakdown.get('intention', 0):,}, Code={token_breakdown.get('code', 0):,}\n")
        f.write("\n详细结果:\n")
        for dr in detailed_results:
            status = "[PASS]" if dr["passed"] else "[FAIL]"
            f.write(f"  {status} {dr['instance_id']} (准确率: {dr['accuracy']*100:.0f}%)\n")
    print(f"报告已保存: {report_file}")


# ============================================================
# 主流程
# ============================================================

def main(args) -> int:
    """主函数，返回退出码"""
    model_name = args.model or os.environ.get("MODEL_C", "deepseek-chat")
    dataset_name = args.dataset
    _suffix_map = {
        "gpt-5-mini-2025-08-07": "gpt5mini",
        "gemini-3-flash-preview": "gemini3flash",
    }
    if model_name in _suffix_map:
        suffix = _suffix_map[model_name]
    elif "qwen" in model_name.lower():
        suffix = "qwen"
    else:
        suffix = "deepseek"
    root_dir = _BASE / "icot_baseline_outputs"
    run_name = f"{dataset_name}_{suffix}"

    # === evaluate 模式 ===
    if args.mode == "evaluate" and args.resume_from:
        run_dir = Path(args.resume_from)
        if not run_dir.exists():
            print(f"错误: 目录不存在: {run_dir}")
            return 1
        if not run_dir.name.startswith("run_"):
            runs = sorted(run_dir.glob("run_*"), key=lambda x: x.name, reverse=True)
            if not runs:
                print(f"错误: 在 {run_dir} 中未找到 run_* 目录")
                return 1
            run_dir = runs[0]
        ckpt_file = run_dir / "generation_checkpoint.json"
        if not ckpt_file.exists():
            print(f"错误: 找不到 {ckpt_file}，请先运行生成模式")
            return 1
        with open(ckpt_file, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        all_results = ckpt["results"]
        eval_dataset_name = args.eval_dataset or ckpt.get("dataset_name", dataset_name)
        if args.eval_dataset:
            print(f"评估模式: 加载 {len(all_results)} 条结果，使用数据集 {eval_dataset_name} 进行评估")
        else:
            print(f"评估模式: 加载 {len(all_results)} 条结果 from {run_dir}")

        eval_start_t = time.time()
        eval_output = _run_evaluation(run_dir, eval_dataset_name, all_results, args.workers)
        eval_time = time.time() - eval_start_t

        out_run_dir = run_dir
        if args.eval_output:
            out_base = Path(args.eval_output)
            out_base.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_run_dir = out_base / f"run_{ts}"
            out_run_dir.mkdir(parents=True, exist_ok=True)
            print(f"评估结果将保存到: {out_run_dir}")

        summary_for_report = {**ckpt, "dataset_name": eval_dataset_name}
        if args.eval_dataset:
            summary_for_report["generation_dataset"] = ckpt.get("dataset_name")
        tu = ckpt.get("token_usage") or {}
        total_tokens = tu.get("total", 0) if isinstance(tu, dict) else 0
        token_breakdown = tu.get("breakdown", {}) if isinstance(tu, dict) else {}
        _write_report_and_summary(
            out_run_dir, summary_for_report, eval_output,
            generation_time=ckpt.get("generation_time", 0),
            total_tokens=total_tokens,
            token_breakdown=token_breakdown,
        )
        (out_run_dir / "eval_results.json").write_text(
            json.dumps({"summary": summary_for_report, "eval": {k: v for k, v in eval_output.items() if k != "eval_results"}}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("=" * 80)
        print(f"Pass@1: {eval_output['pass_count']}/{eval_output['total']} ({eval_output['pass_at_1']*100:.2f}%)")
        print(f"评估耗时: {eval_time:.1f}s")
        print(f"结果: {out_run_dir}")
        print("=" * 80)
        return 0

    # === 生成模式 ===
    print("=" * 80)
    print("ICoT (Implicit Chain-of-Thought) Baseline")
    print("=" * 80)
    print(f"  模式: {args.mode}")
    print(f"  模型: {model_name}")
    print(f"  数据集: {dataset_name}")
    print(f"  进程数: {args.workers}")
    if args.limit:
        print(f"  限制: {args.limit}")
    print("=" * 80)

    try:
        dataset = get_data(dataset_name)
    except FileNotFoundError as e:
        print(f"[ERR] 数据集未找到: {e}")
        print("[TIP] 请确保 Datasets 目录存在且包含对应 jsonl 文件")
        return 1

    if args.limit and args.limit > 0:
        dataset = dataset[: args.limit]
    print(f"加载 {len(dataset)} 题")

    # 确定 run 目录：支持 --resume-from 续写
    completed_ids = set()
    prev_ckpt = {}
    run_dir = None
    if args.resume_from and args.mode != "evaluate":
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            if resume_path.name.startswith("run_"):
                run_dir = resume_path
            else:
                runs = sorted(resume_path.glob("run_*"), key=lambda x: x.name, reverse=True)
                run_dir = runs[0] if runs else None
            if run_dir and (run_dir / "generation_checkpoint.json").exists():
                with open(run_dir / "generation_checkpoint.json", "r", encoding="utf-8") as f:
                    prev_ckpt = json.load(f)
                completed_ids = {r["instance_id"] for r in prev_ckpt.get("results", [])}
                print(f"续写模式: 已完成 {len(completed_ids)} 题，剩余 {len(dataset) - len(completed_ids)} 题")
                logger = DetailedLogger.__new__(DetailedLogger)
                logger.output_dir = run_dir.parent
                logger.run_dir = run_dir
                all_results = prev_ckpt["results"]
            else:
                run_dir = None
                prev_ckpt = {}

    prev_gen_time = prev_ckpt.get("generation_time", 0.0) if prev_ckpt else 0.0
    if run_dir is None:
        log_dir = root_dir / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = DetailedLogger(str(log_dir))
        run_dir = logger.run_dir
        all_results = []

    # 过滤待生成题目
    todo_instances = [inst for inst in dataset if inst.instance_id not in completed_ids]
    if not todo_instances:
        print("所有题目已完成，无需生成")
    else:
        print(f"\n开始生成代码（{args.workers} 线程并行，共 {len(todo_instances)} 题）...")
        use_routing = not getattr(args, "no_routing", False)
        args_list = [
            (model_name, args.temperature, inst, idx + 1, len(todo_instances), logger, use_routing)
            for idx, inst in enumerate(todo_instances)
        ]
        gen_start = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_single_instance, a) for a in args_list]
            for future in as_completed(futures):
                all_results.append(future.result())
        instance_id_order = {inst.instance_id: idx for idx, inst in enumerate(dataset)}
        all_results.sort(key=lambda x: instance_id_order.get(x["instance_id"], 999999))
        generation_time = prev_gen_time + (time.time() - gen_start)
        total_tokens = sum(r.get("token_nums", 0) for r in all_results)
        # 聚合 token_breakdown 用于 R2S/R2I 计算
        token_breakdown_agg = {"routing": 0, "intention": 0, "code": 0}
        for r in all_results:
            tb = r.get("token_breakdown") or {}
            token_breakdown_agg["routing"] += tb.get("routing", 0)
            token_breakdown_agg["intention"] += tb.get("intention", 0)
            token_breakdown_agg["code"] += tb.get("code", 0)

        ckpt_data = {
            "results": all_results,
            "dataset_name": dataset_name,
            "model_name": model_name,
            "generation_time": generation_time,
            "token_usage": {
                "total": total_tokens,
                "average_per_problem": total_tokens / len(all_results) if all_results else 0,
                "breakdown": token_breakdown_agg,
            },
            "timestamp": datetime.now().isoformat(),
        }
        (run_dir / "generation_checkpoint.json").write_text(json.dumps(ckpt_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\ngeneration_checkpoint.json 已保存: {run_dir / 'generation_checkpoint.json'}")
        print(f"生成耗时: {generation_time:.1f}s")

    if args.mode == "generate":
        print(f"\n下一步评估: python icot_baseline.py --mode evaluate --resume-from \"{run_dir}\"")
        return 0

    # === 评估 ===
    ckpt_file = run_dir / "generation_checkpoint.json"
    with open(ckpt_file, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    all_results = ckpt["results"]
    generation_time = ckpt.get("generation_time", 0)
    total_tokens = ckpt.get("token_usage", {}).get("total", 0) if isinstance(ckpt.get("token_usage"), dict) else 0

    # 支持 --eval-dataset：允许用不同数据集评估（如 apps -> apps_eval，code_contests_raw -> code_contests）
    eval_dataset_name = args.eval_dataset or dataset_name
    if args.eval_dataset:
        print(f"评估数据集覆盖: {dataset_name} -> {eval_dataset_name}")

    eval_start_t = time.time()
    eval_output = _run_evaluation(run_dir, eval_dataset_name, all_results, args.workers)
    eval_time = time.time() - eval_start_t

    # 支持 --eval-output：将评估结果输出到独立目录，不覆盖源 run_dir
    out_run_dir = run_dir
    if args.eval_output:
        out_base = Path(args.eval_output)
        out_base.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_run_dir = out_base / f"run_{ts}"
        out_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"评估结果将保存到: {out_run_dir}")

    tu = ckpt.get("token_usage") or {}
    token_breakdown = tu.get("breakdown", {}) if isinstance(tu, dict) else {}
    summary_for_report = {**ckpt, "dataset_name": eval_dataset_name}
    if args.eval_dataset:
        summary_for_report["generation_dataset"] = ckpt.get("dataset_name")
    _write_report_and_summary(out_run_dir, summary_for_report, eval_output, generation_time, total_tokens, token_breakdown=token_breakdown)
    (out_run_dir / "eval_results.json").write_text(
        json.dumps({"summary": summary_for_report, "eval": {k: v for k, v in eval_output.items() if k != "eval_results"}}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 80)
    print(f"Pass@1: {eval_output['pass_count']}/{eval_output['total']} ({eval_output['pass_at_1']*100:.2f}%)")
    print(f"评估耗时: {eval_time:.1f}s")
    print(f"结果: {out_run_dir}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ICoT Baseline Runner")
    parser.add_argument("--model", type=str, default="deepseek-chat", choices=MODEL_OPTIONS,
                        help="deepseek-chat | qwen3-coder-30b-a3b-instruct | gpt-5-mini-2025-08-07 | gemini-3-flash-preview")
    parser.add_argument("--dataset", type=str, default="code_contests", choices=DATASET_OPTIONS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="限制题目数量（测试用）")
    parser.add_argument("--api-key", type=str, default=None, help="API Key")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "generate", "evaluate"])
    parser.add_argument("--resume-from", type=str, default=None,
                        help="evaluate: 指定 run 目录; generate/all: 续写，跳过已完成题目")
    parser.add_argument("--eval-dataset", type=str, default=None,
                        help="evaluate 模式: 指定评估数据集（如从 code_contests_raw 生成、用 code_contests 评估）")
    parser.add_argument("--eval-output", type=str, default=None,
                        help="evaluate 模式: 评估结果输出目录")
    parser.add_argument("--no-routing", action="store_true",
                        help="禁用动态路由，全部题目走 ICoT 路径")
    args = parser.parse_args()

    if args.api_key:
        if args.model == "gpt-5-mini-2025-08-07":
            os.environ["GPT5_MINI_API_KEY"] = args.api_key
        elif args.model == "gemini-3-flash-preview":
            os.environ["GEMINI_FLASH_API_KEY"] = args.api_key
        elif "qwen" in args.model.lower():
            os.environ["DASHSCOPE_API_KEY"] = args.api_key
        else:
            os.environ["DEEPSEEK_API_KEY"] = args.api_key
    _setup_model_env(args.model)

    sys.exit(main(args) or 0)
