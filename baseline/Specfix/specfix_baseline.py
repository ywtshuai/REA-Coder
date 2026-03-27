"""
SpecFix Baseline
Runs the SpecFix method with API models on code_contests_raw, code_contests, apps, apps_eval, xCodeEval, and livecodebench.
Uses SpecFix-main/apps_eval for evaluation.

Examples:
  python SpecFix-main/specfix_baseline.py --model deepseek-chat --dataset xCodeEval --limit 5
  python SpecFix-main/specfix_baseline.py --mode evaluate --resume-from SpecFix-main/specfix_baseline_outputs/xCodeEval_deepseek/run_xxx
  python SpecFix-main/specfix_baseline.py --mode generate --resume-from SpecFix-main/specfix_baseline_outputs/xCodeEval_deepseek
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

# Path setup, aligned with icot_baseline
# 璺緞閰嶇疆锛堜笌 icot_baseline 涓€鑷达級
# ============================================================
_BASE = Path(__file__).resolve().parent
_WORKSPACE = _BASE.parent
_DATASETS = _WORKSPACE / "Self-collaboration-Code-Generation-main" / "Datasets"
sys.path.insert(0, str(_WORKSPACE / "Self-collaboration-Code-Generation-main"))
sys.path.insert(0, str(_BASE))
# Imports

# 瀵煎叆渚濊禆
from apps_eval.data import get_data, InstanceData
from apps_eval.parallel_runner import eval_code
# SpecFix modules (supports running from workspace root or SpecFix-main)

# SpecFix 妯″潡锛堟敮鎸佷粠宸ヤ綔鍖烘牴鐩綍鎴?SpecFix-main 涓嬭繍琛岋級
sys.path.insert(0, str(_BASE))
from specfix.evaluator import SpecFixAccuracyEvaluator
from specfix.tester import differential_tester, ground_truth_tester
from specfix.utils import ensure_print_output_stdio

DATASET_OPTIONS = ["code_contests_raw", "code_contests", "apps", "apps_eval", "xCodeEval", "livecodebench"]
MODEL_OPTIONS = [
    "deepseek-chat",
    "qwen3-coder-30b-a3b-instruct",
    "gpt-5-mini-2025-08-07",
    "gemini-3-flash-preview",
]


def _setup_model_env(model_name: str) -> None:
    """缁熶竴璁剧疆妯″瀷 API 鐜鍙橀噺"""
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
            print("Warn: gemini-3-flash-preview requires GEMINI_FLASH_API_KEY or --api-key")
    elif "qwen" in model_name.lower():
        os.environ["MODEL_API_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        os.environ["MODEL_API_KEY_ENV"] = "DASHSCOPE_API_KEY"
        os.environ["MODEL_C"] = model_name
        if not os.environ.get("DASHSCOPE_API_KEY"):
            print("Warn: Qwen requires DASHSCOPE_API_KEY or --api-key")
    else:
        os.environ.setdefault("MODEL_API_BASE_URL", "https://api.deepseek.com/v1")
        os.environ.setdefault("MODEL_API_KEY_ENV", "DEEPSEEK_API_KEY")
        os.environ["MODEL_C"] = model_name
        key_env = os.environ["MODEL_API_KEY_ENV"]
        if not os.environ.get(key_env):
            print(f"Warn: DeepSeek requires {key_env} or --api-key")


def _sanitize_dirname(name: str) -> str:
    """Convert instance_id into a Windows-safe directory name."""
    invalid_chars = r'<>:"/\|?*'
    for c in invalid_chars:
        name = name.replace(c, "_")
    name = name.strip(" .")
    return name or "unknown"


def _to_json_serializable(obj):
    """Convert non-serializable values such as bytes into JSON-safe values."""
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


def _instance_to_specfix_problem(instance: InstanceData) -> Dict[str, Any]:
    """Convert InstanceData into the SpecFix problem format for StdIO tasks.
    Prefer public_test_cases as prompt examples, and fall back to test_cases when needed."""
    # 浼樺厛浣跨敤 public_test_cases锛堣鏂?Extract_Examples锛宒ata.py 宸叉彁鍙栵級
    if instance.public_test_cases and instance.public_test_cases.get("inputs") and instance.public_test_cases.get("outputs"):
        inputs = instance.public_test_cases["inputs"]
        outputs = instance.public_test_cases["outputs"]
    else:
        inputs = instance.test_cases.get("inputs", [])
        outputs = instance.test_cases.get("outputs", [])
    # grade_stdio 鏈熸湜 outputs 涓?[[out1], [out2], ...]
    outputs_wrapped = [[o] if not isinstance(o, list) else o for o in outputs]
    input_output_examples = [inputs, outputs_wrapped]
    return {
        "task_id": instance.instance_id,
        "requirement": instance.problem_statement,
        "entry_point": "",  # StdIO 椋庢牸
        "input_output_examples": str(input_output_examples),
    }


def _extract_code_robust(response: str) -> str:
    """Extract Python code from a model response."""
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        with_sys = [m for m in matches if "sys" in m]
        cand = with_sys if with_sys else matches
        code = max(cand, key=len).strip()
    else:
        pattern_start = r"```python\s*(.*)$"
        matches_start = re.findall(pattern_start, response, re.DOTALL)
        if matches_start:
            code = matches_start[0].strip()
        else:
            lines = response.split("\n")
            code_lines = []
            in_code = False
            for line in lines:
                if line.strip().startswith(("import ", "from ", "def ", "class ")):
                    in_code = True
                if in_code:
                    code_lines.append(line)
            if code_lines:
                code = "\n".join(code_lines).strip()
                if "import sys" not in code and "from sys" not in code:
                    code = "import sys\n" + code
            else:
                code = response.strip()
    return ensure_print_output_stdio(code)


# ============================================================
# SpecFix agent wrapper
# ============================================================

class SpecFixBaselineAgent:
    """SpecFix pipeline: detect -> repair (if needed) -> generate final code."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.8,
        cluster_size: int = 15,
        process_collector: Optional[List] = None,
        max_tokens_per_problem: Optional[int] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.cluster_size = cluster_size
        self.process_collector = process_collector
        self.evaluator = SpecFixAccuracyEvaluator(
            differential_tester,
            ground_truth_tester,
            model_name,
            temperature,
            process_collector=process_collector,
            max_tokens_per_problem=max_tokens_per_problem,
        )

    def _with_rate_limit_retry(self, func, *args, **kwargs):
        """Retry model calls when rate limiting occurs (HTTP 429 / limit_requests / rate limit)."""
        max_retries = 5
        base_sleep = 10  # seconds
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                is_rate_limit = (
                    "429" in msg
                    or "limit_requests" in msg
                    or "rate limit" in msg.lower()
                )
                if is_rate_limit and attempt < max_retries - 1:
                    wait = base_sleep * (attempt + 1)
                    print(f"[WARN] Rate limited (429), retrying in {wait}s ({attempt + 1}/{max_retries})")
                    time.sleep(wait)
                    continue
                raise

    def generate_final_code(self, instance: InstanceData, simple: bool = False) -> tuple:
        """Run SpecFix on one problem and return (code, full_response, success, tokens)."""
        problem = _instance_to_specfix_problem(instance)
        requirement = problem["requirement"]
        entry_point = problem["entry_point"]
        self.evaluator.model.total_tokens = 0

        if simple:
            self.evaluator._log_process(
                "simple_mode",
                {"requirement_preview": requirement[:300] + "..." if len(requirement) > 300 else requirement},
            )
            try:
                code = self._with_rate_limit_retry(self.evaluator.generate_program, requirement, entry_point)
                code = _extract_code_robust(code) if code else ""
                tokens = getattr(self.evaluator.model, "total_tokens", 0) or 0
                self.evaluator._log_process("simple_final", {"code": code})
                return code, f"[simple]\n[code]\n{code}", True, tokens
            except Exception as e:
                print(f"[ERR] SpecFix simple exception: {e}")
                try:
                    code = self._with_rate_limit_retry(self.evaluator.generate_program, requirement, entry_point)
                    code = _extract_code_robust(code) if code else ""
                    tokens = getattr(self.evaluator.model, "total_tokens", 0) or 0
                except Exception:
                    code = f"# Exception: {e}"
                    tokens = 0
                return code, f"# {e}", False, tokens

        try:
            detect_result, clusters = self._with_rate_limit_retry(
                self.evaluator.specfix_detect, problem, self.cluster_size
            )
            if clusters is None:
                self.evaluator._log_process("detect_fail_fallback", {"reason": "clusters is None, generating directly"})
                code = self._with_rate_limit_retry(self.evaluator.generate_program, requirement, entry_point)
                code = _extract_code_robust(code) if code else ""
                tokens = getattr(self.evaluator.model, "total_tokens", 0) or 0
                self.evaluator._log_process("final_code", {"code": code})
                return code, f"[detect_fail]\n{code}", False, tokens

            final_requirement = requirement
            detect_final_code = self._with_rate_limit_retry(
                self.evaluator.generate_program, requirement, entry_point
            )
            detect_final_code = _extract_code_robust(detect_final_code) if detect_final_code else ""
            self.evaluator._log_process("detect_final_code", {"code": detect_final_code})
            if detect_result and clusters is not None:
                repaired_req, repaired_clusters = self._with_rate_limit_retry(
                    self.evaluator.specfix_repair, clusters, self.cluster_size
                )
                if repaired_req:
                    final_requirement = repaired_req

            tokens_used = getattr(self.evaluator.model, "total_tokens", 0) or 0
            if getattr(self.evaluator, "max_tokens_per_problem", None) and tokens_used >= self.evaluator.max_tokens_per_problem:
                fallback_code = ""
                try:
                    if isinstance(self.process_collector, list) and self.process_collector:
                        for entry in reversed(self.process_collector):
                            code = entry.get("code")
                            if code:
                                fallback_code = _extract_code_robust(code)
                                if fallback_code:
                                    break
                            programs = entry.get("programs")
                            if programs:
                                cand = programs[0]
                                fallback_code = _extract_code_robust(cand) if cand else ""
                                if fallback_code:
                                    break
                except Exception:
                    fallback_code = ""

                self.evaluator._log_process(
                    "final_early_exit",
                    {
                        "reason": f"tokens exceeded before final generation ({tokens_used} >= {self.evaluator.max_tokens_per_problem})",
                        "has_fallback_code": bool(fallback_code),
                    },
                )
                if fallback_code:
                    code = fallback_code
                    tokens = tokens_used
                    full_response = (
                        f"[early_exit_with_fallback]\n"
                        f"reason=tokens_exceeded ({tokens_used} >= {self.evaluator.max_tokens_per_problem})\n"
                        f"[fallback_source]=detect_or_repair_program"
                    )
                    return code, full_response, False, tokens

                code = ""
                tokens = tokens_used
                full_response = f"[early_exit]\nreason=tokens_exceeded ({tokens_used} >= {self.evaluator.max_tokens_per_problem})"
                return code, full_response, False, tokens

            self.evaluator._log_process(
                "final_generation",
                {"requirement_preview": final_requirement[:400] + "..." if len(final_requirement) > 400 else final_requirement},
            )
            code = self._with_rate_limit_retry(self.evaluator.generate_program, final_requirement, entry_point)
            code = _extract_code_robust(code) if code else ""
            tokens = getattr(self.evaluator.model, "total_tokens", 0) or 0
            self.evaluator._log_process("final_code", {"code": code})
            full_response = f"[requirement]\n{final_requirement[:500]}...\n[code]\n{code}"
            return code, full_response, True, tokens
        except Exception as e:
            print(f"[ERR] SpecFix exception: {e}")
            try:
                code = self._with_rate_limit_retry(self.evaluator.generate_program, requirement, entry_point)
                code = _extract_code_robust(code) if code else ""
                tokens = getattr(self.evaluator.model, "total_tokens", 0) or 0
            except Exception:
                code = f"# Exception: {e}"
                tokens = 0
            return code, f"# {e}", False, tokens


# ============================================================
# 璇︾粏鏃ュ織
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

    def save_generation_info(self, problem_dir: Path, code: str, response: str, process_log: Optional[List] = None):
        with open(problem_dir / "generated_code.py", "w", encoding="utf-8") as f:
            f.write(code)
        with open(problem_dir / "full_response.txt", "w", encoding="utf-8") as f:
            f.write(response)
        if process_log:
            with open(problem_dir / "process_log.json", "w", encoding="utf-8") as f:
                json.dump(process_log, f, indent=2, ensure_ascii=False)
            # Also save a readable text version of the process log for debugging.
            lines = []
            for i, entry in enumerate(process_log):
                step = entry.get("step", "?")
                lines.append(f"\n{'='*60}\n[{i+1}] {step}\n{'='*60}")
                for k, v in entry.items():
                    if k == "step":
                        continue
                    if k == "programs" and isinstance(v, list):
                        for j, prog in enumerate(v):
                            lines.append(f"\n--- program_{j+1} ---\n{prog}")
                    elif k == "test_inputs" and isinstance(v, list):
                        lines.append(f"{k} ({len(v)} items): {v[:5]}{'...' if len(v) > 5 else ''}")
                    elif isinstance(v, str) and len(v) > 500:
                        lines.append(f"{k}:\n{v[:500]}...\n")
                    elif isinstance(v, (list, dict)):
                        lines.append(f"{k}: {str(v)[:300]}{'...' if len(str(v)) > 300 else ''}")
                    else:
                        lines.append(f"{k}: {v}")
            with open(problem_dir / "process_log.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(lines))


# ============================================================
# ============================================================
# Single-instance processing
def process_single_instance(args: tuple) -> Dict[str, Any]:
    model_name, temperature, instance, idx, total, logger, cluster_size, simple, max_tokens_per_problem = args
    try:
        print(f"[{idx}/{total}] 寮€濮嬪鐞? {instance.instance_id}")
        problem_dir = logger.create_problem_dir(instance.instance_id)
        logger.save_problem_info(problem_dir, instance)

        process_collector: List[dict] = []
        agent = SpecFixBaselineAgent(
            model_name=model_name,
            temperature=temperature,
            cluster_size=cluster_size,
            process_collector=process_collector,
            max_tokens_per_problem=max_tokens_per_problem,
        )
        start_time = time.time()
        code, response, success, token_nums = agent.generate_final_code(instance, simple=simple)
        generation_time = time.time() - start_time

        logger.save_generation_info(problem_dir, code, response, process_log=process_collector if process_collector else None)

        if code and not code.startswith("# Exception"):
            print(f"[{idx}/{total}] [OK] {instance.instance_id} 鐢熸垚鎴愬姛 (鑰楁椂: {generation_time:.2f}s, tokens: {token_nums})")
        else:
            print(f"[{idx}/{total}] [X] {instance.instance_id} 鐢熸垚澶辫触")

        return {
            "instance_id": instance.instance_id,
            "code": code,
            "test_cases": instance.test_cases,
            "generation_time": generation_time,
            "token_nums": token_nums,
            "problem_dir": str(problem_dir),
            "response": response,
        }
    except Exception as e:
        print(f"[{idx}/{total}] [X] {instance.instance_id} 寮傚父: {e}")
        return {
            "instance_id": instance.instance_id,
            "code": f"# Exception: {e}",
            "test_cases": instance.test_cases,
            "generation_time": 0.0,
            "token_nums": 0,
            "error": str(e),
            "problem_dir": "",
        }


# ============================================================
# ============================================================
# Evaluation and reporting
def _run_evaluation(run_dir: Path, dataset_name: str, all_results: list, workers: int) -> dict:
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
    print(f"\n璇勪及 {len(eval_dataset)} 閬撻鐩?(workers={eval_workers})...")
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


def _write_report_and_summary(run_dir: Path, summary: dict, eval_output: dict, generation_time: float = 0.0, total_tokens: int = 0) -> None:
    pass_count = eval_output["pass_count"]
    total = eval_output["total"]
    pass_at_1 = eval_output["pass_at_1"]
    avg_pass_ratio = eval_output["avg_pass_ratio"]
    detailed_results = eval_output["detailed_results"]

    summary_data = {
        "summary": {
            "method": "SpecFix",
            "dataset": summary.get("dataset_name", ""),
            "model": summary.get("model_name", ""),
            "pass_at_1": pass_at_1,
            "passed": pass_count,
            "total": total,
            "avg_pass_ratio": avg_pass_ratio,
            "generation_time": generation_time,
            "token_usage": {"total": total_tokens, "average_per_problem": total_tokens / total if total and total_tokens else 0},
            "timestamp": datetime.now().isoformat(),
        },
        "results": detailed_results,
    }

    (run_dir / "summary.json").write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")

    report_file = run_dir / "REPORT.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SpecFix Baseline Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Method: SpecFix\n")
        f.write(f"Dataset: {summary.get('dataset_name', '').upper()}\n")
        f.write(f"Model: {summary.get('model_name', '')}\n")
        f.write(f"Problems: {total}\n\n")
        f.write("Summary:\n")
        f.write(f"  - Pass@1: {pass_at_1*100:.2f}% ({pass_count}/{total})\n")
        f.write(f"  - AvgPassRatio: {avg_pass_ratio:.4f}\n")
        f.write(f"  - Generation time: {generation_time:.2f} s\n")
        if total_tokens > 0:
            f.write(f"  - Total tokens: {total_tokens:,}\n")
            avg_tok = int(total_tokens / total) if total else 0
            f.write(f"  - Avg tokens/problem: {avg_tok:,}\n")
        f.write("\nDetailed results:\n")
        for dr in detailed_results:
            status = "[PASS]" if dr["passed"] else "[FAIL]"
            f.write(f"  {status} {dr['instance_id']} (accuracy: {dr['accuracy']*100:.0f}%)\n")
    print(f"Report saved: {report_file}")


# ============================================================
# ============================================================
# Main flow
def main(args) -> int:
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
    root_dir = _BASE / "specfix_baseline_outputs"
    run_name = f"{dataset_name}_{suffix}"

    # === evaluate mode ===
    if args.mode == "evaluate" and args.resume_from:
        run_dir = Path(args.resume_from)
        if not run_dir.exists():
            print(f"Error: directory does not exist: {run_dir}")
            return 1
        if not run_dir.name.startswith("run_"):
            runs = sorted(run_dir.glob("run_*"), key=lambda x: x.name, reverse=True)
            if not runs:
                print(f"Error: no run_* directory found under {run_dir}")
                return 1
            run_dir = runs[0]
            
        ckpt_file = run_dir / "generation_checkpoint.json"
        ckpt = {}
        all_results = []
        if ckpt_file.exists():
            with open(ckpt_file, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            all_results = ckpt.get("results", [])
            
        eval_dataset_name = args.eval_dataset or ckpt.get("dataset_name", dataset_name)
        
        try:
            dataset = get_data(eval_dataset_name)
        except Exception as e:
            dataset = []
            
        dataset_instance_ids = {inst.instance_id: inst for inst in dataset}
        completed_ids = {r.get("instance_id") for r in all_results if "instance_id" in r}
        
        added_count = 0
        for instance_id, inst in dataset_instance_ids.items():
            if instance_id in completed_ids:
                continue
            safe_name = _sanitize_dirname(instance_id)
            problem_dir = run_dir / safe_name
            if (problem_dir / "generated_code.py").exists() and (problem_dir / "full_response.txt").exists():
                completed_ids.add(instance_id)
                added_count += 1
                code = (problem_dir / "generated_code.py").read_text(encoding="utf-8")
                try:
                    response = (problem_dir / "full_response.txt").read_text(encoding="utf-8")
                except Exception:
                    response = code
                all_results.append({
                    "instance_id": instance_id,
                    "code": code,
                    "test_cases": inst.test_cases,
                    "generation_time": 0.0,
                    "token_nums": 0,
                    "problem_dir": str(problem_dir),
                    "response": response,
                })
                
        if not all_results:
            print(f"Error: could not find {ckpt_file}, and no generated code was found in {run_dir}")
            return 1
            
        if added_count > 0:
            print(f"Evaluate mode: recovered {added_count} results from problem directories missing in checkpoint")

        if args.eval_dataset:
            print(f"Evaluate mode: loaded {len(all_results)} results and evaluating on dataset {eval_dataset_name}")
        else:
            print(f"Evaluate mode: loaded {len(all_results)} results from {run_dir}")

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
            print(f"Evaluation outputs will be saved to: {out_run_dir}")

        summary_for_report = {**ckpt, "dataset_name": eval_dataset_name}
        if args.eval_dataset:
            summary_for_report["generation_dataset"] = ckpt.get("dataset_name")
        tu = ckpt.get("token_usage") or {}
        total_tokens = tu.get("total", 0) if isinstance(tu, dict) else 0
        _write_report_and_summary(
            out_run_dir, summary_for_report, eval_output,
            generation_time=ckpt.get("generation_time", 0),
            total_tokens=total_tokens,
        )
        (out_run_dir / "eval_results.json").write_text(
            json.dumps({"summary": summary_for_report, "eval": {k: v for k, v in eval_output.items() if k != "eval_results"}}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("=" * 80)
        print(f"Pass@1: {eval_output['pass_count']}/{eval_output['total']} ({eval_output['pass_at_1']*100:.2f}%)")
        print(f"Evaluation time: {eval_time:.1f}s")
        print(f"Results: {out_run_dir}")
        print("=" * 80)
        return 0

    # === generate mode ===
    print("=" * 80)
    print("SpecFix Baseline")
    print("=" * 80)
    print(f"  Mode: {args.mode}")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Workers: {args.workers}")
    if args.limit:
        print(f"  Limit: {args.limit}")
    print("=" * 80)

    try:
        dataset = get_data(dataset_name)
    except FileNotFoundError as e:
        print(f"[ERR] Dataset not found: {e}")
        print("[TIP] Make sure the Datasets directory exists and contains the required jsonl file")
        return 1

    if args.limit and args.limit > 0:
        dataset = dataset[: args.limit]
    print(f"Loaded {len(dataset)} problems")

    # determine run directory
    completed_ids = set()
    prev_ckpt = {}
    run_dir = None
    all_results = []
    if args.resume_from and args.mode != "evaluate":
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            if resume_path.name.startswith("run_"):
                run_dir = resume_path
            else:
                runs = sorted(resume_path.glob("run_*"), key=lambda x: x.name, reverse=True)
                run_dir = runs[0] if runs else None
            
            if run_dir:
                if (run_dir / "generation_checkpoint.json").exists():
                    with open(run_dir / "generation_checkpoint.json", "r", encoding="utf-8") as f:
                        prev_ckpt = json.load(f)
                    all_results = prev_ckpt.get("results", [])
                    completed_ids = {r.get("instance_id") for r in all_results if "instance_id" in r}
                
                # Scan missing problem directories
                dataset_instance_ids = {inst.instance_id: inst for inst in dataset}
                for instance_id, inst in dataset_instance_ids.items():
                    if instance_id in completed_ids:
                        continue
                    safe_name = _sanitize_dirname(instance_id)
                    problem_dir = run_dir / safe_name
                    if (problem_dir / "generated_code.py").exists() and (problem_dir / "full_response.txt").exists():
                        completed_ids.add(instance_id)
                        try:
                            code = (problem_dir / "generated_code.py").read_text(encoding="utf-8")
                            try:
                                response = (problem_dir / "full_response.txt").read_text(encoding="utf-8")
                            except Exception:
                                response = code
                            all_results.append({
                                "instance_id": instance_id,
                                "code": code,
                                "test_cases": inst.test_cases,
                                "generation_time": 0.0,
                                "token_nums": 0,
                                "problem_dir": str(problem_dir),
                                "response": response,
                            })
                        except Exception as read_err:
                            print(f"[WARN] Failed to read cached code for {instance_id}: {read_err}")
                            completed_ids.remove(instance_id)

                if completed_ids:
                    print(f"Resume mode: completed {len(completed_ids)} problems, remaining {len(dataset) - len(completed_ids)}")
                    logger = DetailedLogger.__new__(DetailedLogger)
                    logger.output_dir = run_dir.parent
                    logger.run_dir = run_dir
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

    todo_instances = [inst for inst in dataset if inst.instance_id not in completed_ids]
    if not todo_instances:
        print("All problems are already completed; skipping generation")
    else:
        simple = getattr(args, "simple", False)
        print(f"\nStarting code generation ({'simple mode' if simple else 'full SpecFix'}, {args.workers} threads, total {len(todo_instances)} problems)...")
        # Per-problem token cap to avoid pathological long-running cases
        max_tokens_per_problem = getattr(args, "max_tokens_per_problem", None)
        args_list = [
            (
                model_name,
                args.temperature,
                inst,
                idx + 1,
                len(todo_instances),
                logger,
                args.cluster_size,
                simple,
                max_tokens_per_problem,
            )
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

        ckpt_data = {
            "results": all_results,
            "dataset_name": dataset_name,
            "model_name": model_name,
            "generation_time": generation_time,
            "token_usage": {"total": total_tokens, "average_per_problem": total_tokens / len(all_results) if all_results else 0},
            "timestamp": datetime.now().isoformat(),
        }
        (run_dir / "generation_checkpoint.json").write_text(json.dumps(ckpt_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\ngeneration_checkpoint.json saved")
        print(f"Generation time: {generation_time:.1f}s")

    if args.mode == "generate":
        print(f"\nNext step: python SpecFix-main/specfix_baseline.py --mode evaluate --resume-from \"{run_dir}\"")
        return 0

    # === evaluation ===
    ckpt_file = run_dir / "generation_checkpoint.json"
    with open(ckpt_file, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    all_results = ckpt["results"]
    generation_time = ckpt.get("generation_time", 0)

    eval_start_t = time.time()
    eval_output = _run_evaluation(run_dir, dataset_name, all_results, args.workers)
    eval_time = time.time() - eval_start_t

    tu = ckpt.get("token_usage") or {}
    total_tokens = tu.get("total", 0) if isinstance(tu, dict) else 0
    _write_report_and_summary(run_dir, ckpt, eval_output, generation_time, total_tokens)
    (run_dir / "eval_results.json").write_text(
        json.dumps({"summary": ckpt, "eval": {k: v for k, v in eval_output.items() if k != "eval_results"}}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 80)
    print(f"Pass@1: {eval_output['pass_count']}/{eval_output['total']} ({eval_output['pass_at_1']*100:.2f}%)")
    print(f"Evaluation time: {eval_time:.1f}s")
    print(f"Results: {run_dir}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SpecFix Baseline Runner")
    parser.add_argument("--model", type=str, default="deepseek-chat", choices=MODEL_OPTIONS,
                        help="deepseek-chat | qwen3-coder-30b-a3b-instruct | gpt-5-mini-2025-08-07 | gemini-3-flash-preview")
    parser.add_argument("--dataset", type=str, default="code_contests", choices=DATASET_OPTIONS)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cluster-size", type=int, default=20, help="SpecFix N (cluster_sample_size)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems for testing")
    parser.add_argument("--max-tokens-per-problem", type=int, default=210000,
                        help="Per-problem token cap; if exceeded, stop early for that problem")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--mode", type=str, default="all", choices=["all", "generate", "evaluate"])
    parser.add_argument("--resume-from", type=str, default=None,
                        help="evaluate: specify a run dir; generate/all: resume and skip completed problems")
    parser.add_argument("--eval-dataset", type=str, default=None,
                        help="evaluate mode: specify evaluation dataset")
    parser.add_argument("--eval-output", type=str, default=None,
                        help="evaluate mode: output directory for evaluation results")
    parser.add_argument("--simple", action="store_true", default=False,
                        help="Simple mode: skip detect+repair and generate directly")
    parser.add_argument("--no-simple", dest="simple", action="store_false",
                        help="Full SpecFix pipeline: detect + repair (default)")
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


