import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent
PARENT_DIR = ROOT_DIR.parent
SCOT_ROOT = PARENT_DIR / "Self-collaboration-Code-Generation-main"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCOT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCOT_ROOT))

from apps_eval.data import InstanceData, get_data
from apps_eval.executor import EvalResult, evaluate_case
from apps_eval.parallel_runner import eval_code
from core.generate_code import build_llm


DATASET_OPTIONS = [
    "code_contests_raw",
    "code_contests",
    "apps",
    "apps_eval",
    "xCodeEval",
    "livecodebench",
]

MODEL_SUFFIX_MAP = {
    "gpt-5-mini-2025-08-07": "gpt5mini",
    "gemini-3-flash-preview": "gemini3flash",
}
_LLM_SEMAPHORE_CACHE: Dict[int, threading.Semaphore] = {}
_LLM_SEMAPHORE_LOCK = threading.Lock()


def _get_llm_semaphore(max_concurrent_requests: int) -> threading.Semaphore:
    size = max(1, max_concurrent_requests)
    with _LLM_SEMAPHORE_LOCK:
        if size not in _LLM_SEMAPHORE_CACHE:
            _LLM_SEMAPHORE_CACHE[size] = threading.Semaphore(size)
        return _LLM_SEMAPHORE_CACHE[size]


def _model_to_suffix(model_name: str) -> str:
    if model_name in MODEL_SUFFIX_MAP:
        return MODEL_SUFFIX_MAP[model_name]
    if "qwen" in (model_name or "").lower():
        return "qwen"
    return "deepseek"


def _setup_model_env(model_name: str, api_key: Optional[str] = None) -> None:
    if model_name == "gpt-5-mini-2025-08-07":
        os.environ["MODEL_API_BASE_URL"] = "http://api.yesapikey.com/v1"
        os.environ["MODEL_API_KEY_ENV"] = "GPT5_MINI_API_KEY"
        os.environ["MODEL_C"] = model_name
        if api_key:
            os.environ["GPT5_MINI_API_KEY"] = api_key
    elif model_name == "gemini-3-flash-preview":
        os.environ["MODEL_API_BASE_URL"] = "http://api.yesapikey.com/v1"
        os.environ["MODEL_API_KEY_ENV"] = "GEMINI_FLASH_API_KEY"
        os.environ["MODEL_C"] = model_name
        if api_key:
            os.environ["GEMINI_FLASH_API_KEY"] = api_key
    elif "qwen" in (model_name or "").lower():
        os.environ["MODEL_API_BASE_URL"] = os.environ.get(
            "MODEL_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        os.environ["MODEL_API_KEY_ENV"] = "DASHSCOPE_API_KEY"
        os.environ["MODEL_C"] = model_name
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key
    else:
        os.environ["MODEL_API_BASE_URL"] = os.environ.get(
            "MODEL_API_BASE_URL", "https://api.deepseek.com/v1"
        )
        os.environ["MODEL_API_KEY_ENV"] = os.environ.get(
            "MODEL_API_KEY_ENV", "DEEPSEEK_API_KEY"
        )
        os.environ["MODEL_C"] = model_name
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key


def _build_llm(model_name: str, temperature: float, phase: str):
    _setup_model_env(model_name)
    if model_name == "gpt-5-mini-2025-08-07":
        return build_llm(
            "MODEL_C",
            temperature=temperature,
            max_tokens=60000,
            reasoning={"effort": "minimal"},
        )
    if model_name == "gemini-3-flash-preview":
        return build_llm("MODEL_C", temperature=temperature, max_tokens=60000)
    if "qwen" in (model_name or "").lower():
        max_tokens = 8192 if phase == "explain" else 8192
        return build_llm(
            "MODEL_C",
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"enable_thinking": False},
        )
    if phase == "explain":
        return build_llm("MODEL_C", temperature=temperature, max_tokens=8192)
    return build_llm("MODEL_C", temperature=temperature, max_tokens=8192)


def _extract_code(response: str) -> str:
    text = (response or "").strip()
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def _format_public_tests(instance: InstanceData) -> str:
    if not instance.public_test_cases:
        return ""
    inputs = instance.public_test_cases.get("inputs", [])
    outputs = instance.public_test_cases.get("outputs", [])
    pairs = []
    for idx, (inp, out) in enumerate(zip(inputs, outputs), 1):
        pairs.append(f"Example {idx}\nInput:\n{inp}\nExpected output:\n{out}")
        if idx >= 3:
            break
    if not pairs:
        return ""
    return "\n\nPublic examples:\n" + "\n\n".join(pairs)


def _build_problem_prompt(instance: InstanceData) -> Tuple[str, str]:
    is_call_based = "fn_name" in instance.test_cases
    starter_block = ""
    if instance.starter_code:
        starter_block = (
            "\n\nStarter code reference:\n```python\n"
            f"{instance.starter_code}\n```"
        )
    examples_block = _format_public_tests(instance)
    system = (
        "You are an expert Python programmer. Solve the programming problem. "
        "Return only one final Python code block."
    )
    if is_call_based:
        fn_name = instance.test_cases["fn_name"]
        user = (
            f"Problem:\n{instance.problem_statement}{starter_block}{examples_block}\n\n"
            "Write a correct Python solution that follows the call-based format.\n"
            f"You must define class Solution with method `{fn_name}`.\n"
            "Do not add explanations outside the code block."
        )
    else:
        user = (
            f"Problem:\n{instance.problem_statement}{starter_block}{examples_block}\n\n"
            "Write a correct Python solution using standard input/output.\n"
            "Read from stdin and write to stdout.\n"
            "Do not add explanations outside the code block."
        )
    return system, user


def _build_explanation_prompt(
    instance: InstanceData, code: str, failure_summary: str
) -> Tuple[str, str]:
    system = (
        "You are a precise Python debugging assistant. "
        "Explain the bug concisely in at most 3 sentences and do not provide code."
    )
    user = (
        f"Problem:\n{instance.problem_statement}\n\n"
        f"Buggy code:\n```python\n{code}\n```\n\n"
        f"Observed failure:\n{failure_summary}\n\n"
        "Explain the root cause briefly."
    )
    return system, user


def _build_repair_prompt(
    instance: InstanceData,
    code: str,
    failure_summary: str,
    explanation: str,
) -> Tuple[str, str]:
    is_call_based = "fn_name" in instance.test_cases
    starter_block = ""
    if instance.starter_code:
        starter_block = (
            "\n\nStarter code reference:\n```python\n"
            f"{instance.starter_code}\n```"
        )
    format_hint = (
        f"Preserve the call-based format and define class Solution with method `{instance.test_cases['fn_name']}`."
        if is_call_based
        else "Preserve the standard input/output format."
    )
    system = (
        "You are an expert Python programmer fixing a buggy solution. "
        "Return only one final Python code block."
    )
    user = (
        f"Problem:\n{instance.problem_statement}{starter_block}\n\n"
        f"Buggy code:\n```python\n{code}\n```\n\n"
        f"Observed failure:\n{failure_summary}\n\n"
        f"Debug explanation:\n{explanation}\n\n"
        f"Fix the code. {format_hint}"
    )
    return system, user


def _build_messages_for_model(
    model_name: str,
    system_prompt: str,
    user_content: str,
) -> List[Dict[str, str]]:
    if model_name == "gemini-3-flash-preview":
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
        return [
            {
                "role": "user",
                "content": json.dumps(request_body, ensure_ascii=False),
            }
        ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _call_llm(
    model_name: str,
    temperature: float,
    phase: str,
    messages: List[Dict[str, str]],
    max_retries: int = 6,
    retry_base_delay: float = 2.0,
    max_concurrent_requests: int = 8,
    request_label: str = "llm",
) -> Tuple[str, int]:
    semaphore = _get_llm_semaphore(max_concurrent_requests)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            with semaphore:
                llm = _build_llm(model_name, temperature, phase)
                before = getattr(llm, "total_tokens", 0)
                response = llm.chat(messages, temperature=temperature)
                after = getattr(llm, "total_tokens", before)
                return response, max(0, after - before)
        except Exception as exc:
            last_error = exc
            error_text = str(exc)
            if "HTTP 400" in error_text or '"code":"bad_request"' in error_text or '"type":"yescale_api_error"' in error_text:
                break
            if attempt >= max_retries:
                break
            sleep_s = retry_base_delay * (2 ** (attempt - 1)) + random.uniform(0.0, retry_base_delay)
            print(
                f"[Retry] {request_label} failed on attempt {attempt}/{max_retries}: {exc}. "
                f"Retrying in {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)

    error_text = str(last_error or "")
    timeout_markers = [
        "Read timed out",
        "read timeout",
        "ConnectTimeout",
        "connect timeout",
        "timed out",
        "HTTPConnectionPool",
        "HTTPSConnectionPool",
    ]
    invalid_request_markers = [
        "HTTP 400",
        "invalid_parameter_error",
        "invalid_request_error",
        "Range of input length should be",
        "maximum context length",
        "token limit",
        "tokens",
    ]
    if any(marker in error_text for marker in timeout_markers):
        print(
            f"[Fallback] {request_label} timed out after {max_retries} attempts. "
            "Returning empty response and treating this candidate as failed."
        )
        return "", 0
    if any(marker in error_text for marker in invalid_request_markers):
        print(
            f"[Fallback] {request_label} hit a non-retriable request error after {max_retries} attempts. "
            "Returning empty response and treating this candidate as failed."
        )
        return "", 0

    raise RuntimeError(
        f"LLM request failed after {max_retries} attempts for {request_label}: {last_error}"
    )

def _evaluate_instance_code(
    instance: InstanceData,
    code: str,
    timeout: float,
    use_public_only: bool = False,
) -> Tuple[float, List[EvalResult]]:
    selected_cases = (instance.public_test_cases or {}) if use_public_only else (instance.test_cases or {})
    inputs = selected_cases.get("inputs", []) if selected_cases else []
    outputs = selected_cases.get("outputs", []) if selected_cases else []

    if not inputs or not outputs:
        return 0.0, []

    results: List[EvalResult] = []
    is_call_based = "fn_name" in instance.test_cases
    entry_func = instance.test_cases.get("fn_name") if is_call_based else None
    for test_input, test_output in zip(inputs, outputs):
        if is_call_based:
            result = evaluate_case(
                code=code,
                input_data=test_input,
                expected=test_output,
                timeout=timeout,
                mode="call",
                entry_func=entry_func,
            )
        else:
            result = evaluate_case(
                code=code,
                input_data=test_input,
                expected=test_output,
                timeout=timeout,
                mode="stdio",
            )
        results.append(result)
    passed = sum(1 for item in results if item.status == "AC")
    accuracy = passed / len(results) if results else 0.0
    return accuracy, results


def _summarize_failure(results: List[EvalResult]) -> str:
    for idx, result in enumerate(results, 1):
        if result.status == "AC":
            continue
        if result.status == "WA":
            return (
                f"Test case {idx} failed with wrong answer.\n"
                f"Input:\n{result.stdin}\n\n"
                f"Expected:\n{result.expected}\n\n"
                f"Actual:\n{result.stdout}"
            )
        return (
            f"Test case {idx} failed with status {result.status}.\n"
            f"Input:\n{result.stdin}\n\n"
            f"Expected:\n{result.expected}\n\n"
            f"Stdout:\n{result.stdout}\n\n"
            f"Stderr:\n{result.stderr}"
        )
    return "No failing test case summary was available."


def _serialize_eval_results(results: List[EvalResult]) -> List[Dict[str, Any]]:
    serialized = []
    for item in results:
        serialized.append(
            {
                "status": item.status,
                "stdin": item.stdin,
                "stdout": item.stdout,
                "stderr": item.stderr,
                "time_cost": item.time_cost,
                "expected": item.expected,
            }
        )
    return serialized


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class StageRecord:
    attempt: int
    explanation: str
    explanation_response: str
    repair_response: str
    repaired_code: str
    accuracy: float
    passed: bool
    failure_summary: str
    eval_results: List[Dict[str, Any]]
    tokens_used: int


INVALID_PATH_CHARS = '<>:"/\\|?*'
INVALID_PATH_CHAR_TABLE = str.maketrans({ch: '_' for ch in INVALID_PATH_CHARS})


def _safe_problem_dir_name(name: Any) -> str:
    safe_name = str(name).translate(INVALID_PATH_CHAR_TABLE).strip().rstrip('. ')
    return safe_name or 'unnamed'

class DetailedLogger:
    def __init__(self, output_dir: Path, run_dir: Optional[Path] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = run_dir or (self.output_dir / f"run_{self.timestamp}")
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_problem(self, result: Dict[str, Any]) -> None:
        problem_dir = self.run_dir / _safe_problem_dir_name(result["instance_id"])
        problem_dir.mkdir(parents=True, exist_ok=True)
        (problem_dir / "problem.txt").write_text(
            result["problem_statement"], encoding="utf-8"
        )
        (problem_dir / "base_response.txt").write_text(
            result["base_response"], encoding="utf-8"
        )
        (problem_dir / "base_solution.py").write_text(
            result["base_code"], encoding="utf-8"
        )
        (problem_dir / "result.json").write_text(
            json.dumps(_json_safe(result), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if result.get("final_code"):
            (problem_dir / "final_solution.py").write_text(
                result["final_code"], encoding="utf-8"
            )
        for stage in result.get("stages", []):
            attempt_dir = problem_dir / f"repair_{stage['attempt']}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            (attempt_dir / "explanation.txt").write_text(
                stage["explanation"], encoding="utf-8"
            )
            (attempt_dir / "explanation_response.txt").write_text(
                stage["explanation_response"], encoding="utf-8"
            )
            (attempt_dir / "repair_response.txt").write_text(
                stage["repair_response"], encoding="utf-8"
            )
            (attempt_dir / "solution.py").write_text(
                stage["repaired_code"], encoding="utf-8"
            )

    def save_json(self, name: str, payload: Dict[str, Any]) -> None:
        (self.run_dir / name).write_text(
            json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _process_instance(
    instance: InstanceData,
    model_name: str,
    generation_temperature: float = 0.8,
    repair_temperature: float = 0.8,
    timeout: float = 10.0,
    max_repair_attempts: int = 1,
    num_base_samples: int = 1,
    per_problem_parallelism: int = 4,
    max_concurrent_llm_calls: int = 8,
    llm_max_retries: int = 6,
    llm_retry_base_delay: float = 2.0,
) -> Dict[str, Any]:
    start_time = time.time()
    base_system, base_user = _build_problem_prompt(instance)

    base_candidates: List[Dict[str, Any]] = []
    repair_stages: List[Dict[str, Any]] = []
    total_tokens = 0
    best_base_candidate: Optional[Dict[str, Any]] = None
    best_candidate_overall: Optional[Dict[str, Any]] = None
    final_candidate: Optional[Dict[str, Any]] = None
    has_public_tests = bool(
        instance.public_test_cases
        and instance.public_test_cases.get("inputs")
        and instance.public_test_cases.get("outputs")
    )

    def _candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[float, float]:
        return (
            candidate.get("public_accuracy", 0.0),
            -float(candidate.get("order", 0)),
        )

    def _update_best(
        candidate: Dict[str, Any], current_best: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if current_best is None or _candidate_sort_key(candidate) > _candidate_sort_key(current_best):
            return candidate
        return current_best

    def _build_base_candidate(sample_idx: int) -> Dict[str, Any]:
        base_response, base_tokens = _call_llm(
            model_name,
            generation_temperature,
            "generate",
            _build_messages_for_model(model_name, base_system, base_user),
            max_retries=llm_max_retries,
            retry_base_delay=llm_retry_base_delay,
            max_concurrent_requests=max_concurrent_llm_calls,
            request_label=f"base:{instance.instance_id}:{sample_idx}",
        )
        base_code = _extract_code(base_response)
        public_accuracy, public_eval_results = _evaluate_instance_code(
            instance,
            base_code,
            timeout,
            use_public_only=True,
        )
        failure_summary = (
            _summarize_failure(public_eval_results)
            if public_eval_results and public_accuracy < 1.0
            else ""
        )
        return {
            "order": sample_idx,
            "phase": "base",
            "response": base_response,
            "code": base_code,
            "public_accuracy": public_accuracy,
            "public_passed": public_accuracy == 1.0 if public_eval_results else False,
            "public_eval_results": _serialize_eval_results(public_eval_results),
            "failure_summary": failure_summary,
            "tokens_used": base_tokens,
        }

    def _repair_candidate(candidate: Dict[str, Any], repair_round: int, stage_attempt: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        failure_summary = candidate["failure_summary"] or "No public failure summary available."
        explain_system, explain_user = _build_explanation_prompt(
            instance,
            candidate["code"],
            failure_summary,
        )
        explanation_response, explain_tokens = _call_llm(
            model_name,
            repair_temperature,
            "explain",
            _build_messages_for_model(model_name, explain_system, explain_user),
            max_retries=llm_max_retries,
            retry_base_delay=llm_retry_base_delay,
            max_concurrent_requests=max_concurrent_llm_calls,
            request_label=f"explain:{instance.instance_id}:{candidate['order']}",
        )
        explanation = explanation_response.strip()

        repair_system, repair_user = _build_repair_prompt(
            instance,
            candidate["code"],
            failure_summary,
            explanation,
        )
        repair_response, repair_tokens = _call_llm(
            model_name,
            repair_temperature,
            "repair",
            _build_messages_for_model(model_name, repair_system, repair_user),
            max_retries=llm_max_retries,
            retry_base_delay=llm_retry_base_delay,
            max_concurrent_requests=max_concurrent_llm_calls,
            request_label=f"repair:{instance.instance_id}:{candidate['order']}",
        )
        repaired_code = _extract_code(repair_response)
        repaired_public_accuracy, repaired_public_eval_results = _evaluate_instance_code(
            instance,
            repaired_code,
            timeout,
            use_public_only=True,
        )
        repaired_failure_summary = (
            _summarize_failure(repaired_public_eval_results)
            if repaired_public_eval_results and repaired_public_accuracy < 1.0
            else ""
        )
        repaired_candidate = {
            "order": num_base_samples + stage_attempt,
            "phase": "repair",
            "parent_order": candidate["order"],
            "repair_round": repair_round,
            "response": repair_response,
            "code": repaired_code,
            "public_accuracy": repaired_public_accuracy,
            "public_passed": repaired_public_accuracy == 1.0 if repaired_public_eval_results else False,
            "public_eval_results": _serialize_eval_results(repaired_public_eval_results),
            "failure_summary": repaired_failure_summary,
            "tokens_used": explain_tokens + repair_tokens,
            "explanation": explanation,
            "explanation_response": explanation_response,
        }
        stage_record = asdict(
            StageRecord(
                attempt=stage_attempt,
                explanation=explanation,
                explanation_response=explanation_response,
                repair_response=repair_response,
                repaired_code=repaired_code,
                accuracy=repaired_public_accuracy,
                passed=repaired_public_accuracy == 1.0 if repaired_public_eval_results else False,
                failure_summary=failure_summary,
                eval_results=_serialize_eval_results(repaired_public_eval_results),
                tokens_used=explain_tokens + repair_tokens,
            )
        )
        return repaired_candidate, stage_record

    base_parallelism = max(1, min(per_problem_parallelism, num_base_samples))
    with ThreadPoolExecutor(max_workers=base_parallelism) as executor:
        in_flight: Dict[Any, int] = {}
        next_sample_idx = 1
        while next_sample_idx <= num_base_samples and len(in_flight) < base_parallelism:
            future = executor.submit(_build_base_candidate, next_sample_idx)
            in_flight[future] = next_sample_idx
            next_sample_idx += 1

        while in_flight:
            done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                candidate = future.result()
                in_flight.pop(future, None)
                base_candidates.append(candidate)
                total_tokens += candidate["tokens_used"]
                best_base_candidate = _update_best(candidate, best_base_candidate)
                best_candidate_overall = _update_best(candidate, best_candidate_overall)

                if has_public_tests and candidate["public_accuracy"] == 1.0:
                    final_candidate = candidate
                    break

                if next_sample_idx <= num_base_samples:
                    new_future = executor.submit(_build_base_candidate, next_sample_idx)
                    in_flight[new_future] = next_sample_idx
                    next_sample_idx += 1

            if final_candidate is not None:
                for pending in list(in_flight.keys()):
                    pending.cancel()
                break

    base_candidates.sort(key=lambda item: item["order"])
    frontier: List[Dict[str, Any]] = [] if final_candidate else list(base_candidates)
    stage_attempt = 0

    for repair_round in range(1, max_repair_attempts + 1):
        if final_candidate or not has_public_tests or not frontier:
            break

        repair_parallelism = max(1, min(per_problem_parallelism, len(frontier)))
        next_frontier: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=repair_parallelism) as executor:
            in_flight: Dict[Any, Dict[str, Any]] = {}
            frontier_index = 0

            while frontier_index < len(frontier) and len(in_flight) < repair_parallelism:
                candidate = frontier[frontier_index]
                stage_attempt += 1
                future = executor.submit(_repair_candidate, candidate, repair_round, stage_attempt)
                in_flight[future] = candidate
                frontier_index += 1

            while in_flight:
                done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    in_flight.pop(future, None)
                    repaired_candidate, stage_record = future.result()
                    repair_stages.append(stage_record)
                    total_tokens += repaired_candidate["tokens_used"]
                    best_candidate_overall = _update_best(repaired_candidate, best_candidate_overall)

                    if has_public_tests and repaired_candidate["public_accuracy"] == 1.0:
                        final_candidate = repaired_candidate
                        break

                    next_frontier.append(repaired_candidate)

                    if frontier_index < len(frontier):
                        next_candidate = frontier[frontier_index]
                        stage_attempt += 1
                        new_future = executor.submit(_repair_candidate, next_candidate, repair_round, stage_attempt)
                        in_flight[new_future] = next_candidate
                        frontier_index += 1

                if final_candidate is not None:
                    for pending in list(in_flight.keys()):
                        pending.cancel()
                    break

        next_frontier.sort(key=lambda item: item["order"])
        frontier = next_frontier

    if final_candidate is None:
        final_candidate = best_candidate_overall or best_base_candidate or {
            "response": "",
            "code": "",
            "public_accuracy": 0.0,
            "public_passed": False,
            "public_eval_results": [],
            "tokens_used": 0,
        }

    base_reference = best_base_candidate or final_candidate

    result: Dict[str, Any] = {
        "instance_id": instance.instance_id,
        "problem_statement": instance.problem_statement,
        "base_response": base_reference.get("response", ""),
        "base_code": base_reference.get("code", ""),
        "base_public_accuracy": base_reference.get("public_accuracy", 0.0),
        "base_public_passed": base_reference.get("public_passed", False),
        "base_public_eval_results": base_reference.get("public_eval_results", []),
        "base_accuracy": None,
        "base_passed": None,
        "base_eval_results": [],
        "base_candidates": base_candidates,
        "stages": repair_stages,
        "tokens_used": total_tokens,
        "generation_time": time.time() - start_time,
        "final_code": final_candidate.get("code", ""),
        "final_public_accuracy": final_candidate.get("public_accuracy", 0.0),
        "final_public_passed": final_candidate.get("public_passed", False),
        "final_public_eval_results": final_candidate.get("public_eval_results", []),
        "final_accuracy": None,
        "final_passed": None,
        "final_eval_results": [],
        "final_phase": final_candidate.get("phase", "base"),
        "num_base_samples": num_base_samples,
        "per_problem_parallelism": per_problem_parallelism,
        "llm_max_retries": llm_max_retries,
        "max_concurrent_llm_calls": max_concurrent_llm_calls,
    }
    return result

def _resolve_output_dir(
    output_dir: Optional[str], dataset_name: str, model_name: str
) -> Path:
    if output_dir:
        return Path(output_dir)
    suffix = _model_to_suffix(model_name)
    return ROOT_DIR / "self_repair_outputs" / f"{dataset_name}_{suffix}"


def _select_instances(
    dataset: List[InstanceData],
    limit: Optional[int],
    instance_ids: Optional[str],
) -> List[InstanceData]:
    selected = dataset
    if instance_ids:
        wanted = {item.strip() for item in instance_ids.split(",") if item.strip()}
        selected = [instance for instance in selected if instance.instance_id in wanted]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _resolve_generation_resume_dir(resume_from: str) -> Path:
    candidate = Path(resume_from)
    if not candidate.exists():
        raise FileNotFoundError(f"Resume path does not exist: {resume_from}")
    if candidate.is_dir() and candidate.name.startswith("run_"):
        return candidate
    run_dirs = sorted(
        [item for item in candidate.glob("run_*") if item.is_dir()],
        key=lambda item: item.name,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found under {resume_from}")
    return run_dirs[0]


def _load_existing_generation_results(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    existing: Dict[str, Dict[str, Any]] = {}
    for problem_dir in sorted(run_dir.iterdir(), key=lambda item: item.name):
        if not problem_dir.is_dir():
            continue
        result_path = problem_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[Resume] Skip unreadable result file: {result_path} ({exc})")
            continue
        instance_id = result.get("instance_id") or problem_dir.name
        existing[instance_id] = result
    return existing


def _infer_dataset_name_from_run_dir(run_dir: Path) -> Optional[str]:
    parent_name = run_dir.parent.name
    for dataset_name in sorted(DATASET_OPTIONS, key=len, reverse=True):
        prefix = f"{dataset_name}_"
        if parent_name.startswith(prefix):
            return dataset_name
    return None


def _ensure_generation_checkpoint(
    run_dir: Path,
    dataset_name: Optional[str],
    model_name: Optional[str],
) -> Path:
    checkpoint_path = run_dir / "generation_checkpoint.json"
    if checkpoint_path.exists():
        return checkpoint_path

    results_map = _load_existing_generation_results(run_dir)
    if not results_map:
        raise FileNotFoundError(
            f"No generation_checkpoint.json or per-problem result.json files found under {run_dir}"
        )

    resolved_dataset_name = dataset_name or _infer_dataset_name_from_run_dir(run_dir)
    if not resolved_dataset_name:
        raise ValueError(
            "Unable to infer dataset name from run directory. Please pass --dataset when rebuilding from result.json files."
        )
    resolved_model_name = model_name
    if not resolved_model_name:
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                resolved_model_name = summary.get("summary", {}).get("model")
            except Exception:
                resolved_model_name = None
    if not resolved_model_name:
        raise ValueError(
            "Unable to infer model name from run directory. Please pass --model when rebuilding from result.json files."
        )

    dataset = get_data(resolved_dataset_name)
    order = {instance.instance_id: idx for idx, instance in enumerate(dataset)}
    results = list(results_map.values())
    results.sort(key=lambda item: order.get(item.get("instance_id"), 10**9))
    generation_time = sum(float(item.get("generation_time", 0.0)) for item in results)

    checkpoint = {
        "results": results,
        "dataset_name": resolved_dataset_name,
        "model_name": resolved_model_name,
        "generation_temperature": None,
        "repair_temperature": None,
        "timeout": 10.0,
        "max_repair_attempts": None,
        "num_base_samples": None,
        "per_problem_parallelism": None,
        "max_concurrent_llm_calls": None,
        "llm_max_retries": None,
        "llm_retry_base_delay": None,
        "workers": None,
        "generation_time": generation_time,
        "timestamp": datetime.now().isoformat(),
        "rebuilt_from_results": True,
    }
    checkpoint_path.write_text(
        json.dumps(_json_safe(checkpoint), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Rebuild] Created generation_checkpoint.json from {len(results)} existing result.json files: {checkpoint_path}")
    return checkpoint_path


def _build_summary(
    results: List[Dict[str, Any]],
    model_name: str,
    dataset_name: str,
    generation_dataset_name: Optional[str],
    total_time: float,
    mode: str,
    hidden_eval_time: float = 0.0,
) -> Dict[str, Any]:
    total = len(results)
    base_public_passed = sum(1 for result in results if result.get("base_public_passed"))
    final_public_passed = sum(1 for result in results if result.get("final_public_passed"))
    public_repaired = sum(
        1
        for result in results
        if (not result.get("base_public_passed")) and result.get("final_public_passed")
    )
    hidden_available = all(
        result.get("base_passed") is not None and result.get("final_passed") is not None
        for result in results
    ) if results else False
    base_passed = sum(1 for result in results if result.get("base_passed") is True)
    final_passed = sum(1 for result in results if result.get("final_passed") is True)
    repaired = sum(
        1
        for result in results
        if result.get("base_passed") is False and result.get("final_passed") is True
    )
    total_tokens = sum(result.get("tokens_used", 0) for result in results)
    generation_time = sum(result.get("generation_time", 0.0) for result in results)
    return {
        "summary": {
            "method": "Self-Repair",
            "mode": mode,
            "dataset": dataset_name,
            "generation_dataset": generation_dataset_name or dataset_name,
            "model": model_name,
            "total": total,
            "base_public_passed": base_public_passed,
            "base_public_pass_at_1": (base_public_passed / total * 100) if total else 0.0,
            "final_public_passed": final_public_passed,
            "final_public_pass_at_1": (final_public_passed / total * 100) if total else 0.0,
            "public_repaired_successes": public_repaired,
            "base_passed": base_passed if hidden_available else None,
            "base_pass_at_1": ((base_passed / total) * 100) if total and hidden_available else None,
            "final_passed": final_passed if hidden_available else None,
            "pass_at_1": ((final_passed / total) * 100) if total and hidden_available else None,
            "repaired_successes": repaired if hidden_available else None,
            "time_cost": {
                "total": total_time,
                "generation": generation_time,
                "hidden_evaluation": hidden_eval_time,
            },
            "token_usage": {
                "total": total_tokens,
                "average_per_problem": (total_tokens / total) if total else 0.0,
            },
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }


def _write_report(run_dir: Path, summary: Dict[str, Any]) -> None:
    stats = summary["summary"]

    def _format_pass_line(label: str, value: Optional[float], passed: Optional[int], total: int) -> str:
        if value is None or passed is None:
            return f"{label}: N/A"
        return f"{label}: {value:.2f}% ({passed}/{total})"

    lines = [
        "=" * 80,
        "Self-Repair Baseline Report",
        "=" * 80,
        "",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Method: {stats['method']}",
        f"Model: {stats['model']}",
        f"Generation dataset: {stats['generation_dataset']}",
        f"Evaluation dataset: {stats['dataset']}",
        f"Problems: {stats['total']}",
        "",
        f"Base Public Pass@1: {stats['base_public_pass_at_1']:.2f}% ({stats['base_public_passed']}/{stats['total']})",
        f"Final Public Pass@1: {stats['final_public_pass_at_1']:.2f}% ({stats['final_public_passed']}/{stats['total']})",
        f"Public repair gains: {stats['public_repaired_successes']}",
        _format_pass_line("Base Pass@1", stats.get('base_pass_at_1'), stats.get('base_passed'), stats['total']),
        _format_pass_line("Final Pass@1", stats.get('pass_at_1'), stats.get('final_passed'), stats['total']),
        f"Hidden repair gains: {stats['repaired_successes'] if stats.get('repaired_successes') is not None else 'N/A'}",
        f"Total time: {stats['time_cost']['total']:.2f}s",
        f"Generation time: {stats['time_cost']['generation']:.2f}s",
        f"Hidden evaluation time: {stats['time_cost'].get('hidden_evaluation', 0.0):.2f}s",
        f"Total tokens: {stats['token_usage']['total']}",
    ]
    (run_dir / "REPORT.txt").write_text("\n".join(lines), encoding="utf-8")


def _run_generation(
    dataset_name: str,
    model_name: str,
    api_key: Optional[str],
    output_dir: Optional[str],
    resume_from: Optional[str],
    limit: Optional[int],
    instance_ids: Optional[str],
    generation_temperature: float,
    repair_temperature: float,
    timeout: float,
    max_repair_attempts: int,
    num_base_samples: int,
    per_problem_parallelism: int,
    max_concurrent_llm_calls: int,
    llm_max_retries: int,
    llm_retry_base_delay: float,
    workers: int,
    mode: str,
) -> Path:
    _setup_model_env(model_name, api_key)
    dataset = get_data(dataset_name)
    dataset = _select_instances(dataset, limit, instance_ids)
    if not dataset:
        raise ValueError("No instances selected.")

    resume_run_dir: Optional[Path] = None
    existing_results_map: Dict[str, Dict[str, Any]] = {}
    previous_generation_time = 0.0
    if resume_from:
        resume_run_dir = _resolve_generation_resume_dir(resume_from)
        existing_results_map = _load_existing_generation_results(resume_run_dir)
        checkpoint_path = resume_run_dir / "generation_checkpoint.json"
        if checkpoint_path.exists():
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                previous_generation_time = float(checkpoint.get("generation_time", 0.0))
            except Exception as exc:
                print(f"[Resume] Failed to read existing checkpoint: {checkpoint_path} ({exc})")
        logger = DetailedLogger(resume_run_dir.parent, run_dir=resume_run_dir)
    else:
        logger = DetailedLogger(_resolve_output_dir(output_dir, dataset_name, model_name))

    selected_ids = {instance.instance_id for instance in dataset}
    existing_results_map = {
        instance_id: result
        for instance_id, result in existing_results_map.items()
        if instance_id in selected_ids
    }
    completed_ids = set(existing_results_map)
    pending_dataset = [instance for instance in dataset if instance.instance_id not in completed_ids]

    print(f"Output directory: {logger.run_dir}")
    print(f"Selected problems: {len(dataset)}")
    if resume_from:
        print(f"Resume source: {logger.run_dir}")
        print(f"Already completed: {len(completed_ids)}")
        print(f"Remaining problems: {len(pending_dataset)}")

    start = time.time()
    results: List[Dict[str, Any]] = list(existing_results_map.values())
    if pending_dataset:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_instance,
                    instance,
                    model_name,
                    generation_temperature,
                    repair_temperature,
                    timeout,
                    max_repair_attempts,
                    num_base_samples,
                    per_problem_parallelism,
                    max_concurrent_llm_calls,
                    llm_max_retries,
                    llm_retry_base_delay,
                ): instance.instance_id
                for instance in pending_dataset
            }
            for future in as_completed(futures):
                result = future.result()
                logger.save_problem(result)
                results.append(result)
                print(
                    f"Completed {result['instance_id']}: "
                    f"base_public={result['base_public_passed']} final_public={result['final_public_passed']}"
                )
    else:
        print("No remaining problems to run. Rebuilding checkpoint and summary from existing results.")

    order = {instance.instance_id: idx for idx, instance in enumerate(dataset)}
    results.sort(key=lambda item: order[item["instance_id"]])
    total_time = previous_generation_time + (time.time() - start)

    checkpoint = {
        "results": results,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "generation_temperature": generation_temperature,
        "repair_temperature": repair_temperature,
        "timeout": timeout,
        "max_repair_attempts": max_repair_attempts,
        "num_base_samples": num_base_samples,
        "per_problem_parallelism": per_problem_parallelism,
        "max_concurrent_llm_calls": max_concurrent_llm_calls,
        "llm_max_retries": llm_max_retries,
        "llm_retry_base_delay": llm_retry_base_delay,
        "workers": workers,
        "generation_time": total_time,
        "timestamp": datetime.now().isoformat(),
    }
    logger.save_json("generation_checkpoint.json", checkpoint)

    if mode in {"generate", "all"}:
        summary = _build_summary(
            results,
            model_name=model_name,
            dataset_name=dataset_name,
            generation_dataset_name=dataset_name,
            total_time=total_time,
            mode=mode,
        )
        logger.save_json("raw_summary.json", summary)
        logger.save_json("summary.json", summary)
        logger.save_json(
            "eval_checkpoint.json",
            {
                "dataset_name": dataset_name,
                "generation_dataset_name": dataset_name,
                "model_name": model_name,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            },
        )
        _write_report(logger.run_dir, summary)
    return logger.run_dir


def _resolve_resume_dir(resume_from: str) -> Path:
    candidate = Path(resume_from)
    if not candidate.exists():
        raise FileNotFoundError(f"Resume path does not exist: {resume_from}")
    if candidate.is_dir() and (candidate / "generation_checkpoint.json").exists():
        return candidate
    run_dirs = sorted(candidate.glob("run_*"), key=lambda item: item.name, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(
            f"No run_* directories with generation_checkpoint.json found under {resume_from}"
        )
    for run_dir in run_dirs:
        if (run_dir / "generation_checkpoint.json").exists():
            return run_dir
    raise FileNotFoundError(
        f"No generation_checkpoint.json found under {resume_from}"
    )


def _run_evaluate(
    resume_from: str,
    workers: int,
    eval_dataset: Optional[str],
    eval_output: Optional[str],
    model_name_override: Optional[str] = None,
    mode_label: str = "evaluate",
    dataset_name_override: Optional[str] = None,
) -> Path:
    candidate = Path(resume_from)
    if candidate.exists() and candidate.is_dir() and not (candidate / "generation_checkpoint.json").exists():
        run_dir = candidate
    else:
        run_dir = _resolve_resume_dir(resume_from)
    checkpoint_path = _ensure_generation_checkpoint(
        run_dir,
        dataset_name=dataset_name_override,
        model_name=model_name_override,
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    generation_results = checkpoint["results"]
    generation_dataset_name = checkpoint["dataset_name"]
    dataset_name = eval_dataset or generation_dataset_name
    model_name = model_name_override or checkpoint["model_name"]

    dataset = get_data(dataset_name)
    index = {instance.instance_id: instance for instance in dataset}
    filtered_results = [item for item in generation_results if item["instance_id"] in index]
    eval_instances = [index[item["instance_id"]] for item in filtered_results]
    base_codes = [item["base_code"] for item in filtered_results]

    eval_start = time.time()
    base_eval_results = eval_code(
        eval_instances,
        base_codes,
        timeout=checkpoint.get("timeout", 10.0),
        workers=workers,
        show_progress=True,
    )

    final_eval_subset_indices: List[int] = []
    final_eval_instances: List[InstanceData] = []
    final_eval_codes: List[str] = []
    for idx, item in enumerate(filtered_results):
        needs_final_eval = bool(item.get("stages")) and item.get("final_code") != item.get("base_code")
        if needs_final_eval:
            final_eval_subset_indices.append(idx)
            final_eval_instances.append(eval_instances[idx])
            final_eval_codes.append(item["final_code"])

    final_eval_subset_results: List[Tuple[float, List[EvalResult]]] = []
    if final_eval_instances:
        final_eval_subset_results = eval_code(
            final_eval_instances,
            final_eval_codes,
            timeout=checkpoint.get("timeout", 10.0),
            workers=workers,
            show_progress=True,
        )
    hidden_eval_time = time.time() - eval_start

    final_eval_map = {
        idx: result for idx, result in zip(final_eval_subset_indices, final_eval_subset_results)
    }

    merged_results: List[Dict[str, Any]] = []
    for idx, (base_item, (base_acc, base_eval_list)) in enumerate(
        zip(filtered_results, base_eval_results)
    ):
        item = dict(base_item)
        item["base_accuracy"] = base_acc
        item["base_passed"] = base_acc == 1.0
        item["base_eval_results"] = _serialize_eval_results(base_eval_list)

        if idx in final_eval_map:
            final_acc, final_eval_list = final_eval_map[idx]
        else:
            final_acc, final_eval_list = base_acc, base_eval_list

        item["eval_accuracy"] = final_acc
        item["final_accuracy"] = final_acc
        item["final_passed"] = final_acc == 1.0
        item["final_eval_results"] = _serialize_eval_results(final_eval_list)
        item["eval_results"] = item["final_eval_results"]
        merged_results.append(item)

    if eval_output:
        output_root = Path(eval_output)
        logger = DetailedLogger(output_root)
        logger.run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.run_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger = DetailedLogger(run_dir.parent)
        logger.run_dir = run_dir
    for item in merged_results:
        logger.save_problem(item)

    generation_total_time = float(checkpoint.get("generation_time", 0.0))
    summary = _build_summary(
        merged_results,
        model_name=model_name,
        dataset_name=dataset_name,
        generation_dataset_name=generation_dataset_name,
        total_time=generation_total_time + hidden_eval_time,
        mode=mode_label,
        hidden_eval_time=hidden_eval_time,
    )
    logger.save_json("summary.json", summary)
    logger.save_json(
        "eval_checkpoint.json",
        {
            "dataset_name": dataset_name,
            "generation_dataset_name": generation_dataset_name,
            "model_name": model_name,
            "results": merged_results,
            "generation_time": generation_total_time,
            "hidden_eval_time": hidden_eval_time,
            "base_eval_count": len(base_eval_results),
            "final_eval_count": len(final_eval_subset_results),
            "timestamp": datetime.now().isoformat(),
        },
    )
    _write_report(logger.run_dir, summary)
    return logger.run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-repair baseline runner")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument(
        "--dataset",
        required=False,
        default="xCodeEval",
        choices=DATASET_OPTIONS,
    )
    parser.add_argument(
        "--mode",
        required=False,
        default="all",
        choices=["all", "generate", "evaluate"],
    )
    parser.add_argument("--api-key", required=False, default=None, type=str)
    parser.add_argument("--output-dir", required=False, default=None, type=str)
    parser.add_argument("--eval-output", required=False, default=None, type=str)
    parser.add_argument("--resume-from", required=False, default=None, type=str)
    parser.add_argument("--eval-dataset", required=False, default=None, type=str)
    parser.add_argument("--limit", required=False, default=None, type=int)
    parser.add_argument("--instance-ids", required=False, default=None, type=str)
    parser.add_argument("--workers", required=False, default=4, type=int)
    parser.add_argument("--num-base-samples", required=False, default=1, type=int)
    parser.add_argument("--per-problem-parallelism", required=False, default=1, type=int)
    parser.add_argument("--max-concurrent-llm-calls", required=False, default=16, type=int)
    parser.add_argument("--llm-max-retries", required=False, default=6, type=int)
    parser.add_argument("--llm-retry-base-delay", required=False, default=2.0, type=float)
    parser.add_argument(
        "--generation-temperature", required=False, default=0.8, type=float
    )
    parser.add_argument(
        "--repair-temperature", required=False, default=0.8, type=float
    )
    parser.add_argument("--timeout", required=False, default=10.0, type=float)
    parser.add_argument(
        "--max-repair-attempts", required=False, default=1, type=int
    )
    args = parser.parse_args()

    _setup_model_env(args.model, args.api_key)

    if args.mode in {"all", "generate"}:
        run_dir = _run_generation(
            dataset_name=args.dataset,
            model_name=args.model,
            api_key=args.api_key,
            output_dir=args.output_dir,
            resume_from=args.resume_from,
            limit=args.limit,
            instance_ids=args.instance_ids,
            generation_temperature=args.generation_temperature,
            repair_temperature=args.repair_temperature,
            timeout=args.timeout,
            max_repair_attempts=args.max_repair_attempts,
            num_base_samples=args.num_base_samples,
            per_problem_parallelism=args.per_problem_parallelism,
            max_concurrent_llm_calls=args.max_concurrent_llm_calls,
            llm_max_retries=args.llm_max_retries,
            llm_retry_base_delay=args.llm_retry_base_delay,
            workers=args.workers,
            mode=args.mode,
        )
        if args.mode == "generate":
            print(f"Generation completed: {run_dir}")
            return

        eval_run_dir = _run_evaluate(
            resume_from=str(run_dir),
            workers=args.workers,
            eval_dataset=args.eval_dataset,
            eval_output=args.eval_output,
            model_name_override=args.model,
            mode_label="all",
            dataset_name_override=args.dataset,
        )
        print(f"All completed: {eval_run_dir}")
        return

    if not args.resume_from:
        raise ValueError("--mode evaluate requires --resume-from")

    eval_run_dir = _run_evaluate(
        resume_from=args.resume_from,
        workers=args.workers,
        eval_dataset=args.eval_dataset,
        eval_output=args.eval_output,
        model_name_override=args.model,
        dataset_name_override=args.dataset,
    )
    print(f"Evaluation completed: {eval_run_dir}")


if __name__ == "__main__":
    main()











