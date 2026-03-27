"""
apps_controller/apps_evaluator.py

Evaluation of generated Python code on APPS-style testcases using apps_eval.
"""
from __future__ import annotations

from time import time
from typing import Dict, List, Any

from apps_controller.types import EvalCaseResult, EvalResult
from apps_eval.data import InstanceData

# External evaluator provided by teammate (apps_eval)
from apps_eval.parallel_runner import eval_code


def evaluate_python(
        code: str,
        test_cases: Dict[str, List[str]],
        timeout: float = 2.0,
        workers: int = 16,
) -> EvalResult:
    inputs = test_cases.get("inputs") or []
    outputs = test_cases.get("outputs") or []
    n = min(len(inputs), len(outputs))

    tasks = [
        InstanceData(
            instance_id="eval_instance",
            problem_statement="",
            starter_code="",
            test_cases=test_cases,
            solutions=[]
        )
    ]
    solutions = [code]

    time_start = time()
    _, raw_results = eval_code(tasks, solutions, timeout=timeout, workers=workers)[0]
    time_end = time()
    cases: List[EvalCaseResult] = []
    passed = True
    counts: Dict[str, int] = {}
    total_time = time_end-time_start

    for i, r in enumerate(raw_results):
        # r is apps_eval.executor.EvalResult
        status = r.status
        stdout = r.stdout
        stderr = r.stderr
        time_cost = r.time_cost
        expected = r.expected
        counts[status] = counts.get(status, 0) + 1
        if status != "AC":
            passed = False
        cases.append(EvalCaseResult(
            status=status,
            stdout=stdout,
            stderr=stderr,
            time_cost=time_cost,
            expected=expected,
            input_data=inputs[i] if i < len(inputs) else None,
        ))

    summary: Dict[str, Any] = {
        "num_cases": len(cases),
        "passed": passed,
        "status_counts": counts,
        "total_time": total_time,
        "timeout": timeout,
    }

    return EvalResult(passed=passed, cases=cases, summary=summary)