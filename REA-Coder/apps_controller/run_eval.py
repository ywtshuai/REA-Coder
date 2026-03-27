"""
apps_controller/run_eval.py

CLI entrypoint for evaluating APPS jsonl with the multi-agent method.

Example:
  python -m apps_controller.run_eval --data apps --max_instances 50 --max_iters 2

Assumes dataset lives at ./Datasets/{name}.jsonl unless --data is a .jsonl path.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import errno
import json
import resource
import time
from pathlib import Path
from typing import List, Dict, Any

from apps_controller.apps_data import load_dataset_jsonl
from apps_controller.code_generation_api import AppsCodeGenAPI
from apps_controller.config_apps import DEFAULT_TASK_PROMPT
from apps_eval.parallel_runner import init_pool


def _now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="apps", help="dataset name (in ./Datasets) or .jsonl path")
    ap.add_argument("--output_dir", type=str, default="", help="base output dir")
    ap.add_argument("--max_instances", type=int, default=0, help="0 means all")
    ap.add_argument("--start", type=int, default=0, help="start index in dataset")
    ap.add_argument("--end", type=int, default=0, help="end index (exclusive), 0 means until end")
    ap.add_argument("--max_iters", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=10.)
    ap.add_argument("--eval_workers", type=int, default=16)
    ap.add_argument("--model", type=str, default="deepseekv3.2")
    ap.add_argument("--show_public_tests", type=lambda x: (str(x).lower() == 'true'),
                    default=True, help="show public test results during eval")
    ap.add_argument("--gen_k", type=int, default=50, help="number of generated test cases per iteration")
    ap.add_argument("--confident_k", type=int, default=20)
    args = ap.parse_args()

    print(args)

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    #resource.setrlimit(resource.RLIMIT_NOFILE, (args.eval_workers ** 2, hard))

    python_executor_pool = init_pool(args.eval_workers)

    problems = load_dataset_jsonl(args.data)
    # print(555)
    # nowlist = [6]
    # problemstmp = []
    # for k in range(0,len(nowlist)):
    #     problemstmp.append(problems[nowlist[k]])
    # # problems = problems[args.start:(args.end or None)]
    # problems = problemstmp
    # # if args.max_instances and args.max_instances > 0:
    # #     problems = problems[:args.max_instances]
    # # problems = problems[args.start:(args.end or None)]
    # print(999)
    if args.max_instances and args.max_instances > 0:
        problems = problems[:args.max_instances]
    # print(666)
    # nowlist = [3, 4, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 20, 26, 28, 29, 30, 33, 34, 37, 38, 43, 44, 46, 47, 48, 50, 51, 54, 55, 56, 61, 62, 70, 76, 77, 78, 79, 82, 84, 87, 88, 89, 93, 94, 95, 97, 99, 100, 101, 103, 108, 112, 115, 117, 118, 119, 120, 122, 127, 129, 132, 134, 135, 136, 138, 141, 142, 148, 149, 154, 155, 156, 160, 163, 35, 36, 59, 60, 98, 133, 137, 139, 140, 143, 146]    
    # exclude_ids = set(str(x) for x in nowlist)
    # problems = [p for p in problems if str(p.problem_id) not in exclude_ids]
    # print(len(problems))
    problems = problems[args.start:(args.end or None)]
    print(len(problems))
    out_dir = args.output_dir or str(Path("./Results") / args.data / args.model /_now_tag())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(777)
    api = AppsCodeGenAPI(
        task_prompt=DEFAULT_TASK_PROMPT,
        iter_state_dir=Path(out_dir) / "iter_states",
        max_iters=args.max_iters,
        timeout=args.timeout,
        eval_workers=args.eval_workers,
        show_public_tests=args.show_public_tests,
        dynamic_stop_gen_k=args.gen_k,
        dynamic_stop_confident_k=args.confident_k
    )

    results: List[Dict[str, Any]] = []
    passed = 0

    def _process_problem(p):
        try:
            while True:
                try:
                    prob_dir = Path(out_dir) / "final" / str(p.problem_id)
                    prob_dir.mkdir(parents=True, exist_ok=True)
                    if (prob_dir / "eval_summary.json").exists():
                        print(f"Skipping problem {p.problem_id} as it has already been processed.")
                        summary = json.loads((prob_dir / "eval_summary.json").read_text(encoding="utf-8"))
                        ok = bool(summary.get("passed"))
                    else:
                        code, summary, state = api.run_one(p, output_dir=out_dir, max_iters=args.max_iters)
                        ok = bool(summary.get("passed"))
                        (prob_dir / "solution.py").write_text(code or "", encoding="utf-8")
                        (prob_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                                    encoding="utf-8")
                        ats = summary.get("all_test_cases_summary")
                        if isinstance(ats, dict):
                            (prob_dir / "eval_all_test_summary.json").write_text(
                                json.dumps(ats, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                    return {
                        "problem_id": p.problem_id,
                        "passed": ok,
                        "eval_summary": summary,
                        "state": state,
                    }
                except OSError as e:
                    if e.errno == errno.EMFILE:
                        time.sleep(60)
                    else:
                        raise
        except Exception as e:
            return {
                "problem_id": p.problem_id,
                "passed": False,
                "error": repr(e),
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.eval_workers) as executor:
        futures = [executor.submit(_process_problem, p) for p in problems]
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            if result.get("passed"):
                passed += 1
            results.append(result)

    python_executor_pool.close()
    python_executor_pool.join()

    summary = {
        "data": args.data,
        "num_instances": len(problems),
        "passed": passed,
        "pass_rate": (passed / len(problems)) if problems else 0.0,
        "output_dir": out_dir,
        "max_iters": args.max_iters,
        "timeout": args.timeout,
        "eval_workers": args.eval_workers,
    }
    (Path(out_dir) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (Path(out_dir) / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(_now_tag())

if __name__ == "__main__":
    main()