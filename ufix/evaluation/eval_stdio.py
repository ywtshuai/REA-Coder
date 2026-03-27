"""
评估 muFiX stdio 模式生成的代码。
从 run 目录读取 generation_checkpoint.json，使用 apps_eval 进行评估。
支持 code_contests、apps、apps_eval、xCodeEval 数据集。

用法:
  python eval_stdio.py --run-dir mufix_outputs_code_contests_qwen/run_20260221_123456
  python eval_stdio.py --output-dir mufix_outputs_xCodeEval_qwen  # 使用最新 run
  # 跨数据集评估：apps 生成，apps_eval 评估，结果写入单独目录（不覆盖原 run）
  python eval_stdio.py --run-dir mufix_outputs_apps_qwen/run_xxx --eval-dataset apps_eval --eval-output mufix_outputs_apps_eval_qwen
"""

import os
import json
import argparse
import platform
from pathlib import Path
from datetime import datetime

sys_path = Path(__file__).resolve().parent
if str(sys_path) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(sys_path))

from apps_eval.data import get_data
from apps_eval.parallel_runner import eval_code


def _extract_index(instance_id: str) -> int:
    """从 instance_id 提取题目索引，如 apps_0 -> 0, apps_eval_123 -> 123"""
    import re
    m = re.search(r'_(\d+)$', str(instance_id))
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None, help="run 目录路径（如 mufix_outputs_code_contests_qwen/run_20260221_123456）")
    parser.add_argument("--output-dir", default=None, help="若指定，则从该目录下找最新的 run_*（与 --run-dir 二选一）")
    parser.add_argument("--eval-dataset", default=None,
                        help="覆盖评估数据集。例如：apps checkpoint 用 apps_eval 评估（更多测试用例）")
    parser.add_argument("--eval-output", default=None,
                        help="评估结果输出目录。若指定，会在该目录下创建 run_{timestamp} 子目录存放结果，不覆盖原 run 目录")
    parser.add_argument("--datasets_dir", default=None, help="Datasets 目录（默认自动检测）")
    parser.add_argument("--workers", default=None, type=int,
                        help="评估并行进程数（默认: Windows=16, Linux/Mac=60）")
    args = parser.parse_args()

    # 平台感知的 workers 默认值
    if args.workers is None:
        args.workers = 16 if platform.system() == 'Windows' else 60
    
    if not args.run_dir and not args.output_dir:
        print("错误: 请指定 --run-dir 或 --output-dir")
        return 1
    
    run_dir = Path(args.run_dir) if args.run_dir else None
    if args.output_dir:
        run_dir = Path(args.output_dir)
        runs = sorted(run_dir.glob('run_*'), key=lambda x: x.name, reverse=True)
        if not runs:
            print(f"错误: 在 {run_dir} 中未找到 run_* 目录")
            return 1
        run_dir = runs[0]
        print(f"使用最新运行: {run_dir}")
    
    if not run_dir.exists():
        print(f"错误: 目录不存在: {run_dir}")
        return 1
    
    ckpt_path = run_dir / "generation_checkpoint.json"
    if not ckpt_path.exists():
        print(f"错误: 找不到 {ckpt_path}")
        return 1
    
    with open(ckpt_path, 'r', encoding='utf-8') as f:
        ckpt = json.load(f)
    
    all_results = ckpt['results']
    dataset_name = ckpt.get('dataset_name', '')
    model_name = ckpt.get('model_name', '')
    generation_time = ckpt.get('generation_time', 0.0)
    token_usage = ckpt.get('token_usage', {})
    total_tokens = token_usage.get('total', 0) if isinstance(token_usage, dict) else (token_usage or 0)

    # 支持用 apps checkpoint 在 apps_eval 上评估（题目相同，apps_eval 有更多测试用例）
    eval_dataset_name = args.eval_dataset or dataset_name
    if args.eval_dataset:
        print(f"评估数据集覆盖: {dataset_name} -> {eval_dataset_name}")

    if args.datasets_dir is None:
        base = Path(__file__).resolve().parent.parent.parent
        args.datasets_dir = str(base / "Self-collaboration-Code-Generation-main" / "Datasets")

    dataset = get_data(eval_dataset_name, args.datasets_dir)
    result_by_id = {r["instance_id"]: r["code"] for r in all_results}

    # 若 checkpoint 与 eval 数据集不同（如 apps vs apps_eval），按题目索引匹配
    match_by_index = False
    if eval_dataset_name != dataset_name and eval_dataset_name in ('apps_eval', 'apps') and dataset_name in ('apps_eval', 'apps'):
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
    
    # 决定输出目录：--eval-output 指定时在该目录下创建 run_{timestamp}，否则写入源 run_dir
    if args.eval_output:
        out_base = Path(args.eval_output)
        out_base.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_run_dir = out_base / f"run_{ts}"
        out_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"评估结果将保存到: {out_run_dir}")
    else:
        out_run_dir = run_dir

    print(f"评估 {len(eval_dataset)} 道题目 (模型: {model_name}, 评估数据集: {eval_dataset_name}, workers={args.workers})...")
    eval_results = eval_code(eval_dataset, eval_solutions, timeout=10.0, workers=args.workers, show_progress=True)
    
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
            "dataset": dataset_name,
            "eval_dataset": eval_dataset_name,
            "model": model_name,
            "pass_at_1": pass_at_1,
            "passed": pass_count,
            "total": total,
            "avg_pass_ratio": avg_pass_ratio,
            "generation_time": generation_time,
            "token_usage": {
                "total": total_tokens,
                "average_per_problem": total_tokens / total if (total > 0 and total_tokens > 0) else 0,
            },
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
        f.write(f"源 run 目录: {run_dir}\n")
        f.write(f"生成数据集: {dataset_name.upper()}\n")
        if eval_dataset_name != dataset_name:
            f.write(f"评估数据集: {eval_dataset_name.upper()} (更多测试用例)\n")
        f.write(f"模型: {model_name}\n")
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


if __name__ == "__main__":
    exit(main())
