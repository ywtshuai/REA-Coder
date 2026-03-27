"""
将 CodeContest、apps、apps_eval、xCodeEval 数据集转换为 muFiX stdio 格式的 prompt 文件。

用法:
  python prepare_stdio_data.py --dataset code_contests --datasets_dir ../Self-collaboration-Code-Generation-main/Datasets
  python prepare_stdio_data.py --dataset code_contests_raw
  python prepare_stdio_data.py --dataset apps
  python prepare_stdio_data.py --dataset apps_eval
  python prepare_stdio_data.py --dataset xCodeEval
  python prepare_stdio_data.py --dataset livecodebench
"""

import os
import json
import argparse
from pathlib import Path


def _format_input_output(inp, out) -> str:
    """将 input/output 格式化为字符串"""
    if isinstance(inp, list):
        inp = '\n'.join(str(x) for x in inp)
    elif not isinstance(inp, str):
        inp = str(inp)
    if isinstance(out, list):
        out = '\n'.join(str(x) for x in out)
    elif not isinstance(out, str):
        out = str(out)
    return inp, out


def build_prompt_content(problem_statement: str, test_cases: dict) -> str:
    """
    构建 prompt 内容，包含问题描述和测试用例（stdio 格式）。
    用于 thought_eliciting_phase 的 LLM 分析。
    """
    inputs = test_cases.get("inputs", [])
    outputs = test_cases.get("outputs", [])
    if not inputs or not outputs:
        return problem_statement.strip()
    
    lines = [problem_statement.strip(), "", "# Test Cases (stdin/stdout format):"]
    for i, (inp, out) in enumerate(zip(inputs, outputs), 1):
        inp_str, out_str = _format_input_output(inp, out)
        lines.append(f"")
        lines.append(f"({i}) Input:")
        lines.append(inp_str)
        lines.append("Expected output:")
        lines.append(out_str)
    return "\n".join(lines)


def _sanitize_filename(name: str) -> str:
    """移除 Windows 文件名中的非法字符"""
    for c in '?*:"<>|/\\':
        name = name.replace(c, '_')
    return name.strip('._ ')


def prepare_dataset(dataset_name: str, datasets_dir: str, output_dir: str):
    """
    准备单个数据集。
    
    Args:
        dataset_name: code_contests | code_contests_raw | apps | apps_eval | xCodeEval | livecodebench
        datasets_dir: Self-collaboration Datasets 目录路径
        output_dir: 输出目录，将创建 {output_dir}/{dataset_name}/
    """
    from apps_eval.data import get_data
    
    datasets_dir = os.path.normpath(datasets_dir)
    out_base = os.path.join(output_dir, dataset_name)
    os.makedirs(out_base, exist_ok=True)
    
    data_list = get_data(dataset_name, datasets_dir)
    print(f"[{dataset_name}] 加载 {len(data_list)} 条数据")
    
    for inst in data_list:
        task_id = _sanitize_filename(inst.instance_id)
        tc = inst.test_cases

        # prompt 用例选取策略：
        # - code_contests / code_contests_raw: 用 public_test_cases，若无则用全部
        # - apps / apps_eval: 用 public_test_cases，若无则用 all_test_cases 的前3个
        # - xCodeEval: 用全部用例
        if dataset_name in ('code_contests', 'code_contests_raw'):
            prompt_tc = inst.public_test_cases if inst.public_test_cases else tc
        elif dataset_name in ('apps', 'apps_eval', 'livecodebench'):
            if inst.public_test_cases:
                prompt_tc = inst.public_test_cases
            else:
                # 无 public_test_cases：取前3个作为 prompt 用例
                inputs = tc.get('inputs', [])[:3]
                outputs = tc.get('outputs', [])[:3]
                prompt_tc = {'inputs': inputs, 'outputs': outputs}
        else:
            prompt_tc = tc

        if "inputs" not in prompt_tc or "outputs" not in prompt_tc:
            print(f"  跳过 {task_id}: 无 inputs/outputs")
            continue

        # 构建 prompt 内容（代码生成阶段可见的测试用例）
        prompt_content = build_prompt_content(inst.problem_statement, prompt_tc)
        
        # 保存 -prompt 和 -prompt-1 (mufix 需要)
        prompt_path = os.path.join(out_base, f"{task_id}-prompt")
        prompt1_path = os.path.join(out_base, f"{task_id}-prompt-1")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt_content)
        with open(prompt1_path, "w", encoding="utf-8") as f:
            f.write(prompt_content)
        
        # 保存 meta.json 供 executor 使用
        test_cases_list = []
        for inp, out in zip(tc["inputs"], tc["outputs"]):
            inp_str, out_str = _format_input_output(inp, out)
            test_cases_list.append({"input": inp_str, "output": out_str})
        
        meta = {"mode": "stdio", "test_cases": test_cases_list, "instance_id": inst.instance_id}
        # CodeContests: 增加 public_test_cases，供 feedback 阶段使用（避免 private 泄漏）
        if inst.public_test_cases:
            public_list = []
            for inp, out in zip(inst.public_test_cases["inputs"], inst.public_test_cases["outputs"]):
                inp_str, out_str = _format_input_output(inp, out)
                public_list.append({"input": inp_str, "output": out_str})
            meta["public_test_cases"] = public_list
        meta_path = os.path.join(out_base, f"{task_id}-meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    
    # 保存 task_id 列表（与文件名一致）
    task_ids = [_sanitize_filename(inst.instance_id) for inst in data_list
                if inst.test_cases and "inputs" in inst.test_cases and "outputs" in inst.test_cases]
    list_path = os.path.join(out_base, "task_ids.json")
    with open(list_path, "w", encoding="utf-8") as f:
        json.dump(task_ids, f, indent=2)
    
    print(f"[{dataset_name}] 已写入 {out_base}, 共 {len(task_ids)} 个任务")
    return task_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["code_contests", "code_contests_raw", "apps", "apps_eval", "xCodeEval", "livecodebench"])
    parser.add_argument("--datasets_dir", default=None, 
                        help="Self-collaboration Datasets 目录，默认自动检测")
    parser.add_argument("--output_dir", default="./dataset/prompts",
                        help="输出目录")
    args = parser.parse_args()
    
    if args.datasets_dir is None:
        # 自动检测：muFiX-main 与 Self-collaboration-Code-Generation-main 同级
        # evaluation/ -> muFiX-main/ -> Self-collaboration-Code-Generation/
        base = Path(__file__).resolve().parent
        datasets_dir = base.parent.parent / "Self-collaboration-Code-Generation-main" / "Datasets"
        args.datasets_dir = str(datasets_dir)
    
    if not os.path.isdir(args.datasets_dir):
        print(f"错误: 数据集目录不存在: {args.datasets_dir}")
        print("请先运行 Self-collaboration-Code-Generation-main 的 prepare_apps_data.py / prepare_apps_eval_data.py")
        return 1
    
    prepare_dataset(args.dataset, args.datasets_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
