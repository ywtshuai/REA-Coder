import ast
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


# 数据集目录：优先使用环境变量，否则使用项目内 Datasets 路径
DATASETS_DIR = os.environ.get(
    "DATASETS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Datasets")
)
if not os.path.isdir(DATASETS_DIR):
    DATASETS_DIR = os.path.join(os.getcwd(), "Datasets")


@dataclass
class InstanceData:
    instance_id: str
    problem_statement: str
    starter_code: str
    test_cases: Dict[str, List[Any]]
    solutions: List[str]
    public_test_cases: Optional[Dict[str, List[Any]]] = None  # 用于 prompt/feedback，避免数据泄漏


def _extract_solution_method_name(starter_code: str) -> Optional[str]:
    """Extract the first public method from a LeetCode-style Solution class."""
    if not starter_code:
        return None
    try:
        tree = ast.parse(starter_code)
    except SyntaxError:
        return None

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'Solution':
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                    return item.name
    return None


def _extract_solution_argument_names(starter_code: str) -> List[str]:
    """Extract method argument names (excluding self) from a LeetCode-style Solution class."""
    if not starter_code:
        return []
    try:
        tree = ast.parse(starter_code)
    except SyntaxError:
        return []

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'Solution':
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                    return [arg.arg for arg in item.args.args if arg.arg != 'self']
    return []


def _serialize_leetcode_stdio_input(raw_input: Any) -> str:
    """Serialize LeetCode call-style arguments into a custom stdio format: one JSON value per line."""
    if isinstance(raw_input, (list, tuple)):
        return '\n'.join(json.dumps(arg, ensure_ascii=False) for arg in raw_input)
    return json.dumps(raw_input, ensure_ascii=False)


def _serialize_leetcode_stdio_output(raw_output: Any) -> str:
    """Serialize expected outputs into JSON text for stable stdio comparison."""
    if isinstance(raw_output, str):
        return raw_output
    return json.dumps(raw_output, ensure_ascii=False)


def _normalize_public_test_cases(public_tc: Any) -> Optional[Dict[str, List[Any]]]:
    """
    将 public_test_cases 规范为 {inputs: [], outputs: []}。
    支持两种格式: 1) {inputs: [], outputs: []}; 2) [{input, output}, ...] (如 muFiX meta)
    """
    if not public_tc:
        return None
    if isinstance(public_tc, dict) and public_tc.get('inputs') and public_tc.get('outputs'):
        return public_tc
    if isinstance(public_tc, list) and len(public_tc) > 0:
        inputs, outputs = [], []
        for tc in public_tc:
            inp = tc.get('input', tc.get('inputs', ''))
            out = tc.get('output', tc.get('outputs', ''))
            if inp is not None and out is not None:
                inputs.append(inp if isinstance(inp, str) else str(inp))
                outputs.append(out if isinstance(out, str) else str(out))
        return {'inputs': inputs, 'outputs': outputs} if inputs and outputs else None
    return None


def load_data(data_name):
    """apps 与 apps_eval 共用 apps.jsonl；code_contests 与 code_contests_raw 共用 code_contests.jsonl"""
    if data_name in ('apps', 'apps_eval'):
        file_name = 'apps.jsonl'
    elif data_name in ('code_contests', 'code_contests_raw'):
        file_name = 'code_contests.jsonl'
    else:
        file_name = f"{data_name}.jsonl"
    test_data = []
    file_path = os.path.join(DATASETS_DIR, file_name)
    for temp in open(file_path, 'r', encoding='utf-8').readlines():
        test_data.append(json.loads(temp))
    return test_data


def get_data(data_name: str) -> List[InstanceData]:
    data_list = []
    raw_data = load_data(data_name)
    for data in raw_data:
        if data_name in ('apps', 'apps_eval'):
            public_test_cases = _normalize_public_test_cases(data.get('public_test_cases'))
            # apps 使用 all_test_cases，apps_eval 使用 all_test_cases_et
            tc_key = 'all_test_cases_et' if data_name == 'apps_eval' else 'all_test_cases'
            test_cases = data.get(tc_key)
            if not test_cases or not test_cases.get('inputs') or not test_cases.get('outputs'):
                raise ValueError(f"Missing or invalid {tc_key} for problem_id {data.get('problem_id')}")
            solutions_raw = data.get('solutions')
            if solutions_raw is None:
                solutions = []
            elif isinstance(solutions_raw, list):
                solutions = solutions_raw
            else:
                solutions = ast.literal_eval(solutions_raw) if solutions_raw else []
            instance = InstanceData(
                instance_id=str(data['problem_id']),
                problem_statement=data['question'],
                starter_code=data.get('starter_code', ''),
                test_cases=test_cases,
                solutions=solutions,
                public_test_cases=public_test_cases
            )
        elif data_name in ('code_contests', 'code_contests_raw'):
            # code_contests 使用 all_test_cases，code_contests_raw 优先 all_test_cases_raw，缺失时回退 all_test_cases
            tc_key = 'all_test_cases_raw' if data_name == 'code_contests_raw' else 'all_test_cases'
            test_cases = data.get(tc_key)
            if not test_cases or not test_cases.get('inputs') or not test_cases.get('outputs'):
                if data_name == 'code_contests_raw' and data.get('all_test_cases'):
                    test_cases = data['all_test_cases']
                if not test_cases or not test_cases.get('inputs') or not test_cases.get('outputs'):
                    raise ValueError(f"Missing or invalid {tc_key} for name {data.get('name')}")
            filtered_solutions = {'language': [], 'solution': []}
            solutions_data = data.get('solutions', {'language': [], 'solution': []})
            for language, solution in zip(solutions_data.get('language', []), solutions_data.get('solution', [])):
                if language == 3:  # 3 表示 Python
                    filtered_solutions['language'].append(language)
                    filtered_solutions['solution'].append(solution)
            public_test_cases = _normalize_public_test_cases(data.get('public_test_cases'))
            instance = InstanceData(
                instance_id=data['name'],
                problem_statement=data['description'],
                starter_code='',
                test_cases=test_cases,
                solutions=filtered_solutions['solution'],
                public_test_cases=public_test_cases
            )
        elif data_name == 'xCodeEval':
            # xCodeEval: description + input_spec + output_spec + notes
            parts = [data['description']]
            if data.get('input_spec'):
                parts.append(f"\nInput format:\n{data['input_spec']}")
            if data.get('output_spec'):
                parts.append(f"\nOutput format:\n{data['output_spec']}")
            if data.get('notes'):
                parts.append(f"\nNote:\n{data['notes']}")
            problem_statement = ''.join(parts)
            public_test_cases = _normalize_public_test_cases(data.get('public_test_cases'))

            # xCodeEval outputs 格式混合：既有 ["50"] 列表也有 "50" 字符串，executor 仅对 str 做规范化
            # 统一转为字符串，便于 executor 正确比较
            def _normalize_xcodeeval_output(out):
                if isinstance(out, list) and len(out) > 0:
                    return str(out[0]) if out[0] is not None else ""
                return str(out) if out is not None else ""

            all_tc = data['all_test_cases']
            normalized_outputs = [_normalize_xcodeeval_output(o) for o in all_tc.get('outputs', [])]
            test_cases = {'inputs': all_tc.get('inputs', []), 'outputs': normalized_outputs}

            if public_test_cases:
                public_test_cases = {
                    'inputs': public_test_cases.get('inputs', []),
                    'outputs': [_normalize_xcodeeval_output(o) for o in public_test_cases.get('outputs', [])]
                }

            instance = InstanceData(
                instance_id=str(data['problem_id']),
                problem_statement=problem_statement,
                starter_code='',
                test_cases=test_cases,
                solutions=[],  # xCodeEval 无 solutions 字段
                public_test_cases=public_test_cases
            )
        elif data_name == 'livecodebench':
            public_test_cases = _normalize_public_test_cases(data.get('public_test_cases'))
            test_cases = data.get('all_test_cases')
            if not test_cases or not test_cases.get('inputs') or not test_cases.get('outputs'):
                raise ValueError(f"Missing or invalid all_test_cases for problem_id {data.get('problem_id')}")
            starter_code = data.get('starter_code', '')
            problem_statement = data['question']
            if data.get('platform') == 'leetcode' and starter_code:
                method_name = _extract_solution_method_name(starter_code)
                arg_names = _extract_solution_argument_names(starter_code)
                serialized_inputs = [
                    _serialize_leetcode_stdio_input(item)
                    for item in test_cases.get('inputs', [])
                ]
                serialized_outputs = [
                    _serialize_leetcode_stdio_output(item)
                    for item in test_cases.get('outputs', [])
                ]
                test_cases = {
                    'inputs': serialized_inputs,
                    'outputs': serialized_outputs,
                }
                if public_test_cases:
                    public_test_cases = {
                        'inputs': [
                            _serialize_leetcode_stdio_input(item)
                            for item in public_test_cases.get('inputs', [])
                        ],
                        'outputs': [
                            _serialize_leetcode_stdio_output(item)
                            for item in public_test_cases.get('outputs', [])
                        ]
                    }
                arg_desc = '\n'.join(
                    f"- Line {idx + 1}: JSON for `{name}`"
                    for idx, name in enumerate(arg_names)
                ) if arg_names else "- Each line is one JSON-serialized argument in method order."
                problem_statement = (
                    f"{problem_statement}\n\n"
                    "Adapted StdIO Format for this benchmark:\n"
                    "Read method arguments from stdin, one JSON value per line, in the same order as the original method signature.\n"
                    f"{arg_desc}\n"
                    "Print the final return value as JSON to stdout.\n\n"
                    "Original Starter Code Reference:\n"
                    f"{starter_code}\n\n"
                    "Your task in this adapted version:\n"
                    f"- Implement a standard-input solution equivalent to the original method"
                    f"{f' `class Solution.{method_name}`' if method_name else ''}.\n"
                    "- Parse each input line as JSON.\n"
                    "- Compute the same result as the original method.\n"
                    "- Output exactly the returned value in JSON-compatible text."
                )
            instance = InstanceData(
                instance_id=str(data['problem_id']),
                problem_statement=problem_statement,
                starter_code=starter_code,
                test_cases=test_cases,
                solutions=[],
                public_test_cases=public_test_cases
            )
        else:
            raise ValueError(f"Unsupported data_name: {data_name}")
        data_list.append(instance)
    return data_list
