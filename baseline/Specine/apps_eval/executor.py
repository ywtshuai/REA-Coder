import ast
import os
import pickle
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Literal, Any, List

from apps_eval.checker import static_security_check


# =========================
# 评估结果结构
# =========================

@dataclass
class EvalResult:
    status: str = 'RUNNING'  # AC / WA / RE / TLE / FORBIDDEN / RUNNING
    stdin: Any = None
    stdout: Any = None
    stderr: str | bytes = ""
    time_cost: float = 10.0
    expected = None

    def __repr__(self):
        return f"EvalResult(status={self.status}, time_cost={self.time_cost:.2f}s, stdin={repr(self.stdin)}, stdout={repr(self.stdout)}, expected={repr(self.expected)}, stderr={repr(self.stderr)})"


# =========================
# 清洗自动调用
# =========================


def remove_main_block(code: str) -> str:
    tree = ast.parse(code)

    new_body = []
    for node in tree.body:
        # 判断是否是 if __name__ == "__main__"
        if isinstance(node, ast.If):
            cond = node.test
            if (
                    isinstance(cond, ast.Compare)
                    and isinstance(cond.left, ast.Name)
                    and cond.left.id == "__name__"
                    and len(cond.ops) == 1
                    and isinstance(cond.ops[0], ast.Eq)
                    and len(cond.comparators) == 1
                    and isinstance(cond.comparators[0], ast.Constant)
                    and cond.comparators[0].value == "__main__"
            ):
                continue  # 跳过这个 if 块（即删除）

        new_body.append(node)

    tree.body = new_body
    return ast.unparse(tree)


# =========================
# 子进程执行
# =========================

def _run_subprocess(
        code: str,
        input_data,
        timeout: float,
        mode: Literal['stdio', 'call'] = "stdio",
        entry_func: Optional[str] = None,
) -> EvalResult:
    check = static_security_check(code)
    if not check.ok:
        return EvalResult(
            status="FORBIDDEN",
            stdout="",
            stderr=f"[{check.reason}]\n{check.detail}",
            time_cost=0.0,
        )

    if mode == 'call':
        code = remove_main_block(code)

    with tempfile.TemporaryDirectory() as tmpdir:

        if mode == "call":
            input_data = pickle.dumps(input_data)
        else:
            input_data = input_data

        code_path = os.path.join(tmpdir, "solution.py")

        if mode == "call":
            final_code = f"""{code}

if __name__ == "__main__":
    import sys
    import pickle

    args = pickle.loads(sys.stdin.buffer.read())
    res = Solution().{entry_func}(*args)
    sys.stdout.buffer.write(pickle.dumps(res))
"""
        else:
            final_code = code
        with open(code_path, "w") as f:
            f.write(final_code)

        start = time.time()
        try:
            p = subprocess.run(
                ["python3", code_path],
                input=input_data,
                text=mode == "stdio",
                capture_output=True,
                timeout=timeout,
                preexec_fn=os.setsid
            )
            cost = time.time() - start

            if p.returncode != 0:
                if mode == "call":
                    stderr_decoded = p.stderr.decode() if isinstance(p.stderr, bytes) else p.stderr
                    return EvalResult("RE", None, p.stdout, stderr_decoded, cost)
                return EvalResult("RE", None, p.stdout, p.stderr, cost)

            return EvalResult("OK", None, p.stdout, p.stderr, cost)

        except subprocess.TimeoutExpired:
            cost = time.time() - start
            return EvalResult("TLE", None, "", "Timeout", cost)


# =========================
# 单测试用例评估
# =========================

def evaluate_case(
        code: str,
        input_data,
        expected,
        timeout: float = 10.0,
        mode: Literal['stdio', 'call'] = "stdio",
        entry_func: Optional[str] = None,
) -> EvalResult:
    if mode == "call" and entry_func is None:
        raise ValueError("entry_func must be provided in 'call' mode.")

    result = _run_subprocess(
        code, input_data, timeout, mode, entry_func
    )
    if result.status != "OK":
        result.stdin = input_data
        result.expected = expected
        return result

    if mode == "call":
        result.stderr = result.stderr.decode()
        try:
            result.stdout = pickle.loads(result.stdout)
        except Exception as e:
            result.status = "RE"
            result.stderr += f"\n[pickle decode error] {e}"
            return result

    if isinstance(input_data, str):
        input_data = '\n'.join([line.strip() for line in input_data.splitlines()])

    if isinstance(result.stdout, str):
        result.stdout = '\n'.join([line.strip() for line in result.stdout.splitlines()])

    if isinstance(expected, str):
        expected = '\n'.join([line.strip() for line in expected.splitlines()])
    elif isinstance(expected, List) and all(isinstance(e, str) for e in expected):
        expected = ['\n'.join([line.strip() for line in e.splitlines()]) for e in expected]

    eval_equal = False
    try:
        eval_stdout = eval(result.stdout)
        eval_expected = eval(expected)
        if eval_stdout == eval_expected:
            eval_equal = True
    except:
        pass

    if result.stdout == expected or eval_equal:
        result.status = "AC"
    elif isinstance(expected, List) and isinstance(result.stdout, str) and result.stdout in expected:
        result.status = "AC"
    else:
        result.status = "WA"

    result.stdin = input_data
    result.expected = expected

    return result
