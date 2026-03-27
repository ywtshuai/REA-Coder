import os
import pickle
import subprocess
import tempfile
import time
import platform
from dataclasses import dataclass
from typing import Optional, Literal, Any

from .checker import static_security_check


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
            if isinstance(input_data, list):
                input_data = '\n'.join(str(item) for item in input_data)
            elif not isinstance(input_data, str):
                input_data = str(input_data)
            
            final_code = code
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(final_code)

        start = time.time()
        try:
            if platform.system() == "Windows":
                python_cmd = "python"
                extra_kwargs = {}
                # Windows 默认 gbk，需强制子进程 stdio 用 UTF-8 以正确读写测试数据
                extra_kwargs["env"] = dict(os.environ, PYTHONIOENCODING="utf-8")
            else:
                python_cmd = "python3"
                extra_kwargs = {"preexec_fn": os.setsid}
            
            run_kwargs = dict(
                capture_output=True,
                timeout=timeout,
                **extra_kwargs
            )
            if mode == "stdio":
                # Windows 默认 gbk，CodeContests 测试用例可能含 Unicode（如 \x85），必须用 UTF-8
                run_kwargs["input"] = input_data
                run_kwargs["text"] = True
                run_kwargs["encoding"] = "utf-8"
            p = subprocess.run([python_cmd, code_path], **run_kwargs)
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

    if isinstance(expected, str):
        result.stdout = '\n'.join([line.strip() for line in result.stdout.splitlines()])
        expected = '\n'.join([line.strip() for line in expected.splitlines()])

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
    else:
        result.status = "WA"

    result.stdin = input_data
    result.expected = expected

    return result


def eval_single_code_stdio(code: str, test_cases: list, timeout: float = 10.0) -> tuple:
    """
    评估单个代码在 stdio 模式下的通过情况。
    
    Args:
        code: 待评估的代码
        test_cases: [{"input": str, "output": str}, ...]
        timeout: 超时时间
    
    Returns:
        (all_passed: bool, feedback_str: str)
        - 若全部通过: (True, "1")
        - 若有失败: (False, "# Execution Results...\nTest case (1) passed.\nTest case (2) failed....")
        - 若执行出错: (False, -1)
    """
    feedback = '# Execution Results of Test Cases:'
    all_passed = True
    try:
        for i, tc in enumerate(test_cases):
            r = evaluate_case(
                code=code,
                input_data=tc["input"],
                expected=tc["output"],
                mode="stdio",
                timeout=timeout
            )
            if r.status == 'AC':
                feedback += f'\nTest case ({i+1}) passed.'
            else:
                feedback += f'\nTest case ({i+1}) failed.'
                all_passed = False
    except Exception:
        return False, -1
    
    if all_passed:
        return True, '1'
    return False, feedback
