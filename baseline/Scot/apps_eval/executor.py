import os
import pickle
import subprocess
import tempfile
import time
import platform
from dataclasses import dataclass
from typing import Optional, Literal, Any

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
            # stdio 模式：确保 input_data 是字符串
            if isinstance(input_data, list):
                # 如果是列表，转换为字符串（每个元素一行）
                input_data = '\n'.join(str(item) for item in input_data)
            elif not isinstance(input_data, str):
                # 如果不是字符串也不是列表，转为字符串
                input_data = str(input_data)
            
            final_code = code
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(final_code)

        start = time.time()
        try:
            # Windows 和 Linux 使用不同的 Python 命令和进程组设置
            if platform.system() == "Windows":
                python_cmd = "python"
                extra_kwargs = {}
            else:
                python_cmd = "python3"
                extra_kwargs = {"preexec_fn": os.setsid}
            
            # 强制子进程使用 UTF-8 的 stdin/stdout/stderr，避免 Windows 下 GBK 编码导致的 UnicodeDecodeError
            # （父进程写 stdin 用 UTF-8，读 stdout/stderr 也需 UTF-8；子进程若用 GBK 输出会解码失败）
            sub_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            
            # stdio 模式：显式指定 encoding='utf-8'
            run_kwargs = dict(
                capture_output=True,
                timeout=timeout,
                env=sub_env,
                **extra_kwargs
            )
            if mode == "stdio":
                run_kwargs["input"] = input_data
                run_kwargs["text"] = True
                run_kwargs["encoding"] = "utf-8"
            else:
                run_kwargs["input"] = None

            p = subprocess.run(
                [python_cmd, code_path],
                **run_kwargs
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
