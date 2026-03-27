# =========================
# 静态安全检查（AST）
# =========================
import ast
from dataclasses import dataclass
from typing import Optional


# =========================
# 静态检查结果结构
# =========================

@dataclass
class StaticCheckResult:
    ok: bool  # 是否通过
    reason: str  # OK / SyntaxError / ForbiddenImport / ForbiddenCall
    detail: Optional[str] = None


FORBIDDEN_MODULES = {
    # "os", "sys", "subprocess", "socket",
    # "multiprocessing", "threading",
    # "signal", "resource"
}

FORBIDDEN_CALLS = {
    # "exec", "eval", "__import__", "open",
    # "fork", "system", "popen"
}


def static_security_check(code: str) -> StaticCheckResult:
    import traceback
    # ---------- 语法检查 ----------
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return StaticCheckResult(
            ok=False,
            reason="SyntaxError",
            detail=traceback.format_exc()
        )

    # ---------- AST 安全扫描 ----------
    for node in ast.walk(tree):

        # import xxx
        if isinstance(node, ast.Import):
            for n in node.names:
                name = n.name.split(".")[0]
                if name in FORBIDDEN_MODULES:
                    return StaticCheckResult(
                        ok=False,
                        reason="ForbiddenImport",
                        detail=f"import {name} at line {node.lineno}"
                    )

        # from xxx import yyy
        if isinstance(node, ast.ImportFrom):
            if node.module:
                name = node.module.split(".")[0]
                if name in FORBIDDEN_MODULES:
                    return StaticCheckResult(
                        ok=False,
                        reason="ForbiddenImport",
                        detail=f"from {name} import ... at line {node.lineno}"
                    )

        # exec / eval / open / system ...
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    return StaticCheckResult(
                        ok=False,
                        reason="ForbiddenCall",
                        detail=f"{node.func.id}() at line {node.lineno}"
                    )

            if isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_CALLS:
                    return StaticCheckResult(
                        ok=False,
                        reason="ForbiddenCall",
                        detail=f"{node.func.attr}() at line {node.lineno}"
                    )

    return StaticCheckResult(ok=True, reason="OK")
