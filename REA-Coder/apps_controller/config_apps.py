"""
apps_controller/config_apps.py

Defaults for APPS evaluation controller.
"""
from __future__ import annotations


DEFAULT_TASK_PROMPT = """You are solving a programming contest problem.
Write a correct and efficient Python3 solution that passes all hidden tests.

IMPORTANT:
- Follow the exact I/O format in the statement.
- Handle edge cases.
- Keep code simple and robust (use input().strip(), split(), etc.).
- Do not import forbidden modules or call forbidden functions (see system message).
"""