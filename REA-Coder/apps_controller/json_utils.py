"""
apps_controller/json_utils.py

Robust JSON extraction utilities for LLM outputs.
We reuse the "loose JSON" strategy used in SWE-bench controller/agents,
but keep it self-contained to avoid modifying existing code.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)
_JSON_ARR_RE = re.compile(r"\[.*\]", re.S)


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    # Remove ```lang ... ``` fences if present
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_\-]*\n", "", s)
        s = s[:-3] if s.endswith("```") else s
    return s.strip()


def extract_json_loose(text: str, default: Optional[Any] = None) -> Any:
    """
    Try to parse JSON from an LLM response.
    - Prefer parsing the whole content.
    - Fallback: parse the first {...} object.
    - Fallback: parse the first [...] array.
    """
    if default is None:
        default = {}
    raw = _strip_code_fences(text)

    # 1) Whole string JSON
    for candidate in (raw, raw.replace("\n", " ")):
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 2) First object
    m = _JSON_OBJ_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # 3) First array
    m = _JSON_ARR_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return default


def ensure_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def ensure_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    return str(x)