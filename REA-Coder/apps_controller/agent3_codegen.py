"""
apps_controller/agent3_codegen.py

Agent3:
- answer requirement questions
- generate final code (single-file Python program)
- fill masked problem statement spans (for Agent4 post-hoc)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from llm.client import LLMClient
from utils.output_logger import OutputLogger
from pathlib import Path
from apps_controller.json_utils import extract_json_loose, ensure_list, ensure_str
from apps_controller.prompts import (
    prompt_agent3_answer_questions,
    prompt_agent3_generate_code,
    prompt_agent3_fill_masked_question,
    prompt_agent3_predict_outputs,
)


@dataclass
class Agent3AppsConfig:
    max_code_tokens: int = 8192


def _strip_markdown_fences(code: str) -> str:
    """
    Backward-compatible wrapper:
    - If the whole reply is fenced code, strip fences.
    - If the reply contains prose + fenced code, extract the best code block.
    - Otherwise return trimmed text.
    Also robustly removes stray fence lines that sometimes appear at the end.
    """
    extracted = _extract_code_from_llm_text(code)
    return _strip_stray_fence_lines(extracted)


def _strip_stray_fence_lines(text: str) -> str:
    """
    Remove standalone markdown fence lines that sometimes leak into outputs, e.g.
    ending with:
        ...
        ```
    or
        ...
        ```   \n
    Also tolerates multiple trailing fence lines.
    """
    import re

    s = (text or "").rstrip()
    if not s:
        return ""

    # Remove trailing standalone fence lines (``` or ~~~) with optional language tag/spaces
    # Repeats to handle multiple such lines.
    trailing_pat = re.compile(r"(?:\r?\n)?(?:```|~~~)\s*[a-zA-Z0-9_\-]*\s*$")
    while True:
        s2 = re.sub(trailing_pat, "", s)
        if s2 == s:
            break
        s = s2.rstrip()

    # Remove leading standalone fence lines too (rare but happens)
    leading_pat = re.compile(r"^(?:```|~~~)\s*[a-zA-Z0-9_\-]*\s*\r?\n")
    while True:
        s2 = re.sub(leading_pat, "", s)
        if s2 == s:
            break
        s = s2.lstrip()

    return s.strip()


def _extract_code_from_llm_text(text: str) -> str:
    import re

    s = (text or "").strip()
    if not s:
        return ""

    # 1) If the reply starts with a fence, keep old behavior but make it more robust (``` or ~~~).
    if s.startswith("```") or s.startswith("~~~"):
        fence = "```" if s.startswith("```") else "~~~"

        # Drop the first fence line (```python\n or ```\n)
        s2 = re.sub(rf"^{re.escape(fence)}[a-zA-Z0-9_\-]*\s*\r?\n", "", s)

        # Drop a trailing fence line even if it has whitespace/newline after it
        s2 = re.sub(rf"(?:\r?\n)?{re.escape(fence)}\s*$", "", s2.rstrip())

        return s2.strip()

    # 2) Otherwise, extract fenced code blocks from anywhere in the text.
    block_pat = re.compile(
        r"(?P<fence>```|~~~)\s*(?P<lang>[a-zA-Z0-9_\-]*)\s*\r?\n(?P<body>.*?)(?:\r?\n)?(?P=fence)\s*",
        re.DOTALL,
    )
    blocks = []
    for m in block_pat.finditer(s):
        lang = (m.group("lang") or "").strip().lower()
        body = (m.group("body") or "").strip("\n").strip()
        if body:
            blocks.append((lang, body))

    if not blocks:
        # 3) No fenced blocks at all: return text but strip stray fence lines just in case
        return _strip_stray_fence_lines(s)

    py_blocks = [b for (lang, b) in blocks if lang in ("python", "py")]
    if py_blocks:
        return _pick_best_python_block(py_blocks)

    return _pick_best_python_block([b for (_, b) in blocks])

def _pick_best_python_block(candidates: list[str]) -> str:
    """
    Heuristic scoring to choose the most likely final solution block.
    """
    import re

    def score(code: str) -> int:
        sc = 0
        t = code

        # Strong signals
        if "if __name__" in t:
            sc += 50
        if re.search(r"^\s*def\s+\w+\(", t, re.M):
            sc += 20
        if re.search(r"^\s*class\s+\w+", t, re.M):
            sc += 10
        if re.search(r"^\s*import\s+\w+|^\s*from\s+\w+\s+import\s+", t, re.M):
            sc += 10

        # Penalize obvious non-code / analysis leakage
        if "Let me" in t or "I need to" in t or "approach" in t:
            sc -= 10
        if "```" in t or "~~~" in t:
            sc -= 5

        # Prefer longer (often the full solution) but not too dominant
        sc += min(len(t) // 200, 30)
        return sc

    best = max(candidates, key=score)
    return best.strip()


class Agent3AppsRunner:
    def __init__(self, llm: LLMClient, cfg: Agent3AppsConfig):
        self.llm = llm
        self.cfg = cfg

    def answer_questions(
        self,
        task_prompt: str,
        questions: List[Dict[str, str]],
        question: str,
        starter_code: str,
        logger: Optional[OutputLogger] = None,
    ) -> List[Dict[str, str]]:
        messages = prompt_agent3_answer_questions(task_prompt, questions, question, starter_code)
        if logger:
            logger.write_prompt_bundle("agent3/B_answers_prompt.txt", messages)
        file_path = Path(logger.root) / "agent3" / "B_answers_raw.txt"
        if file_path.is_file() and file_path.stat().st_size > 0:
            raw = file_path.read_text(encoding="utf-8")
        else: raw = self.llm.chat(messages)
        data = extract_json_loose(raw, default={})
        answers = ensure_list(data.get("answers"))
        out: List[Dict[str, str]] = []
        qids = {q["qid"] for q in questions}
        for a in answers:
            if not isinstance(a, dict):
                continue
            qid = str(a.get("qid") or "").strip()
            if qid not in qids:
                continue
            ans = str(a.get("answer") or "").strip()
            if not ans:
                continue
            out.append({"qid": qid, "answer": ans})

        if logger:
            logger.write_text("agent3/B_answers_raw.txt", raw)
            logger.write_json("agent3/B_answers.json", {"answers": out})
        return out

    def generate_code(
        self,
        task_prompt: str,
        question: str,
        starter_code: str,
        enhanced_requirements: List[str],
        public_examples: Optional[dict] = None,
        logger: Optional[OutputLogger] = None,
    ) -> str:
        messages = prompt_agent3_generate_code(
            task_prompt, question, starter_code, enhanced_requirements, public_examples=public_examples
        )
        if logger:
            logger.write_prompt_bundle("agent3/code_prompt.txt", messages)
        file_path = Path(logger.root) / "agent3" / "code_raw.txt"
        if file_path.is_file() and file_path.stat().st_size > 0:
            raw = file_path.read_text(encoding="utf-8")
        else: raw = self.llm.chat(messages, max_tokens=self.cfg.max_code_tokens)
        code = _strip_markdown_fences(raw)

        if logger:
            logger.write_text("agent3/code_raw.txt", raw)
            logger.write_text("agent3/solution.py", code)
        return code

    def fill_masked_question(
        self,
        task_prompt: str,
        masked_question: str,
        starter_code: str,
        generated_code: str,
        error_info: str,
        logger: Optional[OutputLogger] = None,
    ) -> dict:
        messages = prompt_agent3_fill_masked_question(
            task_prompt, masked_question, starter_code, generated_code, error_info
        )
        if logger:
            logger.write_prompt_bundle("agent3/mask_fill_prompt.txt", messages)
        file_path = Path(logger.root) / "agent3" / "mask_fill_raw.txt"
        if file_path.is_file() and file_path.stat().st_size > 0:
            raw = file_path.read_text(encoding="utf-8")
        else: raw = self.llm.chat(messages)
        try:
            data = extract_json_loose(raw, default={})
            fills = ensure_list(data.get("fills"))
            demasked = ensure_str(data.get("recovered_statement"), default="")
            # normalize fills
            norm_fills = []
            for f in fills:
                if not isinstance(f, dict):
                    continue
                mid = str(f.get("mask_id") or "").strip()
                txt = str(f.get("text") or "").strip()
                if not mid or not txt:
                    continue
                norm_fills.append({"mask_id": mid, "text": txt})

            out = {"fills": norm_fills, "recovered_statement": demasked}
        except Exception as e:
            out = {"fills":"", "recovered_statement": ""}

        if logger:
            logger.write_text("agent3/mask_fill_raw.txt", raw)
            logger.write_json("agent3/mask_fill.json", out)
        return out
    
    def predict_outputs_for_inputs(
        self,
        task_prompt: str,
        aligned_requirement: str,
        starter_code: str,
        inputs: List[str],
        logger: Optional[OutputLogger] = None,
    ) -> List[str]:
        messages = prompt_agent3_predict_outputs(task_prompt, aligned_requirement, starter_code, inputs)
        if logger:
            logger.write_prompt_bundle("agent3/predict_outputs_prompt.txt", messages)
        file_path = Path(logger.root) / "agent3" / "predict_outputs_raw.txt"
        if file_path.is_file() and file_path.stat().st_size > 0:
            raw = file_path.read_text(encoding="utf-8")
        else: raw = self.llm.chat(messages)
        data = extract_json_loose(raw, default={})
        outs = ensure_list(data.get("outputs"))
        outs = [str(x) for x in outs][: len(inputs or [])]
        if logger:
            logger.write_text("agent3/predict_outputs_raw.txt", raw)
            logger.write_json("agent3/predict_outputs.json", {"outputs": outs})
        return outs