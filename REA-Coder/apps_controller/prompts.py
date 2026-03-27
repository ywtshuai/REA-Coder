"""
apps_controller/prompts.py

Prompt builders for the APPS pipeline (multi-agent requirement alignment -> codegen).
All prompts are designed to be repo-agnostic (single-problem) and to work with stdio tasks.

Refactor:
- All long prompt bodies moved to prompts_markdown/*.md
- This file only loads markdown templates + renders {{placeholders}} + returns messages
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# =========================
# Markdown prompt utilities
# =========================

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts_markdown"

_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def _load_md(filename: str) -> str:
    path = _PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt markdown not found: {path}")
    return path.read_text(encoding="utf-8")


def _render_md(template: str, **kwargs: Any) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key not in kwargs:
            raise KeyError(f"Missing prompt variable: {key}")
        val = kwargs[key]
        return "" if val is None else str(val)

    return _VAR_RE.sub(repl, template)


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


# =========================
# System prompt
# =========================

SYSTEM_APPS = _load_md("system_apps.md").strip()


# =========================
# Message helpers
# =========================

def sys_msg(content: str) -> Dict[str, str]:
    return {"role": "system", "content": content}


def user_msg(content: str) -> Dict[str, str]:
    return {"role": "user", "content": content}


# =========================
# Existing helper (kept as-is)
# =========================

def _format_solutions(solutions: List[str], max_solutions: int = 3, max_chars_each: int = 2500) -> str:
    if not solutions:
        return "(none)"
    out: List[str] = []
    for i, s in enumerate(solutions[:max_solutions], start=1):
        ss = (s or "").strip()
        if len(ss) > max_chars_each:
            ss = ss[:max_chars_each] + "\n# ...(truncated)..."
        out.append(f"--- solution_{i} ---\n{ss}")
    if len(solutions) > max_solutions:
        out.append(f"(+{len(solutions) - max_solutions} more solutions omitted)")
    return "\n\n".join(out)


# =========================
# Prompt builders
# =========================

def prompt_agent1_generate_questions(
    task_prompt: str,
    aligned_requirement: str,
    starter_code: str,
    *,
    max_questions: int,
    num_question_test_cases: int,
) -> List[Dict[str, str]]:
    if starter_code:
        full_aligned_requirement = f"{aligned_requirement}\n# START_CODE(The existing code, supplement based on it.)\n{starter_code}"
    else:
        full_aligned_requirement = aligned_requirement
    content = _render_md(
        _load_md("agent1_generate_questions.md"),
        aligned_requirement=full_aligned_requirement,
    )
    return [user_msg(content)]


def prompt_agent1_update_questions(
    task_prompt: str,
    base_questions: List[Dict[str, str]],
    agent4_mask_eval: Dict[str, object],
    aligned_requirement: str,
    starter_code: str,
    max_questions: int,
    masked_text: str,
) -> List[Dict[str, str]]:
    base_txt = "\n".join([f'{q["qid"]}: {q["question"]}' for q in base_questions]) or "(none)"
    if starter_code:
        full_aligned_requirement = f"{aligned_requirement}\n# START_CODE(The existing code, supplement based on it.)\n{starter_code}"
    else:
        full_aligned_requirement = aligned_requirement
    if masked_text:
        full_aligned_requirement = f"{full_aligned_requirement}\n\n# MASKED REQUIREMENT\n{masked_text}"
    content = _render_md(
        _load_md("agent1_update_questions.md"),
        base_txt=base_txt,
        agent4_mask_eval=_json(agent4_mask_eval),
        aligned_requirement=full_aligned_requirement,
        max_questions=max_questions,
    )
    return [user_msg(content)]


def prompt_agent1_generate_gold_answers(
    task_prompt: str,
    questions: List[Dict[str, str]],
    question: str,
    starter_code: str,
) -> List[Dict[str, str]]:
    qs_txt = "\n".join([f'{q.get("qid")}: {q.get("question")}' for q in questions]) or "(none)"
    if starter_code:
        full_aligned_requirement = f"{question}\n# START_CODE(The existing code, supplement based on it.)\n{starter_code}"
    else:
        full_aligned_requirement = question
    content = _render_md(
        _load_md("agent1_generate_gold_answers.md"),
        question=full_aligned_requirement,
        qs_txt=qs_txt,
    )
    return [user_msg(content)]


def prompt_agent3_answer_questions(
    task_prompt: str,
    questions: List[Dict[str, str]],
    question: str,
    starter_code: str,
) -> List[Dict[str, str]]:
    qs_txt = "\n".join([f'{q.get("qid")}: {q.get("question")}' for q in questions]) or "(none)"
    if starter_code:
        full_aligned_requirement = f"{question}\n# START_CODE(The existing code, supplement based on it.)\n{starter_code}"
    else:
        full_aligned_requirement = question
    content = _render_md(
        _load_md("agent3_answer_questions.md"),
        question=full_aligned_requirement,
        qs_txt=qs_txt,
    )
    return [user_msg(content)]


def prompt_agent3_predict_outputs(
    task_prompt: str,
    aligned_requirement: str,
    starter_code: str,
    inputs: List[str],
) -> List[Dict[str, str]]:
    ins = "\n".join([f"[INPUT_{i+1}]\n{inp}" for i, inp in enumerate(inputs or [])]) or "(none)"
    content = _render_md(
        _load_md("agent3_predict_outputs.md"),
        task_prompt=task_prompt,
        aligned_requirement=aligned_requirement,
        starter_code=starter_code or "(empty)",
        ins=ins,
    )
    return [user_msg(content)]


def prompt_judge_agent3_answers(
    task_prompt: str,
    requirement: str,
    questions: List[Dict[str, str]],
    gold_answers: List[Dict[str, str]],
    agent3_answers: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    qmap = {q["qid"]: q["question"] for q in questions or []}
    gmap = {a["qid"]: a["answer"] for a in gold_answers or []}
    mmap = {a["qid"]: a["answer"] for a in agent3_answers or []}

    items: List[str] = []
    for qid in qmap.keys():
        items.append(
            f"QID: {qid}\n"
            f"Q: {qmap.get(qid,'')}\n"
            f"GOLD: {gmap.get(qid,'')}\n"
            f"RELEVANT PERSON'S ANSWER: {mmap.get(qid,'')}\n"
        )
    bundle = "\n---\n".join(items)

    content = _render_md(
        _load_md("judge_agent3_answers.md"),
        REQUIREMENT=requirement,
        bundle=bundle,
    )
    return [user_msg(content)]


def prompt_agent2_summarize_gaps(
    task_prompt: str,
    masked_text: str,
    original_requirement: str,
    gap_blocks_text: str,
) -> List[Dict[str, str]]:
    if masked_text:
        full_aligned_requirement = f"{original_requirement}\n\n# MASKED REQUIREMENT\n{masked_text}"
    else:
        full_aligned_requirement = original_requirement
    content = _render_md(
        _load_md("agent2_summarize_gaps.md"),
        original_requirement=full_aligned_requirement,
        gap_blocks_text=gap_blocks_text,
    )
    return [user_msg(content)]


def prompt_agent3_generate_code(
    task_prompt: str,
    question: str,
    starter_code: str,
    enhanced_requirements: List[str],
    public_examples: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, str]]:
    # Keep your existing prep (even if template chooses not to show it yet).
    req_txt = "\n".join([f"- {r}" for r in enhanced_requirements]) or "(none)"

    ex_txt = ""
    if public_examples and public_examples.get("inputs") and public_examples.get("outputs"):
        pairs = list(zip(public_examples["inputs"], public_examples["outputs"]))[:3]
        ex_lines: List[str] = []
        for i, (inp, outp) in enumerate(pairs, start=1):
            ex_lines.append(f"Example {i} input:\n{inp}\nExample {i} output:\n{outp}")
        ex_txt = "\n\n".join(ex_lines)

    public_block = ""
    if ex_txt:
        public_block = "\n\n[PUBLIC EXAMPLES]\n" + ex_txt

    if starter_code:
        full_aligned_requirement = f"{question}\n# START_CODE(The existing code, supplement based on it.)\n{starter_code}"
    else:
        full_aligned_requirement = question

    content = _render_md(
        _load_md("agent3_generate_code.md"),
        question=full_aligned_requirement,
        req_txt=req_txt,
    )
    return [sys_msg(SYSTEM_APPS), user_msg(content)]


def prompt_agent3_fill_masked_question(
    task_prompt: str,
    masked_question: str,
    starter_code: str,
    generated_code: str,
    error_info: str,
) -> List[Dict[str, str]]:
    if starter_code:
        full_aligned_requirement = f"{masked_question}\n# START_CODE(The existing code, supplement based on it.)\n{starter_code}"
    else:
        full_aligned_requirement = masked_question
    content = _render_md(
        _load_md("agent3_fill_masked_question.md"),
        masked_question=full_aligned_requirement,
        generated_code=generated_code,
    )
    return [user_msg(content)]


def prompt_agent4_evaluate_mask_recovery(
    task_prompt: str,
    aligned_requirement: str,
    masked_requirement: str,
    agent3_fills: Dict[str, object],
) -> List[Dict[str, str]]:
    content = _render_md(
        _load_md("agent4_evaluate_mask_recovery.md"),
        aligned_requirement=aligned_requirement,
        masked_requirement=masked_requirement,
        agent3_fills=_json(agent3_fills),
    )
    return [user_msg(content)]


def prompt_agent4_mask_requirement(
    task_prompt: str,
    requirement: str,
    *,
    masked_token_ratio_range=(0.28, 0.38),
) -> List[Dict[str, str]]:
    low, high = masked_token_ratio_range
    content = _render_md(
        _load_md("agent4_mask_requirement.md"),
        task_prompt=task_prompt,
        requirement=requirement,
        low=low,
        high=high,
    )
    return [user_msg(content)]


def prompt_agent2_generate_stop_test_cases(
    task_prompt: str,
    aligned_requirement: str,
    starter_code: str,
    *,
    k: int,
) -> List[Dict[str, str]]:
    if starter_code:
        full_aligned_requirement = f"{aligned_requirement}\n# START_CODE(The existing code, supplement based on it.)\n{starter_code}"
    else:
        full_aligned_requirement = aligned_requirement
    content = _render_md(
        _load_md("agent2_generate_stop_test_cases.md"),
        aligned_requirement=full_aligned_requirement,
        k=k,
    )
    return [user_msg(content)]


def prompt_agent2_tidy_enhancements(
    task_prompt: str,
    base_problem_statement: str,
    raw_enhancements_text: str,
) -> List[Dict[str, str]]:
    content = _render_md(
        _load_md("agent2_tidy_enhancements.md"),
        base_problem_statement=base_problem_statement,
        raw_enhancements_text=raw_enhancements_text,
    )
    return [user_msg(content)]
