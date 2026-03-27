"""
apps_controller/agent4_feedback.py

Agent4:
- mask aligned requirement text (~1/3 tokens), prioritizing key spec elements
- evaluate Agent3's mask recovery (per-mask verdict/why/difference)
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from llm.client import LLMClient
from utils.output_logger import OutputLogger

from apps_controller.json_utils import extract_json_loose, ensure_list
from apps_controller.prompts import prompt_agent4_evaluate_mask_recovery, prompt_agent4_mask_requirement
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, Any


def _normalize_mask_id(mask_id: Any) -> str:
    """
    Normalize mask id to something like 'MASK_1' and return with brackets: '[MASK_1]'.
    """
    mid = str(mask_id or "").strip()
    if not mid:
        mid = "MASK"
    # keep original if already looks like MASK_123; otherwise just use as-is
    # You can enforce upper-case to be consistent.
    mid = mid.upper()
    if mid.startswith("[") and mid.endswith("]"):
        return mid
    return f"[{mid}]"


def apply_spans_masking(
    original_text: str,
    spans: List[Dict[str, Any]],
    *,
    replace_once_per_span: bool = True,
    use_regex: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Apply deterministic replacement on `original_text`:
      - For each span: find `span['original_text']` in current text, replace with [mask_id].
      - If not found, skip.
      - Default: replace only once per span (first occurrence), preserving span order.

    Returns:
      (masked_text, audit_list)
    where audit_list records what got replaced or skipped.
    """
    text = original_text
    audit: List[Dict[str, Any]] = []

    for i, sp in enumerate(spans or []):
        mask_id = sp.get("mask_id")
        needle = sp.get("original_text")

        rep = _normalize_mask_id(mask_id)
        needle_str = str(needle or "")

        if not needle_str.strip():
            audit.append(
                {
                    "index": i,
                    "mask_id": mask_id,
                    "original_text": needle_str,
                    "replaced": False,
                    "reason": "empty_original_text",
                }
            )
            continue

        # Build pattern
        if use_regex:
            pattern = needle_str
        else:
            pattern = re.escape(needle_str)

        # Replace
        if replace_once_per_span:
            new_text, n = re.subn(pattern, rep, text, count=1)
        else:
            new_text, n = re.subn(pattern, rep, text)

        if n > 0:
            text = new_text
            audit.append(
                {
                    "index": i,
                    "mask_id": mask_id,
                    "original_text": needle_str,
                    "replaced": True,
                    "count": n,
                }
            )
        else:
            audit.append(
                {
                    "index": i,
                    "mask_id": mask_id,
                    "original_text": needle_str,
                    "replaced": False,
                    "reason": "not_found_in_question",
                }
            )

    return text, audit


def compute_masked_token_ratio(masked_text: str) -> float:
    """
    Rough ratio: (# of mask tokens like [MASK_1]) / (# of whitespace tokens).
    If you have a tokenizer, replace this with tokenizer-based computation.
    """
    if not masked_text:
        return 0.0
    total_tokens = len(masked_text.split())
    if total_tokens == 0:
        return 0.0
    masked_tokens = len(re.findall(r"\[MASK_[0-9]+\]", masked_text.upper()))
    return float(masked_tokens) / float(total_tokens)


def _count_tokens_rough(s: str) -> int:
    # rough tokenization by whitespace
    return len([t for t in (s or "").replace("\t", " ").split(" ") if t.strip()])


def mask_text_deterministic(
    text: str,
    *,
    masked_token_ratio_range: Tuple[float, float] = (0.28, 0.38),
    target_ratio: float = 0.33,
    seed: int = 0,
    min_spans: int = 3,
) -> Dict[str, object]:
    """
    Deterministically mask non-contiguous spans of the text, aiming for ~1/3 tokens masked.

    Output:
      {
        "masked_text": "... with [MASK_1] ...",
        "masked_token_ratio": 0.31,
        "spans": [{"mask_id":"MASK_1","start_line":...,"end_line":...,"token_count":...}, ...]
      }
    """
    lines = text.splitlines()
    total_tokens = _count_tokens_rough(text)
    if total_tokens <= 0:
        return {"masked_text": text, "masked_token_ratio": 0.0, "spans": []}

    low, high = masked_token_ratio_range
    target_tokens = int(total_tokens * target_ratio)
    low_tokens = int(total_tokens * low)
    high_tokens = int(total_tokens * high)

    # Candidate spans: 1-2 lines each, with enough tokens
    candidates = []
    for i in range(len(lines)):
        for span_len in (1, 2):
            j = i + span_len
            if j > len(lines):
                continue
            seg = "\n".join(lines[i:j]).strip()
            tok = _count_tokens_rough(seg)
            if tok < 10:
                continue
            # avoid masking pure section separators to keep structure
            if seg.strip("- ").lower().startswith("input") or seg.strip("- ").lower().startswith("output"):
                continue
            candidates.append((i, j, tok))

    rnd = random.Random(seed)
    rnd.shuffle(candidates)

    chosen = []
    masked_tokens = 0

    def overlaps(a, b):
        return not (a[1] <= b[0] or b[1] <= a[0])

    # Greedy pick: reach at least low_tokens, try not exceed high_tokens
    for (i, j, tok) in candidates:
        # no overlaps + keep some separation (no adjacent)
        if any(overlaps((i, j), (ci, cj)) or abs(i - cj) <= 0 or abs(ci - j) <= 0 for (ci, cj, _) in chosen):
            continue
        if masked_tokens + tok > high_tokens and masked_tokens >= low_tokens:
            continue
        chosen.append((i, j, tok))
        masked_tokens += tok
        if masked_tokens >= target_tokens and len(chosen) >= min_spans:
            break

    # If still below low_tokens, allow smaller adjustments (even if exceed high slightly)
    if masked_tokens < low_tokens:
        for (i, j, tok) in candidates:
            if any(overlaps((i, j), (ci, cj)) for (ci, cj, _) in chosen):
                continue
            chosen.append((i, j, tok))
            masked_tokens += tok
            if masked_tokens >= low_tokens and len(chosen) >= min_spans:
                break

    # Build masked text
    chosen.sort(key=lambda x: x[0])
    spans = []
    out_lines = []
    cur = 0
    for k, (i, j, tok) in enumerate(chosen, start=1):
        out_lines.extend(lines[cur:i])
        out_lines.append(f"[MASK_{k}]")
        spans.append({"mask_id": f"MASK_{k}", "start_line": i, "end_line": j - 1, "token_count": tok})
        cur = j
    out_lines.extend(lines[cur:])

    masked_text = "\n".join(out_lines)
    ratio = masked_tokens / max(1, total_tokens)

    return {"masked_text": masked_text, "masked_token_ratio": ratio, "spans": spans}


@dataclass
class Agent4AppsConfig:
    masked_token_ratio_range: Tuple[float, float] = (0.28, 0.38)
    target_ratio: float = 0.33


class Agent4AppsRunner:
    def __init__(self, llm: LLMClient, cfg: Agent4AppsConfig):
        self.llm = llm
        self.cfg = cfg

    def mask_question(
        self,
        question: str,
        logger: Optional[OutputLogger] = None,
    ) -> Dict[str, object]:
        # 1) LLM-based mask
        messages = prompt_agent4_mask_requirement(
            task_prompt="", # task prompt already included by caller usually; safe to keep empty or pass through if you want
            requirement=question,
            masked_token_ratio_range=self.cfg.masked_token_ratio_range,
        )
        if logger:
            logger.write_prompt_bundle("agent4/mask_llm_prompt.txt", messages)

        file_path = Path(logger.root) / "agent4" / "mask_llm_raw.txt"
        if file_path.is_file() and file_path.stat().st_size > 0:
            raw = file_path.read_text(encoding="utf-8")
        else: raw = self.llm.chat(messages)
        try:
            data = extract_json_loose(raw, default={})
            masked_text = str(data.get("masked_text") or "").strip()
            spans = ensure_list(data.get("spans"))
            ratio = data.get("masked_token_ratio")
            if ratio is None:
                ratio = 0.33
        except Exception as e:
            ratio = 0.33 

        masked = {"masked_text": masked_text, "masked_token_ratio": float(ratio), "spans": spans, "raw": raw}

        if logger:
            logger.write_text("agent4/mask_llm_raw.txt", raw)
            logger.write_json("agent4/mask_meta.json", masked)
            logger.write_text("agent4/masked_question.txt", str(masked.get("masked_text") or ""))

        return masked

    def mask_requirement(self, requirement: str, logger: Optional[OutputLogger] = None) -> Dict[str, object]:
        return self.mask_question(requirement, logger=logger)

    def evaluate_mask_recovery(
        self,
        task_prompt: str,
        original_requirement: str,
        masked_requirement: str,
        agent3_fills: dict,
        logger: Optional[OutputLogger] = None,
    ) -> Dict[str, Any]:
        messages = prompt_agent4_evaluate_mask_recovery(
            task_prompt=task_prompt,
            aligned_requirement=original_requirement,
            masked_requirement=masked_requirement,
            agent3_fills=agent3_fills,
        )
        if logger:
            logger.write_prompt_bundle("agent4/mask_eval_prompt.txt", messages)
        file_path = Path(logger.root) / "agent4" / "mask_eval_raw.txt"
        if file_path.is_file() and file_path.stat().st_size > 0:
            raw = file_path.read_text(encoding="utf-8")
        else: raw = self.llm.chat(messages)
        data = extract_json_loose(raw, default={})

        # expected:
        # {"mask_evaluations":[{"mask_id":"MASK_1","verdict":"correct|incorrect","why":"...","difference":"..."}], "notes":""}
        out = {
            "mask_evals": ensure_list(data.get("mask_evals")),
            "notes": str(data.get("notes") or "").strip(),
            "raw": raw,
        }
        if logger:
            logger.write_text("agent4/mask_eval_raw.txt", raw)
            logger.write_json("agent4/mask_eval.json", out)
        return out