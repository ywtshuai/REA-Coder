"""
apps_controller/agent2_requirements.py

Agent2:
- summarize ONLY incorrect judged items (gap blocks) into aspect-based enhancements
- output an enhanced requirement TEXT:
    original_requirement + "\n" + "<Aspect>: <content>" lines
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from llm.client import LLMClient
from utils.output_logger import OutputLogger

from apps_controller.json_utils import extract_json_loose, ensure_list
from apps_controller.prompts import prompt_agent2_summarize_gaps, prompt_agent2_generate_stop_test_cases, prompt_agent2_tidy_enhancements
import re
from pathlib import Path
@dataclass
class Agent2AppsConfig:
    pass


class Agent2AppsRunner:
    def __init__(self, llm: LLMClient, cfg: Agent2AppsConfig):
        self.llm = llm
        self.cfg = cfg

    @staticmethod
    def _build_gap_blocks(
        incorrect_results: List[Dict[str, Any]],
        questions: List[Dict[str, str]],
        gold_answers: List[Dict[str, str]],
        agent3_answers: List[Dict[str, str]],
    ) -> str:
        """
        Build text in the required format:

        [gap|judge_incorrect|QID]
        Q: ...
        GOLD: ...
        RELEVANT PERSON'S ANSWER: ...
        WHY: ...
        DIFF: ...

        Only for incorrect items.
        """
        qmap = {q.get("qid"): q.get("question", "") for q in questions or []}
        gmap = {a.get("qid"): a.get("answer", "") for a in gold_answers or []}
        mmap = {a.get("qid"): a.get("answer", "") for a in agent3_answers or []}

        blocks: List[str] = []
        for it in incorrect_results or []:
            qid = str(it.get("qid") or "").strip()
            if not qid:
                continue
            why = str(it.get("why") or "").strip()
            diff = str(it.get("difference") or "").strip()
            blocks.append(
                (
                    f"[gap|judge_incorrect|{qid}]\n"
                    f"Q: {qmap.get(qid, '')}\n"
                    f"GOLD: {gmap.get(qid, '')}\n"
                    f"RELEVANT PERSON'S ANSWER: {mmap.get(qid, '')}\n"
                    f"WHY: {why}\n"
                    f"DIFF: {diff}"
                ).strip()
            )

        return "\n\n".join(blocks) or "(none)"

    @staticmethod
    def build_mask_gap_blocks(
        masked_spans: List[Dict[str, Any]],
        agent3_fills: Dict[str, Any],
        mask_eval: Dict[str, Any],
    ) -> str:
        # gold: mask_id -> original_text
        gold_map = {}
        for sp in masked_spans or []:
            mid = str(sp.get("mask_id") or "").strip()
            if not mid:
                continue
            gold_map[mid] = str(sp.get("original_text") or "").strip()

        # model: mask_id -> filled text
        model_map = {}
        fills_list = agent3_fills.get("fills") if isinstance(agent3_fills, dict) else []
        if isinstance(fills_list, list):
            for f in fills_list:
                if not isinstance(f, dict):
                    continue
                mid = str(f.get("mask_id") or "").strip()
                txt = str(f.get("text") or "").strip()
                if mid:
                    model_map[mid] = txt

        # eval: only incorrect
        blocks = []
        evals = mask_eval.get("mask_evals") if isinstance(mask_eval, dict) else []
        if not isinstance(evals, list):
            evals = []

        for it in evals:
            if not isinstance(it, dict):
                continue
            mid = str(it.get("mask_id") or "").strip()
            verdict = str(it.get("verdict") or "").lower().strip()
            if not mid or verdict != "incorrect":
                continue
            why = str(it.get("why") or "").strip()
            diff = str(it.get("difference") or "").strip()

            blocks.append(
                (
                    f"[gap|mask_recovery|{mid}]\n"
                    f"Q: Recover the missing requirement span {mid}.\n"
                    f"GOLD: {gold_map.get(mid, '')}\n"
                    f"RELEVANT PERSON'S ANSWER: {model_map.get(mid, '')}\n"
                    f"WHY: {why}\n"
                    f"DIFF: {diff}"
                ).strip()
            )

        return "\n\n".join(blocks) or ""

    def generate_stop_test_cases(
        self,
        task_prompt: str,
        aligned_requirement: str,
        starter_code: str,
        k: int,
        logger: Optional[OutputLogger] = None,
    ) -> Dict[str, List[str]]:
        messages = prompt_agent2_generate_stop_test_cases(
            task_prompt=task_prompt,
            aligned_requirement=aligned_requirement,
            starter_code=starter_code,
            k=k,
        )
        if logger:
            logger.write_prompt_bundle("agent2/stop_cases_prompt.txt", messages)

        file_path = Path(logger.root) / "agent2" / "stop_cases_raw.txt"
        if file_path.is_file() and file_path.stat().st_size > 0:
            raw = file_path.read_text(encoding="utf-8")
        else: raw = self.llm.chat(messages)
        try:
            data = extract_json_loose(raw, default={})

            ins = ensure_list(data.get("inputs"))
            outs = ensure_list(data.get("outputs"))
            ins = [str(x) for x in ins]
            outs = [str(x) for x in outs]
            n = min(len(ins), len(outs))
            out = {"inputs": ins[:n], "outputs": outs[:n]}
        except Exception as e:
            out = {"inputs": [], "outputs": []}

        if logger:
            logger.write_text("agent2/stop_cases_raw.txt", raw)
            logger.write_json("agent2/stop_cases.json", out)

        return out

    @staticmethod
    def _stitch_as_text(original_requirement: str, aspects: Dict[str, Any]) -> str:
        """
        New behavior:
        - Parse existing '<Aspect>: ...' blocks from original_requirement
        - Merge with `aspects` (dedupe by aspect key)
        - Render back so each aspect header appears at most once
        """
        preferred_order = [
            "Requirement Background",
            "Requirement Purpose",
            "Terminology Explanation",
            "Input Requirements",
            "Output Requirements",
            "Explanations of examples",
            "Edge/Corner Cases",
            "Core Functionality",
            "APIs",
            "Error Handling Requirements",
            "Hints or Tips",
        ]

        def _norm_ws(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "").strip())

        def _merge_text(old: str, new: str) -> str:
            o = (old or "").strip()
            n = (new or "").strip()
            if not o:
                return n
            if not n:
                return o
            on = _norm_ws(o)
            nn = _norm_ws(n)
            if on == nn:
                return o
            # keep the more informative one if it contains the other
            if nn in on:
                return o
            if on in nn:
                return n
            # otherwise concatenate deterministically (old first)
            return (o + "\n" + n).strip()

        # ---------- parse existing aspects from original_requirement ----------
        text = (original_requirement or "").strip()
        if not text:
            core_text = ""
            existing_aspects: Dict[str, str] = {}
        else:
            headers = preferred_order[:]  # allow deterministic parse keys
            # also accept any extra aspect keys that may appear
            header_pat = r"^(" + "|".join([re.escape(h) for h in headers]) + r")\s*:\s*(.*)$"
            header_re = re.compile(header_pat)

            lines = text.splitlines()
            core_lines: List[str] = []
            existing_aspects = {}

            i = 0
            while i < len(lines):
                m = header_re.match(lines[i].strip())
                if not m:
                    core_lines.append(lines[i])
                    i += 1
                    continue
                key = m.group(1).strip()
                buf = [m.group(2).rstrip()]
                i += 1
                while i < len(lines):
                    m2 = header_re.match(lines[i].strip())
                    if m2:
                        break
                    buf.append(lines[i].rstrip())
                    i += 1
                val = "\n".join(buf).strip()
                if val:
                    if key in existing_aspects:
                        existing_aspects[key] = _merge_text(existing_aspects[key], val)
                    else:
                        existing_aspects[key] = val

            core_text = "\n".join(core_lines).strip()

        # ---------- merge existing aspects with new aspects ----------
        merged: Dict[str, str] = dict(existing_aspects)
        if isinstance(aspects, dict):
            for k, v in aspects.items():
                kk = str(k).strip()
                vv = str(v or "").strip()
                if not kk or not vv:
                    continue
                if kk in merged:
                    merged[kk] = _merge_text(merged[kk], vv)
                else:
                    merged[kk] = vv

        # ---------- render back (indent multiline values to keep parsing stable) ----------
        out_lines: List[str] = []
        if core_text:
            out_lines.append(core_text)

        def _emit_block(key: str, val: str):
            parts = (val or "").splitlines()
            if not parts:
                return
            first = parts[0].strip()
            out_lines.append(f"{key}: {first}")
            for rest in parts[1:]:
                rr = rest.rstrip()
                if rr.strip():
                   out_lines.append(f"  {rr}")

        # preferred order first
        for k in preferred_order:
            if k in merged and merged[k].strip():
                _emit_block(k, merged[k])
        # then extras in sorted order for determinism
        for k in sorted([x for x in merged.keys() if x not in preferred_order]):
            if merged[k].strip():
                _emit_block(k, merged[k])

        return "\n".join(out_lines).strip()

    @staticmethod
    def _parse_core_and_aspects(text: str) -> tuple[str, Dict[str, str]]:
        preferred_order = [
            "Requirement Background",
            "Requirement Purpose",
            "Terminology Explanation",
            "Input Requirements",
            "Output Requirements",
            "Explanations of examples",
            "Edge/Corner Cases",
            "Core Functionality",
            "APIs",
            "Error Handling Requirements",
            "Hints or Tips",
        ]

        text = (text or "").strip()
        if not text:
            return "", {}

        # NEW: accept ANY "Key: value" header line
        # - key cannot start with whitespace
        # - key must contain at least 2 non-colon chars
        # - key cannot contain newline
        header_re = re.compile(r"^([^\s:\n][^:\n]{1,80})\s*:\s*(.*)$")

        lines = text.splitlines()
        core_lines: List[str] = []
        aspects: Dict[str, str] = {}

        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            m = header_re.match(line.strip())
            if not m:
                core_lines.append(lines[i])
                i += 1
                continue

            key = m.group(1).strip()
            first_val = m.group(2).rstrip()

            buf = [first_val]
            i += 1
            while i < len(lines):
                m2 = header_re.match(lines[i].strip())
                if m2:
                    break
                buf.append(lines[i].rstrip())
                i += 1

            val = "\n".join(buf).strip()
            if val:
                if key in aspects:
                    aspects[key] = (aspects[key].rstrip() + "\n" + val).strip()
                else:
                    aspects[key] = val

        core_text = "\n".join(core_lines).strip()
        return core_text, aspects

    def tidy_enhancements_under_base(
        self,
        task_prompt: str,
        base_problem_statement: str,
        stitched_requirement_text: str,
        logger: Optional[OutputLogger] = None,
    ) -> str:
        # 1) 解析出当前 stitched 文本里的 aspects（我们只整理它）
        _, aspects = self._parse_core_and_aspects(stitched_requirement_text)

        # 2) 把 aspects 渲染成“原始增强段文本”（不带 core）
        raw_enh = self._stitch_as_text("", aspects).strip()
        if not raw_enh:
            return (base_problem_statement or "").strip()

        # 3) 让 Agent2 整理增强段（只整理下面内容）
        messages = prompt_agent2_tidy_enhancements(
            task_prompt=task_prompt,
            base_problem_statement=(base_problem_statement or "").strip(),
            raw_enhancements_text=raw_enh,
        )
        if logger:
            logger.write_prompt_bundle("agent2/tidy_enhancements_prompt.txt", messages)

        file_path = Path(logger.root) / "agent2" / "tidy_enhancements_raw.txt"
        if file_path.is_file() and file_path.stat().st_size > 0:
            #print(567867989021)
            raw = file_path.read_text(encoding="utf-8")
        else: raw = self.llm.chat(messages)
        data = extract_json_loose(raw, default={})
        tidied = str(data.get("tidied_enhancements") or "").strip()

        if logger:
            logger.write_text("agent2/tidy_enhancements_raw.txt", raw)
            logger.write_text("agent2/tidied_enhancements.txt", tidied)

        # 4) 回填：base_problem_statement + tidied_enhancements
        if not tidied:
            return (base_problem_statement or "").strip()

        return ((base_problem_statement or "").strip() + "\n\n" + tidied).strip()


    def build_enhanced_requirement_text(
        self,
        task_prompt: str,
        original_requirement: str,
        incorrect_results: List[Dict[str, Any]],
        questions: List[Dict[str, str]],
        gold_answers: List[Dict[str, str]],
        agent3_answers: List[Dict[str, str]],
        extra_gap_blocks_text: str = "",
        base_problem_statement: Optional[str] = None,   # <-- NEW
        masked_text: str = "",
        logger: Optional[OutputLogger] = None,
    ) -> str:
        """
        Main entry:
        - Convert incorrect judge results into gap blocks
        - Ask Agent2 to summarize into aspect requirements
        - Stitch into final enhanced requirement TEXT
        """
        gap_blocks_text = self._build_gap_blocks(
            incorrect_results=incorrect_results,
            questions=questions,
            gold_answers=gold_answers,
            agent3_answers=agent3_answers,
        )
        if extra_gap_blocks_text.strip():
            gap_blocks_text = (gap_blocks_text.strip() + "\n\n" + extra_gap_blocks_text.strip()).strip()

        messages = prompt_agent2_summarize_gaps(
            task_prompt=task_prompt,
            masked_text=masked_text,
            original_requirement=self._stitch_as_text(original_requirement, {}),
            gap_blocks_text=gap_blocks_text,
        )
        if logger:
            logger.write_prompt_bundle("agent2/agent2_prompt.txt", messages)
        file_path = Path(logger.root) / "agent2" / "agent2_raw.txt"
        if file_path.is_file() and file_path.stat().st_size > 0:
            raw = file_path.read_text(encoding="utf-8")
        else: raw = self.llm.chat(messages)
        data = extract_json_loose(raw, default={})

        # Expected JSON:
        # {
        #   "aspects": { ... }
        # }
        orig = str(original_requirement or "").strip()
        aspects = data.get("aspects") if isinstance(data.get("aspects"), dict) else {}

        stitched = self._stitch_as_text(orig, aspects)

        # NEW: tidy only the enhancement part, keep base statement frozen
        base_stmt = (base_problem_statement or "").strip()
        if base_stmt:
            enhanced_text = self.tidy_enhancements_under_base(
                task_prompt=task_prompt,
                base_problem_statement=base_stmt,
                stitched_requirement_text=stitched,
                logger=logger,
            )
        else:
            # fallback: no frozen base provided -> keep existing behavior
            enhanced_text = stitched

        if logger:
            logger.write_text("agent2/gaps_input.txt", gap_blocks_text)
            logger.write_text("agent2/agent2_raw.txt", raw)
            logger.write_json(
                "agent2/agent2_aspects.json",
                {"original_requirement": orig, "aspects": aspects},
            )
            logger.write_text("agent2/enhanced_requirement_text.txt", enhanced_text)

        return enhanced_text
    
    
    