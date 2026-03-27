"""
apps_controller/agent1_questions.py

Agent1:
- generate requirement questions + dedicated question-list test cases
- generate gold answers
- judge alignment between Agent3 answers and gold (correct/incorrect + why/difference + summary_missing_knowledge)
- update question template based on:
    - previous round incorrect questions as BASE
    - Agent4 mask-recovery evaluation feedback
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from llm.client import LLMClient
from utils.output_logger import OutputLogger

from apps_controller.json_utils import extract_json_loose, ensure_list
from apps_controller.prompts import (
    prompt_agent1_generate_questions,
    prompt_agent1_update_questions,          
    prompt_agent1_generate_gold_answers,
    prompt_judge_agent3_answers,             
)


@dataclass
class Agent1AppsConfig:
    max_questions: int = 20
    num_question_test_cases: int = 6  # dedicated for question-list sanity check


class Agent1AppsRunner:
    def __init__(self, llm: LLMClient, cfg: Agent1AppsConfig):
        self.llm = llm
        self.cfg = cfg

    # -----------------------------
    # helpers
    # -----------------------------
    def _normalize_questions(
        self,
        questions: List[Any],
        max_questions: int,
    ) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        seen = set()
        for i, q in enumerate(questions or []):
            if not isinstance(q, dict):
                continue
            qid = str(q.get("qid") or f"Q{i+1}").strip()
            if not qid or qid in seen:
                continue
            qq = str(q.get("question") or "").strip()
            if not qq:
                continue
            seen.add(qid)
            out.append({"qid": qid, "question": qq})
            if len(out) >= max_questions:
                break
        return out

    def _normalize_test_cases(self, tc_obj: Any) -> Dict[str, List[str]]:
        """
        Expected:
          {"inputs":[...], "outputs":[...]}
        """
        if not isinstance(tc_obj, dict):
            return {"inputs": [], "outputs": []}
        ins = ensure_list(tc_obj.get("inputs"))
        outs = ensure_list(tc_obj.get("outputs"))
        ins = [str(x) for x in ins if isinstance(x, (str, int, float))]
        outs = [str(x) for x in outs if isinstance(x, (str, int, float))]
        # keep aligned length
        n = min(len(ins), len(outs))
        return {"inputs": ins[:n], "outputs": outs[:n]}

    # -----------------------------
    # round-1 generation
    # -----------------------------
    def generate_questions_and_tests(
        self,
        task_prompt: str,
        aligned_requirement: str,
        starter_code: str,
        logger: Optional[OutputLogger] = None,
    ) -> List[Dict[str, str]]:
        """
        Round-1: generate initial question list + dedicated question-list test cases.
        """
        messages = prompt_agent1_generate_questions(
            task_prompt=task_prompt,
            aligned_requirement=aligned_requirement,
            starter_code=starter_code,
            max_questions=self.cfg.max_questions,
            num_question_test_cases=self.cfg.num_question_test_cases,
        )
        if logger:
            logger.write_prompt_bundle("agent1/A_questions_prompt.txt", messages)
        file_path = Path(logger.root) / "agent1" / "A_questions_raw.txt"
        if file_path.is_file() and file_path.stat().st_size > 0:
            raw = file_path.read_text(encoding="utf-8")
        else: raw = self.llm.chat(messages)
        data = extract_json_loose(raw, default={})

        questions = self._normalize_questions(
            ensure_list(data.get("questions")),
            max_questions=self.cfg.max_questions,
        )

        if logger:
            logger.write_text("agent1/A_questions_raw.txt", raw)
            logger.write_json(
                "agent1/A_questions.json",
                {"questions": questions},
            )
        return questions

    # -----------------------------
    # later-round update
    # -----------------------------
    def update_questions(
        self,
        task_prompt: str,
        base_questions: List[Dict[str, str]],
        agent4_mask_eval: Dict[str, Any],
        aligned_requirement: str,
        starter_code: str,
        masked_text,
        logger: Optional[OutputLogger] = None,
    ) -> List[Dict[str, str]]:
        """
        Later rounds:
        - base_questions are previous round INCORRECT questions; must be kept.
        - use Agent4 mask-recovery evaluation feedback to add more questions.
        - return the FULL updated list (base + new), NOT monotonic append.
        """
        messages = prompt_agent1_update_questions(
            task_prompt=task_prompt,
            base_questions=base_questions,
            agent4_mask_eval=agent4_mask_eval,
            aligned_requirement=aligned_requirement,
            starter_code=starter_code,
            max_questions=self.cfg.max_questions,
            masked_text=masked_text,
        )
        if logger:
            logger.write_prompt_bundle("agent1/A_update_questions_prompt.txt", messages)
        file_path = Path(logger.root) / "agent1" / "D_update_questions_raw.txt"
        if file_path.is_file() and file_path.stat().st_size > 0:
            raw = file_path.read_text(encoding="utf-8")
        else: raw = self.llm.chat(messages)
        data = extract_json_loose(raw, default={})
        model_questions = self._normalize_questions(
            ensure_list(data.get("questions")),
            max_questions=self.cfg.max_questions,
        )
        # --------- NEW: merge base_questions back (must keep) ----------
        base_norm = self._normalize_questions(base_questions, max_questions=self.cfg.max_questions)

        def _next_qid(existing_qids: set) -> str:
            # Generate Q<number> not in existing_qids
            i = 1
            while f"Q{i}" in existing_qids:
                i += 1
            return f"Q{i}"

        base_by_qid = {q.get("qid"): q for q in base_norm if q.get("qid")}
        existing_qids = set(base_by_qid.keys())

        merged: List[Dict[str, str]] = []
        merged.extend(base_norm)  # base must be kept and placed first

        for q in model_questions:
            qid = q.get("qid")
            if not qid:
                qid = _next_qid(existing_qids)
                q = {**q, "qid": qid}
                existing_qids.add(qid)
                merged.append(q)
                continue

            if qid in base_by_qid:
                # conflict: keep base version, but still keep this new question by re-qid
                new_qid = _next_qid(existing_qids)
                existing_qids.add(new_qid)
                merged.append({**q, "qid": new_qid})
            else:
                existing_qids.add(qid)
                merged.append(q)

            if len(merged) >= self.cfg.max_questions:
                break

        # Final normalize to enforce schema/order constraints (and max_questions)
        updated = self._normalize_questions(merged, max_questions=self.cfg.max_questions)
        # -------------------------------------------------------------

        if logger:
            logger.write_text("agent1/D_update_questions_raw.txt", raw)
            logger.write_json("agent1/D_questions_updated.json", {"questions": updated})
        return updated

    # -----------------------------
    # gold answers
    # -----------------------------
    def generate_gold_answers(
        self,
        task_prompt: str,
        questions: List[Dict[str, str]],
        aligned_requirement: str,
        starter_code: str,
        logger: Optional[OutputLogger] = None,
    ) -> List[Dict[str, str]]:
        messages = prompt_agent1_generate_gold_answers(
            task_prompt=task_prompt,
            questions=questions,
            question=aligned_requirement,
            starter_code=starter_code,
        )
        if logger:
            logger.write_prompt_bundle("agent1/B_gold_prompt.txt", messages)
        file_path = Path(logger.root) / "agent1" / "B_gold_raw.txt"
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
            logger.write_text("agent1/B_gold_raw.txt", raw)
            logger.write_json("agent1/B_gold.json", {"answers": out})
        return out

    # -----------------------------
    # judge Agent3 answers vs gold
    # -----------------------------
    def judge_agent3_answers(
        self,
        task_prompt: str,
        requirement: str,
        questions: List[Dict[str, str]],
        gold_answers: List[Dict[str, str]],
        agent3_answers: List[Dict[str, str]],
        logger: Optional[OutputLogger] = None,
    ) -> Dict[str, Any]:
        """
        Output format:
        {
          "results":[{"qid":"...","verdict":"correct|incorrect","why":"...","difference":""},...],
          "summary_missing_knowledge":"..."
        }
        """
        messages = prompt_judge_agent3_answers(
            task_prompt=task_prompt,
            requirement=requirement,
            questions=questions,
            gold_answers=gold_answers,
            agent3_answers=agent3_answers,
        )
        if logger:
            logger.write_prompt_bundle("agent1/C_judge_prompt.txt", messages)
        file_path = Path(logger.root) / "agent1" / "C_judge_raw.txt"
        if file_path.is_file() and file_path.stat().st_size > 0:
            raw = file_path.read_text(encoding="utf-8")
        else: raw = self.llm.chat(messages)
        data = extract_json_loose(raw, default={})

        results = ensure_list(data.get("results"))
        qids = {q["qid"] for q in questions}

        out_results: List[Dict[str, str]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            qid = str(item.get("qid") or "").strip()
            if qid not in qids:
                continue
            verdict = str(item.get("verdict") or "incorrect").strip().lower()
            if verdict not in {"correct", "incorrect"}:
                verdict = "incorrect"
            out_results.append({
                "qid": qid,
                "verdict": verdict,
                "why": str(item.get("why") or "").strip(),
                "difference": str(item.get("difference") or "").strip(),
            })

        out = {
            "results": out_results,
            "summary_missing_knowledge": str(data.get("summary_missing_knowledge") or "").strip(),
        }

        if logger:
            logger.write_text("agent1/C_judge_raw.txt", raw)
            logger.write_json("agent1/C_judge.json", out)
        return out