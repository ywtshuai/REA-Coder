"""
apps_controller/code_generation_api.py

Orchestrate the APPS multi-agent pipeline with iterative requirement alignment.

NEW PIPELINE (per user's method):
- Two disjoint testcase categories:
  (1) Agent1 "question-list test cases" (sanity-check only; NOT used for stopping criteria)
  (2) Stop-criteria test cases + (optional) public_test_cases (used for iteration stopping)

Agents:
- Agent1: generate questions (+ question-list testcases on round-1), generate gold answers, judge Agent3 answers,
          update questions (base=incorrect questions + Agent4 mask-recovery eval)
- Agent2: summarize ONLY incorrect judged items into aspect-based enhanced requirement TEXT
- Agent3: answer questions, generate code using enhanced requirement TEXT, fill masked spans
- Agent4: mask enhanced requirement, evaluate mask recovery (NO test-failure-info based feedback)

Iteration rule:
- If stop-criteria (stop cases + public cases) fails: next round base_requirement becomes THIS round enhanced requirement.
- In the same round:
    * QA (Agent3 answers / gold / judge) uses base_requirement (= previous enhanced)
    * Masking/eval uses current enhanced requirement (full)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from apps_controller.agent1_questions import Agent1AppsRunner, Agent1AppsConfig
from apps_controller.agent2_requirements import Agent2AppsRunner, Agent2AppsConfig
from apps_controller.agent3_codegen import Agent3AppsRunner, Agent3AppsConfig
from apps_controller.agent4_feedback import Agent4AppsRunner, Agent4AppsConfig
from apps_controller.apps_evaluator import evaluate_python
from apps_controller.types import ProblemData, IterationState
from llm.client import LLMClient, LLMConfig
from utils.output_logger import OutputLogger


def _build_llm(model_env: str, *, temperature: float, max_tokens: int) -> LLMClient:
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model = os.environ.get(model_env, "deepseek-chat")
    return LLMClient(
        LLMConfig(
            provider="openai_compatible",
            base_url=base_url,
            api_key_env="DEEPSEEK_API_KEY",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items or []:
        xx = str(x).strip()
        if not xx:
            continue
        if xx in seen:
            continue
        seen.add(xx)
        out.append(xx)
    return out


def _load_state(state_dir: Path, problem_id: str) -> IterationState:
    p = Path(state_dir) / f"{problem_id}.json"
    if not p.exists():
        return IterationState()
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return IterationState(
            question_template=obj.get("question_template") or [],
            last_incorrect_qids=obj.get("last_incorrect_qids") or [],
            requirement_text=obj.get("requirement_text") or "",
            pending_mask_gap_blocks_text=obj.get("pending_mask_gap_blocks_text") or "",
            dynamic_stop_test_cases=obj.get("dynamic_stop_test_cases") or {"inputs": [], "outputs": []},
            dynamic_stop_streak=obj.get("dynamic_stop_streak") or {},
            pending_masked_text=obj.get("pending_masked_text") or ""
        )
    except Exception:
        return IterationState()


def _save_state(state_dir: Path, problem_id: str, state: IterationState) -> None:
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    p = Path(state_dir) / f"{problem_id}.json"
    p.write_text(
        json.dumps(
            {
                "question_template": state.question_template,
                "last_incorrect_qids": state.last_incorrect_qids,
                "requirement_text": state.requirement_text,
                "pending_mask_gap_blocks_text": state.pending_mask_gap_blocks_text,
                "dynamic_stop_test_cases": state.dynamic_stop_test_cases,
                "dynamic_stop_streak": state.dynamic_stop_streak,
                "pending_masked_text": state.pending_masked_text,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_stop_test_cases(problem_id: str) -> Optional[Dict[str, List[str]]]:
    """
    Pre-generated stop-criteria test cases live in:
      test_case/<problem_id>  (either a JSON file or a directory containing a JSON)
    Expected JSON format: {"inputs":[...], "outputs":[...]}  (same as public_test_cases)
    """
    base = Path(f"test_case_code_contest/{problem_id}_test_case")
    candidates: List[Path] = []
    if base.is_file():
        candidates.append(base)
    if base.is_dir():
        candidates += [
            base / "test_cases.json",
            base / "stop_test_cases.json",
            base / "cases.json",
        ]
        candidates += list(base.glob("*.json"))

    for p in candidates:
        if not p.exists():
            continue
        try:
            # print(777)
            obj = json.loads(p.read_text(encoding="utf-8"))
            ins = obj.get("inputs") or []
            outs = obj.get("outputs") or []
            if isinstance(ins, list) and isinstance(outs, list) and len(ins) > 0 and len(ins) == len(outs):
                return {"inputs": [str(x) for x in ins], "outputs": [str(x) for x in outs]}
        except Exception:
            continue
    return None


def _merge_test_cases(
        a: Optional[Dict[str, List[str]]],
        b: Optional[Dict[str, List[str]]],
) -> Optional[Dict[str, List[str]]]:
    if not a and not b:
        return None
    ins: List[str] = []
    outs: List[str] = []
    for tc in (a, b):
        if not tc:
            continue
        ins += list(tc.get("inputs") or [])
        outs += list(tc.get("outputs") or [])
    n = min(len(ins), len(outs))
    return {"inputs": ins[:n], "outputs": outs[:n]}


def _canon_inp(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    return "\n".join([ln.strip() for ln in s.splitlines()]).strip()


def _tc_signature(inp: str, out: str) -> str:
    # IMPORTANT: streak key only depends on input
    return f"INP::{_canon_inp(inp)}::OUT::{_canon_inp(out)}"


def _dedupe_by_io(tc: Dict[str, List[str]]) -> Dict[str, List[str]]:
    ins = tc.get("inputs") or []
    outs = tc.get("outputs") or []
    seen = set()
    new_in, new_out = [], []
    for i, o in zip(ins, outs):
        raw_i = str(i)
        raw_o = str(o)
        key = _tc_signature(raw_i, raw_o)
        if key in seen:
            continue
        seen.add(key)
        new_in.append(raw_i)
        new_out.append(raw_o)
    return {"inputs": new_in, "outputs": new_out}


def _filter_dynamic_stop_cases(
        dynamic_tc: Dict[str, List[str]],
        streaks: Dict[str, int],
        min_streak: int = -2,
        max_streak: int = 2,  # keep if streak <=2; drop if >=3
) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    ins = dynamic_tc.get("inputs") or []
    outs = dynamic_tc.get("outputs") or []
    keep_in, keep_out = [], []
    confident_in, confident_out = [], []
    for i, o in zip(ins, outs):
        key = _tc_signature(i, o)  # input-only key
        streak = int(streaks.get(key, 0))
        if min_streak <= streak <= max_streak:
            keep_in.append(i)
            keep_out.append(o)
        elif streak > max_streak:
            confident_in.append(i)
            confident_out.append(o)
    return {"inputs": keep_in, "outputs": keep_out}, {"inputs": confident_in, "outputs": confident_out}


@dataclass
class AppsCodeGenAPI:
    task_prompt: str
    iter_state_dir: Path
    max_iters: int = 10
    show_public_tests: bool = True

    # evaluation params
    timeout: float = 10.0
    eval_workers: int = 16

    # NEW: run all_test_cases every iteration (for monitoring)
    run_all_tests_each_iter: bool = True
    all_tests_timeout: Optional[float] = None  # None => use timeout
    all_tests_workers: Optional[int] = None  # None => use eval_workers

    dynamic_stop_gen_k: int = 50
    dynamic_stop_confident_k: int = 50

    def __post_init__(self):
        llm_a = _build_llm("MODEL_A", temperature=0.2, max_tokens=8192)
        llm_b = _build_llm("MODEL_B", temperature=0.2, max_tokens=8192)
        llm_c = _build_llm("MODEL_C", temperature=0.0, max_tokens=8192)

        self.agent1 = Agent1AppsRunner(llm=llm_a, cfg=Agent1AppsConfig())
        self.agent2 = Agent2AppsRunner(llm=llm_b, cfg=Agent2AppsConfig())
        self.agent3 = Agent3AppsRunner(llm=llm_c, cfg=Agent3AppsConfig())
        self.agent4 = Agent4AppsRunner(llm=llm_a, cfg=Agent4AppsConfig())

    @staticmethod
    def base_constraints(mode: str = "stdio") -> List[str]:
        if mode == "call":
            io_rule = (
                "Use Python3. This is call-based: do NOT read stdin or write stdout; "
                "implement only the starter_code API."
            )
        else:
            io_rule = "Use Python3. Read from stdin and write to stdout exactly as specified."
        return [
            io_rule,
            "Do not import or use: os, sys, subprocess, socket, multiprocessing, threading, signal, resource.",
            "Do not call: exec, eval, __import__, open, fork, system, popen.",
            "Avoid interactive prompts; no extra prints.",
            "Must handle edge cases and maximum constraints efficiently.",
        ]

    def run_one(
            self,
            problem: ProblemData,
            output_dir: str,
            max_iters: Optional[int] = None,
    ) -> Tuple[str, dict, dict]:
        """
        Returns:
          (final_code, final_eval_summary, final_state_dict)
        """
        max_iters = self.max_iters if max_iters is None else max_iters

        # rebuild state
        state_path = Path(self.iter_state_dir) / f"{problem.problem_id}.json"
        state_path.unlink(missing_ok=True)

        state = _load_state(self.iter_state_dir, problem.problem_id)

        # Ensure base requirement exists (Round-1 base is original problem statement)
        if not (state.requirement_text or "").strip():
            state.requirement_text = problem.question
            _save_state(self.iter_state_dir, problem.problem_id, state)

        # Stop criteria testcases = stop_test_cases + (optional) public_test_cases
        stop_cases = _load_stop_test_cases(problem.problem_id)
        fixed_stop_cases = _load_stop_test_cases(problem.problem_id)
        # print(f"{problem.problem_id}:{stop_cases}")
        # NOTE: public_cases is used as the ONLY stopping criterion (per new request)
        public_cases = problem.public_test_cases if problem.public_test_cases else None

        # Keep stop_eval_cases as a non-stopping monitoring pool (optional)
        stop_eval_cases = _merge_test_cases(stop_cases, public_cases)

        # If stop cases are missing, fallback to dataset-provided eval cases (best effort)
        if not stop_eval_cases:
            fallback = problem.best_eval_test_cases()
            stop_eval_cases = fallback if fallback else None

        last_code = ""
        last_eval: Dict[str, Any] = {"passed": False}
        #fixed_stop_cases = stop_eval_cases

        # ---- NEW: initialize unified stop pool with fixed stop cases (once) ----
        if fixed_stop_cases and not (state.dynamic_stop_test_cases.get("inputs") or []):
            state.dynamic_stop_test_cases = {
                "inputs": list(fixed_stop_cases.get("inputs") or []),
                "outputs": list(fixed_stop_cases.get("outputs") or []),
            }
            n = min(len(state.dynamic_stop_test_cases["inputs"]), len(state.dynamic_stop_test_cases["outputs"]))
            state.dynamic_stop_test_cases["inputs"] = state.dynamic_stop_test_cases["inputs"][:n]
            state.dynamic_stop_test_cases["outputs"] = state.dynamic_stop_test_cases["outputs"][:n]
            _save_state(self.iter_state_dir, problem.problem_id, state)

        for it in range(1, max_iters + 1):
            iter_dir = str(Path(output_dir) / f"iter_{it}")
            logger = OutputLogger(iter_dir, problem.problem_id)

            base_requirement = (state.requirement_text or "").strip() or problem.question

            logger.write_text("problem/original_question.txt", problem.question)
            logger.write_text("problem/base_requirement.txt", base_requirement)
            logger.write_text("problem/starter_code.txt", problem.starter_code or "")
            logger.write_json("problem/meta.json",
                              {"problem_id": problem.problem_id, "difficulty": problem.difficulty, "url": problem.url})

            # -----------------------
            # Agent1 round-1 init: questions + question-list testcases (once)
            # -----------------------
            if not state.question_template:
                qs = self.agent1.generate_questions_and_tests(
                    self.task_prompt,
                    aligned_requirement=base_requirement,
                    starter_code=problem.starter_code or "",
                    logger=logger,
                )
                state.question_template = qs
                _save_state(self.iter_state_dir, problem.problem_id, state)

            questions = state.question_template

            # -----------------------
            # Agent3 answers questions (using base_requirement = previous enhanced requirement)
            # -----------------------
            a3_answers = self.agent3.answer_questions(
                self.task_prompt,
                questions,
                base_requirement,
                problem.starter_code or "",
                logger=logger,
            )

            # -----------------------
            # Agent1 gold + judge (STRICT correct/incorrect format)
            # -----------------------
            gold = self.agent1.generate_gold_answers(
                self.task_prompt,
                questions,
                base_requirement,
                problem.starter_code or "",
                logger=logger,
            )

            judge_obj = self.agent1.judge_agent3_answers(
                self.task_prompt,
                base_requirement,
                questions,
                gold,
                a3_answers,
                logger=logger,
            )

            judge_results = judge_obj.get("results") or []
            incorrect_results = [r for r in judge_results if str(r.get("verdict")).lower() == "incorrect"]
            incorrect_qids = [r.get("qid") for r in incorrect_results if r.get("qid")]

            logger.write_json("agent1/incorrect_qids.json", {"incorrect_qids": incorrect_qids})

            # -----------------------
            # Agent2: build FULL enhanced requirement TEXT for this round
            # -----------------------
            extra_blocks = []
            if (state.pending_mask_gap_blocks_text or "").strip():
                extra_blocks.append(state.pending_mask_gap_blocks_text.strip())

            enhanced_requirement_text = self.agent2.build_enhanced_requirement_text(
                task_prompt=self.task_prompt,
                original_requirement=base_requirement,
                incorrect_results=incorrect_results,
                questions=questions,
                gold_answers=gold,
                agent3_answers=a3_answers,
                extra_gap_blocks_text="\n\n".join(extra_blocks).strip(),
                base_problem_statement=problem.question,  # <-- NEW
                masked_text=state.pending_masked_text,
                logger=logger,
            )

            logger.write_text("agent2/enhanced_requirement_text.txt", enhanced_requirement_text)
            state.pending_mask_gap_blocks_text = ""

            # -----------------------
            # Agent3: generate code using enhanced_requirement_text as the "problem statement"
            # (base constraints only; enhanced requirement already embedded in text)
            # -----------------------
            mode = "call" if (problem.starter_code or "").strip() else "stdio"
            reqs = _dedupe_keep_order(self.base_constraints(mode))

            code = self.agent3.generate_code(
                self.task_prompt,
                enhanced_requirement_text,
                problem.starter_code or "",
                reqs,
                public_examples=public_cases,
                logger=logger,
            )

            last_code = code

            # -----------------------
            # NEW: generate dynamic stop test cases after round-1
            # -----------------------
            # if it >= 2 and self.dynamic_stop_gen_k > 0:
            #     new_dyn = self.agent2.generate_stop_test_cases(
            #         task_prompt=self.task_prompt,
            #         aligned_requirement=base_requirement,
            #         starter_code=problem.starter_code or "",
            #         k=self.dynamic_stop_gen_k,
            #         logger=logger,
            #     )

            #     new_ins = [str(x) for x in (new_dyn.get("inputs") or [])]
            #     new_outs = [str(x) for x in (new_dyn.get("outputs") or [])]
            #     n = min(len(new_ins), len(new_outs))
            #     new_ins, new_outs = new_ins[:n], new_outs[:n]

            #     state.dynamic_stop_test_cases["inputs"] += new_ins
            #     state.dynamic_stop_test_cases["outputs"] += new_outs
            #     state.dynamic_stop_test_cases = _dedupe_by_io(state.dynamic_stop_test_cases)
            #     _save_state(self.iter_state_dir, problem.problem_id, state)

            #     eval_pool_inputs = state.dynamic_stop_test_cases.get("inputs") or []
            #     eval_pool_outputs = state.dynamic_stop_test_cases.get("outputs") or []
            #     eval_sig_set = set(
            #         _tc_signature(i, o)
            #         for i, o in zip(eval_pool_inputs, eval_pool_outputs)
            #         if str(i).strip()
            #     )
            #     #print(12983789127389127389123)
            #     eval_res = evaluate_python(
            #         code,
            #         state.dynamic_stop_test_cases,
            #         timeout=self.timeout,
            #         workers=self.eval_workers
            #     )

            #     for case in eval_res.cases:
            #         inp_raw = case.input_data
            #         out_raw = case.expected
            #         sig = _tc_signature(inp_raw, out_raw)
            #         if sig not in eval_sig_set:
            #             continue

            #         if case.status == "AC":
            #             state.dynamic_stop_streak[sig] = int(state.dynamic_stop_streak.get(sig, 0)) + 1
            #         else:
            #             state.dynamic_stop_streak[sig] = int(state.dynamic_stop_streak.get(sig, 0)) - 1

            # unconfident_test_cases, confident_test_cases = _filter_dynamic_stop_cases(
            #     state.dynamic_stop_test_cases,
            #     state.dynamic_stop_streak
            # )

            # state.dynamic_stop_test_cases = _merge_test_cases(unconfident_test_cases, confident_test_cases)
            # state.dynamic_stop_test_cases = _dedupe_by_io(state.dynamic_stop_test_cases)

            # kept_sig_set = set(
            #     _tc_signature(i, o)
            #     for i, o in zip((state.dynamic_stop_test_cases.get("inputs") or []),
            #                     (state.dynamic_stop_test_cases.get("outputs") or []))
            #     if str(i).strip()
            # )
            # state.dynamic_stop_streak = {
            #     k: v
            #     for k, v in state.dynamic_stop_streak.items()
            #     if k in kept_sig_set
            # }

            # _save_state(self.iter_state_dir, problem.problem_id, state)

            # -----------------------
            # NEW STOP RULE:
            # Stop iteration as long as PUBLIC test cases pass.
            # Then run all_test_cases once to determine final correctness.
            # -----------------------

            # If public cases exist, they are the only stopping gate.
            # If missing, fall back to previous stop_eval_cases behavior (best effort).
            stop_gate_cases = public_cases if (public_cases and public_cases.get("inputs") and public_cases.get("outputs")) else stop_eval_cases

            stop_gate_cases = _dedupe_by_io(stop_gate_cases) if stop_gate_cases else stop_gate_cases
            eval_res = evaluate_python(
                code,
                stop_gate_cases,
                timeout=self.timeout,
                workers=self.eval_workers
            )
            last_eval = eval_res.summary
            logger.write_json("eval_summary.json", eval_res.summary)

            # -----------------------
            # NEW: Evaluate on all_test_cases EVERY iteration (if available)
            # -----------------------
            all_tests_summary = None
            passed_all = None
            if self.run_all_tests_each_iter:
                if problem.all_test_cases and problem.all_test_cases.get("inputs") and problem.all_test_cases.get(
                        "outputs"):
                    all_timeout = self.all_tests_timeout if self.all_tests_timeout is not None else self.timeout
                    all_workers = self.all_tests_workers if self.all_tests_workers is not None else self.eval_workers
                    all_eval = evaluate_python(
                        code,
                        problem.all_test_cases,
                        timeout=all_timeout,
                        workers=all_workers,
                    )
                    all_tests_summary = all_eval.summary
                    passed_all = bool(all_eval.summary.get("passed", False))
                    logger.write_json("eval_all_test_summary.json", all_eval.summary)

                    # all_test_details = []
                    # for case in all_eval.cases:
                    #     all_test_details.append({
                    #         "input": case.input_data,
                    #         "expected": case.expected,
                    #         "status": case.status,
                    #         "stdout": case.stdout,
                    #         "stderr": case.stderr,
                    #     })
                    # logger.write_json("eval_all_test_details.json", {"cases": all_test_details})

            # Stitch a combined per-iter summary for easy tracking
            iter_summary = dict(eval_res.summary)
            iter_summary["passed_stop"] = bool(eval_res.summary.get("passed", False))
            iter_summary["passed_all"] = passed_all
            if all_tests_summary is not None:
                iter_summary["all_test_cases_summary"] = all_tests_summary

            # Optional: write a single merged json per iter (nice for dashboards)
            logger.write_json("eval_merged_summary.json", iter_summary)

            # Keep last_eval as merged so outer loop returns richer info on failure too
            last_eval = iter_summary

            if eval_res.passed:
                # PUBLIC passed => stop iterating immediately.
                # Still run all_test_cases once to determine correctness.

                # Persist successful enhanced requirement as new state
                state.requirement_text = enhanced_requirement_text
                state.last_incorrect_qids = incorrect_qids
                _save_state(self.iter_state_dir, problem.problem_id, state)

                final_summary = dict(eval_res.summary)
                final_summary["passed_public"] = True  # explicit

                passed_all = None
                all_tests_summary = None
                if problem.all_test_cases and problem.all_test_cases.get("inputs") and problem.all_test_cases.get("outputs"):
                    all_eval = evaluate_python(
                        code,
                        problem.all_test_cases,
                        timeout=self.timeout,
                        workers=self.eval_workers,
                    )
                    all_tests_summary = all_eval.summary
                    passed_all = bool(all_eval.summary.get("passed", False))
                    logger.write_json("eval_all_test_summary.json", all_eval.summary)

                if all_tests_summary is not None:
                    final_summary["all_test_cases_summary"] = all_tests_summary
                    final_summary["passed_all"] = passed_all
                    final_summary["passed"] = passed_all  # FINAL correctness is based on all_test_cases
                else:
                    final_summary["passed_all"] = None
                    # If no all_test_cases, treat as passed (since public passed and that's our only gate)
                    final_summary["passed"] = True

                return code, final_summary, {
                    "question_template": state.question_template,
                    "last_incorrect_qids": state.last_incorrect_qids,
                    "requirement_text": state.requirement_text
                }
            # -----------------------
            # Failure: Agent4 mask (on THIS round full enhanced requirement)
            #         Agent3 fill masked spans
            #         Agent4 evaluate mask recovery (NO test-based feedback)
            #         Agent1 update questions (base=incorrect questions + mask eval)
            #         Next-round base_requirement becomes THIS enhanced requirement
            # -----------------------
            masked = self.agent4.mask_requirement(
                enhanced_requirement_text,
                logger=logger,
            )
            masked_text = str(masked.get("masked_text") or "")

            fills = self.agent3.fill_masked_question(
                self.task_prompt,
                masked_text,
                problem.starter_code or "",
                code,
                error_info="",  # per new method: Agent4 feedback should NOT rely on test failure info
                logger=logger,
            )

            mask_eval = self.agent4.evaluate_mask_recovery(
                task_prompt=self.task_prompt,
                original_requirement=enhanced_requirement_text,
                masked_requirement=masked_text,
                agent3_fills=fills,
                logger=logger,
            )
            mask_gap_blocks_text = self.agent2.build_mask_gap_blocks(
                masked_spans=masked.get("spans") or [],
                agent3_fills=fills,
                mask_eval=mask_eval,
            )
            logger.write_text("agent2/pending_mask_gaps.txt", mask_gap_blocks_text)

            state.pending_mask_gap_blocks_text = mask_gap_blocks_text
            state.pending_masked_text = masked_text

            # Update questions: keep incorrect questions as base
            incorrect_set = set(incorrect_qids)
            base_incorrect_questions = [q for q in questions if q.get("qid") in incorrect_set]

            updated_questions = self.agent1.update_questions(
                task_prompt=self.task_prompt,
                base_questions=base_incorrect_questions,
                agent4_mask_eval=mask_eval,
                aligned_requirement=enhanced_requirement_text,  # current round full enhanced requirement
                starter_code=problem.starter_code or "",
                masked_text=masked_text,
                logger=logger,
            )

            # State update per rule:
            # next round base requirement = this round enhanced requirement
            state.question_template = updated_questions
            state.last_incorrect_qids = incorrect_qids
            state.requirement_text = enhanced_requirement_text
            _save_state(self.iter_state_dir, problem.problem_id, state)

        if problem.all_test_cases and problem.all_test_cases.get("inputs") and problem.all_test_cases.get("outputs"):
            try:
                all_timeout = self.all_tests_timeout if self.all_tests_timeout is not None else self.timeout
                all_workers = self.all_tests_workers if self.all_tests_workers is not None else self.eval_workers
                all_eval = evaluate_python(
                    last_code,
                    problem.all_test_cases,
                    timeout=all_timeout,
                    workers=all_workers,
                )
                all_tests_summary = all_eval.summary

                # Merge into last_eval for final return
                if not isinstance(last_eval, dict):
                    last_eval = {"passed": False}
                last_eval = dict(last_eval)
                last_eval["all_test_cases_summary"] = all_tests_summary
                last_eval["passed_all"] = bool(all_tests_summary.get("passed", False))
                # Make "passed" reflect FINAL correctness when all_test_cases exists
                last_eval["passed"] = last_eval["passed_all"]
            except Exception as e:
                # Best-effort: don't crash final return
                if not isinstance(last_eval, dict):
                    last_eval = {"passed": False}
                last_eval = dict(last_eval)
                last_eval["all_test_cases_summary"] = {"passed": False, "error": str(e)}
                last_eval["passed_all"] = False
                last_eval["passed"] = False

        return last_code, last_eval, {
            "question_template": state.question_template,
            "last_incorrect_qids": state.last_incorrect_qids,
            "requirement_text": state.requirement_text
        }