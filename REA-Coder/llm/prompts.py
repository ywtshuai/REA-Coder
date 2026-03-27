# 功能：所有Agent使用的提示词模板
# 作用：
# - Agent1: 需求问题生成和回答模板
# - Agent2: 仓库上下文总结模板  
# - Agent3: 代码补丁生成模板
# - Agent4: 事后需求发现模板
# - 统一的系统消息和格式约束
"""
llm/prompts.py

All prompt templates used by multi-agent repo-level requirement alignment.
Design goals:
- Repo-level (not single-function)
- Explicit requirement questions (Q/A)
- DSL is monotonic (only add, never delete)
- Agent4 supports mask -> demask style post-hoc requirement discovery
"""

from typing import List, Dict


# -----------------------------
# Basic message helpers
# -----------------------------

def sys_msg(content: str) -> Dict[str, str]:
    return {"role": "system", "content": content}


def user_msg(content: str) -> Dict[str, str]:
    return {"role": "user", "content": content}


# -----------------------------
# Global system instruction
# -----------------------------

SYSTEM_BASE = """You are a senior software engineer and program repair agent.
You operate at REPOSITORY LEVEL (not just a single function).
You must strictly follow instructions and output ONLY the requested format.
Avoid speculation. Be precise and minimal.
"""


# =============================
# Agent 1: Requirement gap detection (pre-code)
# =============================

def prompt_agent1_generate_qa(
    problem_statement: str,
    question_rules: List[str],
) -> str:
    """
    Agent1-Stage1:
    Generate requirement clarification questions + gold answers.
    These questions will later be used to test whether an LLM
    is aligned with the requirements.
    """

    rules_block = "\n".join([f"- {r}" for r in question_rules]) if question_rules else "- (No explicit rules provided)"

    return f"""# TASK
Design a checklist of requirement-clarifying questions AND their correct answers.
These questions are used to detect requirement misalignment BEFORE code generation.

# PROBLEM_STATEMENT
{problem_statement}

# QUESTION_GENERATION_RULES
{rules_block}

# REQUIREMENTS
- Focus on functional behavior, constraints, edge cases, APIs, data flow, and repo-level assumptions
- Questions must be answerable from the problem statement
- Answers must be concise, deterministic, and testable
- DO NOT include any repository code
- DO NOT include implementation details

# OUTPUT_FORMAT (JSON ONLY)
```json
{{
  "qa": [
    {{
      "question": "…",
      "answer": "…",
      "category": "functional | constraint | api | data_flow | edge_case | repo_assumption"
    }}
  ]
}}
```

- Produce 3–8 Q/A pairs.
"""


def prompt_answer_questions(
    problem_statement: str,
    questions: List[str],
) -> str:
    """
    Used by Agent1 and Agent4:
    Ask the model to answer requirement questions.
    """

    qs = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

    return f"""# TASK
Answer the following requirement questions using ONLY the given problem statement.

# PROBLEM_STATEMENT
{problem_statement}

# QUESTIONS
{qs}

# OUTPUT_FORMAT (JSON ONLY)
```json
{{
  "answers": [
    {{
      "question_index": 1,
      "answer": "..."
    }}
  ]
}}
```

- Do NOT explain your answers.
- If the problem statement is ambiguous, answer with the most reasonable interpretation.
"""

# =============================
# Agent 3: Patch generation
# =============================

def format_repo_context(
    repo_brief: str,
    glossary: str,
    retrieved_chunks: str,
) -> str:
    """
    Assemble repo-level context for code generation.
    """

    sections = []

    if repo_brief:
        sections.append(f"# REPO_BRIEF\n{repo_brief}")

    if glossary:
        sections.append(f"# DOMAIN_GLOSSARY\n{glossary}")

    if retrieved_chunks:
        sections.append(f"# PATCH_TARGET_FILES_FULLTEXT\n{retrieved_chunks}")

    return "\n\n".join(sections)


def prompt_agent3_generate_patch(problem_statement: str, enhanced_dsl: str, repo_context: str) -> str:
    return f"""You are an automated code-editing system running in a real git checkout.

You MUST output ONLY a single JSON object (no markdown, no explanation, no extra whitespace before '{{', no trailing text).
If you output anything other than valid JSON, the result will be discarded.

# PROBLEM_STATEMENT
{problem_statement}

# ENHANCED_DSL
{enhanced_dsl}

# REPO_CONTEXT
{repo_context}

# CRITICAL: MINIMAL CHANGES REQUIRED
⚠️ **MOST IMPORTANT RULE: Make MINIMAL, SURGICAL changes only!**
- DO NOT refactor or rewrite entire functions/files
- DO NOT change function signatures or add new functions unless absolutely necessary
- DO NOT restructure code organization
- ONLY modify the specific lines that fix the bug
- Think: "What is the SMALLEST change that fixes the issue?"
- Example: If a bug is "change = 1 to = right", ONLY change that one line, keep everything else identical

# HARD RULES
- Do NOT output a git diff.
- Output ONLY JSON that matches the schema below.
- Each edit MUST target an existing file path in the repository.
  - DO NOT create new files.
  - DO NOT use paths like "a/..." or "b/...".
- "content" must be the FULL FILE CONTENT after your changes (not partial snippets).
- **MINIMAL CHANGES**: Only modify the exact lines necessary to fix FAIL_TO_PASS tests while keeping PASS_TO_PASS passing.
  - If the fix requires changing 1 line, change ONLY that 1 line.
  - If the fix requires changing 3 lines, change ONLY those 3 lines.
  - Do NOT rewrite entire functions or files.
- Do not include placeholders like "..." or "TODO".
- Preserve all existing code structure, comments, and formatting except for the minimal necessary changes.

# OUTPUT JSON SCHEMA (return exactly this structure)
{{
  "edits": [
    {{
      "path": "relative/path/in/repo.py",
      "content": "FULL FILE CONTENT HERE"
    }}
  ]
}}
"""


# =============================
# Agent 1 (ABC): Requirement gap detection with separate LLMs
# =============================

from typing import Dict, List

def _json_only_contract() -> str:
    return (
        "Return ONLY valid JSON. No markdown fences, no extra text.\n"
        "If you are unsure, still return JSON with best-effort fields.\n"
    )

def prompt_A_generate_questions(
    task_prompt: str,
    question_template: Dict,
    problem_statement: str,
    patch: str,
    test_patch: str,
    repo_context: str,
    pass_to_pass: str,
    fail_to_pass: str,
) -> str:
    """
    A1: generate question pack (questions only, no answers).
    """
    return f"""# ROLE
You are Agent A (Question Constructor). Your job is to produce a requirement-alignment question pack.

# TASK_PROMPT
{task_prompt}

# INPUTS
## QUESTION_TEMPLATE (JSON)
{question_template}

## PROBLEM_STATEMENT
{problem_statement}

## REPO_CONTEXT (repo tree + retrieved snippets)
{repo_context}

## TESTS
FAIL_TO_PASS:
{fail_to_pass}

PASS_TO_PASS:
{pass_to_pass}

# INSTRUCTIONS
- Produce a minimal but sufficient checklist to detect requirement misunderstanding BEFORE coding.
- Questions MUST be answerable from the problem statement + repo_context + tests.
- Cover: task type, bug/failure category, observed vs expected behavior, invariants/must-not-break, IO contract/core APIs.
- Use the template fields and QIDs when possible.

# OUTPUT_SCHEMA (JSON)
{_json_only_contract()}
{{
  "question_pack_id": "AUTO_{{instance_id}}",
  "questions": [
    {{
      "qid": "SOME_QID_FROM_TEMPLATE",
      "question": "..."
    }}
  ]
}}
"""


def prompt_A_generate_gold_answers(
    task_prompt: str,
    question_pack: Dict,
    problem_statement: str,
    patch: str,
    test_patch: str,
    repo_context: str,
    pass_to_pass: str,
    fail_to_pass: str,
) -> str:
    """
    A2: generate gold answers for the produced question pack.
    """
    return f"""# ROLE
You are Agent A (Gold Answer Author). You must provide correct, deterministic answers to the question pack.

# TASK_PROMPT
{task_prompt}

# QUESTION_PACK (JSON)
{question_pack}

# INPUTS
## PROBLEM_STATEMENT
{problem_statement}

## REPO_CONTEXT
{repo_context}

## TESTS
FAIL_TO_PASS:
{fail_to_pass}

PASS_TO_PASS:
{pass_to_pass}

# INSTRUCTIONS
- Answer each question precisely.
- Answers must be testable and should match enums/constraints if any.
- If the question requires listing APIs or invariants, be explicit.
- Do NOT include implementation details beyond what is necessary for a correct requirement answer.

# BUG_CATEGORY_ENUM (must pick EXACTLY one when task_type == "Bug fix")
[
  "Conditional / branching logic error",
  "State / context lost or not propagated",
  "Boundary / indexing error",
  "Null / missing value handling error",
  "Missing input validation or incorrect validation logic",
  "Exception type / error message not conforming to contract",
  "Type conversion / numerical computation error (precision, overflow, divide-by-zero, instability)",
  "Structure / shape / matrix construction error",
  "Semantic error in composed / nested structures",
  "Compatibility / platform difference / API semantic mismatch",
  "Concurrency / caching / resource / performance defect",
  "Unit / time / coordinate frame handling error",
  "Security vulnerability"
]

# OUTPUT_SCHEMA (JSON)
{_json_only_contract()}
{{
  "answer_key_id": "AUTO_{{instance_id}}",
  "answers": [
    {{
      "qid": "Q1_TASK_TYPE",
      "answer": {{ }}
    }}
  ]
}}
"""


def prompt_B_answer_questions(
    task_prompt: str,
    problem_statement:str,
    question_pack: Dict,
    repo_context: str,
) -> str:
    """
    B: answer the question pack (this is the "code model" role, but only answering questions here).
    """
    return f"""# ROLE
You are Agent B (Solver Model). You answer requirement questions before writing code.

# TASK_PROMPT
{task_prompt}

## PROBLEM_STATEMENT
{problem_statement}

# REPO_CONTEXT
{repo_context}

# QUESTION_PACK (JSON)
{question_pack}

# INSTRUCTIONS
- Answer each question in JSON according to its requested format.
- Follow enum constraints exactly when present.
- No explanations.

# BUG_CATEGORY_ENUM (must pick EXACTLY one when task_type == "Bug fix")
[
  "Conditional / branching logic error",
  "State / context lost or not propagated",
  "Boundary / indexing error",
  "Null / missing value handling error",
  "Missing input validation or incorrect validation logic",
  "Exception type / error message not conforming to contract",
  "Type conversion / numerical computation error (precision, overflow, divide-by-zero, instability)",
  "Structure / shape / matrix construction error",
  "Semantic error in composed / nested structures",
  "Compatibility / platform difference / API semantic mismatch",
  "Concurrency / caching / resource / performance defect",
  "Unit / time / coordinate frame handling error",
  "Security vulnerability"
]

# OUTPUT_SCHEMA (JSON)
{_json_only_contract()}
{{
  "answers": [
    {{
      "qid": "Q1_TASK_TYPE",
      "answer": {{ }}
    }}
  ]
}}
"""


def prompt_C_judge_alignment(
    task_prompt: str,
    question_pack: Dict,
    gold_answers: Dict,
    b_answers: Dict,
) -> str:
    return f"""# ROLE
You are Agent C (Strict Judge). You grade Agent B's answers against the gold answers.

# TASK_PROMPT
{task_prompt}

# QUESTION_PACK (JSON)
{question_pack}

# GOLD_ANSWERS (JSON)
{gold_answers}

# B_ANSWERS (JSON)
{b_answers}

# GRADING_RULES (STRICT, PER-QUESTION ONLY)
• Grade each question independently using ONLY:
  - that question's text (from QUESTION_PACK),
  - the corresponding gold answer for the same qid,
  - the corresponding B answer for the same qid.

• DO NOT use other questions to influence the verdict of this qid.

• verdict must be exactly one of: "correct" or "incorrect".

• If B violates schema or enum constraints for THIS qid -> incorrect.
• If B answer is missing for THIS qid -> incorrect.
• If verdict == incorrect, you MUST describe the key differences between B answer and gold answer.

# OUTPUT_SCHEMA (JSON)
Return ONLY valid JSON. No markdown fences, no extra text.
{{
  "per_question": [
    {{
      "qid": "Q1_TASK_TYPE",
      "verdict": "correct",
      "why": "...",
      "difference": ""
    }}
  ],
  "summary_missing_knowledge": [
    "..."
  ]
}}
"""

def prompt_agent3_generate_replacements(problem_statement: str, enhanced_dsl: str, repo_context: str) -> str:
    return f"""You are an bug-fixed system running in a real git checkout.

You MUST output ONLY a single JSON object (no markdown, no explanation).
If you output anything other than valid JSON, the result will be discarded.

# PROBLEM_STATEMENT
{problem_statement}

# ENHANCED_DSL
{enhanced_dsl}

# REPO_CONTEXT
{repo_context}

# GOAL
Produce minimal, surgical code changes that:
- Fix all FAIL_TO_PASS tests
- Do not break PASS_TO_PASS tests

# HARD RULES
- Do NOT output a git diff.
- Do NOT output full file contents.
- Output ONLY "replacements" operations that can be applied to existing files.
- Prefer modifying files shown in TARGET_FILES_FULLTEXT. If a path is referenced in PROBLEM_STATEMENT and exists in the repo, you may modify it.
- Each "before" MUST match an exact substring in the current file content.
- Keep each replacement SMALL (ideally a few lines).
- Prefer unique anchors: include enough surrounding context so "before" appears exactly once.
- You MUST copy-paste every "before" EXACTLY from TARGET_FILES_FULLTEXT. Do NOT retype it.
- If the exact "before" text is not present in TARGET_FILES_FULLTEXT, return {{"edits": []}}.

# OUTPUT JSON SCHEMA (exactly)
{{
  "edits": [
    {{
      "path": "relative/path/in/repo.py",
      "replacements": [
        {{
          "before": "exact text to be replaced",
          "after": "replacement text"
        }}
      ]
    }}
  ]
}}

# NOTES
- Use \\n for newlines.
- Escape quotes properly.
- Do not create new files unless necessary. If the problem really requires it, then the "before" content should be "" and the "after" content should be the content of the new file.
- If you cannot find a safe minimal replacement, return {{"edits": []}}.
"""


def _prompt_mask_problem_statement(
    problem_statement: str,
    max_masks: int,
    *,
    total_tokens: int,
    target_ratio: float,
    ratio_min: float,
    ratio_max: float,
    min_span_tokens: int,
    max_span_tokens: int,
    suggested_span_tokens: list[int],
    require_distinct_lengths: bool = True,
) -> str:
    """
    Mask rules are explicitly specified here so the LLM does NOT free-style masking.
    """
    return f"""# TASK
You will MASK the PROBLEM_STATEMENT by replacing spans with placeholders [MASK_1]..[MASK_6].

# IMPORTANT
- You MUST use EXACTLY N=6 masks: [MASK_1]..[MASK_6] (no fewer, no more).
- Each MASK span should be requirement-bearing and non-overlapping.
- Try to match these suggested per-span token lengths (best-effort):
  suggested_span_tokens={total_tokens}

# MASKING RULES (MUST FOLLOW)
1) Coverage ratio: mask about 1/3 of the content.
   - total_tokens={total_tokens}
   - target_ratio={target_ratio}
   - Acceptable masked_token_ratio in [{ratio_min}, {ratio_max}]
2) Mask content does not require continuity
3) Spans must be requirement-bearing (expected/observed behavior, constraints, IO, examples, outputs).
4) each masked span should include context to ensure the masked content is logically coherent and meaningful.
5) No overlap: spans must not overlap each other.
6) Output MUST be JSON only (no markdown, no commentary).
7) CRITICAL: In "masked_problem_statement", you MUST replace the selected spans with the corresponding [MASK_i] tokens. Do NOT keep the original text for those spans there.

# PROBLEM_STATEMENT
------------------
{problem_statement}
------------------

# EXAMPLE OF WHAT MASKED_PROBLEM_STATEMENT SHOULD LOOK LIKE:
Original: "We have experienced recurring issues raised by folks that want to observe satellites regarding the apparent inaccuracy."
If MASK_1 = "regarding the apparent inaccuracy", then masked_problem_statement should contain:
"We have experienced recurring issues raised by folks that want to observe satellites [MASK_1]."

# MUST_FOLLOW
You must keep masked_token_ratio in [0.28, 0.38]

# OUTPUT (JSON ONLY)
{{
  "masked_problem_statement": "the full text with [MASK_i] inserted",
  "mask_map": {{
    "MASK_1": "original span 1",
    "MASK_2": "original span 2",
    "...": "...",
    "MASK_N": "original span N"
  }},
  "stats": {{
    "total_tokens": {total_tokens},
    "masked_tokens_estimate": 0,
    "masked_ratio_estimate": 0.0
  }}
}}
"""

import json

def prompt_agent4_demask_problem_statement(
    *,
    instance_id: str,
    masked_problem_statement: str,
    max_masks: int = 10,
    fail_to_pass_failed: list[str],
    pass_to_pass_failed: list[str],
    eval_report: dict,
    test_output_txt: str,
    fail_to_pass_files_text: str,
    pass_to_pass_files_text: str,
    model_patch: str,
) -> str:
    mask_fill_lines = ",\n".join([f'    "MASK_{i}": "."' for i in range(1, max_masks + 1)])
    return f"""
You are Agent4 Stage-A (Demask).
Your ONLY job is to fill [MASK_i] placeholders in the MASKED_PROBLEM_STATEMENT.

Rules:
- Use ONLY evidence from: failing tests list, report.json, test_output.txt, and test file contents.
- Do NOT propose code changes.
- Do NOT write requirements checklists.
- If a mask cannot be confidently filled, fill it with a conservative, minimal phrase that does not over-specify.

instance_id: {instance_id}

# MASKED_PROBLEM_STATEMENT
{masked_problem_statement}

# FAILURES
FAIL_TO_PASS failures: {fail_to_pass_failed}
PASS_TO_PASS failures: {pass_to_pass_failed}

# report.json
{json.dumps(eval_report, ensure_ascii=False, indent=2)}

# test_output.txt
{test_output_txt}

# FAIL_TO_PASS_FILES (full text)
{fail_to_pass_files_text}

# PASS_TO_PASS_FILES (full text)
{pass_to_pass_files_text}

# Generated model_patch (context only)
{model_patch}

# OUTPUT (JSON ONLY)
{{
  "mask_fills": {{
    {mask_fill_lines}
  }},
  "demasked_problem_statement_text": "the full problem statement with all [MASK_i] replaced"
}}
""".strip()

import json

def prompt_agent4_posthoc_v2(
    *,
    task_prompt: str,
    instance_id: str,
    problem_statement: str,
    demasked_problem_statement_text: str,
    mask_fills: dict,
    model_patch: str,
    agent1_question_pack: dict,
    agent1_judgement: dict,
    agent2_enhanced_dsl: str,
    eval_report: dict,
    test_output_txt: str,
    fail_to_pass_failed: list[str],
    pass_to_pass_failed: list[str],
    fail_to_pass_files_text: str,
    pass_to_pass_files_text: str,
) -> str:
    return f"""
# TASK_PROMPT
{task_prompt}

You are Agent4 Stage-B (Post-hoc requirement discovery).
Goal:
1) Semantically cluster the failing tests into a small number of root-cause categories.
2) Produce Checklist v2: a new set of testable requirement statements for the NEXT iteration.
3) Produce DSL delta: extra hard rules to append to Agent2 enhanced DSL.

instance_id: {instance_id}

# PROBLEM_STATEMENT (original)
{problem_statement}

# DEMASKED_PROBLEM_STATEMENT (use this as the clarified spec)
{demasked_problem_statement_text}

# MASK_FILLS (audit trail)
{json.dumps(mask_fills, ensure_ascii=False, indent=2)}

# FIRST_ITERATION_ARTIFACTS
[Agent1 Question Pack]
{json.dumps(agent1_question_pack, ensure_ascii=False, indent=2)}

[Agent1 Judgement] (context only)
{json.dumps(agent1_judgement, ensure_ascii=False, indent=2)}

[Agent2 Enhanced DSL]
{agent2_enhanced_dsl}

[Generated model_patch diff]
{model_patch}

# EVALUATION
FAIL_TO_PASS failures: {fail_to_pass_failed}
PASS_TO_PASS failures: {pass_to_pass_failed}

[report.json]
{json.dumps(eval_report, ensure_ascii=False, indent=2)}

# TEST FILES (extract intent statements from these)
[FAIL_TO_PASS_FILES]
{fail_to_pass_files_text}

[PASS_TO_PASS_FILES]
{pass_to_pass_files_text}

# OUTPUT (JSON ONLY)
Return ONLY valid JSON:
{{
  "failure_taxonomy": [
    {{
      "label": "short category name",
      "tests": ["nodeid", "..."],
      "evidence": ["short quote from test_output", "..."],
      "hypothesis": "what requirement was violated"
    }}
  ],
  "checklist_v2": [
    {{
      "rid": "R1",
      "requirement": "testable requirement statement",
      "linked_tests": ["nodeid", "..."]
    }}
  ],
  "dsl_delta": [
    "bullet rule 1",
    "bullet rule 2"
  ]
}}
""".strip()


