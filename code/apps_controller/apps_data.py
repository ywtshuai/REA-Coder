"""
apps_controller/apps_data.py

Load APPS-style jsonl data into AppsProblem objects.
Compatible with teammate's apps_eval.data.load_data behavior (./Datasets/{name}.jsonl).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from apps_controller.types import ProblemData

def load_dataset_jsonl(data_name_or_path: str) -> List[ProblemData]:
    """
    data_name_or_path:
      - "apps"  => ./Datasets/apps.jsonl
      - "/abs/path/to/apps.jsonl" or "./Datasets/apps.jsonl"
    """
    p = Path(data_name_or_path)
    if p.suffix.lower() != ".jsonl":
        p = Path("./Datasets") / f"{data_name_or_path}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"APPS dataset not found: {p}")

    problems: List[ProblemData] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj: Dict[str, Any] = json.loads(line)
            pid = str(obj.get("problem_id") or "")
            if not pid:
                pid = str(0)
            problems.append(ProblemData(
                problem_id=pid,
                question=str(obj.get("question") or obj.get("description") or ""),
                starter_code=str(obj.get("starter_code") or ""),
                public_test_cases=obj.get("public_test_cases"),
                all_test_cases=obj.get("all_test_cases"),
                difficulty=obj.get("difficulty"),
                url=obj.get("url") or "",
                raw=obj,
            ))
    return problems