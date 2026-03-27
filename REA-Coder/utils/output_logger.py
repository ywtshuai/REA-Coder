import json
from pathlib import Path
from typing import Any, List, Dict

class OutputLogger:
    def __init__(self, base_dir: str, instance_id: str):
        self.root = Path(base_dir) / instance_id
        self.root.mkdir(parents=True, exist_ok=True)

    def write_text(self, relative_path: str, content: str):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content or "", encoding="utf-8")

    def write_json(self, relative_path: str, obj: Any):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(obj, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    # -----------------------------
    # NEW: prompt logging helpers
    # -----------------------------
    def write_messages(self, relative_path: str, messages: List[Dict[str, str]]):
        """
        Save OpenAI-style chat messages as JSON.
        """
        safe = []
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            safe.append({
                "role": m.get("role", ""),
                "content": m.get("content", "")
            })
        self.write_json(relative_path, safe)

    def write_prompt_bundle(self, base_path_no_ext: str, messages: List[Dict[str, str]]):
        """
        Save both:
          - <base>.json : structured messages
          - <base>.txt  : human-readable rendered prompt
        """
        self.write_messages(base_path_no_ext + ".json", messages)

        parts = []
        for m in messages or []:
            role = (m.get("role") or "").upper()
            content = m.get("content") or ""
            parts.append(f"===== {role} =====\n{content}".rstrip())
        rendered = "\n\n".join(parts).strip() + "\n"
        self.write_text(base_path_no_ext + ".txt", rendered)