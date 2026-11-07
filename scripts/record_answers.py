"""Freeze the generator answers of a live eval run into a small committed recording.

    python scripts/record_answers.py            # reads reports/checkpoint_<prompt>_<model>.jsonl

The offline CI gate (``dronedocs eval --offline``) replays these answers through the real retrieval,
rerank, evidence gate and citation validator, so the build is deterministic and needs no API key.
Only rows that completed without error and actually called the generator are recorded.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedocs.config import get_settings, load_prompt_bundle  # noqa: E402


def main() -> None:
    cfg = get_settings()
    version = load_prompt_bundle(cfg["app"]["prompt_file"])["version"]
    model = cfg["llm"]["model"]
    ck = ROOT / cfg["eval"]["report_dir"] / f"checkpoint_{version}_{model.replace('/', '_')}.jsonl"
    last: dict[str, dict] = {}
    for line in ck.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row.get("error") and row.get("raw_answer"):
            last[row["id"]] = row
    out = ROOT / cfg["eval"]["recordings_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rid in sorted(last):
            r = last[rid]
            rec = {
                "id": rid,
                "question": r["question"],
                "model": model,
                "prompt_version": version,
                "raw_answer": r["raw_answer"],
                "context_ids": r["context_ids"],
                "usage": {k: r["usage"].get(k, 0) for k in ("prompt_tokens", "completion_tokens")},
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"recorded {len(last)} answers -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
