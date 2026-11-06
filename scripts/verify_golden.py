"""Verify every golden answer's evidence quote exists verbatim on its cited page/section."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedocs.loaders import discover, load_file  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def main() -> int:
    sections = {}
    for p in discover(ROOT / "data" / "raw"):
        sections[p.name] = load_file(p)
    rows = [json.loads(line) for line in (ROOT / "eval" / "golden.jsonl").read_text(encoding="utf-8").splitlines()]
    bad = 0
    for r in rows:
        if not r["answerable"]:
            continue
        refs = [{"source": r["source"], "locator": r["locator"], "evidence": r["evidence"]}, *r.get("alt_evidence", [])]
        for ref in refs:
            secs = sections.get(ref["source"])
            if secs is None:
                print(f"FAIL {r['id']}: unknown source {ref['source']}")
                bad += 1
                continue
            loc = ref["locator"]
            cand = [s for s in secs if s.locator == loc or s.locator.startswith(loc)]
            if not any(norm(ref["evidence"]) in norm(s.text) for s in cand):
                where = [s.locator for s in secs if norm(ref["evidence"]) in norm(s.text)]
                print(f"FAIL {r['id']}: evidence not on {ref['source']} {loc}; found at {where}")
                bad += 1
    n_ans = sum(r["answerable"] for r in rows)
    n_alt = sum(len(r.get("alt_evidence") or []) for r in rows)
    print(
        f"checked {n_ans} answerable items + {n_alt} alternate evidence passages: {bad} failures; "
        f"{len(rows) - n_ans} unanswerable items"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
