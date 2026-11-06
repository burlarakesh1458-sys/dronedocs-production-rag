"""Ragas cross-check of an existing eval report, with no retrieval models loaded.

Reads the answered rows of reports/latest.json, scores a small sample with Ragas (faithfulness +
LLM context recall) and stores the result next to the built-in judge's scores for the same items:
reports/ragas_crosscheck.json, and the ``ragas`` block of reports/latest.json.

    python scripts/ragas_crosscheck.py --sample 3 --model qwen/qwen3.8-27b
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedocs.chunking import Chunk  # noqa: E402
from dronedocs.config import get_settings  # noqa: E402
from dronedocs.eval.ragas_eval import run_ragas  # noqa: E402
from dronedocs.eval.run_eval import load_golden  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=3)
    ap.add_argument("--model", default=None, help="Ragas LLM (default: eval.ragas_model)")
    args = ap.parse_args()
    cfg = get_settings()
    if args.model:
        cfg["eval"]["ragas_model"] = args.model
    report_path = ROOT / "reports" / "latest.json"
    report = json.loads(report_path.read_text())
    items = load_golden(cfg["eval"]["golden_path"])
    manifest = ROOT / cfg["paths"]["index_dir"] / "chunks.jsonl"
    chunks = {d["chunk_id"]: Chunk(**d) for d in map(json.loads, manifest.read_text(encoding="utf-8").splitlines())}
    rows = [r for r in report["rows"] if not r.get("error")]
    out = run_ragas(cfg, items, rows, chunks, args.sample)
    out["generator_model"] = report["generator_model"]
    (ROOT / "reports" / "ragas_crosscheck.json").write_text(json.dumps(out, indent=2))
    report["ragas"] = out
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
