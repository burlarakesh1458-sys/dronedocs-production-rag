"""Render charts from reports/latest.json and reports/retrieval_ablation.json into docs/."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def eval_chart(rep: dict) -> None:
    s, thr = rep["summary"], rep["thresholds"]
    keys = list(thr)
    vals = [s.get(k) or 0 for k in keys]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    colors = ["#55a868" if v >= thr[k] else "#d62728" for k, v in zip(keys, vals, strict=True)]
    bars = ax.bar(keys, vals, color=colors)
    for i, k in enumerate(keys):
        ax.hlines(thr[k], i - 0.4, i + 0.4, colors="black", linestyles="dotted")
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1)
    skipped = s.get("n_not_evaluated_quota") or 0
    part = f", {skipped} not evaluated (provider quota)" if skipped else ""
    ax.set_title(
        f"Golden-set eval: n={s['n_items']} ({s['n_answerable']} answerable){part}\n"
        f"gen {rep['generator_model']}, judge {rep['judge_model']}  (dotted = CI threshold)",
        fontsize=10,
    )
    plt.xticks(rotation=15, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "eval_metrics.png", dpi=120)


def latency_chart(rep: dict) -> None:
    lat = rep["summary"]["latency"]
    stages = [k for k in ("retrieve", "rerank", "generate", "total") if k in lat]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = range(len(stages))
    ax.bar([i - 0.2 for i in x], [lat[k]["p50_ms"] for k in stages], 0.4, label="p50", color="#4c72b0")
    ax.bar([i + 0.2 for i in x], [lat[k]["p95_ms"] for k in stages], 0.4, label="p95", color="#dd8452")
    ax.set_xticks(list(x), stages)
    ax.set_ylabel("ms")
    ax.set_title("Per-stage latency (CPU retrieval/rerank; generate includes Groq free-tier 429 backoff)", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "latency.png", dpi=120)


def ablation_chart(ab: dict) -> None:
    res = ab["results"]
    names = list(res)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = range(len(names))
    k = ab.get("k", 5)
    for off, key, color in ((-0.27, "hit@k", "#4c72b0"), (0, "mrr@k", "#55a868"), (0.27, "recall@20", "#8172b3")):
        vals = [res[n][key] for n in names]
        bars = ax.bar(
            [i + off for i in x], vals, 0.27, label=key.replace("k", str(k)) if key != "recall@20" else key, color=color
        )
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(x), names)
    ax.set_ylim(0, 1.22)
    ax.set_title(
        f"Retrieval ablation ({ab['n_questions']} answerable golden questions, {ab['chunks_in_index']} chunks)"
    )
    ax.legend(loc="upper center", ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "retrieval_ablation.png", dpi=120)


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    rep_path = ROOT / "reports" / "latest.json"
    if rep_path.exists():
        rep = json.loads(rep_path.read_text())
        eval_chart(rep)
        latency_chart(rep)
    ab_path = ROOT / "reports" / "retrieval_ablation.json"
    if ab_path.exists():
        ablation_chart(json.loads(ab_path.read_text()))
    print(sorted(p.name for p in DOCS.glob("*.png")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
