"""Rows for the model-change case page (docs/case_model_change.md), from the sixth set.

For each candidate that replaced multilingual-e5-base on the XPQA product-answer sets, this
reads the cached model-change record of ``sixth.py`` and reports, next to the full-label
nDCG@10 difference and the label-free verdict, what the labeled gate decides with 100 seeded
labeled queries at ``retrieval_confidence=0.95`` (200 subsets). Nothing here is a
prediction; the preregistered criteria are P18 in ``sixth_summary.json``.

    python examples/validation/case_model_change.py
Writes examples/validation/case_model_change.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATASETS = ("xpqa-it", "xpqa-pl", "xpqa-pt")


def main(argv: list[str] | None = None) -> int:
    from run import out_dir
    from third import label_gate

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", help="run.py cache directory")
    args = p.parse_args(argv)
    folder = out_dir(args.out) / "sixth_changes"
    rows = []
    for name in DATASETS:
        for f in sorted(folder.glob(f"{name}__*.json")):
            c = json.loads(f.read_text(encoding="utf-8"))
            share = label_gate([c])["share"]["confidence"]
            kind = "harmful" if c["ndcg_drop"] > 0.01 else "harmless"
            rows.append(
                {
                    "dataset": name,
                    "candidate": c["model"],
                    "languages": c["languages"],
                    "baseline_ndcg": round(c["baseline_ndcg"], 3),
                    "candidate_ndcg": round(c["candidate_ndcg"], 3),
                    "ndcg_change": round(-c["ndcg_drop"], 3),
                    "lost_share": round(c["top_hit"]["queries"]["lost_share"], 3),
                    "reference_regressions": bool(c["reference_regressions"]),
                    "reference_review": bool(c["reference_review"]),
                    "labeled_gate": {k: round(v, 3) for k, v in share[kind].items()},
                }
            )
    rows.sort(key=lambda r: (r["dataset"], -r["ndcg_change"]))
    (HERE / "case_model_change.json").write_text(
        json.dumps(rows, indent=1) + "\n", encoding="utf-8"
    )
    for r in rows:
        g = r["labeled_gate"]
        print(
            f"{r['dataset']:<8} {r['candidate']:<30} {r['ndcg_change']:+.3f} "
            f"lost {r['lost_share']:.3f} {'FLAG' if r['reference_regressions'] else '    '} "
            f"{'REVIEW' if r['reference_review'] else '      '} "
            f"pass {g['pass']:.2f} inc {g['inconclusive']:.2f} fail {g['fail']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
