"""
analyze_agreement.py
--------------------
Computes inter-rater and rater-vs-LLM agreement on the held-out validation
sample. Inputs come from `annotate.py --export`.

Outputs:
    - agreement_summary.json with kappa, weighted kappa, accuracy, and
      per-class precision/recall/F1 for each principle, comparing
      (a) rater vs rater (inter-rater reliability) and
      (b) consensus rater score vs LLM (validation of the pipeline)
    - confusion_matrices.csv for the paper

Consensus rule:
    - If both raters agree (and neither marked N/A), consensus = that value
    - If they disagree, the row is flagged for tiebreaker (you).
    - Tiebreaker decisions can be supplied via --tiebreaker tiebreak.csv
      (columns: method_id, principle, tiebreak_score)

Usage (two-pass workflow):

    # Pass 1: surface unresolved disagreements
    python analyze_agreement.py \\
        --results results.csv \\
        --output agreement_summary.json \\
        --confusion confusion.csv \\
        --disagreements disagreements_to_resolve.csv

    # ...you fill in tiebreak.csv based on disagreements_to_resolve.csv...

    # Pass 2: with tiebreaks applied
    python analyze_agreement.py \\
        --results results.csv \\
        --tiebreaker tiebreak.csv \\
        --output agreement_summary.json \\
        --confusion confusion.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)


PRINCIPLES = ["srp", "ocp", "dip"]
CLASSES = [0, 1, 2]


def pivot_by_rater(results: pd.DataFrame) -> pd.DataFrame:
    """Reshape long results into one row per method with columns per rater."""
    keep_cols = ["method_id", "project"] + [f"{p}_score_llm" for p in PRINCIPLES]
    keep_cols = [c for c in keep_cols if c in results.columns]
    base = results[keep_cols].drop_duplicates(subset=["method_id"])

    pivots = []
    for p in PRINCIPLES:
        wide = results.pivot_table(
            index="method_id",
            columns="rater",
            values=f"{p}_score",
            aggfunc="first",
        ).add_prefix(f"{p}_rater_")
        wide_na = results.pivot_table(
            index="method_id",
            columns="rater",
            values=f"{p}_na",
            aggfunc="first",
        ).add_prefix(f"{p}_na_")
        pivots.append(wide)
        pivots.append(wide_na)

    out = base.set_index("method_id").join(pivots).reset_index()
    return out


def build_consensus(wide: pd.DataFrame, tiebreak):
    """For each principle, build a consensus score column."""
    rater_cols = {}
    for p in PRINCIPLES:
        rater_cols[p] = sorted([
            c for c in wide.columns
            if c.startswith(f"{p}_rater_")
        ])

    disagreements = []

    for p in PRINCIPLES:
        rcols = rater_cols[p]
        if len(rcols) < 2:
            print(f"WARN: <2 raters for {p}; skipping consensus.", file=sys.stderr)
            wide[f"{p}_consensus"] = np.nan
            continue

        consensus = []
        for _, row in wide.iterrows():
            scores = [row[c] for c in rcols]
            na_cols = [c.replace(f"{p}_rater_", f"{p}_na_") for c in rcols]
            na_flags = [row.get(nc, 0) for nc in na_cols]

            if any(pd.isna(s) or n == 1 for s, n in zip(scores, na_flags)):
                consensus.append(np.nan)
                disagreements.append({
                    "method_id": row["method_id"],
                    "principle": p,
                    "rater_scores": dict(zip(rcols, scores)),
                    "na_flags": dict(zip(na_cols, na_flags)),
                    "reason": "na_or_missing",
                })
                continue

            scores_int = [int(s) for s in scores]
            if len(set(scores_int)) == 1:
                consensus.append(scores_int[0])
            else:
                tb = None
                if tiebreak is not None:
                    match = tiebreak[
                        (tiebreak["method_id"].astype(str) == str(row["method_id"]))
                        & (tiebreak["principle"] == p)
                    ]
                    if not match.empty:
                        tb = int(match.iloc[0]["tiebreak_score"])
                if tb is None:
                    consensus.append(np.nan)
                    disagreements.append({
                        "method_id": row["method_id"],
                        "principle": p,
                        "rater_scores": dict(zip(rcols, scores_int)),
                        "na_flags": dict(zip(na_cols, na_flags)),
                        "reason": "disagreement_no_tiebreak",
                    })
                else:
                    consensus.append(tb)

        wide[f"{p}_consensus"] = consensus

    return wide, disagreements


def compute_agreement(wide: pd.DataFrame) -> dict:
    summary = {}
    for p in PRINCIPLES:
        rcols = sorted([c for c in wide.columns if c.startswith(f"{p}_rater_")])
        per_principle = {"raters": rcols}

        # Inter-rater kappa per pair
        if len(rcols) >= 2:
            pairs = []
            for i in range(len(rcols)):
                for j in range(i + 1, len(rcols)):
                    a, b = rcols[i], rcols[j]
                    valid = wide[[a, b]].dropna()
                    if len(valid) == 0:
                        continue
                    ya = valid[a].astype(int)
                    yb = valid[b].astype(int)
                    pairs.append({
                        "rater_a": a, "rater_b": b,
                        "n": int(len(valid)),
                        "kappa": float(cohen_kappa_score(ya, yb)),
                        "weighted_kappa_linear": float(
                            cohen_kappa_score(ya, yb, weights="linear")
                        ),
                        "raw_agreement": float(accuracy_score(ya, yb)),
                    })
            per_principle["inter_rater"] = pairs

        # Consensus vs LLM
        cons_col = f"{p}_consensus"
        llm_col = f"{p}_score_llm"
        if cons_col in wide.columns and llm_col in wide.columns:
            valid = wide[[cons_col, llm_col]].dropna()
            if len(valid) > 0:
                y_true = valid[cons_col].astype(int)
                y_pred = valid[llm_col].astype(int)
                prec, rec, f1, support = precision_recall_fscore_support(
                    y_true, y_pred, labels=CLASSES, zero_division=0
                )
                cm = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()
                per_principle["llm_vs_consensus"] = {
                    "n": int(len(valid)),
                    "kappa": float(cohen_kappa_score(y_true, y_pred)),
                    "weighted_kappa_linear": float(
                        cohen_kappa_score(y_true, y_pred, weights="linear")
                    ),
                    "raw_agreement": float(accuracy_score(y_true, y_pred)),
                    "per_class": [
                        {
                            "class": c,
                            "precision": float(prec[i]),
                            "recall": float(rec[i]),
                            "f1": float(f1[i]),
                            "support": int(support[i]),
                        }
                        for i, c in enumerate(CLASSES)
                    ],
                    "confusion_matrix": cm,
                    "confusion_matrix_axes": {"rows": "consensus (truth)", "cols": "llm"},
                }
        summary[p] = per_principle
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--tiebreaker", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confusion", type=Path, default=None)
    parser.add_argument("--disagreements", type=Path, default=None)
    args = parser.parse_args()

    results = pd.read_csv(args.results)
    results["method_id"] = results["method_id"].astype(str)
    tiebreak = None
    if args.tiebreaker and args.tiebreaker.exists():
        tiebreak = pd.read_csv(args.tiebreaker)

    wide = pivot_by_rater(results)
    wide, disagreements = build_consensus(wide, tiebreak)
    summary = compute_agreement(wide)

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {args.output}", file=sys.stderr)

    if args.disagreements:
        # Flatten dict-valued cols for CSV
        rows = []
        for d in disagreements:
            rows.append({
                "method_id": d["method_id"],
                "principle": d["principle"],
                "rater_scores": json.dumps(d["rater_scores"], default=str),
                "na_flags": json.dumps(d["na_flags"], default=str),
                "reason": d["reason"],
            })
        pd.DataFrame(rows).to_csv(args.disagreements, index=False)
        print(f"Wrote {len(rows)} unresolved cases to {args.disagreements}", file=sys.stderr)

    if args.confusion:
        rows = []
        for p, info in summary.items():
            cm = info.get("llm_vs_consensus", {}).get("confusion_matrix")
            if cm is None:
                continue
            for i, true_c in enumerate(CLASSES):
                for j, pred_c in enumerate(CLASSES):
                    rows.append({
                        "principle": p,
                        "consensus_class": true_c,
                        "llm_class": pred_c,
                        "count": cm[i][j],
                    })
        pd.DataFrame(rows).to_csv(args.confusion, index=False)
        print(f"Wrote confusion matrices to {args.confusion}", file=sys.stderr)


if __name__ == "__main__":
    main()