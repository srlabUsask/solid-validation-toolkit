"""
sample_holdout.py
-----------------
Draws a stratified held-out validation sample from the unified corpus
produced by build_corpus.py.

Stratification design:
    1. Drop trivial methods (default LOC < 3) by default. Empty-bodied
       methods like "void test() {}" produce uninformative LLM scores
       (almost always Compliant with high confidence) and would inflate
       agreement metrics without testing the pipeline on substantive cases.
       Override with --no-loc-floor if you need to include them.
    2. Sample is drawn in two phases:
         (a) ~half proportional to the joint distribution of LLM scores
             (preserves external-validity-style representativeness)
         (b) the rest oversampled from minority cells (any principle in
             {0=Violated, 1=Partial}) so per-class precision/recall metrics
             are estimable rather than dominated by class-2 cases.
    3. Within each phase, sampling is also balanced across the eight
       projects (round-robin top-up if any project is under-represented).
    4. Methods used during prompt calibration must be excluded; provide
       their IDs via --exclude-ids.

Output is a CSV used by annotate.py, plus a JSON manifest with sampling
parameters for the paper's methods section.

Usage:
    python sample_holdout.py \\
        --corpus corpus.csv \\
        --exclude-ids calibration_ids.txt \\
        --output holdout_sample.csv \\
        --manifest holdout_manifest.json \\
        --n 200 \\
        --seed 42
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PRINCIPLES = ["srp", "ocp", "dip"]


def load_exclude_ids(path):
    if path is None:
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def stratified_sample(df, n, seed, min_per_cell=15):
    """
    Three-phase stratified sampler designed for ordinal-classifier validation.

    Phase 1 (per-cell minimums): for each (principle, class) cell, guarantee
        at least `min_per_cell` methods. This is what makes per-class
        precision/recall metrics estimable. With 3 principles x 3 classes
        and a target of 15 each, the floor is 9 cells; methods can satisfy
        multiple cells simultaneously, so the actual budget consumed is
        usually well below 9 * min_per_cell.

    Phase 2 (proportional fill): of the remaining budget, ~half goes
        proportionally to the joint LLM-score distribution to preserve
        representativeness for overall agreement metrics.

    Phase 3 (minority top-up): the rest oversamples cells where any
        principle is < 2, to bias the sample further toward informative
        cases without abandoning representativeness from phase 2.

    Within each phase, draws are without replacement and methods already
    selected in earlier phases are excluded from later phases.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_joint"] = list(zip(df["srp_score"], df["ocp_score"], df["dip_score"]))

    selected_ids = set()
    selected_dfs = []

    def take(pool, k, label):
        """Sample k rows from pool, excluding already-selected, return df."""
        avail = pool[~pool["method_id"].isin(selected_ids)]
        if len(avail) == 0 or k <= 0:
            return pd.DataFrame(columns=df.columns)
        actual_k = min(int(k), len(avail))
        picked = avail.sample(n=actual_k, random_state=int(rng.integers(2**32)))
        selected_ids.update(picked["method_id"].tolist())
        return picked

    # ---------- Phase 1: per-cell minimums ----------
    # For each (principle, class), top up to min_per_cell.
    cell_status = {}  # for logging
    for principle in PRINCIPLES:
        for cls in (0, 1, 2):
            current = sum(
                1 for sid in selected_ids
                if df.loc[df["method_id"] == sid, f"{principle}_score"].iloc[0] == cls
            ) if selected_ids else 0
            need = max(0, min_per_cell - current)
            if need == 0:
                cell_status[(principle, cls)] = ("already_met", 0)
                continue
            pool = df[df[f"{principle}_score"] == cls]
            picked = take(pool, need, f"{principle}={cls}")
            cell_status[(principle, cls)] = ("filled", len(picked))
            if len(picked) > 0:
                selected_dfs.append(picked)
            shortfall_in_cell = need - len(picked)
            if shortfall_in_cell > 0:
                cell_status[(principle, cls)] = (
                    f"SHORT_{len(picked)}_of_{need}", len(picked)
                )

    phase1_count = len(selected_ids)
    if phase1_count > n:
        # The minimums alone exceeded the budget; trim back to n by removing
        # rows that satisfied multiple cells but are over-represented in
        # any one cell. Simplest deterministic approach: drop excess from
        # the highest-population cells until we hit n.
        merged = pd.concat(selected_dfs)
        merged = merged.sample(n=n, random_state=int(rng.integers(2**32)))
        selected_ids = set(merged["method_id"].tolist())
        out = merged.drop(columns=["_joint"])
        out = out.sample(frac=1, random_state=int(rng.integers(2**32))).reset_index(drop=True)
        return out, cell_status

    # ---------- Phase 2: proportional fill ----------
    remaining = n - phase1_count
    proportional_budget = remaining // 2
    strata_sizes = df["_joint"].value_counts(normalize=True)
    for stratum, frac in strata_sizes.items():
        k = int(round(frac * proportional_budget))
        if k <= 0:
            continue
        pool = df[df["_joint"] == stratum]
        picked = take(pool, k, f"prop_{stratum}")
        if len(picked) > 0:
            selected_dfs.append(picked)

    # ---------- Phase 3: minority top-up + final fill ----------
    remaining = n - len(selected_ids)
    if remaining > 0:
        minority_pool = df[
            (df["srp_score"] < 2) | (df["ocp_score"] < 2) | (df["dip_score"] < 2)
        ]
        picked = take(minority_pool, remaining, "minority_topup")
        if len(picked) > 0:
            selected_dfs.append(picked)

    # Final fill from anywhere if still short
    remaining = n - len(selected_ids)
    if remaining > 0:
        picked = take(df, remaining, "final_topup")
        if len(picked) > 0:
            selected_dfs.append(picked)

    out = pd.concat(selected_dfs).drop(columns=["_joint"])
    out = out.sample(frac=1, random_state=int(rng.integers(2**32))).reset_index(drop=True)
    return out, cell_status


def write_manifest(sample, corpus_size, n_excluded, n_dropped_loc, loc_floor,
                   seed, min_per_cell, cell_status, output_path):
    # Translate cell_status into a JSON-serializable structure
    cell_report = {
        f"{p}_class_{c}": status
        for (p, c), status in cell_status.items()
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "loc_floor": loc_floor,
        "min_per_cell": min_per_cell,
        "corpus_size_after_filters": corpus_size,
        "n_calibration_excluded": n_excluded,
        "n_dropped_below_loc_floor": n_dropped_loc,
        "sample_size": len(sample),
        "stratification": {
            "method": "three-phase: per-cell minimums + proportional fill + minority top-up",
            "min_per_cell_target": min_per_cell,
            "minority_definition": "any principle score < 2",
        },
        "phase1_cell_status": cell_report,
        "score_distribution": {
            p: {str(k): int(v) for k, v in Counter(sample[f"{p}_score"]).items()}
            for p in PRINCIPLES
        },
        "label_distribution": {
            p: {str(k): int(v) for k, v in Counter(sample[f"{p}_label"]).items()}
            for p in PRINCIPLES
        },
        "project_distribution": {
            str(k): int(v) for k, v in Counter(sample["project"]).items()
        },
        "needs_more_context_in_sample": int(
            sample["overall_flags"].astype(str).str.contains(
                "needs_more_context", na=False
            ).sum()
        ),
        "method_loc_stats": {
            "min": int(sample["method_loc"].min()),
            "max": int(sample["method_loc"].max()),
            "median": float(sample["method_loc"].median()),
            "mean": float(sample["method_loc"].mean()),
        },
    }
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, type=Path,
                        help="Unified corpus CSV from build_corpus.py")
    parser.add_argument("--exclude-ids", type=Path, default=None,
                        help="Text file with calibration method IDs, one per line")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loc-floor", type=int, default=3,
                        help="Drop methods with method_loc below this (default 3). "
                             "Set to 0 or use --no-loc-floor to keep trivial methods.")
    parser.add_argument("--no-loc-floor", action="store_true",
                        help="Disable LOC floor entirely.")
    parser.add_argument("--min-per-cell", type=int, default=15,
                        help="Minimum methods per (principle, class) cell. "
                             "Ensures stable per-class precision/recall. Default 15.")
    args = parser.parse_args()

    df = pd.read_csv(args.corpus)
    df["method_id"] = df["method_id"].astype(str)

    n_before_loc = len(df)
    loc_floor = 0 if args.no_loc_floor else args.loc_floor
    if loc_floor > 0:
        df = df[df["method_loc"] >= loc_floor]
    n_dropped_loc = n_before_loc - len(df)

    exclude_ids = load_exclude_ids(args.exclude_ids)
    if exclude_ids:
        df = df[~df["method_id"].isin(exclude_ids)]
        print(f"Excluded {len(exclude_ids)} calibration IDs.", file=sys.stderr)

    print(f"Eligible pool: {len(df)} methods (dropped {n_dropped_loc} below LOC floor {loc_floor})",
          file=sys.stderr)

    if len(df) < args.n:
        sys.exit(f"Eligible pool ({len(df)}) < requested n ({args.n})")

    sample, cell_status = stratified_sample(
        df, args.n, args.seed, min_per_cell=args.min_per_cell
    )

    # Print per-cell status for visibility
    print("\nPhase 1 per-cell status (target: "
          f"{args.min_per_cell} methods per cell):", file=sys.stderr)
    for principle in PRINCIPLES:
        for cls in (0, 1, 2):
            status = cell_status.get((principle, cls), ("not_run", 0))
            print(f"  {principle}={cls}: {status[0]}", file=sys.stderr)
    print("", file=sys.stderr)

    # Rename score cols to *_llm so the annotation app keeps a clean
    # separation between LLM scores and rater scores.
    rename = {}
    for p in PRINCIPLES:
        rename[f"{p}_score"]      = f"{p}_score_llm"
        rename[f"{p}_label"]      = f"{p}_label_llm"
        rename[f"{p}_confidence"] = f"{p}_confidence_llm"
        rename[f"{p}_notes"]      = f"{p}_notes_llm"
    sample = sample.rename(columns=rename)

    sample.to_csv(args.output, index=False)
    print(f"Wrote sample of {len(sample)} methods to {args.output}", file=sys.stderr)

    # Recompute manifest with renamed cols
    write_sample = sample.rename(columns={v: k for k, v in rename.items()})
    write_manifest(
        sample=write_sample,
        corpus_size=len(df),
        n_excluded=len(exclude_ids),
        n_dropped_loc=n_dropped_loc,
        loc_floor=loc_floor,
        seed=args.seed,
        min_per_cell=args.min_per_cell,
        cell_status=cell_status,
        output_path=args.manifest,
    )


if __name__ == "__main__":
    main()