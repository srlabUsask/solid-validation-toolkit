"""
build_corpus.py
---------------
Reads the per-project LLM score files (JSONL, one record per method) and the
per-project NiCad-style extracted XML files, joins them on (file_path,
startline, endline), and emits a single unified CSV ready for sampling.

Expected layout (relative to --root):
    Evaluated_Solid_Scores_Jsons/
        commonslang_SOLID_Eval.json
        fitnesse_SOLID_Eval.json
        hibernate-orm_SOLID_Eval.json
        jackson-databind__SOLID_Eval.json
        jmeter_SOLID_Eval.json
        junit5_SOLID_Eval.json
        selenium-trunk_SOLID_Eval.json
        struts_SOLID_Eval.json
    Extracted_functions/
        commons-lang-master_functions.xml
        fitnesse_functions.xml
        hibernate-orm-main_functions.xml
        jackson-databind-2.19_functions.xml
        jmeter-master_functions.xml
        junit5-main_functions.xml
        selenium-trunk_functions.xml
        struts-main_functions.xml

Each JSON file contains one JSON object per line (JSONL), with fields:
    id, file_path, startline, endline,
    srp:{score,label,confidence,evidence,notes},
    ocp:{...}, dip:{...},
    overall:{solid_score, flags},
    model, prompt_version, run_settings, timestamp_utc

Each XML file is a flat sequence of <source file="..." startline="..."
endline="...">...</source> blocks (NiCad's standard output for function
granularity).

Usage:
    python build_corpus.py --root ~/Desktop/SOLID/Research --output corpus.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from collections import defaultdict


# Mapping: project label used downstream  ->  (json_filename, xml_filename)
# Project labels follow the convention used in the paper.
PROJECTS = {
    "commons-lang":     ("commonslang_SOLID_Eval.json",       "commons-lang-master_functions.xml"),
    "fitnesse":         ("fitnesse_SOLID_Eval.json",          "fitnesse_functions.xml"),
    "hibernate-orm":    ("hibernate-orm_SOLID_Eval.json",     "hibernate-orm-main_functions.xml"),
    "jackson-databind": ("jackson-databind__SOLID_Eval.json", "jackson-databind-2.19_functions.xml"),
    "jmeter":           ("jmeter_SOLID_Eval.json",            "jmeter-master_functions.xml"),
    "junit5":           ("junit5_SOLID_Eval.json",            "junit5-main_functions.xml"),
    "selenium":         ("selenium-trunk_SOLID_Eval.json",    "selenium-trunk_functions.xml"),
    "struts":           ("struts_SOLID_Eval.json",            "struts-main_functions.xml"),
}

# Regex for parsing NiCad's XML <source ...>...</source> blocks.
# NiCad output is not strict XML at the document level (no single root) so a
# real XML parser would refuse it. A regex over the file is the standard
# approach; the inner method body can contain anything including "<" and ">"
# in operators, so we use a non-greedy match terminated by the literal close
# tag </source>.
SOURCE_RE = re.compile(
    r'<source\s+file="(?P<file>[^"]+)"\s+startline="(?P<sl>\d+)"\s+endline="(?P<el>\d+)">'
    r'(?P<body>.*?)'
    r'</source>',
    re.DOTALL,
)


def load_json_scores(path: Path) -> dict:
    """
    Read a JSONL file of LLM scores. Returns dict keyed by
    (file_path, startline, endline) tuple -> score record.
    """
    out = {}
    n_dup = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  warn: {path.name}:{lineno} JSON parse failed: {e}",
                      file=sys.stderr)
                continue
            try:
                key = (rec["file_path"], int(rec["startline"]), int(rec["endline"]))
            except (KeyError, ValueError, TypeError) as e:
                print(f"  warn: {path.name}:{lineno} missing/bad key: {e}",
                      file=sys.stderr)
                continue
            if key in out:
                n_dup += 1
                # Keep the later one; this matches typical re-run-overwrite semantics
            out[key] = rec
    if n_dup:
        print(f"  note: {path.name} had {n_dup} duplicate keys (kept latest)", file=sys.stderr)
    return out


def load_xml_sources(path: Path) -> dict:
    """
    Read a NiCad <source>...</source> XML-ish file. Returns dict keyed by
    (file_path, startline, endline) -> source body string.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    n_dup = 0
    for m in SOURCE_RE.finditer(text):
        key = (m.group("file"), int(m.group("sl")), int(m.group("el")))
        body = m.group("body").strip()
        if key in out:
            n_dup += 1
        out[key] = body
    if n_dup:
        print(f"  note: {path.name} had {n_dup} duplicate keys (kept latest)", file=sys.stderr)
    return out


def extract_method_name(body: str) -> str:
    """
    Cheap extraction of a display name from the method source. Looks for the
    first identifier followed by '(' in the first ~3 non-empty lines. Used
    only for human display in the annotation app; failures are non-fatal.
    """
    lines = [ln for ln in body.splitlines() if ln.strip()][:3]
    head = " ".join(lines)
    m = re.search(r'(\w+)\s*\(', head)
    return m.group(1) if m else "(unnamed)"


def process_project(project: str, root: Path, writer: csv.DictWriter) -> dict:
    """Process one project, write rows to CSV, return summary stats."""
    json_name, xml_name = PROJECTS[project]
    json_path = root / "Evaluated_Solid_Scores_Jsons" / json_name
    xml_path  = root / "Extracted_functions" / xml_name

    if not json_path.exists():
        print(f"  ERROR: missing {json_path}", file=sys.stderr)
        return {"project": project, "matched": 0, "unmatched_score": 0, "unmatched_source": 0}
    if not xml_path.exists():
        print(f"  ERROR: missing {xml_path}", file=sys.stderr)
        return {"project": project, "matched": 0, "unmatched_score": 0, "unmatched_source": 0}

    print(f"[{project}] reading scores: {json_path.name}", file=sys.stderr)
    scores = load_json_scores(json_path)
    print(f"[{project}] reading sources: {xml_path.name}", file=sys.stderr)
    sources = load_xml_sources(xml_path)

    score_keys = set(scores.keys())
    source_keys = set(sources.keys())
    common = score_keys & source_keys

    print(f"[{project}] scores: {len(score_keys)}, sources: {len(source_keys)}, "
          f"matched: {len(common)}, score-only: {len(score_keys - source_keys)}, "
          f"source-only: {len(source_keys - score_keys)}", file=sys.stderr)

    n_written = 0
    for key in sorted(common):
        rec = scores[key]
        body = sources[key]
        try:
            row = {
                "method_id": rec["id"],
                "project": project,
                "file_path": rec["file_path"],
                "startline": rec["startline"],
                "endline": rec["endline"],
                "method_loc": int(rec["endline"]) - int(rec["startline"]) + 1,
                "method_name": extract_method_name(body),
                "method_source": body,
                "srp_score": int(rec["srp"]["score"]),
                "srp_label": rec["srp"].get("label", ""),
                "srp_confidence": float(rec["srp"].get("confidence", 0.0)),
                "srp_notes": rec["srp"].get("notes", ""),
                "ocp_score": int(rec["ocp"]["score"]),
                "ocp_label": rec["ocp"].get("label", ""),
                "ocp_confidence": float(rec["ocp"].get("confidence", 0.0)),
                "ocp_notes": rec["ocp"].get("notes", ""),
                "dip_score": int(rec["dip"]["score"]),
                "dip_label": rec["dip"].get("label", ""),
                "dip_confidence": float(rec["dip"].get("confidence", 0.0)),
                "dip_notes": rec["dip"].get("notes", ""),
                "overall_solid_score": int(rec.get("overall", {}).get("solid_score", -1)),
                "overall_flags": "|".join(rec.get("overall", {}).get("flags", []) or []),
            }
            writer.writerow(row)
            n_written += 1
        except (KeyError, ValueError, TypeError) as e:
            print(f"  warn: skipping {key}: {e}", file=sys.stderr)
            continue

    return {
        "project": project,
        "matched": n_written,
        "unmatched_score": len(score_keys - source_keys),
        "unmatched_source": len(source_keys - score_keys),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path,
                        help="Path to ~/Desktop/SOLID/Research (or wherever)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output unified corpus CSV")
    args = parser.parse_args()

    fieldnames = [
        "method_id", "project", "file_path", "startline", "endline", "method_loc",
        "method_name", "method_source",
        "srp_score", "srp_label", "srp_confidence", "srp_notes",
        "ocp_score", "ocp_label", "ocp_confidence", "ocp_notes",
        "dip_score", "dip_label", "dip_confidence", "dip_notes",
        "overall_solid_score", "overall_flags",
    ]

    summaries = []
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for project in PROJECTS:
            summary = process_project(project, args.root, writer)
            summaries.append(summary)

    print("\n=== Build summary ===", file=sys.stderr)
    total_matched = 0
    for s in summaries:
        print(f"  {s['project']:20s}  matched={s['matched']:>7,}  "
              f"score-only={s['unmatched_score']:>5,}  "
              f"source-only={s['unmatched_source']:>5,}", file=sys.stderr)
        total_matched += s["matched"]
    print(f"  {'TOTAL':20s}  matched={total_matched:>7,}", file=sys.stderr)
    print(f"\nWrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()