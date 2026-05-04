"""
annotate.py
-----------
Local Flask app for blind manual annotation of methods on SRP, OCP, and DIP.

Schema match: this app expects the holdout_sample.csv produced by
sample_holdout.py, with these columns:

    method_id, project, file_path, startline, endline, method_loc,
    method_name, method_source,
    srp_score_llm, srp_label_llm, srp_confidence_llm, srp_notes_llm,
    ocp_score_llm, ocp_label_llm, ocp_confidence_llm, ocp_notes_llm,
    dip_score_llm, dip_label_llm, dip_confidence_llm, dip_notes_llm,
    overall_solid_score, overall_flags

Design:
    - Each rater logs in with their identifier (e.g., student_a, student_b).
    - Methods are presented one at a time in the same shuffled order.
    - The LLM's score, label, confidence, and notes are HIDDEN during
      annotation. The rater sees only the method source and the rubric.
    - Annotations save to SQLite. Raters can close & reopen freely.
    - URL param ?method_id=ID jumps to a specific method to revise.

Usage:
    pip install flask pandas

    # Start the app
    python annotate.py --sample holdout_sample.csv --db annotations.db --rubric rubric.html

    # When raters are done, export their annotations to CSV
    python annotate.py --sample holdout_sample.csv --db annotations.db \\
        --export results.csv
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from flask import (
    Flask, request, redirect, url_for, render_template_string, session, jsonify
)

app = Flask(__name__)
app.secret_key = "change-me-for-deployment"

# Module-level state populated at startup
SAMPLE_DF: "pd.DataFrame | None" = None
DB_PATH: "Path | None" = None
RUBRIC_HTML: str = ""

# Set True only if you want students to see the LLM's verdict AFTER they
# submit their own score (e.g., for training rounds). For real validation,
# leave this False.
REVEAL_LLM_AFTER_SUBMIT: bool = False


SCHEMA = """
CREATE TABLE IF NOT EXISTS annotations (
    rater TEXT NOT NULL,
    method_id TEXT NOT NULL,
    srp_score INTEGER,
    ocp_score INTEGER,
    dip_score INTEGER,
    srp_na INTEGER DEFAULT 0,
    ocp_na INTEGER DEFAULT 0,
    dip_na INTEGER DEFAULT 0,
    comment TEXT,
    seconds_spent INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (rater, method_id)
);
"""


def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_init():
    with db_connect() as conn:
        conn.executescript(SCHEMA)


def get_annotation(rater, method_id):
    with db_connect() as conn:
        row = conn.execute(
            "SELECT * FROM annotations WHERE rater = ? AND method_id = ?",
            (rater, method_id),
        ).fetchone()
        return dict(row) if row else None


def save_annotation(rater, method_id, scores, na, comment, seconds_spent):
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO annotations (
                rater, method_id, srp_score, ocp_score, dip_score,
                srp_na, ocp_na, dip_na, comment, seconds_spent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(rater, method_id) DO UPDATE SET
                srp_score=excluded.srp_score,
                ocp_score=excluded.ocp_score,
                dip_score=excluded.dip_score,
                srp_na=excluded.srp_na,
                ocp_na=excluded.ocp_na,
                dip_na=excluded.dip_na,
                comment=excluded.comment,
                seconds_spent=COALESCE(excluded.seconds_spent, seconds_spent),
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                rater, method_id,
                scores.get("srp"), scores.get("ocp"), scores.get("dip"),
                int(na.get("srp", False)),
                int(na.get("ocp", False)),
                int(na.get("dip", False)),
                comment,
                seconds_spent,
            ),
        )


def progress_for(rater):
    with db_connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM annotations WHERE rater = ?", (rater,)
        ).fetchone()
        return row["c"], len(SAMPLE_DF)


# ---------- routes ----------

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        rater = request.form.get("rater", "").strip().lower().replace(" ", "_")
        if not rater:
            return redirect(url_for("login"))
        session["rater"] = rater
        return redirect(url_for("annotate"))
    return render_template_string(LOGIN_TEMPLATE)


@app.route("/annotate")
def annotate():
    if "rater" not in session:
        return redirect(url_for("login"))
    rater = session["rater"]

    requested_id = request.args.get("method_id")
    if requested_id:
        idx_match = SAMPLE_DF.index[SAMPLE_DF["method_id"].astype(str) == str(requested_id)]
        if len(idx_match) == 0:
            return f"Method {requested_id} not in sample.", 404
        idx = int(idx_match[0])
    else:
        with db_connect() as conn:
            done = {
                str(r["method_id"])
                for r in conn.execute(
                    "SELECT method_id FROM annotations WHERE rater = ?", (rater,)
                )
            }
        idx = next(
            (
                i for i, mid in enumerate(SAMPLE_DF["method_id"].astype(str))
                if mid not in done
            ),
            None,
        )
        if idx is None:
            return render_template_string(DONE_TEMPLATE, rater=rater)

    method = SAMPLE_DF.iloc[idx].to_dict()
    existing = get_annotation(rater, str(method["method_id"]))
    completed, total = progress_for(rater)

    return render_template_string(
        ANNOTATE_TEMPLATE,
        method=method,
        existing=existing,
        rater=rater,
        idx=idx,
        total=total,
        completed=completed,
        rubric_html=RUBRIC_HTML,
        reveal_llm=REVEAL_LLM_AFTER_SUBMIT and existing is not None,
        prev_id=(SAMPLE_DF.iloc[idx - 1]["method_id"] if idx > 0 else None),
        next_id=(SAMPLE_DF.iloc[idx + 1]["method_id"] if idx < len(SAMPLE_DF) - 1 else None),
    )


@app.route("/submit", methods=["POST"])
def submit():
    if "rater" not in session:
        return redirect(url_for("login"))
    rater = session["rater"]
    method_id = request.form["method_id"]

    scores, na = {}, {}
    for p in PRINCIPLES:
        val = request.form.get(f"{p}_score")
        if val == "na":
            na[p] = True
            scores[p] = None
        elif val in ("0", "1", "2"):
            scores[p] = int(val)
            na[p] = False
        else:
            return f"Missing score for {p.upper()}", 400

    comment = request.form.get("comment", "").strip()
    seconds_spent = request.form.get("seconds_spent", "")
    try:
        seconds_spent = int(seconds_spent) if seconds_spent else None
    except ValueError:
        seconds_spent = None

    save_annotation(rater, method_id, scores, na, comment, seconds_spent)

    action = request.form.get("action", "next")
    if action == "stay":
        return redirect(url_for("annotate", method_id=method_id))
    return redirect(url_for("annotate"))


@app.route("/progress")
def progress():
    if "rater" not in session:
        return jsonify({"error": "not logged in"}), 401
    rater = session["rater"]
    completed, total = progress_for(rater)
    return jsonify({"completed": completed, "total": total})


@app.route("/logout")
def logout():
    session.pop("rater", None)
    return redirect(url_for("login"))


PRINCIPLES = ["srp", "ocp", "dip"]


# ---------- templates ----------

LOGIN_TEMPLATE = """
<!doctype html>
<html><head><meta charset="utf-8"><title>SOLID Annotation</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 480px; margin: 80px auto; padding: 0 20px; color: #222; }
  h1 { font-size: 22px; }
  input[type=text] { width: 100%; padding: 10px; font-size: 16px; box-sizing: border-box; }
  button { margin-top: 16px; padding: 10px 20px; font-size: 16px; cursor: pointer; }
  .note { color: #666; font-size: 14px; margin-top: 16px; }
</style></head><body>
<h1>SOLID Method Annotation</h1>
<p>Enter your rater identifier (e.g. <code>student_a</code> or <code>student_b</code>).
Use the same identifier every session so your progress is preserved.</p>
<form method="post">
  <input type="text" name="rater" placeholder="rater identifier" required autofocus>
  <button type="submit">Start</button>
</form>
<p class="note">Annotations save automatically. You can close the browser and resume later.</p>
</body></html>
"""

ANNOTATE_TEMPLATE = """
<!doctype html>
<html><head><meta charset="utf-8"><title>{{ idx + 1 }} / {{ total }} &middot; SOLID</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; color: #222; background: #f5f5f7; }
  .topbar { background: white; padding: 10px 20px; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; align-items: center; font-size: 14px; }
  .topbar code { background: #eee; padding: 1px 6px; border-radius: 3px; }
  .topbar a { color: #2469d4; text-decoration: none; }
  .container { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 16px; padding: 16px; max-width: 1500px; margin: 0 auto; }
  .panel { background: white; border: 1px solid #ddd; border-radius: 6px; padding: 16px; }
  .meta { font-size: 12px; color: #555; margin-bottom: 10px; line-height: 1.6; }
  .meta code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px; word-break: break-all; }
  pre.code { background: #1e1e1e; color: #e8e8e8; padding: 14px; border-radius: 4px; overflow-x: auto; font-size: 13px; line-height: 1.5; max-height: 65vh; margin: 0; }
  .principle { margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid #eee; }
  .principle:last-of-type { border-bottom: none; }
  .principle h3 { margin: 0 0 4px; font-size: 15px; }
  .principle .desc { font-size: 12px; color: #555; margin-bottom: 8px; }
  .options { display: flex; flex-wrap: wrap; gap: 12px; }
  .options label { font-size: 13px; cursor: pointer; padding: 4px 8px; border: 1px solid #ddd; border-radius: 4px; background: #fafafa; }
  .options label:hover { background: #f0f0f0; }
  .options input[type=radio] { margin-right: 4px; }
  .options input[type=radio]:checked + span { font-weight: 600; }
  textarea { width: 100%; min-height: 50px; padding: 6px; font-family: inherit; font-size: 13px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
  .actions { margin-top: 14px; display: flex; gap: 8px; }
  button { padding: 8px 16px; font-size: 14px; border: 1px solid #888; background: #fff; border-radius: 4px; cursor: pointer; }
  button.primary { background: #2469d4; color: white; border-color: #2469d4; }
  button:hover { opacity: 0.9; }
  details.rubric { margin-bottom: 14px; }
  details.rubric summary { cursor: pointer; font-weight: 600; font-size: 13px; color: #2469d4; }
  details.rubric > div { margin-top: 8px; font-size: 13px; line-height: 1.5; padding: 10px; background: #fafafa; border-radius: 4px; }
  .nav { font-size: 13px; padding-top: 10px; border-top: 1px solid #eee; margin-top: 14px; }
  .nav a { color: #2469d4; text-decoration: none; margin-right: 12px; }
  .llm-reveal { margin-top: 6px; padding: 6px 8px; background: #f0f7ff; border-left: 3px solid #2469d4; font-size: 12px; color: #345; }
</style></head><body>
<div class="topbar">
  <div><strong>SOLID Annotation</strong> &middot; rater: <code>{{ rater }}</code></div>
  <div>
    <strong>{{ idx + 1 }}</strong> of {{ total }} &middot;
    <strong>{{ completed }}</strong> completed &middot;
    <a href="/logout">log out</a>
  </div>
</div>

<div class="container">
  <div class="panel">
    <div class="meta">
      <strong>{{ method.method_name }}</strong> &middot;
      project <code>{{ method.project }}</code> &middot;
      LOC <code>{{ method.method_loc }}</code><br>
      <code>{{ method.file_path }}:{{ method.startline }}-{{ method.endline }}</code>
    </div>
    <pre class="code">{{ method.method_source }}</pre>
  </div>

  <div class="panel">
    <details class="rubric">
      <summary>Rubric (click to expand)</summary>
      <div>{{ rubric_html|safe }}</div>
    </details>

    <form method="post" action="/submit" id="ann-form">
      <input type="hidden" name="method_id" value="{{ method.method_id }}">
      <input type="hidden" name="seconds_spent" id="seconds_spent" value="">

      {% for p, name, desc in [
        ('srp', 'SRP — Single Responsibility', 'Does this method do one thing, or mix multiple concerns?'),
        ('ocp', 'OCP — Open/Closed', 'Is behavior closed to modification (extends cleanly) or does it hard-code variant logic?'),
        ('dip', 'DIP — Dependency Inversion', 'Does the method depend on abstractions, or instantiate/access concrete dependencies directly?')
      ] %}
      <div class="principle">
        <h3>{{ name }}</h3>
        <div class="desc">{{ desc }}</div>
        <div class="options">
          {% set existing_score = existing[p + '_score'] if existing else None %}
          {% set existing_na = existing[p + '_na'] if existing else 0 %}
          <label><input type="radio" name="{{ p }}_score" value="0" {% if existing_score == 0 and not existing_na %}checked{% endif %} required><span>0 — Violated</span></label>
          <label><input type="radio" name="{{ p }}_score" value="1" {% if existing_score == 1 and not existing_na %}checked{% endif %}><span>1 — Partial</span></label>
          <label><input type="radio" name="{{ p }}_score" value="2" {% if existing_score == 2 and not existing_na %}checked{% endif %}><span>2 — Compliant</span></label>
          <label><input type="radio" name="{{ p }}_score" value="na" {% if existing_na %}checked{% endif %}><span>N/A — Cannot tell</span></label>
        </div>
        {% if reveal_llm %}
        <div class="llm-reveal">
          LLM: score <strong>{{ method[p + '_score_llm'] }}</strong>
          ({{ method[p + '_label_llm'] }}, conf {{ method[p + '_confidence_llm'] }})
          {% if method[p + '_notes_llm'] %} &middot; <em>{{ method[p + '_notes_llm'] }}</em>{% endif %}
        </div>
        {% endif %}
      </div>
      {% endfor %}

      <label style="font-size:13px; font-weight:600;">Comment (optional, short)</label>
      <textarea name="comment" placeholder="Edge cases, why N/A, uncertainty notes...">{{ existing.comment if existing else '' }}</textarea>

      <div class="actions">
        <button type="submit" class="primary" name="action" value="next">Save &amp; next</button>
        <button type="submit" name="action" value="stay">Save &amp; stay</button>
      </div>
    </form>

    <div class="nav">
      {% if prev_id %}<a href="?method_id={{ prev_id }}">← previous</a>{% endif %}
      {% if next_id %}<a href="?method_id={{ next_id }}">next →</a>{% endif %}
      <a href="/annotate" style="float:right;">jump to first un-annotated</a>
    </div>
  </div>
</div>

<script>
  // Track time on page so per-method seconds_spent is logged.
  const t0 = Date.now();
  document.getElementById('ann-form').addEventListener('submit', () => {
    document.getElementById('seconds_spent').value = Math.round((Date.now() - t0) / 1000);
  });
</script>
</body></html>
"""

DONE_TEMPLATE = """
<!doctype html>
<html><head><meta charset="utf-8"><title>All done</title>
<style>body { font-family: -apple-system, sans-serif; max-width: 600px; margin: 80px auto; text-align: center; padding: 0 20px; }</style>
</head><body>
<h1>All methods annotated 🎉</h1>
<p>Rater <code>{{ rater }}</code> has completed every method. You can revisit any item from the URL bar with <code>?method_id=ID</code> to revise.</p>
<p><a href="/logout">Log out</a></p>
</body></html>
"""


@app.context_processor
def inject_globals():
    return {"SAMPLE_DF": SAMPLE_DF}


# ---------- export ----------

def export_results(out_path):
    with db_connect() as conn:
        ann = pd.read_sql_query("SELECT * FROM annotations", conn)
    if ann.empty:
        print("No annotations yet.", file=sys.stderr)
        return
    ann["method_id"] = ann["method_id"].astype(str)
    merged = SAMPLE_DF.merge(ann, on="method_id", how="left")
    merged.to_csv(out_path, index=False)
    print(
        f"Exported {len(ann)} annotations across {ann['rater'].nunique()} raters "
        f"to {out_path}", file=sys.stderr
    )


# ---------- main ----------

def main():
    global SAMPLE_DF, DB_PATH, RUBRIC_HTML

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--rubric", type=Path, default=None)
    parser.add_argument("--export", type=Path, default=None,
                        help="Export annotations to CSV and exit")
    parser.add_argument("--reveal-llm", action="store_true",
                        help="Show LLM score after rater submits (training mode only)")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    SAMPLE_DF = pd.read_csv(args.sample)
    SAMPLE_DF["method_id"] = SAMPLE_DF["method_id"].astype(str)
    DB_PATH = args.db
    db_init()

    if args.rubric and args.rubric.exists():
        RUBRIC_HTML = args.rubric.read_text()
    else:
        RUBRIC_HTML = (
            "<p><em>(No rubric supplied. Pass --rubric path/to/rubric.html with the "
            "exact 0/1/2 rubric used in the LLM prompt so raters score against the "
            "same definitions.)</em></p>"
        )

    if args.reveal_llm:
        global REVEAL_LLM_AFTER_SUBMIT
        REVEAL_LLM_AFTER_SUBMIT = True
        print("WARNING: --reveal-llm enabled. Use only for training rounds, "
              "not real validation.", file=sys.stderr)

    if args.export:
        export_results(args.export)
        return

    print(f"Loaded {len(SAMPLE_DF)} methods from {args.sample}", file=sys.stderr)
    print(f"Database: {DB_PATH}", file=sys.stderr)
    print(f"Open http://{args.host}:{args.port} in a browser.", file=sys.stderr)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()