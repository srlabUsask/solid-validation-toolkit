# SOLID Validation Toolkit

Manual rater toolkit for validating an LLM-based SOLID-principle scoring
pipeline at the method level. This repository contains everything a rater
needs to score 200 Java methods on Single Responsibility (SRP),
Open/Closed (OCP), and Dependency Inversion (DIP) principles.

## Quick start (for raters)

You will need Python 3.10 or later.

    git clone https://github.com/your-org/solid-validation-toolkit.git
    cd solid-validation-toolkit
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/annotate.py --sample data/holdout_sample.csv --db annotations.db --rubric rubric.html

Then open http://127.0.0.1:5000 in a browser.

## How the annotation works

- Pick a rater identifier when prompted (e.g., rater_1). Use the same
  identifier every session so the app remembers your progress.
- Score each method independently. Do not look at another rater's
  scores.
- The 0/1/2 scale and N/A option are explained in the in-app rubric
  (click "Rubric" at the top of the right panel to expand it).
- When in doubt, score 1. Score 1 is the correct answer when the method
  alone does not give you enough signal.
- Use the comment field for unusual cases or notes for the tiebreaker.
- Annotations save automatically. Close and resume any time.

## How long this takes

Roughly 5 to 10 minutes per method, 200 methods total: about 17 to 33
hours, spread over several days.

## When you are done

Send annotations.db back to the principal investigator.

    python -c "import sqlite3; c=sqlite3.connect('annotations.db'); print(c.execute('SELECT COUNT(*) FROM annotations').fetchone()[0], 'annotations saved')"

## Repository layout

    solid-validation-toolkit/
    |-- README.md
    |-- LICENSE
    |-- requirements.txt
    |-- rubric.html
    |-- scripts/
    |   |-- annotate.py
    |   |-- analyze_agreement.py
    |   |-- sample_holdout.py
    |   `-- build_corpus.py
    `-- data/
        |-- holdout_sample.csv
        `-- holdout_manifest.json

Raters only need annotate.py, holdout_sample.csv, and rubric.html.

## License

Apache License 2.0. See LICENSE.
