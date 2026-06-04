#!/usr/bin/env python3
"""
Scoring engine smoke tests.
Run this after any change to scoring.py:

    python3 test_scoring.py

All tests must pass before a change to scoring.py is considered safe.
"""

import sys
from pathlib import Path

from scoring import (
    _normalize,
    find_resume_file,
    detect_keywords,
    compute_fit_composite,
)

PASS = "✓"
FAIL = "✗"
_failures = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f"  [{detail}]" if detail else ""))
        _failures.append(label)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------
print("\n_normalize")
check("lowercases",                  _normalize("Alice Smith") == "alice smith")
check("strips underscores",          _normalize("Alice_Smith") == "alice smith")
check("strips hyphens",              _normalize("Alice-Smith") == "alice smith")
check("strips apostrophes",          _normalize("O'Brien")    == "obrien")
check("strips smart apostrophes",    _normalize("O’Brien") == "obrien")
check("collapses whitespace",        _normalize("Alice  Smith") == "alice smith")


# ---------------------------------------------------------------------------
# find_resume_file
# ---------------------------------------------------------------------------
print("\nfind_resume_file")

def _fake_files(*names):
    return [Path(n) for n in names]

files = _fake_files("Alice_Smith.pdf", "Jordan_Williams.pdf", "Brennan_OBrien.pdf",
                    "Priya_Sharma.pdf", "Marcus_Chen.pdf")

check("exact match",                find_resume_file("Alice Smith",      files) == Path("Alice_Smith.pdf"))
check("all-parts match",            find_resume_file("Jordan Williams",  files) == Path("Jordan_Williams.pdf"))
check("apostrophe in name",         find_resume_file("Brennan O'Brien",  files) == Path("Brennan_OBrien.pdf"))
check("smart-apostrophe in name",   find_resume_file("Brennan O’Brien", files) == Path("Brennan_OBrien.pdf"))
check("returns None when no match", find_resume_file("Nobody Here",      files) is None)


# ---------------------------------------------------------------------------
# detect_keywords
# ---------------------------------------------------------------------------
print("\ndetect_keywords")

KEYWORD_RESUME = """
Alice Smith
alice@example.com | Brooklyn, NY

Experience

LangGraph Engineer | Acme AI  |  2022 – Present
• Built multi-agent workflows with LangGraph and Amazon Bedrock
• Deployed ML models to production on AWS using CI/CD pipelines
• Implemented MLOps observability with MLflow and Kubernetes
• Used LLMOps tooling for monitoring and LLM evaluation

Skills
Python, AWS, DevOps, Machine Learning, AI Agents
"""

NO_KEYWORD_RESUME = """
Carol Davis
carol@example.com

Experience

Data Analyst | Numbers Co  |  2020 – Present
• Wrote SQL reports and Excel dashboards
"""

kw = detect_keywords(KEYWORD_RESUME)
no_kw = detect_keywords(NO_KEYWORD_RESUME)

check("LangGraph detected",          "LangGraph"      in kw, str(kw))
check("Amazon Bedrock detected",     "Amazon Bedrock" in kw, str(kw))
check("AWS detected",                "AWS"            in kw, str(kw))
check("MLOps detected",              "MLOps"          in kw, str(kw))
check("no false positives on plain resume", no_kw == [], str(no_kw))
check("empty string returns []",     detect_keywords("") == [])


# ---------------------------------------------------------------------------
# compute_fit_composite
# ---------------------------------------------------------------------------
print("\ncompute_fit_composite")
check("perfect scores → 10",
      compute_fit_composite(10, 10) == 10.0,
      f"got {compute_fit_composite(10, 10)}")
check("zero scores → 0",
      compute_fit_composite(0, 0) == 0.0,
      f"got {compute_fit_composite(0, 0)}")
check("L1 only (no bonus): 10*0.4 = 4.0",
      compute_fit_composite(10, 0) == 4.0,
      f"got {compute_fit_composite(10, 0)}")
check("L2 only (no bonus): 10*0.6 = 6.0",
      compute_fit_composite(0, 10) == 6.0,
      f"got {compute_fit_composite(0, 10)}")
check("both >= 5 receives bonus above base",
      compute_fit_composite(7, 7) > round(7 * 0.4 + 7 * 0.6, 1),
      f"got {compute_fit_composite(7, 7)}, base would be 7.0")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
print()
if _failures:
    print(f"FAILED — {len(_failures)} test(s) failed:")
    for f in _failures:
        print(f"  • {f}")
    sys.exit(1)
else:
    print(f"All tests passed.")
