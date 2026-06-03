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
    score_jd_fit,
    compute_fit_composite,
    _classify_resume_lines,
    _depth_for_pattern,
    TIER1_KEYWORDS,
    TIER2_KEYWORDS,
    TIER3_KEYWORDS,
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
# score_jd_fit — depth scoring
# ---------------------------------------------------------------------------
print("\nscore_jd_fit — depth scoring")

DEEP_RESUME = """
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

SKILLS_ONLY_RESUME = """
Bob Jones
bob@example.com

Experience

Software Engineer | Generic Corp  |  2021 – Present
• Built web services and REST APIs

Skills
AWS, LangGraph, MLOps, DevOps, CI/CD, LLM, Machine Learning
"""

NO_MATCH_RESUME = """
Carol Davis
carol@example.com

Experience

Data Analyst | Numbers Co  |  2020 – Present
• Wrote SQL reports and Excel dashboards
"""

deep  = score_jd_fit(DEEP_RESUME)
skill = score_jd_fit(SKILLS_ONLY_RESUME)
none_ = score_jd_fit(NO_MATCH_RESUME)

check("deep resume: T1 > skills-only T1",
      deep["fit_tier1"] > skill["fit_tier1"],
      f"deep={deep['fit_tier1']} skill={skill['fit_tier1']}")
check("deep resume: T2 > 0",
      deep["fit_tier2"] > 0,
      f"got {deep['fit_tier2']}")
check("deep resume: T3 > 0",
      deep["fit_tier3"] > 0,
      f"got {deep['fit_tier3']}")
check("skills-only: T1 > 0 (found in skills section)",
      skill["fit_tier1"] > 0,
      f"got {skill['fit_tier1']}")
check("no-match resume: T1 == 0",
      none_["fit_tier1"] == 0,
      f"got {none_['fit_tier1']}")
check("no-match resume: T2 == 0",
      none_["fit_tier2"] == 0,
      f"got {none_['fit_tier2']}")
check("no-match resume: T3 == 0",
      none_["fit_tier3"] == 0,
      f"got {none_['fit_tier3']}")
check("deep resume: LangGraph in keywords_found",
      "LangGraph" in deep["fit_keywords_found"],
      str(deep["fit_keywords_found"]))
check("empty resume returns zeros",
      score_jd_fit("")["fit_tier1"] == 0)
check("error placeholder returns zeros",
      score_jd_fit("[PDF read error: ...]")["fit_tier1"] == 0)


# ---------------------------------------------------------------------------
# compute_fit_composite
# ---------------------------------------------------------------------------
print("\ncompute_fit_composite")
check("perfect scores → 10",  compute_fit_composite(10, 10, 10) == 10.0)
check("zero scores → 0",      compute_fit_composite(0,  0,  0)  == 0.0)
check("T1 weighted 60%",
      compute_fit_composite(10, 0, 0) == 6.0,
      f"got {compute_fit_composite(10, 0, 0)}")
check("T2 weighted 30%",
      compute_fit_composite(0, 10, 0) == 3.0,
      f"got {compute_fit_composite(0, 10, 0)}")
check("T3 weighted 10%",
      compute_fit_composite(0, 0, 10) == 1.0,
      f"got {compute_fit_composite(0, 0, 10)}")


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
