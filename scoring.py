#!/usr/bin/env python3
"""
Resume Scoring Engine
=====================
All scoring logic lives here. This file is the source of truth for:
  • resume text extraction
  • candidate ↔ resume file matching
  • supplementary keyword detection (display only, not used in composite score)
  • stability tier + job hopper flag + semantic Layer 1/2 fit (Claude API)
  • per-candidate result assembly

DO NOT modify this file for UI or web-route changes.
UI work belongs in templates/ and app.py only.
Run test_scoring.py after any change to verify correctness.
"""

import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import anthropic
import pdfplumber
from docx import Document


# ---------------------------------------------------------------------------
# Config — change these here, no logic edits needed elsewhere
# ---------------------------------------------------------------------------

MODEL               = "claude-opus-4-8"   # swap model version here only
MAX_RESUME_CHARS    = 14_000
MAX_RETRIES         = 3
RETRY_BASE_DELAY    = 2.0                 # seconds; doubles on each retry
ERROR_KEY           = "_error"            # sentinel key for failed Claude responses
LOCALITY_THRESHOLD  = 2.0                 # fit-composite gap needed for non-local to outrank local within same tier


# ---------------------------------------------------------------------------
# Supplementary keyword detection
# Display only — these do NOT drive the composite score.
# Claude's semantic Layer 1/2 scores are the scoring source of truth.
# ---------------------------------------------------------------------------

# Pre-compiled at module load — no per-call recompilation
_KEYWORDS: list[tuple[str, re.Pattern]] = [
    ("LangGraph",        re.compile(r"\blanggraph\b",                    re.IGNORECASE)),
    ("LangChain",        re.compile(r"\blangchain\b",                    re.IGNORECASE)),
    ("Amazon Bedrock",   re.compile(r"\bamazon\s+bedrock\b|\bbedrock\b", re.IGNORECASE)),
    ("AWS",              re.compile(r"\baws\b",                          re.IGNORECASE)),
    ("MLOps",            re.compile(r"\bmlops\b",                        re.IGNORECASE)),
    ("LLMOps",           re.compile(r"\bllmops\b",                       re.IGNORECASE)),
    ("DevOps",           re.compile(r"\bdevops\b",                       re.IGNORECASE)),
    ("CI/CD",            re.compile(r"\bci[-/]?cd\b",                    re.IGNORECASE)),
    ("AI Agents",        re.compile(r"\bai\s+agents?\b",                 re.IGNORECASE)),
    ("Agentic",          re.compile(r"\bagentic\b",                      re.IGNORECASE)),
    ("Generative AI",    re.compile(r"\bgenerative\s+ai\b|\bgenai\b",    re.IGNORECASE)),
    ("RAG",              re.compile(r"\brag\b|\bretrieval.augmented",    re.IGNORECASE)),
    ("LLM",              re.compile(r"\bllm\b",                          re.IGNORECASE)),
    ("Machine Learning", re.compile(r"\bmachine\s+learning\b",           re.IGNORECASE)),
    ("Kubernetes",       re.compile(r"\bkubernetes\b|\bk8s\b",           re.IGNORECASE)),
    ("MLflow",           re.compile(r"\bmlflow\b",                       re.IGNORECASE)),
    ("Observability",    re.compile(r"\bobservability\b",                re.IGNORECASE)),
    ("Monitoring",       re.compile(r"\bmonitoring\b",                   re.IGNORECASE)),
]


def detect_keywords(resume_text: str) -> list[str]:
    """Return keyword display names found anywhere in resume text (for display only)."""
    if not resume_text:
        return []
    return [name for name, rx in _KEYWORDS if rx.search(resume_text)]


# ---------------------------------------------------------------------------
# Resume text extraction
# ---------------------------------------------------------------------------

def extract_pdf(path: Path) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
    except Exception as e:
        return f"[PDF read error: {e}]"


def extract_docx(path: Path) -> str:
    try:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception as e:
        return f"[DOCX read error: {e}]"


def extract_resume_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in (".docx", ".doc"):
        return extract_docx(path)
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    return f"[Unsupported file type: {suffix}]"


# ---------------------------------------------------------------------------
# Candidate ↔ resume file matching
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = s.lower().replace("_", " ").replace("-", " ").replace("'", "").replace("’", "")
    return " ".join(s.split())


def find_resume_file(candidate_name: str, resume_files: list, warn_fn=None) -> "Path | None":
    name  = _normalize(candidate_name)
    parts = name.split()

    # Exact stem match
    for f in resume_files:
        if name == _normalize(f.stem):
            return f

    # All name parts present in stem
    for f in resume_files:
        if all(p in _normalize(f.stem) for p in parts):
            return f

    # Last-name fallback — only if unambiguous
    if parts:
        last    = parts[-1]
        matches = [f for f in resume_files if last in _normalize(f.stem).split()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            msg = (
                f"Name collision on '{last}' for candidate '{candidate_name}' — "
                f"matched {[f.name for f in matches]}. Skipping fallback to avoid mismatch."
            )
            (warn_fn or print)(f"⚠  {msg}")
            return None

    return None


# ---------------------------------------------------------------------------
# Claude scoring — all signals in one API call per candidate
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are an expert technical recruiter's assistant evaluating resumes for a
senior-level AI/ML infrastructure role (LLMOps Engineer or similar).

The ideal candidate has TWO layers of experience:

  LAYER 1 — Infrastructure Foundation (table stakes)
    DevOps, MLOps, AWS/cloud infrastructure, CI/CD pipelines, Kubernetes,
    system reliability, monitoring, observability. Evidence must appear in
    actual job responsibilities and bullets, not just a skills list.

  LAYER 2 — AI/ML Application (the differentiator)
    Has applied that infrastructure foundation to real AI/ML systems:
    agentic workflows, generative AI infrastructure, LLM deployment and
    operations, RAG pipelines, model inference at scale, AI observability,
    prompt engineering, foundation model integration (Bedrock, GPT, Llama,
    Gemini, etc.). Evidence must appear in job bullets, not skills lists.

Analyze the resume and return ONLY a JSON object. No explanation, no markdown.

━━━ SIGNAL 0: Candidate Name ━━━
Extract the candidate's full name from the resume header or contact section
(typically the very first line or the largest text at the top of the page).
Return it properly capitalized. If the name cannot be determined, return "".

━━━ SIGNAL 1: Stability & Degree Check ━━━
All thresholds calculate dynamically from today's date ({today}).
Three years ago = {three_years_ago}.

─── Tenure Check ───
Based on the candidate's MOST RECENT role only (latest end date or 'Present'):
  Pass — most recent role started on or before {three_years_ago}
  Fail — most recent role started after {three_years_ago}

Rules:
  • Full-time, non-internship roles only
  • Currently employed (no end date or 'Present') = count from start to today
  • Internships, co-ops, and contract/temp stints do NOT count

─── Degree Check ───
Any bachelor's or master's degree listed must have a graduation date
older than {three_years_ago} to pass.
  Pass — graduation date is before {three_years_ago}, or no degree listed
  Fail — graduation date is on or after {three_years_ago}

If a master's degree (including MBA) is listed without a graduation date,
use the bachelor's degree graduation date for this check instead.
If neither degree has a graduation date, treat as Pass.

─── Tier Assignment ───
  Tier 1 — Passes both tenure AND degree check
  Tier 2 — Fails tenure only; degree is fine or not listed.
            Slight score reduction but strong JD fit can compensate.
  Tier 3 — Fails degree check regardless of tenure.
            HARD FLOOR: automatically ranks below all Tier 1 and Tier 2
            candidates no matter how strong the resume.
  Tier 4 — Fails both tenure AND degree check.
            Ranks below Tier 3. Absolute bottom of the list.
            Flagged for potential future elimination pending team approval.

━━━ SIGNAL 2: Locality ━━━
Detect any reference to NY, New York, NYC, Brooklyn, Queens, Manhattan,
Bronx, Staten Island, Long Island, New Jersey, or CT in the resume.
  local: true — NY metro reference detected
  local: false — no reference found
Also extract the candidate's stated city/state (e.g. "Brooklyn, NY" or
"Austin, TX"). Return an empty string if no location is found.

━━━ SIGNAL 3: Layer 1 Score (Infrastructure Foundation) ━━━
Score 0–10 based on depth and consistency of evidence in job bullets:
  0  = no infrastructure experience evident
  4  = some exposure, mostly surface-level or skills-list only
  7  = solid hands-on infrastructure background across roles
  10 = deep, consistent infrastructure engineering, clearly a core strength

━━━ SIGNAL 4: Layer 2 Score (AI/ML Application) ━━━
Score 0–10 based on whether the candidate has applied infrastructure
skills to real AI/ML systems. Skills-list mentions score low;
job bullet evidence scores high.
  0  = no AI/ML application experience evident
  4  = some exposure, mostly buzzwords or skills-list only
  7  = clear evidence of applying infrastructure to AI/ML systems
  10 = deep, consistent AI/ML application work, clearly a core strength

━━━ SIGNAL 5: Job Hopper Flag ━━━
Flag true if career history shows a pattern of instability — multiple
stints under 18 months with no clear context (layoffs, closures, or
explicit contract roles are acceptable explanations).
A strong recent role reduces but does not erase an extreme earlier pattern.

━━━ RANKING WEIGHTS ━━━
Within each tier, candidates rank by composite score in this priority order:
  1. Tier assignment (1 is highest, 4 is lowest — hard boundaries)
  2. Locality (local candidates rank above non-local within the same tier)
  3. JD fit composite (Layer 1 at 40%, Layer 2 at 60%, bonus if both >= 5)

Non-local candidates may rank above local candidates within the same tier
only if their JD fit composite is significantly higher.
Locality threshold gap: {locality_threshold} [TUNABLE — calibrate after
validation run against recruiter eye test]

Return this JSON structure exactly:
{{
  "candidate_name": "Full name from resume header, or empty string",
  "stability_tier": 1 | 2 | 3 | 4,
  "tenure_pass": true | false,
  "degree_pass": true | false,
  "local": true | false,
  "detected_location": "City, State or empty string",
  "layer1_score": 0-10,
  "layer2_score": 0-10,
  "fit_composite": 0-10,
  "job_hopper": true | false,
  "summary": "2-3 sentence plain English explanation of why this candidate ranked where they did"
}}"""


def _build_system_prompt() -> str:
    """Inject runtime values into the prompt template."""
    today = datetime.now()
    try:
        three_years_ago = today.replace(year=today.year - 3)
    except ValueError:
        three_years_ago = today.replace(year=today.year - 3, day=28)
    return _SYSTEM_TEMPLATE.format(
        today=today.strftime("%Y-%m-%d"),
        three_years_ago=three_years_ago.strftime("%Y-%m-%d"),
        locality_threshold=LOCALITY_THRESHOLD,
    )


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # Extract the first complete {...} object so trailing text doesn't break parsing.
    start = text.find("{")
    if start != -1:
        depth, in_str, escape = 0, False, False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
            elif ch == "\\" and in_str:
                escape = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        text = text[start:i + 1]
                        break

    return json.loads(text)


def _error_result(reason: str) -> dict:
    return {
        "candidate_name":             "",
        "stability_tier":             None,
        "stability_rationale":        reason,
        "most_recent_role_years":     None,
        "job_hopper_flag":            False,
        "job_hopper_rationale":       reason,
        "layer1_score":               0,
        "layer1_rationale":           reason,
        "layer2_score":               0,
        "layer2_rationale":           reason,
        "estimated_experience_years": None,
        "ny_signal":                  False,
        "ny_rationale":               reason,
        "detected_location":          "Unknown",
        ERROR_KEY:                    reason,
    }


def score_candidate(
    client: anthropic.Anthropic,
    candidate_name: str,
    resume_text: str,
) -> dict:
    """Score one candidate via Claude. Retries on transient API errors."""
    if not resume_text or resume_text.startswith("["):
        return _error_result(resume_text or "Empty resume")
    if len(resume_text) > MAX_RESUME_CHARS:
        resume_text = resume_text[:MAX_RESUME_CHARS] + "\n\n[truncated]"

    system_prompt = _build_system_prompt()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=500,
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{
                    "role": "user",
                    "content": f"Candidate: {candidate_name}\n\n--- RESUME ---\n{resume_text}",
                }],
            )
            return _parse_json(response.content[0].text)

        except json.JSONDecodeError as e:
            return _error_result(f"JSON parse error: {e}")

        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
            else:
                return _error_result("Rate limit exceeded after retries")

        except anthropic.APIStatusError as e:
            if e.status_code in (503, 529) and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
            else:
                return _error_result(f"API error {e.status_code}: {e.message}")

        except Exception as e:
            return _error_result(f"Unexpected error: {e}")

    return _error_result("Max retries exceeded")


# ---------------------------------------------------------------------------
# Composite fit score
# ---------------------------------------------------------------------------

def compute_fit_composite(layer1: float, layer2: float) -> float:
    """
    Layer 1 is the foundation (40%), Layer 2 is the differentiator (60%).
    Candidates with both layers strong (both >= 5) receive a small bonus,
    rewarding the combination rather than just summing two independent scores.
    Max composite is 10.0.
    """
    base = layer1 * 0.4 + layer2 * 0.6
    if layer1 >= 5 and layer2 >= 5:
        bonus = min(1.5, (layer1 + layer2) / 20 * 1.5)
        return round(min(10.0, base + bonus), 1)
    return round(base, 1)


# ---------------------------------------------------------------------------
# Per-candidate processor — runs in thread pool
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def process_candidate(
    idx: int,
    total: int,
    client: anthropic.Anthropic,
    name: str,
    resume_file,
    gh_row: dict,
) -> dict:
    resume_text  = extract_resume_text(resume_file) if resume_file else ""
    resume_found = resume_file is not None

    if not resume_found:
        result        = _error_result("No resume file found")
        keywords      = []
        fit_composite = None
        status        = "⚠  no resume matched"
    else:
        result   = score_candidate(client, name, resume_text)
        keywords = detect_keywords(resume_text)

        if ERROR_KEY in result:
            fit_composite = None
            status = f"✗  {result[ERROR_KEY][:60]}"
        else:
            l1 = result.get("layer1_score", 0)
            l2 = result.get("layer2_score", 0)
            fit_composite = compute_fit_composite(l1, l2)
            hopper_tag = " 🚩" if result.get("job_hopper") else ""
            status = (
                f"✓  tier={result['stability_tier']}  "
                f"local={'Y' if result.get('local') else 'N'}  "
                f"L1={l1} L2={l2} fit={fit_composite}{hopper_tag}"
            )

    with _print_lock:
        print(f"[{idx:>3}/{total}] {name} … {status}", flush=True)

    tier  = result.get("stability_tier")
    l1    = result.get("layer1_score", 0) or 0
    l2    = result.get("layer2_score", 0) or 0
    local = result.get("local", False)
    ok    = resume_found and ERROR_KEY not in result

    return {
        # Internal sort keys — not displayed
        "_stability_tier": tier,
        "_fit_composite":  fit_composite,
        "_local":          local,

        # Display columns
        "Candidate Name":       result.get("candidate_name") or name,
        "Stability Tier":       tier or "",
        "Tenure Pass":          ("Yes" if result.get("tenure_pass") else "No") if ok else "",
        "Degree Pass":          ("Yes" if result.get("degree_pass") else "No") if ok else "",
        "Local":                "Yes" if local else "No",
        "Job Hopper":           "⚠ Review" if result.get("job_hopper") else "",
        "JD Fit Composite":     fit_composite,
        "Layer 1 (Foundation)": l1 if ok else None,
        "Layer 2 (AI/ML)":      l2 if ok else None,
        "Keywords Detected":    ", ".join(keywords),
        "Summary":              result.get("summary", ""),
        "Resume Found":         "Yes" if resume_found else "No",
        "Resume Link":          resume_file.name if resume_file else "",
        **gh_row,
    }


# ---------------------------------------------------------------------------
# Sort key: tier ASC → locality-adjusted fit composite DESC
# ---------------------------------------------------------------------------

def sort_key(row: dict) -> tuple:
    tier     = row.get("_stability_tier") or 99
    fit      = row.get("_fit_composite") or 0
    adjusted = fit if row.get("_local") else fit - LOCALITY_THRESHOLD
    return (
        tier,
        -adjusted,
    )
