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

MODEL            = "claude-opus-4-8"   # swap model version here only
MAX_RESUME_CHARS = 14_000
MAX_RETRIES      = 3
RETRY_BASE_DELAY = 2.0                 # seconds; doubles on each retry
ERROR_KEY        = "_error"            # sentinel key for failed Claude responses


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
You are an expert technical recruiter's assistant evaluating resumes for a \
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
    Gemini, etc.). Again — evidence in job bullets, not skills lists.

Analyze the resume and return ONLY a JSON object. No explanation, no markdown.

━━━ SIGNAL 1: Stability Tier ━━━
Based on the candidate's MOST RECENT role (latest end date, or 'Present'):
  Tier A — most recent role lasted 3+ years
  Tier B — most recent role lasted less than 3 years

Rules:
  • Full-time, non-internship roles only
  • If currently employed (no end date or says 'Present'), count from start
    date to today ({today})
  • Internships, co-ops, and pure contract/temp stints do NOT count

━━━ SIGNAL 2: Job Hopper Flag ━━━
Flag true if the career history shows a pattern of instability — multiple
stints under 18 months with no clear context (layoffs, closures, or explicit
contract roles are acceptable explanations). A strong recent role reduces but
does not erase an extreme earlier pattern.

━━━ SIGNAL 3: Layer 1 Score (Infrastructure Foundation) ━━━
0–10 based on depth and consistency of evidence in job bullets:
  0  = no infrastructure experience evident
  4  = some exposure, mostly surface-level or skills-list only
  7  = solid hands-on infrastructure background across roles
  10 = deep, consistent infrastructure engineering, clearly a core strength

━━━ SIGNAL 4: Layer 2 Score (AI/ML Application) ━━━
0–10 based on whether the candidate has applied infrastructure skills to
real AI/ML systems. Skills-list mentions score low; job bullet evidence scores high.
  0  = no AI/ML application evidence
  4  = peripheral or conceptual exposure only
  7  = clear evidence of building or operating AI/ML systems
  10 = deep, consistent AI/ML systems work, a defining part of their career

━━━ SIGNAL 5: Estimated Experience ━━━
Estimate total years of relevant full-time professional experience
(exclude internships and education time).

━━━ SIGNAL 6: NY Metro Signal ━━━
Report true if ANY of the following:
  • Current/listed address in NYC, Long Island, northern NJ, or Fairfield County CT
  • Any employer based in NY metro
  • Any university in NY metro

Return EXACTLY this JSON (no extra fields):
{{
  "stability_tier":             "A" | "B",
  "stability_rationale":        "<one concise sentence>",
  "most_recent_role_years":     <float or null>,
  "job_hopper_flag":            true | false,
  "job_hopper_rationale":       "<one concise sentence>",
  "layer1_score":               <int 0–10>,
  "layer1_rationale":           "<one concise sentence>",
  "layer2_score":               <int 0–10>,
  "layer2_rationale":           "<one concise sentence>",
  "estimated_experience_years": <float or null>,
  "ny_signal":                  true | false,
  "ny_rationale":               "<one concise sentence>",
  "detected_location":          "<city/region or 'Unknown'>"
}}"""


def _build_system_prompt() -> str:
    """Inject runtime values into the prompt template."""
    return _SYSTEM_TEMPLATE.format(today=datetime.now().strftime("%Y-%m-%d"))


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def _error_result(reason: str) -> dict:
    return {
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
            hopper_tag = " 🚩" if result.get("job_hopper_flag") else ""
            status = (
                f"✓  tier={result['stability_tier']}  "
                f"ny={'Y' if result['ny_signal'] else 'N'}  "
                f"L1={l1} L2={l2} fit={fit_composite}{hopper_tag}"
            )

    with _print_lock:
        print(f"[{idx:>3}/{total}] {name} … {status}", flush=True)

    tier = result.get("stability_tier")
    l1   = result.get("layer1_score", 0) or 0
    l2   = result.get("layer2_score", 0) or 0
    ok   = resume_found and ERROR_KEY not in result

    return {
        # Internal sort keys — not displayed
        "_stability_tier":  tier,
        "_fit_composite":   fit_composite,
        "_job_hopper_flag": result.get("job_hopper_flag", False),

        # Display columns
        "Candidate Name":          name,
        "Stability Tier":          tier or "",
        "Job Hopper":              "⚠ Review" if result.get("job_hopper_flag") else "",
        "JD Fit Composite":        fit_composite,
        "Layer 1 (Foundation)":    l1 if ok else None,
        "Layer 2 (AI/ML)":         l2 if ok else None,
        "Keywords Detected":       ", ".join(keywords),
        "Est. Experience (yrs)":   result.get("estimated_experience_years"),
        "Stability Rationale":     result.get("stability_rationale", ""),
        "Job Hopper Rationale":    result.get("job_hopper_rationale", ""),
        "Layer 1 Rationale":       result.get("layer1_rationale", ""),
        "Layer 2 Rationale":       result.get("layer2_rationale", ""),
        "NY Signal":               "Yes" if result.get("ny_signal") else "No",
        "NY Rationale":            result.get("ny_rationale", ""),
        "Detected Location":       result.get("detected_location", ""),
        "Most Recent Role (yrs)":  result.get("most_recent_role_years"),
        "Resume Found":            "Yes" if resume_found else "No",
        **gh_row,
    }


# ---------------------------------------------------------------------------
# Sort key: Tier A > B → fit composite DESC → job hoppers last within tier
# ---------------------------------------------------------------------------

def sort_key(row: dict) -> tuple:
    tier_order = {"A": 0, "B": 1, None: 2, "": 2}
    return (
        tier_order.get(row.get("_stability_tier"), 2),
        -(row.get("_fit_composite") or 0),
        1 if row.get("_job_hopper_flag") else 0,
    )
