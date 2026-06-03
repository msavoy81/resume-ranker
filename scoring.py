#!/usr/bin/env python3
"""
Resume Scoring Engine
=====================
All scoring logic lives here. This file is the source of truth for:
  • resume text extraction
  • candidate ↔ resume file matching
  • JD fit keyword scoring (pure Python, zero API calls)
  • stability tier + NY signal (Claude API)
  • per-candidate result assembly

DO NOT modify this file for UI or web-route changes.
UI work belongs in templates/ and app.py only.
Run test_scoring.py after any change to verify correctness.
"""

import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path

import anthropic
import pdfplumber
from docx import Document


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


def find_resume_file(candidate_name: str, resume_files: list) -> "Path | None":
    name  = _normalize(candidate_name)
    parts = name.split()

    for f in resume_files:
        if name == _normalize(f.stem):
            return f
    for f in resume_files:
        if all(p in _normalize(f.stem) for p in parts):
            return f
    if parts:
        last = parts[-1]
        for f in resume_files:
            if last in _normalize(f.stem).split():
                return f
    return None


# ---------------------------------------------------------------------------
# JD fit — pure-code keyword depth scoring (no Claude)
# ---------------------------------------------------------------------------

# (display_name, regex_pattern)
TIER1_KEYWORDS = [
    ("LangGraph",      r"\blanggraph\b"),
    ("Amazon Bedrock", r"\bamazon\s+bedrock\b"),
    ("AWS",            r"\baws\b"),
]
TIER2_KEYWORDS = [
    ("MLOps",     r"\bmlops\b"),
    ("LLMOps",    r"\bllmops\b"),
    ("DevOps",    r"\bdevops\b"),
    ("CI/CD",     r"\bci[-/]?cd\b"),
    ("AI Agents", r"\bai\s+agents?\b"),
]
TIER3_KEYWORDS = [
    ("Machine Learning",        r"\bmachine\s+learning\b"),
    ("LLM",                     r"\bllm\b"),
    ("Artificial Intelligence", r"\bartificial\s+intelligence\b"),
    ("Observability",           r"\bobservability\b"),
    ("Monitoring",              r"\bmonitoring\b"),
    ("Production Deployment",   r"\bprod(?:uction)?\s+deploy|\bdeploy\w*\s+to\s+prod"),
    ("Kubernetes",              r"\bkubernetes\b|\bk8s\b"),
    ("MLflow",                  r"\bmlflow\b"),
]

_BULLET_RE  = re.compile(r"^\s*[•\-\*•◦⁃◦]\s")
_DATE_RE    = re.compile(r"\b(19|20)\d{2}\b")
_SKILLS_SEC = re.compile(
    r"^\s*(skills?|technical\s+skills?|technologies|tech\s+stack|tools?|expertise|competencies)\s*:?\s*$",
    re.IGNORECASE,
)
_EXP_SEC = re.compile(
    r"^\s*(experience|work\s+experience|employment|professional\s+experience|work\s+history|career)\s*:?\s*$",
    re.IGNORECASE,
)
_OTHER_SEC = re.compile(
    r"^\s*(education|academic|projects?|certifications?|summary|objective|profile|"
    r"about|languages?|awards?|publications?|volunteer)\s*:?\s*$",
    re.IGNORECASE,
)


def _classify_resume_lines(text: str) -> list:
    """Tag every line with its structural role and section context."""
    classified   = []
    current_sec  = "preamble"

    for line in text.split("\n"):
        stripped = line.strip()
        info = {"stripped": stripped, "section": current_sec, "type": "body"}

        if not stripped:
            info["type"] = "blank"
        elif _SKILLS_SEC.match(stripped):
            current_sec = "skills"
            info.update(type="section_header", section="skills")
        elif _EXP_SEC.match(stripped):
            current_sec = "experience"
            info.update(type="section_header", section="experience")
        elif _OTHER_SEC.match(stripped):
            current_sec = "other"
            info.update(type="section_header", section="other")
        elif _BULLET_RE.match(line):
            info["type"] = "bullet"
        elif _DATE_RE.search(stripped) and len(stripped) < 180 and not _BULLET_RE.match(line):
            info["type"] = "title_line"

        info["section"] = current_sec
        classified.append(info)

    return classified


def _depth_for_pattern(pattern: str, classified: list) -> int:
    """
    Depth score 0–10 based on WHERE a keyword appears in the resume:
      0  = not found
      2  = found only in skills / summary / other section
      5  = found in experience section (body text)
      8  = found on a job title line in experience
      10 = found on a job title line AND at least one bullet follows
    """
    rx   = re.compile(pattern, re.IGNORECASE)
    hits = [i for i, item in enumerate(classified) if rx.search(item["stripped"])]
    if not hits:
        return 0

    exp_title_hits = [i for i in hits
                      if classified[i]["section"] == "experience"
                      and classified[i]["type"] == "title_line"]
    exp_body_hits  = [i for i in hits
                      if classified[i]["section"] == "experience"
                      and classified[i]["type"] in ("body", "bullet")]
    non_exp_hits   = [i for i in hits
                      if classified[i]["section"] in ("skills", "preamble", "other")]

    if exp_title_hits:
        for title_idx in exp_title_hits:
            window = classified[title_idx + 1: title_idx + 20]
            for item in window:
                if item["type"] == "title_line":
                    break
                if item["type"] == "bullet":
                    return 10
        return 8

    if exp_body_hits:
        return 5

    if non_exp_hits:
        return 2

    return 1


def score_jd_fit(resume_text: str) -> dict:
    """Pure-code JD fit scoring — no API call."""
    if not resume_text or resume_text.startswith("["):
        return {"fit_tier1": 0, "fit_tier2": 0, "fit_tier3": 0,
                "fit_keywords_found": [], "fit_rationale": "No resume text"}

    classified     = _classify_resume_lines(resume_text)
    keywords_found = []

    def _tier_score(kw_list: list) -> float:
        depths = []
        for display, pattern in kw_list:
            d = _depth_for_pattern(pattern, classified)
            depths.append(d)
            if d > 0:
                keywords_found.append(display)
        if not depths or max(depths) == 0:
            return 0.0
        return round(max(depths) * 0.6 + (sum(depths) / len(depths)) * 0.4, 1)

    t1 = min(10.0, _tier_score(TIER1_KEYWORDS))
    t2 = min(10.0, _tier_score(TIER2_KEYWORDS))

    t3_depths = [_depth_for_pattern(p, classified) for _, p in TIER3_KEYWORDS]
    found_t3  = sum(1 for d in t3_depths if d > 0)
    t3        = min(10.0, round(found_t3 * 1.5, 1))
    for (display, _), depth in zip(TIER3_KEYWORDS, t3_depths):
        if depth > 0:
            keywords_found.append(display)

    seen, kw_deduped = set(), []
    for kw in keywords_found:
        if kw not in seen:
            seen.add(kw)
            kw_deduped.append(kw)

    rationale = f"T1={t1:.0f} T2={t2:.0f} T3={t3:.0f}"
    if kw_deduped:
        rationale = f"Found: {', '.join(kw_deduped)} | {rationale}"

    return {
        "fit_tier1": t1,
        "fit_tier2": t2,
        "fit_tier3": t3,
        "fit_keywords_found": kw_deduped,
        "fit_rationale": rationale,
    }


def compute_fit_composite(t1: float, t2: float, t3: float) -> float:
    """Tier1=60%, Tier2=30%, Tier3=10% → 0–10."""
    return round(t1 * 0.6 + t2 * 0.3 + t3 * 0.1, 1)


# ---------------------------------------------------------------------------
# Claude scoring — stability tier + NY signal only
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """\
You are an expert recruiter's assistant. Analyze the resume for two signals.
Return ONLY a JSON object — no explanation, no markdown fences.

━━━ SIGNAL 1: Stability Tier ━━━
Base the tier on the candidate's MOST RECENT role (latest end date, or 'Present'):

  Tier A — most recent role lasted 3+ years (regardless of current employment status)
  Tier B — most recent role lasted <3 years, AND NO recent undergrad or master's degree
            completed in {cutoff_year} or later
  Tier C — most recent role lasted <3 years, AND candidate has an undergrad OR master's
            degree completed in {cutoff_year} or later (likely OPT/student visa — flag for review)

Rules:
  • Count only full-time, non-internship roles for duration
  • If currently employed (no end date or says 'Present'), count from start date to today ({today})
  • Internships, co-ops, and contract/temp stints do NOT count toward tenure
  • Recent degree types that trigger Tier C: Bachelor's, B.S., B.A., B.Eng., Master's, M.S.,
    M.A., M.Eng., MBA, MIS, MEng, MS, MA — any undergrad or graduate degree completed in
    {cutoff_year} or later
  • IMPORTANT: when the graduation year is ambiguous or unclear, err toward Tier C (flag for review)
  • IMPORTANT: set has_recent_degree=true whenever you find any undergrad or master's degree
    completed in {cutoff_year} or later, even if you are uncertain about the exact year

━━━ SIGNAL 2: New York Signal ━━━
Report true if ANY NY-metro connection is present:
  • Current/listed address in NYC, Long Island, northern NJ, or Fairfield County CT
  • Any employer based in NY metro
  • Any university in NY metro

Return EXACTLY this JSON (no extra fields):
{{
  "stability_tier": "A" | "B" | "C",
  "stability_rationale": "<one concise sentence>",
  "most_recent_role_years": <float or null>,
  "recent_degree_year": <int or null>,
  "has_recent_degree": true | false,
  "ny_signal": true | false,
  "ny_rationale": "<one concise sentence>",
  "detected_location": "<city/region or 'Unknown'>"
}}"""


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def _error_claude(reason: str) -> dict:
    return {
        "stability_tier": None, "stability_rationale": reason,
        "most_recent_role_years": None, "recent_degree_year": None,
        "has_recent_degree": False,
        "ny_signal": False, "ny_rationale": reason,
        "detected_location": "Unknown", "_error": reason,
    }


MAX_RESUME_CHARS = 14_000


def score_stability_ny(
    client: anthropic.Anthropic,
    candidate_name: str,
    resume_text: str,
    system_prompt: str,
) -> dict:
    if not resume_text or resume_text.startswith("["):
        return _error_claude(resume_text or "Empty resume")
    if len(resume_text) > MAX_RESUME_CHARS:
        resume_text = resume_text[:MAX_RESUME_CHARS] + "\n\n[truncated]"

    try:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=350,
            system=[{"type": "text", "text": system_prompt,
                      "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user",
                       "content": f"Candidate: {candidate_name}\n\n--- RESUME ---\n{resume_text}"}],
        )
        return _parse_json(response.content[0].text)
    except json.JSONDecodeError as e:
        return _error_claude(f"JSON parse error: {e}")
    except Exception as e:
        return _error_claude(f"API error: {e}")


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
    system_prompt: str,
    gh_row: dict,
) -> dict:
    resume_text  = extract_resume_text(resume_file) if resume_file else ""
    resume_found = resume_file is not None

    if not resume_found:
        claude = _error_claude("No resume file found")
        fit    = {"fit_tier1": 0, "fit_tier2": 0, "fit_tier3": 0,
                  "fit_keywords_found": [], "fit_rationale": "No resume"}
        status = "⚠  no resume matched"
    else:
        claude = score_stability_ny(client, name, resume_text, system_prompt)
        fit    = score_jd_fit(resume_text)

        if "_error" in claude:
            status = f"✗  {claude['_error'][:60]}"
        else:
            t1, t2, t3 = fit["fit_tier1"], fit["fit_tier2"], fit["fit_tier3"]
            composite  = compute_fit_composite(t1, t2, t3)
            kw_preview = ", ".join(fit["fit_keywords_found"][:4]) or "none"
            status = (
                f"✓  tier={claude['stability_tier']}  "
                f"ny={'Y' if claude['ny_signal'] else 'N'}  "
                f"fit={composite} (t1={t1:.0f} t2={t2:.0f} t3={t3:.0f})  [{kw_preview}]"
            )

    with _print_lock:
        print(f"[{idx:>3}/{total}] {name} … {status}", flush=True)

    tier      = claude.get("stability_tier")
    ny_signal = claude.get("ny_signal", False) if "_error" not in claude else False

    # Safety net: override to Tier C if Claude's own fields contradict its tier label.
    if tier in ("A", "B") and "_error" not in claude:
        degree_year    = claude.get("recent_degree_year")
        has_recent_deg = claude.get("has_recent_degree", False)
        role_years     = claude.get("most_recent_role_years")
        cutoff         = datetime.now().year - 3
        recent_deg     = has_recent_deg or (degree_year is not None and degree_year >= cutoff)
        short_tenure   = role_years is not None and role_years < 3
        if recent_deg and short_tenure:
            tier = "C"

    t1 = fit.get("fit_tier1", 0) or 0
    t2 = fit.get("fit_tier2", 0) or 0
    t3 = fit.get("fit_tier3", 0) or 0
    fit_composite = compute_fit_composite(t1, t2, t3) if resume_found and "_error" not in claude else None

    return {
        "_stability_tier": tier,
        "_ny_signal":      ny_signal,
        "_fit_composite":  fit_composite,
        "FLAG":            "⚠ OPT REVIEW" if tier == "C" else "",
        "Candidate Name":  name,
        "Stability Tier":  tier or "",
        "NY Signal":       "Yes" if ny_signal else "No",
        "JD Fit Composite": fit_composite,
        "JD Fit Tier 1\n(LangGraph/Bedrock/AWS)":   t1 if resume_found and "_error" not in claude else None,
        "JD Fit Tier 2\n(MLOps/DevOps/AI Agents)":  t2 if resume_found and "_error" not in claude else None,
        "JD Fit Tier 3\n(General AI/ML)":            t3 if resume_found and "_error" not in claude else None,
        "Keywords Found":  ", ".join(fit.get("fit_keywords_found", [])),
        "Stability Rationale": claude.get("stability_rationale", ""),
        "NY Rationale":    claude.get("ny_rationale", ""),
        "Fit Rationale":   fit.get("fit_rationale", ""),
        "Detected Location": claude.get("detected_location", ""),
        "Most Recent Role (yrs)": claude.get("most_recent_role_years"),
        "Recent Degree Year":     claude.get("recent_degree_year"),
        "Resume Found":    "Yes" if resume_found else "No",
        **gh_row,
    }


# ---------------------------------------------------------------------------
# Sort key: Tier A > B > C → NY=Yes → fit composite DESC
# ---------------------------------------------------------------------------

def sort_key(row: dict) -> tuple:
    tier_order = {"A": 0, "B": 1, "C": 2, None: 3, "": 3}
    return (
        tier_order.get(row.get("_stability_tier"), 3),
        0 if row.get("_ny_signal") else 1,
        -(row.get("_fit_composite") or 0),
    )
