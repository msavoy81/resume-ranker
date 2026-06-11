#!/usr/bin/env python3
"""
Greenhouse Harvest API Integration
===================================
Fetches candidates, applications, and resumes for a given job directly from
the Greenhouse Harvest API — no manual CSV export or ZIP upload needed.

OUTPUT CONTRACT
---------------
All public functions return data in the exact shape that ranker.py already
expects:

  tasks : list of tuples
      (idx: int, total: int, name: str, resume_file: Path | None, gh_row: dict)

  gh_row keys (all optional — ranker.py guards each with `if col:` checks):
      "Email"               : str
      "Applied At"          : str   (YYYY-MM-DD)
      "Stage"               : str
      "Greenhouse Location" : str

  resume_file : Path pointing to a local file on disk
                (downloaded to a temp dir in live mode, from resumes/ in test mode)

TEST MODE
---------
Set TEST_MODE = True (default) to use the local candidates.csv + resumes/ folder
as a mock data source. No API calls are made. Safe to run without credentials.

Set TEST_MODE = False to hit the real Greenhouse Harvest API. Requires:
  - GREENHOUSE_API_KEY env var  (or populated .env file)
  - GREENHOUSE_BASE_URL env var (optional, defaults to standard harvest URL)
"""

import csv
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TEST_MODE = True   # ← flip to False to use the real Greenhouse API

_BASE_DIR         = Path(__file__).parent
_DEFAULT_BASE_URL = "https://harvest.greenhouse.io/v1"
_REQUEST_TIMEOUT  = 30   # seconds per HTTP request
_DOWNLOAD_TIMEOUT = 60   # seconds for resume file downloads
_RETRY_STATUSES   = {429, 500, 502, 503, 504}
_MAX_RETRIES      = 3
_RETRY_DELAY      = 2.0  # seconds; doubles on each retry


# ---------------------------------------------------------------------------
# Test-mode data paths
# ---------------------------------------------------------------------------

_TEST_CSV     = _BASE_DIR / "candidates.csv"
_TEST_RESUMES = _BASE_DIR / "resumes"

# Fake job list returned in test mode
_FAKE_JOBS = [
    {"id": 1001, "name": "LLMOps Engineer", "status": "open",
     "departments": [{"name": "Engineering"}], "offices": [{"name": "Brooklyn, NY"}]},
    {"id": 1002, "name": "Senior ML Engineer", "status": "open",
     "departments": [{"name": "Engineering"}], "offices": [{"name": "Brooklyn, NY"}]},
    {"id": 1003, "name": "AI Product Manager", "status": "open",
     "departments": [{"name": "Product"}], "offices": [{"name": "Brooklyn, NY"}]},
]


# ---------------------------------------------------------------------------
# Greenhouse API client
# ---------------------------------------------------------------------------

class GreenhouseClient:
    """
    Thin wrapper around the Greenhouse Harvest API.
    Handles authentication, pagination, and retries.
    Authentication: HTTP Basic with API key as username, empty password.
    """

    def __init__(self, api_key: str, base_url: str = _DEFAULT_BASE_URL):
        self._auth     = (api_key, "")
        self._base_url = base_url.rstrip("/")
        self._session  = requests.Session()
        self._session.auth = self._auth

    def _get(self, path: str, params: dict = None) -> dict | list:
        """GET a single page. Retries on transient errors."""
        url = f"{self._base_url}/{path.lstrip('/')}"
        for attempt in range(_MAX_RETRIES):
            try:
                r = self._session.get(url, params=params, timeout=_REQUEST_TIMEOUT)
                if r.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY * (2 ** attempt))
                    continue
                r.raise_for_status()
                return r.json()
            except requests.exceptions.Timeout:
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY * (2 ** attempt))
                else:
                    raise GreenhouseError(f"Timeout after {_MAX_RETRIES} attempts: {url}")
            except requests.exceptions.HTTPError as e:
                raise GreenhouseError(f"HTTP {r.status_code} from {url}: {e}")
            except requests.exceptions.RequestException as e:
                raise GreenhouseError(f"Request failed: {e}")
        raise GreenhouseError(f"Max retries exceeded: {url}")

    def _get_all_pages(self, path: str, params: dict = None) -> list:
        """
        Fetch all pages of a paginated Greenhouse endpoint.
        Greenhouse uses Link headers for pagination:
          Link: <https://...?page=2&per_page=100>; rel="next"
        """
        params  = dict(params or {})
        params.setdefault("per_page", 100)
        params.setdefault("page", 1)
        results = []

        while True:
            url = f"{self._base_url}/{path.lstrip('/')}"
            for attempt in range(_MAX_RETRIES):
                try:
                    r = self._session.get(url, params=params, timeout=_REQUEST_TIMEOUT)
                    if r.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES - 1:
                        time.sleep(_RETRY_DELAY * (2 ** attempt))
                        continue
                    r.raise_for_status()
                    break
                except requests.exceptions.HTTPError as e:
                    raise GreenhouseError(f"HTTP {r.status_code}: {e}")
                except requests.exceptions.RequestException as e:
                    raise GreenhouseError(f"Request failed: {e}")

            page_data = r.json()
            if not page_data:
                break
            results.extend(page_data if isinstance(page_data, list) else [page_data])

            # Check Link header for next page
            link = r.headers.get("Link", "")
            if 'rel="next"' not in link:
                break
            params["page"] += 1

        return results

    def list_jobs(self, status: str = "open") -> list[dict]:
        """Return all jobs matching status ('open', 'closed', 'draft')."""
        return self._get_all_pages("jobs", params={"status": status})

    def list_applications(self, job_id: int) -> list[dict]:
        """Return all applications for a job, including candidate details."""
        return self._get_all_pages("applications", params={"job_id": job_id})

    def get_candidate(self, candidate_id: int) -> dict:
        """Return full candidate record (includes addresses, attachments)."""
        return self._get(f"candidates/{candidate_id}")

    def download_resume(self, url: str, dest_dir: Path) -> Optional[Path]:
        """
        Download a resume from a signed S3 URL to dest_dir.
        Returns the local Path, or None if the download fails.
        Greenhouse resume URLs are pre-signed and expire — call this promptly.
        """
        try:
            r = requests.get(url, timeout=_DOWNLOAD_TIMEOUT, stream=True)
            r.raise_for_status()

            # Infer file extension from Content-Disposition or URL
            ext = ".pdf"
            cd  = r.headers.get("Content-Disposition", "")
            if "filename=" in cd:
                fname = cd.split("filename=")[-1].strip().strip('"')
                ext   = Path(fname).suffix or ext
            elif "." in url.split("?")[0].split("/")[-1]:
                ext = "." + url.split("?")[0].split("/")[-1].rsplit(".", 1)[-1]

            dest = dest_dir / f"resume_{abs(hash(url)) % 10**8}{ext}"
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return dest

        except requests.exceptions.RequestException as e:
            print(f"⚠  Resume download failed ({url[:60]}...): {e}", flush=True)
            return None


class GreenhouseError(Exception):
    """Raised for unrecoverable Greenhouse API errors."""


# ---------------------------------------------------------------------------
# API key / config loading
# ---------------------------------------------------------------------------

def _load_greenhouse_key() -> str:
    """Return Greenhouse API key from env var or .env file."""
    key = os.environ.get("GREENHOUSE_API_KEY", "")
    if key:
        return key
    env_file = _BASE_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("GREENHOUSE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    return ""


def _load_base_url() -> str:
    return os.environ.get("GREENHOUSE_BASE_URL", _DEFAULT_BASE_URL)


# ---------------------------------------------------------------------------
# Shared output helpers
# ---------------------------------------------------------------------------

def _parse_applied_date(iso: str) -> str:
    """Normalize Greenhouse ISO timestamp to YYYY-MM-DD."""
    if not iso:
        return ""
    return iso[:10]  # 2025-04-01T00:00:00.000Z → 2025-04-01


def _candidate_location(candidate: dict) -> str:
    """
    Extract a human-readable location string from a Greenhouse candidate record.
    Greenhouse stores address as: candidate["addresses"][0]["value"] (free text)
    or candidate["phone_numbers"], or keyed_custom_fields.
    """
    addresses = candidate.get("addresses") or []
    if addresses:
        val = addresses[0].get("value", "").strip()
        if val:
            return val
    # Fallback: check keyed_custom_fields for a "location" key
    kcf = candidate.get("keyed_custom_fields") or {}
    loc = kcf.get("location") or kcf.get("current_location") or {}
    if isinstance(loc, dict):
        return str(loc.get("value", "")).strip()
    return ""


def _resume_attachment(candidate: dict) -> Optional[str]:
    """
    Return the URL of the candidate's most recent resume attachment.
    Greenhouse attachments have type: 'resume' | 'cover_letter' | 'other'.
    """
    attachments = candidate.get("attachments") or []
    # Prefer explicit resume type; fall back to first attachment
    for a in attachments:
        if a.get("type") == "resume" and a.get("url"):
            return a["url"]
    for a in attachments:
        if a.get("url"):
            return a["url"]
    return None


# ---------------------------------------------------------------------------
# Public interface — TEST MODE
# ---------------------------------------------------------------------------

def list_jobs_test() -> list[dict]:
    """Return fake job list (mirrors real list_jobs() output shape)."""
    return _FAKE_JOBS


def fetch_tasks_test() -> list[tuple]:
    """
    Build task list from local candidates.csv + resumes/ folder.
    Returns exactly the (idx, total, name, resume_file, gh_row) tuples
    that ranker.py's ThreadPoolExecutor expects.
    """
    if not _TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found: {_TEST_CSV}")

    resume_files = [
        f for f in _TEST_RESUMES.iterdir()
        if f.suffix.lower() in (".pdf", ".docx", ".doc", ".txt") and f.is_file()
    ] if _TEST_RESUMES.exists() else []

    from scoring import find_resume_file

    tasks = []
    with _TEST_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows   = list(reader)

    total = len(rows)
    for i, row in enumerate(rows, 1):
        name = row.get("Candidate Name") or row.get("Name") or row.get("Full Name", "")
        name = name.strip()
        if not name:
            continue

        resume_file = find_resume_file(name, resume_files)

        gh_row = {
            "Email":               row.get("Email", ""),
            "Applied At":          row.get("Applied At", ""),
            "Stage":               row.get("Stage", ""),
            "Greenhouse Location": row.get("Location", ""),
        }
        tasks.append((i, total, name, resume_file, gh_row))

    return tasks


# ---------------------------------------------------------------------------
# Public interface — LIVE MODE
# ---------------------------------------------------------------------------

def list_jobs_live(client: GreenhouseClient) -> list[dict]:
    """Return open jobs from the Greenhouse API."""
    return client.list_jobs(status="open")


def fetch_tasks_live(
    client: GreenhouseClient,
    job_id: int,
    resume_dir: Optional[Path] = None,
) -> list[tuple]:
    """
    Pull all applications for job_id, download resumes, build task list.

    Parameters
    ----------
    client      : authenticated GreenhouseClient
    job_id      : Greenhouse job ID
    resume_dir  : directory to save downloaded resumes; a temp dir is created
                  if not provided (caller is responsible for cleanup if needed)

    Returns
    -------
    list of (idx, total, name, resume_file, gh_row) tuples
    """
    if resume_dir is None:
        resume_dir = Path(tempfile.mkdtemp(prefix="gh_resumes_"))
    resume_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching applications for job {job_id}...", flush=True)
    applications = client.list_applications(job_id)
    total        = len(applications)
    print(f"  {total} applications found", flush=True)

    tasks = []
    for i, app in enumerate(applications, 1):
        # Each application object contains a nested "candidate" sub-object
        # with name and basic info. We fetch the full record for address + attachments.
        candidate_id   = app.get("candidate_id") or (app.get("candidate") or {}).get("id")
        candidate_name = (app.get("candidate") or {}).get("name", f"Candidate_{candidate_id}")

        try:
            candidate = client.get_candidate(candidate_id)
        except GreenhouseError as e:
            print(f"⚠  [{i}/{total}] {candidate_name}: could not fetch candidate — {e}", flush=True)
            candidate = app.get("candidate") or {}

        # Email: Greenhouse stores as list of {value, type} dicts
        emails     = candidate.get("email_addresses") or []
        email      = next((e["value"] for e in emails if e.get("type") == "personal"), "")
        if not email and emails:
            email = emails[0].get("value", "")

        applied_at = _parse_applied_date(app.get("applied_at", ""))
        stage      = (app.get("current_stage") or {}).get("name", "")
        location   = _candidate_location(candidate)

        # Download resume
        resume_url  = _resume_attachment(candidate)
        resume_file = None
        if resume_url:
            resume_file = client.download_resume(resume_url, resume_dir)
            if resume_file:
                # Rename to candidate name for easier matching
                safe_name   = candidate_name.replace(" ", "_").replace("/", "-")
                new_path    = resume_dir / f"{safe_name}{resume_file.suffix}"
                resume_file = resume_file.rename(new_path)

        gh_row = {
            "Email":               email,
            "Applied At":          applied_at,
            "Stage":               stage,
            "Greenhouse Location": location,
        }
        tasks.append((i, total, candidate_name, resume_file, gh_row))

    return tasks


# ---------------------------------------------------------------------------
# Unified public API — respects TEST_MODE flag
# ---------------------------------------------------------------------------

def list_jobs() -> list[dict]:
    """
    Return available jobs.
    In test mode: returns fake job list.
    In live mode: fetches from Greenhouse API.

    Each job dict has at minimum:
        id   : int    — pass to fetch_tasks()
        name : str    — display name
    """
    if TEST_MODE:
        return list_jobs_test()

    key = _load_greenhouse_key()
    if not key:
        raise GreenhouseError(
            "GREENHOUSE_API_KEY not set. Add it to your .env file or environment."
        )
    client = GreenhouseClient(key, _load_base_url())
    return list_jobs_live(client)


def fetch_tasks(
    job_id: int,
    resume_dir: Optional[Path] = None,
) -> list[tuple]:
    """
    Build the full task list for ranker.py's ThreadPoolExecutor.

    Parameters
    ----------
    job_id      : Greenhouse job ID (from list_jobs())
    resume_dir  : where to save downloaded resumes (live mode only;
                  ignored in test mode which uses local resumes/ folder)

    Returns
    -------
    list of (idx, total, name, resume_file, gh_row) tuples — same shape
    ranker.py builds from its CSV + resumes_dir path today.
    """
    if TEST_MODE:
        return fetch_tasks_test()

    key = _load_greenhouse_key()
    if not key:
        raise GreenhouseError(
            "GREENHOUSE_API_KEY not set. Add it to your .env file or environment."
        )
    client = GreenhouseClient(key, _load_base_url())
    return fetch_tasks_live(client, job_id, resume_dir)


# ---------------------------------------------------------------------------
# Quick smoke test — python3 greenhouse.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Greenhouse module — TEST_MODE={'ON' if TEST_MODE else 'OFF'}\n")

    print("list_jobs():")
    jobs = list_jobs()
    for j in jobs:
        print(f"  [{j['id']}] {j['name']}")

    if not jobs:
        print("  (no jobs returned)")
    else:
        first_job = jobs[0]
        print(f"\nfetch_tasks(job_id={first_job['id']}) [{first_job['name']}]:")
        tasks = fetch_tasks(first_job["id"])
        for idx, total, name, resume_file, gh_row in tasks:
            resume_status = resume_file.name if resume_file else "NO RESUME"
            print(f"  [{idx:>2}/{total}] {name:<22}  "
                  f"stage={gh_row.get('Stage',''):<20}  "
                  f"resume={resume_status}")

    print("\nDone.")
