#!/usr/bin/env python3
"""
Resume Ranker — Web Interface
Run with: python3 app.py
Then open http://localhost:5000
"""

import json
import os
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, send_file

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def _load_api_key() -> str:
    """Return the Anthropic API key from env var or local .env file."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    return ""

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

# Active jobs: job_id -> {"queue": Queue, "output_file": Path, "status": str}
_jobs: dict = {}

# ---------------------------------------------------------------------------
# Recruiting activity notifications (fake seed data)
# ---------------------------------------------------------------------------

_notifications = [
    # ── Today (2026-05-15) — all unread ──────────────────────────────────
    {
        "id": "1",
        "type": "availability",
        "candidate_name": "Marcus Chen",
        "role": "LLMOps Engineer",
        "timestamp": "2026-05-15T09:23:00",
        "read": False,
    },
    {
        "id": "2",
        "type": "scorecard",
        "candidate_name": "Priya Sharma",
        "interviewer_name": "Rachel Torres",
        "interviewer_title": "Sr. Engineering Manager",
        "role": "LLMOps Engineer",
        "timestamp": "2026-05-15T10:15:00",
        "read": False,
    },
    {
        "id": "3",
        "type": "availability",
        "candidate_name": "Jordan Williams",
        "role": "LLMOps Engineer",
        "timestamp": "2026-05-15T11:07:00",
        "read": False,
    },
    {
        "id": "4",
        "type": "scorecard",
        "candidate_name": "Marcus Chen",
        "interviewer_name": "David Kim",
        "interviewer_title": "VP, Artificial Intelligence",
        "role": "LLMOps Engineer",
        "timestamp": "2026-05-15T14:30:00",
        "read": False,
    },
    {
        "id": "5",
        "type": "availability",
        "candidate_name": "Kevin Nguyen",
        "role": "LLMOps Engineer",
        "timestamp": "2026-05-15T15:52:00",
        "read": False,
    },
    # ── Yesterday (2026-05-14) — all read ────────────────────────────────
    {
        "id": "6",
        "type": "scorecard",
        "candidate_name": "Alex Torres",
        "interviewer_name": "James Park",
        "interviewer_title": "AI Engineer",
        "role": "LLMOps Engineer",
        "timestamp": "2026-05-14T13:30:00",
        "read": True,
    },
    {
        "id": "7",
        "type": "availability",
        "candidate_name": "Priya Sharma",
        "role": "LLMOps Engineer",
        "timestamp": "2026-05-14T15:20:00",
        "read": True,
    },
    {
        "id": "8",
        "type": "scorecard",
        "candidate_name": "Jordan Williams",
        "interviewer_name": "David Kim",
        "interviewer_title": "VP, Artificial Intelligence",
        "role": "LLMOps Engineer",
        "timestamp": "2026-05-14T16:45:00",
        "read": True,
    },
    # ── May 13 — all read ────────────────────────────────────────────────
    {
        "id": "9",
        "type": "availability",
        "candidate_name": "Alex Torres",
        "role": "LLMOps Engineer",
        "timestamp": "2026-05-13T10:30:00",
        "read": True,
    },
    {
        "id": "10",
        "type": "scorecard",
        "candidate_name": "Kevin Nguyen",
        "interviewer_name": "Rachel Torres",
        "interviewer_title": "Sr. Engineering Manager",
        "role": "LLMOps Engineer",
        "timestamp": "2026-05-13T14:00:00",
        "read": True,
    },
]


def _run_ranker(job_id: str, csv_path: Path, resumes_dir: Path, jd_path: Path,
                output_path: Path, api_key: str) -> None:
    q = _jobs[job_id]["queue"]
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = api_key

    cmd = [
        sys.executable,
        str(BASE_DIR / "ranker.py"),
        str(csv_path),
        str(resumes_dir),
        "--jd", str(jd_path),
        "--output", str(output_path),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        for line in proc.stdout:
            q.put({"type": "log", "text": line.rstrip()})
        proc.wait()
        if proc.returncode == 0:
            _jobs[job_id]["status"] = "done"
            q.put({"type": "done", "filename": output_path.name})
        else:
            _jobs[job_id]["status"] = "error"
            q.put({"type": "error", "text": f"ranker.py exited with code {proc.returncode}"})
    except Exception as exc:
        _jobs[job_id]["status"] = "error"
        q.put({"type": "error", "text": str(exc)})
    finally:
        q.put(None)  # sentinel — tells the SSE generator to stop


@app.route("/")
def index():
    has_key = bool(_load_api_key())
    return render_template("index.html", has_api_key=has_key)


@app.route("/run", methods=["POST"])
def run():
    api_key = request.form.get("api_key", "").strip() or _load_api_key()
    if not api_key:
        return jsonify({"error": "Anthropic API key is required."}), 400

    csv_file = request.files.get("csv")
    resume_files = request.files.getlist("resumes")
    jd_file = request.files.get("jd")
    jd_text = request.form.get("jd_text", "").strip()

    if not resume_files or not any(f.filename for f in resume_files):
        return jsonify({"error": "Resume files are required."}), 400
    if not jd_file and not jd_text:
        return jsonify({"error": "Job description (file or text) is required."}), 400

    job_id = uuid.uuid4().hex[:10]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True)

    # Save CSV
    csv_path = job_dir / "candidates.csv"
    if csv_file and csv_file.filename:
        csv_file.save(str(csv_path))

    # Save uploaded resume files directly
    resumes_dir = job_dir / "resumes"
    resumes_dir.mkdir()
    for f in resume_files:
        fname = Path(f.filename).name
        if not fname or fname.startswith("."):
            continue
        if Path(fname).suffix.lower() in (".pdf", ".docx", ".doc", ".txt"):
            f.save(str(resumes_dir / fname))

    if not (csv_file and csv_file.filename):
        stems = sorted(
            f.stem for f in resumes_dir.iterdir()
            if f.suffix.lower() in (".pdf", ".docx", ".doc", ".txt")
        )
        csv_path.write_text("Candidate Name\n" + "\n".join(stems), encoding="utf-8")

    # Save JD
    jd_path = job_dir / "job.txt"
    if jd_file and jd_file.filename:
        jd_file.save(str(jd_path))
    else:
        jd_path.write_text(jd_text, encoding="utf-8")

    output_path = OUTPUT_DIR / f"{job_id}_ranked.xlsx"

    _jobs[job_id] = {
        "queue": queue.Queue(),
        "output_file": output_path,
        "status": "running",
    }

    threading.Thread(
        target=_run_ranker,
        args=(job_id, csv_path, resumes_dir, jd_path, output_path, api_key),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id: str):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        q = _jobs[job_id]["queue"]
        while True:
            msg = q.get()
            if msg is None:
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                break
            yield f"data: {json.dumps(msg)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.route("/download/<job_id>")
def download(job_id: str):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404
    f = _jobs[job_id]["output_file"]
    if not f.exists():
        return jsonify({"error": "Output file not found"}), 404
    return send_file(str(f), as_attachment=True, download_name="ranked_candidates.xlsx")


@app.route("/resume/<job_id>")
def serve_resume(job_id: str):
    # Allow disk-only lookup so panel still works after a server restart
    resumes_dir = UPLOAD_DIR / job_id / "resumes"
    if job_id not in _jobs and not resumes_dir.exists():
        return jsonify({"error": "Job not found"}), 404

    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"error": "name parameter required"}), 400

    if not resumes_dir.exists():
        return jsonify({"error": "Resumes not found for this job"}), 404

    from scoring import find_resume_file
    exts = (".pdf", ".docx", ".doc", ".txt")
    resume_files = [f for f in resumes_dir.iterdir() if f.suffix.lower() in exts]
    matched = find_resume_file(name, resume_files)
    if matched is None:
        return jsonify({"error": f"No resume found for '{name}'"}), 404

    as_attachment = request.args.get("download") == "1"
    return send_file(
        str(matched),
        mimetype="application/pdf",
        as_attachment=as_attachment,
        download_name=matched.name if as_attachment else None,
    )


@app.route("/api/notifications")
def get_notifications():
    return jsonify(sorted(_notifications, key=lambda n: n["timestamp"], reverse=True))


@app.route("/api/notifications/<nid>/read", methods=["POST"])
def mark_notification_read(nid: str):
    for n in _notifications:
        if n["id"] == nid:
            n["read"] = True
            return jsonify({"ok": True})
    return jsonify({"error": "not found"}), 404


@app.route("/api/notifications/read-all", methods=["POST"])
def mark_all_read():
    for n in _notifications:
        n["read"] = True
    return jsonify({"ok": True})


def _job_title_from_jd(jd_path: Path) -> str:
    try:
        for line in jd_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                return line
    except OSError:
        pass
    return "Open Role"


@app.route("/sample-results")
def sample_results():
    xlsx_path = BASE_DIR / "sample_ranked.xlsx"
    jd_path   = BASE_DIR / "job.txt"
    job_title = _job_title_from_jd(jd_path)

    if not xlsx_path.exists():
        return render_template("sample_results.html", rows=None, job_title=job_title)

    df = pd.read_excel(xlsx_path)
    df = df.where(pd.notna(df), None)

    rows = []
    for record in df.to_dict(orient="records"):
        name = str(record.get("Candidate Name") or "")
        tier_raw = record.get("Stability Tier")
        try:
            tier = int(tier_raw) if tier_raw is not None else None
        except (ValueError, TypeError):
            tier = None
        fit_raw = record.get("JD Fit Composite")
        try:
            fit = round(float(fit_raw), 1) if fit_raw is not None else None
        except (ValueError, TypeError):
            fit = None
        rows.append({
            "rank":         record.get("Rank"),
            "name":         name,
            "name_encoded": quote_plus(name),
            "tier":         tier,
            "fit":          fit,
            "local":        str(record.get("Local") or "No"),
            "job_hopper":   bool(str(record.get("Job Hopper") or "").strip()),
            "summary":      str(record.get("Summary") or ""),
        })

    return render_template("sample_results.html", rows=rows)


@app.route("/sample-resume")
def sample_resume():
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"error": "name parameter required"}), 400

    resumes_dir = BASE_DIR / "sample_resumes"
    if not resumes_dir.exists():
        return jsonify({"error": "sample_resumes directory not found"}), 404

    from scoring import find_resume_file
    exts = (".pdf", ".docx", ".doc", ".txt")
    resume_files = [f for f in resumes_dir.iterdir() if f.suffix.lower() in exts]
    matched = find_resume_file(name, resume_files)
    if matched is None:
        return jsonify({"error": f"No resume found for '{name}'"}), 404

    return send_file(str(matched), mimetype="application/pdf")


if __name__ == "__main__":
    print("Resume Ranker — Web UI")
    print("Open http://localhost:5001 in your browser\n")
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))