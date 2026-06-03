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
import zipfile
from pathlib import Path

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
    resumes_zip = request.files.get("resumes")
    jd_file = request.files.get("jd")
    jd_text = request.form.get("jd_text", "").strip()

    if not csv_file or not csv_file.filename:
        return jsonify({"error": "Greenhouse CSV file is required."}), 400
    if not resumes_zip or not resumes_zip.filename:
        return jsonify({"error": "Resumes ZIP file is required."}), 400
    if not jd_file and not jd_text:
        return jsonify({"error": "Job description (file or text) is required."}), 400

    job_id = uuid.uuid4().hex[:10]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True)

    # Save CSV
    csv_path = job_dir / "candidates.csv"
    csv_file.save(str(csv_path))

    # Save and unzip resumes
    resumes_dir = job_dir / "resumes"
    resumes_dir.mkdir()
    zip_path = job_dir / "resumes.zip"
    resumes_zip.save(str(zip_path))
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                fname = Path(member).name
                if not fname or fname.startswith("."):
                    continue
                ext = Path(fname).suffix.lower()
                if ext in (".pdf", ".docx", ".doc", ".txt"):
                    (resumes_dir / fname).write_bytes(zf.read(member))
    except zipfile.BadZipFile:
        return jsonify({"error": "The resumes file is not a valid ZIP archive."}), 400

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


if __name__ == "__main__":
    print("Resume Ranker — Web UI")
    print("Open http://localhost:5001 in your browser\n")
    app.run(debug=False, port=5001)
