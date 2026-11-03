"""Whiteboard2Repo web app (FastAPI)."""
from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import re

from . import agent, llm, scaffold, vision
from . import diff as diffmod
from . import github_pr

HERE = Path(__file__).parent
ROOT = HERE.parent
SAMPLES = ROOT / "samples"
JOBS = Path(os.getenv("JOBS_DIR", Path(tempfile.gettempdir()) / "w2r_jobs"))
JOBS.mkdir(parents=True, exist_ok=True)
HOURLY_LIMIT = int(os.getenv("HOURLY_BUILD_LIMIT", "20"))

app = FastAPI(title="Whiteboard2Repo")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")

_builds: deque[float] = deque()
_lock = threading.Lock()


def _rate_ok() -> bool:
    now = time.time()
    with _lock:
        while _builds and now - _builds[0] > 3600:
            _builds.popleft()
        if len(_builds) >= HOURLY_LIMIT:
            return False
        _builds.append(now)
        return True


def _job(job_id: str) -> Path:
    if not job_id.replace("-", "").isalnum():
        raise HTTPException(400, "bad job id")
    p = JOBS / job_id
    if not p.exists():
        raise HTTPException(404, "job not found")
    return p


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model": llm.MODEL}


@app.get("/api/samples")
def samples() -> list[dict[str, Any]]:
    out = []
    for p in sorted(SAMPLES.glob("*.jpg")):
        m = re.match(r"(.+)_v(\d+)$", p.stem)
        out.append({"name": p.stem, "image": f"/samples/{p.name}",
                    "update_of": m.group(1) if m else None,
                    "sam": (SAMPLES / f"{p.stem}.sam.json").exists(),
                    "sam_overlay": f"/samples/{p.stem}.sam.png" if (SAMPLES / f"{p.stem}.sam.png").exists() else None})
    return out


@app.post("/api/analyze")
async def analyze(file: UploadFile | None = File(None), sample: str | None = Form(None),
                  use_cache: bool = Form(False)) -> JSONResponse:
    sam_hints = None
    if sample:
        src = SAMPLES / f"{Path(sample).name}.jpg"
        if not src.exists():
            raise HTTPException(404, "unknown sample")
        data = src.read_bytes()
        sam_file = SAMPLES / f"{src.stem}.sam.json"
        if sam_file.exists():
            sam_hints = json.loads(sam_file.read_text()).get("regions")
        cached = SAMPLES / f"{src.stem}.graph.json"
        if use_cache and cached.exists():
            graph = json.loads(cached.read_text())
        else:
            graph = vision.read_board(data, sam_hints)
    else:
        if file is None:
            raise HTTPException(400, "upload an image or pick a sample")
        data = await file.read()
        if len(data) > 8_000_000:
            raise HTTPException(413, "image too large (8 MB max)")
        graph = vision.read_board(data)
    job_id = uuid.uuid4().hex[:12]
    d = JOBS / job_id
    d.mkdir()
    (d / "board.jpg").write_bytes(data)
    (d / "graph.json").write_text(json.dumps(graph))
    return JSONResponse({"job_id": job_id, "graph": graph, "mermaid": vision.to_mermaid(graph),
                         "sam_regions": sam_hints or []})


@app.post("/api/build/{job_id}")
async def build(job_id: str, request: Request) -> StreamingResponse:
    d = _job(job_id)
    body = await request.json()
    graph = vision.normalize_graph(body.get("graph") or json.loads((d / "graph.json").read_text()))
    for k in ("image_size", "model", "sam_grounded"):
        graph[k] = json.loads((d / "graph.json").read_text()).get(k)
    if not _rate_ok():
        raise HTTPException(429, "demo build limit reached for this hour - try again later")
    repo = d / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    # the graph the repo was built from (after user edits) is version 1 of the board history
    (d / "built.json").write_text(json.dumps(graph))
    (d / "graph.v1.json").write_text(json.dumps(graph))
    (d / "version").write_text("1")

    def stream():
        def sse(ev: dict) -> str:
            return f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        t0 = time.time()
        files = scaffold.build(graph, repo)
        yield sse({"type": "scaffold", "files": files, "mermaid": vision.to_mermaid(graph)})
        try:
            for ev in agent.run(graph, repo):
                ev["t"] = round(time.time() - t0, 1)
                yield sse(ev)
        except Exception as e:  # pragma: no cover
            yield sse({"type": "error", "text": str(e)[:500]})
        if graph.get("notes"):
            yield sse({"type": "proving", "t": round(time.time() - t0, 1)})
            try:
                proofs = agent.prove_invariants(graph, repo)
            except Exception as e:  # pragma: no cover
                proofs = [{"invariant": "-", "caught": None, "detail": str(e)[:200]}]
            yield sse({"type": "proof", "results": proofs, "t": round(time.time() - t0, 1)})
        cov = agent.coverage(graph, repo)
        cov["t"] = round(time.time() - t0, 1)
        (d / "coverage.json").write_text(json.dumps(cov))
        yield sse({"type": "coverage", **cov, "files": agent.Workspace(repo).list_files().splitlines(),
                   "download": f"/api/download/{job_id}"})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _read_upload_or_sample(data: bytes | None, sample: str | None, use_cache: bool) -> dict[str, Any]:
    if sample:
        src = SAMPLES / f"{Path(sample).name}.jpg"
        if not src.exists():
            raise HTTPException(404, "unknown sample")
        cached = SAMPLES / f"{src.stem}.graph.json"
        if use_cache and cached.exists():
            return json.loads(cached.read_text())
        sam_file = SAMPLES / f"{src.stem}.sam.json"
        hints = json.loads(sam_file.read_text()).get("regions") if sam_file.exists() else None
        return vision.read_board(src.read_bytes(), hints)
    if not data:
        raise HTTPException(400, "upload an image or pick a sample")
    if len(data) > 8_000_000:
        raise HTTPException(413, "image too large (8 MB max)")
    return vision.read_board(data)


@app.post("/api/resnap/{job_id}")
async def resnap(job_id: str, file: UploadFile | None = File(None), sample: str | None = Form(None),
                 use_cache: bool = Form(False)) -> JSONResponse:
    """Read an updated photo of the same whiteboard and diff it against the built board."""
    d = _job(job_id)
    if not (d / "built.json").exists():
        raise HTTPException(409, "build the first version before re-snapping")
    old = json.loads((d / "built.json").read_text())
    data = await file.read() if file is not None else None
    new = _read_upload_or_sample(data, sample, use_cache)
    if data:
        (d / "board.next.jpg").write_bytes(data)
    elif sample and (SAMPLES / f"{Path(sample).name}.jpg").exists():
        shutil.copy(SAMPLES / f"{Path(sample).name}.jpg", d / "board.next.jpg")
    df = diffmod.diff_graphs(old, new)
    (d / "pending.json").write_text(json.dumps(df["aligned"]))
    return JSONResponse({"job_id": job_id, "graph": df["aligned"], "diff": {k: df[k] for k in
                         ("components", "edges", "invariants", "empty", "count")},
                         "summary": diffmod.summary_lines(df),
                         "mermaid": diffmod.to_mermaid_diff(old, df)})


@app.post("/api/update/{job_id}")
async def update(job_id: str, request: Request) -> StreamingResponse:
    """Apply the re-snapped board to the existing repo (SSE stream like /api/build)."""
    d = _job(job_id)
    repo = d / "repo"
    if not repo.exists() or not (d / "built.json").exists():
        raise HTTPException(409, "nothing built yet")
    body = await request.json()
    old = json.loads((d / "built.json").read_text())
    pending = body.get("graph") or json.loads((d / "pending.json").read_text())
    new = vision.normalize_graph(pending)
    for k in ("image_size", "model", "sam_grounded"):
        new[k] = pending.get(k)
    df = diffmod.diff_graphs(old, new)
    graph = df["aligned"]
    if not _rate_ok():
        raise HTTPException(429, "demo build limit reached for this hour - try again later")

    def stream():
        def sse(ev: dict) -> str:
            return f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        t0 = time.time()
        tests: dict[str, Any] = {}
        changed: list[str] = []
        try:
            github_pr.ensure_baseline(repo, d / "board.jpg")
        except Exception as e:  # pragma: no cover - git missing
            yield sse({"type": "thought", "text": f"(git snapshot skipped: {str(e)[:120]})"})
        try:
            for ev in agent.update(old, new, repo, df):
                ev["t"] = round(time.time() - t0, 1)
                if ev.get("type") == "tests":
                    tests = {"ok": ev.get("ok"), **(ev.get("counts") or {})}
                if ev.get("type") == "changes":
                    changed = [f"+ {f}" for f in ev.get("added", [])] + [f"~ {f}" for f in ev.get("modified", [])] + \
                              [f"- {f}" for f in ev.get("deleted", []) or []]
                yield sse(ev)
        except Exception as e:  # pragma: no cover
            yield sse({"type": "error", "text": str(e)[:500]})
        new_notes = df["invariants"]["added"]
        proofs: list[dict[str, Any]] = []
        if new_notes:
            yield sse({"type": "proving", "t": round(time.time() - t0, 1)})
            try:
                proofs = agent.prove_invariants(graph, repo, new_notes)
            except Exception as e:  # pragma: no cover
                proofs = [{"invariant": "-", "caught": None, "detail": str(e)[:200]}]
            yield sse({"type": "proof", "results": proofs, "t": round(time.time() - t0, 1)})
        cov = agent.coverage(graph, repo)
        cov["t"] = round(time.time() - t0, 1)
        v = int((d / "version").read_text() or "1") + 1
        (d / f"graph.v{v}.json").write_text(json.dumps(graph))
        (d / "version").write_text(str(v))
        (d / "built.json").write_text(json.dumps(graph))
        (d / "coverage.json").write_text(json.dumps(cov))
        (d / f"report.v{v}.json").write_text(json.dumps({
            "version": v, "summary": diffmod.summary_lines(df), "mermaid": diffmod.to_mermaid_diff(old, df),
            "proofs": proofs, "tests": tests, "coverage": cov, "changed_files": changed}))
        try:
            github_pr.snapshot(repo, v, d / "board.next.jpg", f"Re-snap: board v{v-1} -> v{v}\n\n" +
                               "\n".join(diffmod.summary_lines(df)), branch=f"w2r/resnap-{job_id}-v{v}")
        except Exception as e:  # pragma: no cover
            yield sse({"type": "thought", "text": f"(git snapshot skipped: {str(e)[:120]})"})
        yield sse({"type": "coverage", **cov, "version": v, "can_pr": True,
                   "files": agent.Workspace(repo).list_files().splitlines(),
                   "download": f"/api/download/{job_id}"})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/github/{job_id}/pr")
async def github_pr_open(job_id: str, request: Request) -> JSONResponse:
    """Push the re-snapped update to a dedicated GitHub repo and open a Pull Request."""
    d = _job(job_id)
    try:
        body = await request.json()
    except Exception:
        body = {}
    target = github_pr.target_repo() or body.get("repo")
    if not target or not re.fullmatch(r"[\w.-]+/[\w.-]+", target):
        raise HTTPException(400, "set W2R_GITHUB_REPO=owner/repo (a dedicated, initially empty repo)")
    try:
        return JSONResponse(github_pr.open_pr(d, target, os.getenv("GITHUB_TOKEN")))
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.get("/api/file/{job_id}")
def get_file(job_id: str, path: str) -> JSONResponse:
    ws = agent.Workspace(_job(job_id) / "repo")
    return JSONResponse({"path": path, "content": ws.read_file(path)})


@app.get("/api/download/{job_id}")
def download(job_id: str) -> StreamingResponse:
    repo = _job(job_id) / "repo"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in repo.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts and ".git" not in p.parts:
                z.write(p, f"whiteboard2repo-{job_id}/{p.relative_to(repo)}")
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename=whiteboard2repo-{job_id}.zip"})
