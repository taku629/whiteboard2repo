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

from . import agent, llm, scaffold, vision

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
        out.append({"name": p.stem, "image": f"/samples/{p.name}",
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
            if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
                z.write(p, f"whiteboard2repo-{job_id}/{p.relative_to(repo)}")
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename=whiteboard2repo-{job_id}.zip"})
