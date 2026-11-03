"""Step 1 - read the whiteboard.

Muse Spark Open (multimodal, open weights) looks at the photo and returns a
typed architecture graph: components with bounding boxes, directed edges with
labels/protocols, and free-floating handwritten notes. Notes are promoted to
*invariants* that later become executable tests.

If SAM 3.1 detections are available (see `sam/`), they are passed to the model
as grounding hints so that every component is anchored to a real region of the
board.
"""
from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

from PIL import Image

from . import llm

KINDS = ["client", "gateway", "service", "worker", "database", "cache", "queue", "external"]

SYSTEM = f"""You are Whiteboard2Repo's vision module. You read photos of hand-drawn
software architecture sketches and convert them into a precise, typed graph.
Rules:
- Every box/cylinder/cloud with a label is a component. kind must be one of {KINDS}.
  Cylinders are usually databases. "queue", "kafka", "redis queue", "sqs" are queues.
  Clouds or third-party things (Email, Stripe, S3) are external. Mobile/Web/React are client.
- Every arrow is an edge from the tail to the head. A double-headed arrow means read+write
  from the box that owns the data access (service -> database). Copy the arrow's text label exactly.
- protocol is one of: http, event, sql, cache, smtp, other. Infer from context
  (service->database = sql, service->queue or labels like "order.created" = event).
- Handwritten notes that are not boxes (e.g. "no double booking!") are notes; rewrite each as a
  testable invariant in plain English.
- bbox: [x0, y0, x1, y1] in a 0-1000 normalized coordinate system of the full image.
- If something is ambiguous (unreadable label, unclear arrow direction), add a short question to
  "questions" instead of guessing silently.
Return ONLY JSON with this shape:
{{"title": str, "components": [{{"id": slug, "label": str, "kind": str, "bbox": [int,int,int,int], "responsibility": str}}],
 "edges": [{{"from": id, "to": id, "label": str, "protocol": str}}],
 "notes": [{{"text": str, "invariant": str, "bbox": [int,int,int,int]}}],
 "questions": [str]}}"""


def _encode(img_bytes: bytes, max_side: int = 1600) -> tuple[str, tuple[int, int]]:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode(), im.size


def slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip().lower()).strip("_")
    return s or "component"


def normalize_graph(g: dict[str, Any]) -> dict[str, Any]:
    comps = []
    seen: dict[str, str] = {}
    for c in g.get("components", []):
        cid = slug(c.get("id") or c.get("label", ""))
        base, i = cid, 2
        while cid in seen.values():
            cid = f"{base}_{i}"; i += 1
        seen[c.get("id", cid)] = cid
        kind = c.get("kind", "service")
        if kind not in KINDS:
            kind = "service"
        comps.append({"id": cid, "label": c.get("label", cid), "kind": kind,
                      "bbox": c.get("bbox"), "responsibility": c.get("responsibility", "")})
    ids = {c["id"] for c in comps}
    edges = []
    for e in g.get("edges", []):
        a = seen.get(e.get("from"), slug(str(e.get("from", ""))))
        b = seen.get(e.get("to"), slug(str(e.get("to", ""))))
        if a in ids and b in ids and a != b:
            edges.append({"from": a, "to": b, "label": e.get("label") or "",
                          "protocol": e.get("protocol") or "other"})
    return {"title": g.get("title") or "Whiteboard system", "components": comps,
            "edges": edges, "notes": g.get("notes", []), "questions": g.get("questions", [])}


def read_board(img_bytes: bytes, sam_hints: list[dict] | None = None) -> dict[str, Any]:
    b64, size = _encode(img_bytes)
    text = "Convert this whiteboard sketch into the architecture graph JSON."
    if sam_hints:
        text += ("\nSAM 3.1 segmentation found these regions (bbox in 0-1000 coords, prompt used). "
                 "Use them to anchor each component's bbox; every component should match one region:\n"
                 + json.dumps(sam_hints)[:6000])
    msg = llm.chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]},
    ], max_tokens=6000, temperature=0.1)
    graph = normalize_graph(llm.extract_json(msg.get("content") or ""))
    graph["image_size"] = list(size)
    graph["model"] = llm.MODEL
    graph["sam_grounded"] = bool(sam_hints)
    return graph


def to_mermaid(graph: dict[str, Any]) -> str:
    shape = {"database": ("[(", ")]"), "queue": ("[/", "/]"), "external": ("((", "))"),
             "client": ("([", "])"), "cache": ("[(", ")]")}
    lines = ["flowchart LR"]
    for c in graph["components"]:
        l, r = shape.get(c["kind"], ("[", "]"))
        lines.append(f'  {c["id"]}{l}"{c["label"]}"{r}')
    for e in graph["edges"]:
        lab = e.get("label")
        arrow = "-.->" if e.get("protocol") == "event" else "-->"
        lines.append(f'  {e["from"]} {arrow}' + (f'|"{lab}"| ' if lab else " ") + e["to"])
    return "\n".join(lines)
