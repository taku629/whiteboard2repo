"""Re-snap - what changed on the whiteboard since the last build?

Pure, deterministic Python. Components are matched by id first, then by a
fuzzy label match within the same kind, so a box the model re-read as
"Booking Service" instead of "Booking API" keeps its code. Edges are matched
on their (from, to) pair after that id mapping, notes on their text.
"""
from __future__ import annotations

import copy
import re
from difflib import SequenceMatcher
from typing import Any

FUZZY = 0.72


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _sim(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _note_key(n: dict[str, Any]) -> str:
    return n.get("text") or n.get("invariant") or ""


def match_components(old: dict[str, Any], new: dict[str, Any]) -> dict[str, str]:
    """Map new component id -> old component id for components that are the same box."""
    old_by_id = {c["id"]: c for c in old.get("components", [])}
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for c in new.get("components", []):
        if c["id"] in old_by_id and c["id"] not in used:
            mapping[c["id"]] = c["id"]
            used.add(c["id"])
    def nbrs(g: dict[str, Any], cid: str, m: dict[str, str]) -> set[tuple[str, str]]:
        out = set()
        for e in g.get("edges", []):
            if e["from"] == cid:
                out.add(("out", m.get(e["to"], e["to"])))
            if e["to"] == cid:
                out.add(("in", m.get(e["from"], e["from"])))
        return out

    for c in new.get("components", []):
        if c["id"] in mapping:
            continue
        best, score = None, 0.0
        for o in old.get("components", []):
            if o["id"] in used:
                continue
            s = max(_sim(c.get("label", ""), o.get("label", "")), _sim(c["id"], o["id"]))
            if o.get("kind") != c.get("kind"):
                s -= 0.15
            else:
                # same kind + same connections on the board = same box, even if relabelled
                a, b = nbrs(new, c["id"], mapping), nbrs(old, o["id"], {})
                if a and b:
                    s += 0.3 * len(a & b) / len(a | b)
            if s > score:
                best, score = o, s
        if best is not None and score >= FUZZY:
            mapping[c["id"]] = best["id"]
            used.add(best["id"])
    return mapping


def align(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `new` that reuses the old ids for matched components."""
    mapping = match_components(old, new)
    g = copy.deepcopy(new)
    taken = set(mapping.values())
    for c in g.get("components", []):
        if c["id"] in mapping:
            c["id"] = mapping[c["id"]]
        else:
            base, i = c["id"], 2
            while c["id"] in taken:
                c["id"] = f"{base}_{i}"
                i += 1
            mapping.setdefault(base, c["id"])
            taken.add(c["id"])
    for e in g.get("edges", []):
        e["from"] = mapping.get(e["from"], e["from"])
        e["to"] = mapping.get(e["to"], e["to"])
    return g


def diff_graphs(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Diff two board graphs. `new` is aligned to `old` ids first."""
    new = align(old, new)
    oc = {c["id"]: c for c in old.get("components", [])}
    nc = {c["id"]: c for c in new.get("components", [])}
    comps = {"added": [nc[i] for i in nc if i not in oc],
             "removed": [oc[i] for i in oc if i not in nc],
             "changed": []}
    for i in nc:
        if i in oc:
            # responsibility is the model's own prose and varies between reads: not a board change
            fields = [f for f in ("label", "kind") if _norm(oc[i].get(f, "")) != _norm(nc[i].get(f, ""))]
            if fields:
                comps["changed"].append({"id": i, "fields": fields,
                                         "old": {f: oc[i].get(f) for f in fields},
                                         "new": {f: nc[i].get(f) for f in fields}})
    oe = {(e["from"], e["to"]): e for e in old.get("edges", [])}
    ne = {(e["from"], e["to"]): e for e in new.get("edges", [])}
    edges = {"added": [ne[k] for k in ne if k not in oe],
             "removed": [oe[k] for k in oe if k not in ne],
             "changed": []}
    for k in ne:
        if k in oe:
            f = [x for x in ("label", "protocol") if _norm(oe[k].get(x, "")) != _norm(ne[k].get(x, ""))]
            if f:
                edges["changed"].append({"from": k[0], "to": k[1], "fields": f,
                                         "old": {x: oe[k].get(x) for x in f},
                                         "new": {x: ne[k].get(x) for x in f}})
    old_notes = list(old.get("notes", []))
    notes = {"added": [], "removed": [], "kept": []}
    used: set[int] = set()
    for n in new.get("notes", []):
        best, score = None, 0.0
        for j, o in enumerate(old_notes):
            if j in used:
                continue
            s = _sim(_note_key(n), _note_key(o))
            if s > score:
                best, score = j, s
        if best is not None and score >= FUZZY:
            used.add(best)
            notes["kept"].append(n)
        else:
            notes["added"].append(n)
    notes["removed"] = [o for j, o in enumerate(old_notes) if j not in used]
    n_changes = (sum(len(comps[k]) for k in ("added", "removed", "changed"))
                 + sum(len(edges[k]) for k in ("added", "removed", "changed"))
                 + len(notes["added"]) + len(notes["removed"]))
    return {"components": comps, "edges": edges, "invariants": notes,
            "aligned": new, "empty": n_changes == 0, "count": n_changes}


def summary_lines(d: dict[str, Any]) -> list[str]:
    """Human/LLM-readable bullet list of the diff."""
    out = []
    for c in d["components"]["added"]:
        out.append(f"+ component {c['id']} ({c['kind']}): {c.get('label')} - {c.get('responsibility', '')}")
    for c in d["components"]["removed"]:
        out.append(f"- component {c['id']} ({c['kind']}): {c.get('label')}")
    for c in d["components"]["changed"]:
        out.append(f"~ component {c['id']}: {c['old']} -> {c['new']}")
    for e in d["edges"]["added"]:
        out.append(f"+ edge {e['from']} -> {e['to']} [{e['protocol']}] {e.get('label', '')}")
    for e in d["edges"]["removed"]:
        out.append(f"- edge {e['from']} -> {e['to']} [{e['protocol']}] {e.get('label', '')}")
    for e in d["edges"]["changed"]:
        out.append(f"~ edge {e['from']} -> {e['to']}: {e['old']} -> {e['new']}")
    for n in d["invariants"]["added"]:
        out.append(f"+ invariant: {n.get('invariant') or n.get('text')}")
    for n in d["invariants"]["removed"]:
        out.append(f"- invariant: {n.get('invariant') or n.get('text')}")
    return out


def to_mermaid_diff(old: dict[str, Any], d: dict[str, Any]) -> str:
    """Mermaid of the new board with removed parts kept as ghosts and changes coloured."""
    from .vision import to_mermaid
    new = d["aligned"]
    ghost = {"title": new.get("title"), "components": list(new["components"]) + d["components"]["removed"],
             "edges": list(new["edges"]) + d["edges"]["removed"]}
    lines = to_mermaid(ghost).splitlines()
    lines += ["  classDef added fill:#e3f6ea,stroke:#1a8f55,stroke-width:3px",
              "  classDef removed fill:#fdeaea,stroke:#d23b3b,stroke-dasharray:5 5,color:#999",
              "  classDef changed fill:#fff4d6,stroke:#d99a00,stroke-width:3px"]
    for k in ("added", "removed", "changed"):
        ids = [c["id"] for c in d["components"][k]]
        if ids:
            lines.append(f"  class {','.join(ids)} {k}")
    added = {(e["from"], e["to"]) for e in d["edges"]["added"]}
    changed = {(e["from"], e["to"]) for e in d["edges"]["changed"]}
    removed = {(e["from"], e["to"]) for e in d["edges"]["removed"]}
    for i, e in enumerate(ghost["edges"]):
        k = (e["from"], e["to"])
        if k in added:
            lines.append(f"  linkStyle {i} stroke:#1a8f55,stroke-width:3px")
        elif k in removed:
            lines.append(f"  linkStyle {i} stroke:#d23b3b,stroke-dasharray:5 5")
        elif k in changed:
            lines.append(f"  linkStyle {i} stroke:#d99a00,stroke-width:3px")
    return "\n".join(lines)
