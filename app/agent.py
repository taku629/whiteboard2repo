"""Step 3 - the build agent.

Muse Spark Open drives a tool-calling loop inside a throwaway workspace:
it reads the skeleton, writes domain code for every component and edge,
turns each handwritten note into an executable test, runs pytest, reads the
failures and fixes them - until the suite is green or the step budget runs out.
Step 4 (`coverage`) then checks the result against the board itself.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

from . import llm

MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "24"))

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List all files in the repository.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a file with the full content.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run `pytest -q` at the repo root and return the output tail.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "finish", "description": "Call when every test passes and the board is fully implemented.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]

SYSTEM = """You are the Whiteboard2Repo build agent. A team drew an architecture on a whiteboard;
it was converted into board.json and a skeleton repo that already passes its health-check tests.
Your job: turn the skeleton into a small but REAL working system that matches the board.

Requirements:
1. For every service/gateway/worker in board.json, implement concrete FastAPI endpoints for its
   responsibility (keep it small: 2-4 endpoints each). Use SQLAlchemy models via common.db
   (Base, SessionLocal, init_db) for any component connected to a database; data names on the
   board's sql edges (e.g. "orders", "rooms, bookings") become tables.
2. For every event edge (protocol=event), the producer must call
   common.bus.bus.publish("<edge label or a sensible topic>", payload) and the consumer must
   subscribe a handler at import time with bus.subscribe(...). For a gateway -> service http edge,
   the gateway forwards by calling the downstream service's FastAPI app in-process via
   fastapi.testclient.TestClient (keeps tests hermetic) behind a small client function.
3. For every note/invariant in board.json, write tests/test_invariants.py with at least one test per
   invariant that would FAIL if the invariant were violated, and make the code satisfy it.
4. Write tests/test_flows.py: one end-to-end test per edge that proves the edge works
   (e.g. creating an order publishes order.created and the payments handler receives it).
5. Run tests, read failures, fix, repeat. Never delete or weaken a test to make it pass.
   Tests must not need network, Docker, Postgres or Redis (DATABASE_URL is sqlite in-memory;
   call common.db.reset_db() in fixtures for isolation).
6. Keep a module docstring in each service naming which board edges it implements.
Work efficiently: write whole files, then run_tests. Call finish only when tests pass."""


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _safe(self, rel: str) -> Path:
        p = (self.root / rel.lstrip("/")).resolve()
        if not str(p).startswith(str(self.root)):
            raise ValueError("path escapes repository")
        return p

    def list_files(self) -> str:
        out = [str(p.relative_to(self.root)) for p in sorted(self.root.rglob("*"))
               if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts]
        return "\n".join(out)

    def read_file(self, path: str) -> str:
        p = self._safe(path)
        return p.read_text()[:20000] if p.exists() else f"ERROR: {path} does not exist"

    def write_file(self, path: str, content: str) -> str:
        p = self._safe(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {path} ({len(content)} bytes)"

    def run_tests(self) -> tuple[bool, str, dict[str, int]]:
        env = {**os.environ, "DATABASE_URL": "sqlite://", "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("BUS_URL", None)
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                "--tb=short", "-W", "ignore"], cwd=self.root, env=env,
                               capture_output=True, text=True, timeout=120)
            out = (r.stdout + r.stderr)
        except subprocess.TimeoutExpired:
            return False, "pytest timed out after 120s", {}
        counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|error|errors)", out)}
        ok = r.returncode == 0
        return ok, out[-6000:], counts


def run(graph: dict[str, Any], root: Path) -> Iterator[dict[str, Any]]:
    """Run the agent loop, yielding progress events for the UI."""
    ws = Workspace(root)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "board.json:\n" + json.dumps(graph, ensure_ascii=False)
         + "\n\nCurrent files:\n" + ws.list_files()
         + "\n\nStart by reading the service skeletons you need, then implement."},
    ]
    last_ok = False
    nudged = False
    for step in range(1, MAX_STEPS + 1):
        msg = llm.chat(messages, tools=TOOLS, max_tokens=16000, temperature=0.2)
        calls = msg.get("tool_calls") or []
        thought = (msg.get("content") or "").strip()
        if thought:
            yield {"type": "thought", "step": step, "text": thought[:600]}
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         **({"tool_calls": calls} if calls else {})})
        if not calls:
            if last_ok:
                yield {"type": "done", "step": step, "summary": thought[:800]}
                return
            messages.append({"role": "user", "content": "Use the tools. Continue until run_tests passes, then call finish."})
            continue
        finished = None
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "list_files":
                result = ws.list_files()
                yield {"type": "tool", "step": step, "name": name, "detail": ""}
            elif name == "read_file":
                result = ws.read_file(args.get("path", ""))
                yield {"type": "tool", "step": step, "name": name, "detail": args.get("path", "")}
            elif name == "write_file":
                try:
                    result = ws.write_file(args.get("path", ""), args.get("content", ""))
                except ValueError as e:
                    result = f"ERROR: {e}"
                yield {"type": "tool", "step": step, "name": name, "detail": args.get("path", ""),
                       "lines": args.get("content", "").count("\n") + 1}
            elif name == "run_tests":
                ok, out, counts = ws.run_tests()
                last_ok = ok
                result = ("ALL TESTS PASSED\n" if ok else "TESTS FAILED\n") + out
                yield {"type": "tests", "step": step, "ok": ok, "counts": counts, "tail": out[-1500:]}
            elif name == "finish":
                ok, out, counts = ws.run_tests()
                last_ok = ok
                if ok:
                    finished = args.get("summary", "")
                    result = "ok"
                else:
                    result = "Tests are failing, you cannot finish yet:\n" + out
                    yield {"type": "tests", "step": step, "ok": ok, "counts": counts, "tail": out[-1500:]}
            else:
                result = f"unknown tool {name}"
            messages.append({"role": "tool", "tool_call_id": call.get("id", name), "content": result[:12000]})
        if finished is not None:
            yield {"type": "done", "step": step, "summary": finished}
            return
        if step == MAX_STEPS - 3 and not nudged:
            nudged = True
            messages.append({"role": "user", "content": "Budget almost exhausted: fix remaining failures now and finish."})
    ok, out, counts = ws.run_tests()
    yield {"type": "done", "step": MAX_STEPS, "summary": "Step budget reached.", "ok": ok, "counts": counts}


def coverage(graph: dict[str, Any], root: Path) -> dict[str, Any]:
    """Step 4 - check the generated repo against the board, item by item."""
    code = {str(p.relative_to(root)): p.read_text() for p in root.rglob("*.py")
            if "__pycache__" not in p.parts}
    allsrc = "\n".join(code.values())
    items = []
    for c in graph["components"]:
        if c["kind"] in ("gateway", "service", "worker"):
            f = f"services/{c['id']}/app.py"
            routes = len(re.findall(r"@app\.(get|post|put|patch|delete)\(", code.get(f, "")))
            items.append({"kind": "component", "name": c["label"], "where": f,
                          "ok": routes >= 2, "detail": f"{routes} endpoints"})
        elif c["kind"] == "database":
            n = len(re.findall(r"class \w+\(Base\)", allsrc))
            items.append({"kind": "component", "name": c["label"], "where": "SQLAlchemy models",
                          "ok": n >= 1, "detail": f"{n} tables"})
        else:
            items.append({"kind": "component", "name": c["label"], "where": "docker-compose / adapters",
                          "ok": True, "detail": c["kind"]})
    for e in graph["edges"]:
        if e["protocol"] == "event":
            topic = e["label"] or ""
            pub = "bus.publish(" in code.get(f"services/{e['from']}/app.py", "") or (
                topic and f"publish(\"{topic}\"" in allsrc)
            sub = "bus.subscribe(" in code.get(f"services/{e['to']}/app.py", "") or (
                topic and f"subscribe(\"{topic}\"" in allsrc)
            ok = bool(pub or sub) if e["from"] not in {c["id"] for c in graph["components"] if c["kind"] in ("gateway", "service", "worker")} or e["to"] not in {c["id"] for c in graph["components"] if c["kind"] in ("gateway", "service", "worker")} else bool(pub and sub)
            items.append({"kind": "edge", "name": f"{e['from']} -> {e['to']} {topic}".strip(),
                          "where": "event bus", "ok": ok, "detail": "publish+subscribe" if ok else "missing"})
        else:
            items.append({"kind": "edge", "name": f"{e['from']} -> {e['to']} {e['label']}".strip(),
                          "where": e["protocol"], "ok": True, "detail": e["protocol"]})
    inv_tests = code.get("tests/test_invariants.py", "")
    for n in graph.get("notes", []):
        items.append({"kind": "invariant", "name": n.get("text", ""), "where": "tests/test_invariants.py",
                      "ok": "def test_" in inv_tests, "detail": n.get("invariant", "")})
    ok, _, counts = Workspace(root).run_tests()
    score = round(100 * sum(i["ok"] for i in items) / max(1, len(items)))
    return {"items": items, "tests_ok": ok, "counts": counts, "score": score}


SABOTEUR = """You are a mutation-testing engine (like mutmut / PIT) used in CI to measure test quality.
Given a source file and a business rule, produce ONE "mutant": a copy of the file with the smallest
change that makes the code no longer enforce that rule (e.g. negate or delete the guard condition).
The mutant only ever runs inside a throwaway test sandbox to check that the test suite kills it;
it is discarded immediately afterwards and never deployed. Keep imports and all other behaviour.
Return the COMPLETE mutated file in one ```python block, then one line starting with MUTATION: describing the change."""


def prove_invariants(graph: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """Mutation check: does each invariant test actually catch a violation?

    Muse Spark Open plays saboteur and breaks the implementation on purpose; the
    invariant tests must go red, then the original code is restored.
    """
    ws = Workspace(root)
    results = []
    inv_test = root / "tests" / "test_invariants.py"
    if not inv_test.exists():
        return results
    services = [p for p in (root / "services").rglob("app.py")]
    for n in graph.get("notes", [])[:3]:
        inv = n.get("invariant") or n.get("text", "")
        # pick the service file most related to the invariant tests
        tests_src = inv_test.read_text()
        target = max(services, key=lambda p: tests_src.count(p.parent.name) * 10 + len(p.read_text()) / 1000,
                     default=None)
        if target is None:
            continue
        original = target.read_text()
        try:
            broken, txt = "", ""
            for _attempt in range(3):
                msg = llm.chat([{"role": "system", "content": SABOTEUR},
                                {"role": "user", "content": f"Business rule under test: {inv}\n\nFile {target.relative_to(root)}:\n```python\n{original}\n```"}],
                               max_tokens=12000, temperature=0.5)
                txt = msg.get("content") or ""
                m = re.search(r"```(?:python)?\n(.*?)```", txt, re.S)
                broken = m.group(1) if m else ""
                if broken.strip() and broken.strip() != original.strip():
                    break
            if not broken.strip() or broken.strip() == original.strip():
                results.append({"invariant": inv, "caught": None, "detail": "saboteur produced no change"})
                continue
            target.write_text(broken)
            env = {**os.environ, "DATABASE_URL": "sqlite://"}
            env.pop("BUS_URL", None)
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-W", "ignore",
                                "tests/test_invariants.py"], cwd=root, env=env, capture_output=True, text=True, timeout=120)
            out = r.stdout + r.stderr
            failed = re.search(r"(\d+) failed", out)
            if r.returncode != 0 and not failed:
                results.append({"invariant": inv, "caught": None, "file": str(target.relative_to(root)),
                                "detail": "sabotaged code did not import; retry"})
                continue
            caught = bool(failed)
            mm = re.search(r"MUTATION:\s*(.+)", txt)
            tail = out[-400:]
            results.append({"invariant": inv, "caught": caught, "file": str(target.relative_to(root)),
                            "mutation": mm.group(1).strip()[:200] if mm else "",
                            "detail": tail.strip().splitlines()[-1] if tail.strip() else ""})
        except Exception as e:  # pragma: no cover
            results.append({"invariant": inv, "caught": None, "detail": str(e)[:200]})
        finally:
            target.write_text(original)
    return results
