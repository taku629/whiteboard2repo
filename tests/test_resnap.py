"""Offline tests for Re-snap: board diff + incremental update with a mocked model."""
import copy
import json
from pathlib import Path

from app import agent, diff, scaffold, vision

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _old():
    return vision.normalize_graph(json.loads((SAMPLES / "board_rooms.graph.json").read_text()))


def _new(old):
    g = copy.deepcopy(old)
    g["components"].append({"id": "payments_api", "label": "Payments API (external)", "kind": "external",
                            "bbox": None, "responsibility": "charges bookings"})
    g["components"].append({"id": "audit_worker", "label": "Audit worker", "kind": "worker",
                            "bbox": None, "responsibility": "logs every booking"})
    api = next(c["id"] for c in g["components"] if c["kind"] == "service")
    g["edges"].append({"from": api, "to": "payments_api", "label": "charge", "protocol": "http"})
    g["notes"].append({"text": "max 3 active bookings per user!",
                       "invariant": "A user may hold at most 3 active bookings"})
    return g


def test_identical_boards_have_empty_diff():
    old = _old()
    d = diff.diff_graphs(old, copy.deepcopy(old))
    assert d["empty"] and d["count"] == 0


def test_diff_finds_added_parts():
    old = _old()
    d = diff.diff_graphs(old, _new(old))
    assert {c["id"] for c in d["components"]["added"]} == {"payments_api", "audit_worker"}
    assert [e["to"] for e in d["edges"]["added"]] == ["payments_api"]
    assert len(d["invariants"]["added"]) == 1 and len(d["invariants"]["kept"]) == len(old["notes"])
    assert not d["components"]["removed"] and not d["edges"]["removed"]


def test_relabelled_box_keeps_its_id():
    old = _old()
    new = copy.deepcopy(old)
    svc = next(c for c in new["components"] if c["kind"] == "service")
    old_id = svc["id"]
    svc["id"], svc["label"] = "booking_service", svc["label"].replace("API", "Service")
    for e in new["edges"]:
        e["from"] = "booking_service" if e["from"] == old_id else e["from"]
        e["to"] = "booking_service" if e["to"] == old_id else e["to"]
    d = diff.diff_graphs(old, new)
    assert not d["components"]["added"] and not d["components"]["removed"], diff.summary_lines(d)
    assert not d["edges"]["added"] and not d["edges"]["removed"]
    assert [c["fields"] for c in d["components"]["changed"]] == [["label"]]
    assert old_id in {c["id"] for c in d["aligned"]["components"]}


def test_removed_component_and_mermaid_marks():
    old = _old()
    new = copy.deepcopy(old)
    gone = next(c["id"] for c in new["components"] if c["kind"] == "worker")
    new["components"] = [c for c in new["components"] if c["id"] != gone]
    new["edges"] = [e for e in new["edges"] if gone not in (e["from"], e["to"])]
    d = diff.diff_graphs(old, new)
    assert [c["id"] for c in d["components"]["removed"]] == [gone]
    m = diff.to_mermaid_diff(old, d)
    assert f"class {gone} removed" in m and "stroke-dasharray" in m


def test_update_applies_diff_with_mocked_model(tmp_path, monkeypatch):
    old = _old()
    scaffold.build(old, tmp_path)
    keep = (tmp_path / "ARCHITECTURE.md").read_text()
    new = _new(old)
    script = iter([
        {"content": "Add a payments client and the new invariant test.", "tool_calls": [
            {"id": "1", "function": {"name": "write_file", "arguments": json.dumps({
                "path": "services/payments_client.py",
                "content": "CHARGES = []\n\ndef charge(booking_id, amount):\n    CHARGES.append((booking_id, amount))\n    return True\n"})}},
            {"id": "2", "function": {"name": "write_file", "arguments": json.dumps({
                "path": "tests/test_invariants.py",
                "content": "from services.payments_client import charge, CHARGES\n\n\ndef test_charge_recorded():\n    assert charge(1, 10)\n    assert CHARGES[-1] == (1, 10)\n"})}},
            {"id": "3", "function": {"name": "run_tests", "arguments": "{}"}}]},
        {"content": "", "tool_calls": [
            {"id": "4", "function": {"name": "finish", "arguments": json.dumps({"summary": "done"})}}]},
    ])
    monkeypatch.setattr(agent.llm, "chat", lambda *a, **k: next(script))
    events = list(agent.update(old, new, tmp_path))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "scaffold_diff" and "done" in kinds and kinds[-1] == "changes"
    assert (tmp_path / "services/audit_worker/app.py").exists()
    ch = events[-1]
    assert "services/payments_client.py" in ch["added"]
    assert "services/audit_worker/app.py" in ch["added"]
    assert "docker-compose.yml" in ch["modified"]
    assert (tmp_path / "ARCHITECTURE.md").read_text() != keep
    ok, out, counts = agent.Workspace(tmp_path).run_tests()
    assert ok, out


def test_update_with_no_changes_skips_the_model(tmp_path, monkeypatch):
    old = _old()
    scaffold.build(old, tmp_path)

    def boom(*a, **k):
        raise AssertionError("model must not be called for an unchanged board")
    monkeypatch.setattr(agent.llm, "chat", boom)
    events = list(agent.update(old, copy.deepcopy(old), tmp_path))
    assert events[-1]["type"] == "changes"
    assert not events[-1]["added"] and not events[-1]["deleted"]
