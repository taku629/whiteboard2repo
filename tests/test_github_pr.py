import json, subprocess
from pathlib import Path
from app import github_pr


def test_snapshot_and_preview(tmp_path: Path):
    job = tmp_path / "job"; repo = job / "repo"; repo.mkdir(parents=True)
    (repo / "a.py").write_text("x = 1\n")
    (job / "board.jpg").write_bytes(b"jpg1"); (job / "board.next.jpg").write_bytes(b"jpg2")
    github_pr.ensure_baseline(repo, job / "board.jpg")
    (repo / "a.py").write_text("x = 2\n")
    github_pr.snapshot(repo, 2, job / "board.next.jpg", "v2", branch=f"w2r/resnap-{job.name}-v2")
    log = subprocess.run(["git", "log", "--oneline", "--all"], cwd=repo, capture_output=True, text=True).stdout
    assert len(log.strip().splitlines()) == 2
    assert (repo / "board" / "v1.jpg").exists() and (repo / "board" / "v2.jpg").exists()
    (job / "version").write_text("2")
    (job / "report.v2.json").write_text(json.dumps({
        "version": 2, "summary": ["+ component payments_api"], "mermaid": "graph LR\n a-->b",
        "proofs": [{"invariant": "max 3 active bookings", "caught": True, "file": "services/booking_api/app.py",
                    "mutation": "guard disabled"}],
        "tests": {"ok": True, "passed": 24}, "coverage": {"score": 100, "items": []}, "changed_files": ["+ x.py"]}))
    out = github_pr.open_pr(job, "me/demo", None)
    assert out["preview"] and "KILLED" in out["body"] and "tests green" in out["body"]
    assert (job / "PR_PREVIEW.md").exists()


def test_unverified_pr_is_not_green():
    body = github_pr.pr_body({"version": 2, "proofs": [{"invariant": "r", "caught": False}],
                              "tests": {"ok": True, "passed": 3}}, "me/demo", "b")
    assert "NOT fully verified" in body and "SURVIVED" in body


def test_token_is_redacted():
    assert "secret" not in github_pr._redact("fatal: https://x-access-token:secret@github.com/a/b.git")
