"""Offline tests: the deterministic scaffold must be green before any LLM touches it."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app import scaffold, vision

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


@pytest.mark.parametrize("name", ["board_rooms", "board_shop"])
def test_scaffold_is_green(name, tmp_path):
    graph = vision.normalize_graph(json.loads((SAMPLES / f"{name}.graph.json").read_text()))
    files = scaffold.build(graph, tmp_path)
    assert "docker-compose.yml" in files
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                       cwd=tmp_path, capture_output=True, text=True,
                       env={"DATABASE_URL": "sqlite://", "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stdout + r.stderr


def test_mermaid_has_every_edge():
    graph = vision.normalize_graph(json.loads((SAMPLES / "board_rooms.graph.json").read_text()))
    m = vision.to_mermaid(graph)
    assert m.count("-->") + m.count("-.->") + m.count("==>") >= len(graph["edges"])
