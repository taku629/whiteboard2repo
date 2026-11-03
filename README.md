# Whiteboard2Repo

**Snap the whiteboard after the design meeting. Get a running, tested repo before everyone has left the room.**

Built for the Meta Global AI Developer Hackathon (Nov 2026) · Track: Build for Your World · Solo · Everything in this repo was written during the hackathon.

![Read the board](media/01_read.png)
![Built and proven](media/02_built.png)

## The problem

Every system starts as boxes and arrows on a whiteboard. Then someone takes a phone photo, it rots in a chat thread, and a day later an engineer re-types the same design into boilerplate by hand: services, a database, a queue, a compose file, tests. The constraints people wrote in the corner ("no double booking!") are the first thing to get lost.

## What it does

1. **Read the board** – Meta **Muse Spark Open** looks at the photo and returns a typed graph: components (client / service / database / worker / queue / external) with bounding boxes, every arrow with its label and protocol, and the handwritten notes. Each note is rewritten as a testable **invariant**. Optional Meta **SAM 3.1** segmentation regions are passed in as grounding hints.
2. **Confirm** – you see the boxes drawn over your photo, a Mermaid diagram, and an editable table. Fix anything the model misread before building.
3. **Build** – a deterministic scaffolder emits a green skeleton (FastAPI per service, SQLAlchemy models, an event bus, `docker-compose.yml`, Dockerfile). Then a Muse Spark Open **tool-calling agent** (`list_files`, `read_file`, `write_file`, `run_tests`) implements every endpoint and edge and writes tests until `pytest` is green.
4. **Prove the invariants** – for each invariant the model plays a mutation-testing engine: it injects the smallest change that would break that rule, the tests run, and the mutant must be **KILLED**. Then the code is restored. A rule only counts as covered when breaking it makes the tests fail.
5. **Board coverage** – every box, arrow and note on the board is traced to the file / endpoint / test that implements it. Download the repo as a zip.

Example run (board_rooms sample): read in 8.9 s → build 78 s → 16 tests passing, board coverage 100 %, "no double booking" mutant KILLED.

## Meta models used

| Model | Where | Why |
|---|---|---|
| **Muse Spark Open** (open weights) | `app/vision.py`, `app/agent.py` | One model for vision (sketch → graph + bboxes + notes), planning, multi-step tool calling, and adversarial mutation. Open weights means a team can self-host it next to their code. |
| **SAM 3.1** (open weights) | `sam/sam31_regions.py` | Text-prompted segmentation ("rectangle box", "cylinder", "arrow", "handwritten text") gives pixel regions that ground the graph read. Run offline on a free GPU; results are cached as `samples/*.sam.json`. |

Muse Spark Open is called through any OpenAI-compatible endpoint (we used Nebius Token Factory). Nothing is hard-wired to one provider.

## Run it

```bash
pip install -r requirements.txt coverage
export MUSE_API_KEY=...            # key for an OpenAI-compatible endpoint serving Muse Spark Open
# optional: MUSE_BASE_URL (default https://api.tokenfactory.nebius.com/v1), MUSE_MODEL (default meta-models/muse-spark-open)
uvicorn app.main:app --port 8765
```

Open http://localhost:8765, pick a sample or upload a photo. Or with Docker: `docker build -t w2r . && docker run -p 8000:8000 -e MUSE_API_KEY=... w2r`.

Offline tests (no model needed): `pytest tests`.

## Layout

```
app/llm.py        OpenAI-compatible client for Muse Spark Open (vision + tools)
app/vision.py     photo -> graph JSON (components, edges, notes -> invariants), Mermaid
app/scaffold.py   deterministic, always-green skeleton generator
app/agent.py      tool-calling build loop, mutation proof, board coverage
app/main.py       FastAPI + SSE progress stream
app/static/       single-page UI
sam/              SAM 3.1 region extraction script (GPU or >=8 GB CPU)
samples/          demo boards + cached graph reads
```

## Honest notes

- The two sample whiteboards in `samples/` are **AI-generated test images**, not photos of a real meeting. Upload your own photo to try a real board.
- Generated code runs in a temporary directory per job. For a public deployment, run builds in a container sandbox.
- Mutation proof needs the model to cooperate; if it refuses or the mutant survives, the UI says so instead of claiming coverage.

## License

MIT
