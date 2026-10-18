"""Daemon API end to end: real training subprocesses on a tiny CPU job."""

import io
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from conftest import tiny_classifier
from weightsmith_runtime.server import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "ws", web_dist=tmp_path / "nodist")) as c:
        yield c


def wait(client, rid, states=("completed", "failed", "stopped"), timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/api/runs/{rid}").json()
        if r["status"] in states:
            return r
        time.sleep(0.2)
    raise TimeoutError(r["status"])


def submit(client, job, **extra):
    r = client.post("/api/runs", json={**job, "project": "test", **extra})
    assert r.status_code == 200, r.text
    return r.json()


def test_validate_compile_catalog(client):
    job = tiny_classifier()
    v = client.post("/api/validate", json={"graph": job["graph"]}).json()
    assert v["ok"] and v["totals"]["total"] == 8 * 16 + 16 + 16 * 3 + 3 and v["hash"]
    c = client.post("/api/compile", json={"graph": job["graph"]}).json()
    assert c["ok"] and c["smoke"]["ok"] and "class Model" in c["model_py"]
    types = {t["type"] for t in client.get("/api/catalog").json()["types"]}
    assert {"Linear", "Attention", "PythonModule", "Subgraph"} <= types


def test_run_is_reproducible_and_identified_by_hashes(client):
    job = tiny_classifier()
    rec = submit(client, job)
    done = wait(client, rec["id"])
    assert done["status"] == "completed", done
    assert done["summary"]["last"]["step"] == 2 * 15  # 461 train samples / 32, two epochs
    again = submit(client, job)
    assert again["id"] == rec["id"] and again["existing"]
    other = submit(client, tiny_classifier(seed=1))
    assert other["id"] != rec["id"]
    wait(client, other["id"])
    # the metric series is replayable and sequence-numbered
    evs = client.get(f"/api/runs/{rec['id']}/events").json()
    seqs = [e["seq"] for e in evs]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    later = client.get(f"/api/runs/{rec['id']}/events", params={"since": seqs[5]}).json()
    assert later[0]["seq"] == seqs[6]
    assert (client.get(f"/api/runs/{rec['id']}/files/model.py").status_code) == 200


def test_websocket_resumes_from_seq(client):
    rec = submit(client, tiny_classifier())
    wait(client, rec["id"])
    with client.websocket_connect(f"/ws/runs/{rec['id']}?since=3") as ws:
        first = ws.receive_json()
    assert first["seq"] == 4


def test_pause_resume_stop_and_resume_from_checkpoint(client):
    job = tiny_classifier(max_steps=10**7, log_every=1)
    rec = submit(client, job)
    rid = rec["id"]
    wait(client, rid, ("running",))
    assert client.post(f"/api/runs/{rid}/control", json={"cmd": "pause"}).status_code == 200
    wait(client, rid, ("paused",))
    client.post(f"/api/runs/{rid}/control", json={"cmd": "checkpoint"})
    client.post(f"/api/runs/{rid}/control", json={"cmd": "resume"})
    wait(client, rid, ("running",))
    client.post(f"/api/runs/{rid}/control", json={"cmd": "stop"})
    stopped = wait(client, rid, ("stopped",))
    step_at_stop = stopped["summary"]["last"]["step"]
    evs = client.get(f"/api/runs/{rid}/events", params={"types": "checkpoint"}).json()
    assert len(evs) >= 2
    r = client.post(f"/api/runs/{rid}/control", json={"cmd": "resume"})
    assert r.status_code == 200, r.text
    wait(client, rid, ("running",))
    logs = client.get(f"/api/runs/{rid}/events", params={"types": "log"}).json()
    assert any("resumed from" in e["message"] for e in logs)
    client.post(f"/api/runs/{rid}/control", json={"cmd": "stop"})
    final = wait(client, rid, ("stopped",))
    assert final["summary"]["last"]["step"] >= step_at_stop


def test_ablation_links_to_parent(client):
    rec = submit(client, tiny_classifier())
    wait(client, rec["id"])
    r = client.post(f"/api/runs/{rec['id']}/ablate", json={"nodes": ["act"]})
    assert r.status_code == 200, r.text
    child = r.json()
    assert child["parent"] == rec["id"] and child["id"] != rec["id"]
    wait(client, child["id"])
    assert child["id"] in client.get(f"/api/runs/{rec['id']}").json()["children"]
    d = client.post("/api/diff", json={"a": tiny_classifier()["graph"], "b": client.get(f"/api/runs/{child['id']}").json()["job"]["graph"]}).json()
    assert [c["kind"] for c in d["changes"]] == ["disabled_changed"]


def test_export_with_and_without_merge(client):
    job = tiny_classifier()
    job["graph"]["nodes"][1]["adapt"] = [{"kind": "lora", "r": 2, "alpha": 4}]
    rec = submit(client, job)
    wait(client, rec["id"])
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/runs/{rec['id']}/export").content))
    assert {"model.py", "train.py", "config.yaml", "model.pt", "adapters.pt"} <= set(z.namelist())
    zm = zipfile.ZipFile(io.BytesIO(client.get(f"/api/runs/{rec['id']}/export", params={"merge": True}).content))
    assert "adapters.pt" not in zm.namelist()
    assert "LoRALinear" not in zm.read("model.py").decode()


def test_nan_is_classified(client):
    rec = submit(client, tiny_classifier(optimizer={"name": "sgd", "lr": 1e30}, grad_clip=None))
    done = wait(client, rec["id"])
    assert done["status"] == "failed" and done["error"]["class"] == "nan"


def test_invalid_graph_is_rejected(client):
    job = tiny_classifier()
    job["graph"]["nodes"][3]["params"]["in"] = 5
    r = client.post("/api/runs", json=job)
    assert r.status_code == 422


def test_run_note_and_final_status_after_error(client):
    rec = submit(client, tiny_classifier(optimizer={"name": "sgd", "lr": 1e30}, grad_clip=None))
    done = wait(client, rec["id"])
    for _ in range(50):  # the final status event follows when the process has exited
        states = [e["state"] for e in client.get(f"/api/runs/{rec['id']}/events", params={"types": "status"}).json()]
        if states[-1] == "failed":
            break
        time.sleep(0.1)
    assert states[-1] == "failed"  # announced as an event, not only stored
    r = client.patch(f"/api/runs/{rec['id']}", json={"note": "lr far too high on purpose", "name": "nan probe"}).json()
    assert r["summary"]["note"] == "lr far too high on purpose" and r["name"] == "nan probe"
    assert done["status"] == "failed"
