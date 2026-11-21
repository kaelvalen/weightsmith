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


# --------------------------------------------------------------- trainscope recording


def scope_rows(client, rid, name="global.arrow"):
    from trainscope.io.writer import read_arrow_rows_sync

    run_dir = client.app.state.jobs.store.run_dir(rid)
    return read_arrow_rows_sync(run_dir / "scope" / name)


def scope_dir(client, rid):
    return client.app.state.jobs.store.run_dir(rid) / "scope"


def test_run_records_trainscope(client):
    rec = submit(client, tiny_classifier())
    assert wait(client, rec["id"])["status"] == "completed"
    assert [r["step"] for r in scope_rows(client, rec["id"])] == list(range(30))


def test_runs_without_trainscope(tmp_path):
    import subprocess
    import sys

    from weightsmith_codegen.project import write_run

    out = tmp_path / "export"
    write_run(tiny_classifier(), out)
    blocked = "import runpy, sys; sys.modules['trainscope'] = None; sys.argv = ['train.py']; runpy.run_path('train.py', run_name='__main__')"
    r = subprocess.run([sys.executable, "-c", blocked], cwd=out, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert not (out / "scope").exists()


def test_spike_event_on_lr_ramp(client):
    rec = submit(client, tiny_classifier(epochs=8, optimizer={"name": "sgd", "lr": 1.0}, scheduler={"name": "constant"},
                                         grad_clip=None))
    done = wait(client, rec["id"])
    spikes = client.get(f"/api/runs/{rec['id']}/events", params={"types": "spike"}).json()
    assert spikes, done
    assert done["summary"]["first_spike_step"] == spikes[0]["step"]
    assert done["summary"]["spikes"] == len(spikes)


def test_resume_appends_scope(client):
    rec = submit(client, tiny_classifier(max_steps=10**7, log_every=1))
    rid = rec["id"]
    for _ in range(2):  # run, stop, resume, stop
        wait(client, rid, ("running",))
        time.sleep(0.5)
        client.post(f"/api/runs/{rid}/control", json={"cmd": "stop"})
        final = wait(client, rid, ("stopped",))
        if _ == 0:
            assert client.post(f"/api/runs/{rid}/control", json={"cmd": "resume"}).status_code == 200
    steps = [r["step"] for r in scope_rows(client, rid)]
    assert steps == list(range(final["summary"]["last"]["step"]))


def test_compile_keeps_layer_names(client):
    rec = submit(client, tiny_classifier(compile=True))
    done = wait(client, rec["id"], timeout=300)
    if done["status"] != "completed":
        pytest.skip(f"torch.compile unavailable here: {(done.get('error') or {}).get('message', '')[:200]}")
    names = [p.name for p in (scope_dir(client, rec["id"]) / "layers").iterdir()]
    assert names and not any("_orig_mod" in n for n in names)


def test_nan_flushes_scope(client):
    rec = submit(client, tiny_classifier(optimizer={"name": "sgd", "lr": 1e30}, grad_clip=None))
    done = wait(client, rec["id"])
    assert done["error"]["class"] == "nan"
    assert len(scope_rows(client, rec["id"])) >= 1


def test_lora_job_records(client):
    job = tiny_classifier()
    job["graph"]["nodes"][1]["adapt"] = [{"kind": "lora", "r": 2, "alpha": 4}]
    rec = submit(client, job)
    assert wait(client, rec["id"])["status"] == "completed"
    from urllib.parse import quote

    def last_grad(layer):
        return scope_rows(client, rec["id"], f"layers/{quote(layer, safe='')}.arrow")[-1]["grad_l2_norm"]

    assert last_grad("fc1.base.weight") == 0.0  # frozen base: recorded, never updated
    assert last_grad("fc1.lora_B") > 0.0


# --------------------------------------------------------------- /api/scope


def test_scope_api_serves_run(client):
    rec = submit(client, tiny_classifier())
    wait(client, rec["id"])
    layers = client.get(f"/api/scope/runs/{rec['id']}/layers").json()
    assert "fc1.weight" in layers
    assert len(client.get(f"/api/scope/runs/{rec['id']}/global").json()) == 30


def test_scope_api_404_for_run_without_scope(client):
    rec = submit(client, tiny_classifier(diagnostics={"trainscope": False}))
    assert wait(client, rec["id"])["status"] == "completed"
    assert client.get(f"/api/scope/runs/{rec['id']}/global").status_code == 404
    assert client.get("/api/scope/runs/nope/global").status_code == 404


def test_scope_cluster_over_runs(client):
    a = submit(client, tiny_classifier())
    b = submit(client, tiny_classifier(seed=1))
    wait(client, a["id"])
    wait(client, b["id"])
    data = client.get("/api/scope/cluster").json()
    names = set(data["unclustered"]) | {n for c in data["clusters"] for n in c["runs"]}
    assert {a["id"], b["id"]} <= names
    assert {r["name"] for r in client.get("/api/scope/runs").json()} >= {a["id"], b["id"]}
