"""Experiment store: SQLite for run records, the file system for everything else.

Layout under the workspace (default ~/.weightsmith, override WEIGHTSMITH_HOME)::

    weightsmith.db
    projects/<slug>.json          project file: variants (graph IR each) + job defaults
    runs/<run id>/                job.json, model.py, train.py, data.py, config.yaml,
                                  events.jsonl, checkpoints/, model.pt, adapters.pt

A run is identified by graph hash + config hash + data hash + seed: the same
four always give the same run id.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
create table if not exists runs (
    id text primary key,
    project text,
    variant text,
    name text,
    graph_hash text not null,
    config_hash text not null,
    data_hash text not null,
    seed integer not null,
    status text not null,
    parent text,
    ablation text,
    created real not null,
    updated real not null,
    summary text not null default '{}',
    error text
);
create index if not exists runs_project on runs(project);
"""

STATUSES = ("queued", "preparing", "running", "paused", "completed", "stopped", "failed", "interrupted")


def workspace_root() -> Path:
    return Path(os.environ.get("WEIGHTSMITH_HOME", Path.home() / ".weightsmith")).expanduser()


def run_id(graph_hash: str, config_hash: str, data_hash: str, seed: int | str) -> str:
    return hashlib.sha256(f"{graph_hash}|{config_hash}|{data_hash}|{seed}".encode()).hexdigest()[:16]


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "project"


class Store:
    def __init__(self, root: Path | None = None):
        self.root = (root or workspace_root()).expanduser()
        (self.root / "projects").mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "weightsmith.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ------------------------------------------------------------- projects

    def project_path(self, slug: str) -> Path:
        return self.root / "projects" / f"{slugify(slug)}.json"

    def list_projects(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.root.glob("projects/*.json")):
            try:
                d = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            out.append({"slug": p.stem, "name": d.get("name", p.stem), "variants": list(d.get("variants", {})), "updated": p.stat().st_mtime})
        return sorted(out, key=lambda d: -d["updated"])

    def load_project(self, slug: str) -> dict[str, Any] | None:
        p = self.project_path(slug)
        return json.loads(p.read_text()) if p.exists() else None

    def save_project(self, project: dict[str, Any]) -> str:
        slug = slugify(project.get("slug") or project["name"])
        project["slug"] = slug
        p = self.project_path(slug)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(project, indent=2))
        tmp.replace(p)
        return slug

    def delete_project(self, slug: str) -> None:
        self.project_path(slug).unlink(missing_ok=True)

    # ----------------------------------------------------------------- runs

    def run_dir(self, rid: str) -> Path:
        return self.root / "runs" / rid

    def _row(self, r: sqlite3.Row | None) -> dict[str, Any] | None:
        if r is None:
            return None
        d = dict(r)
        d["summary"] = json.loads(d["summary"] or "{}")
        d["ablation"] = json.loads(d["ablation"]) if d["ablation"] else None
        d["error"] = json.loads(d["error"]) if d["error"] else None
        return d

    def get_run(self, rid: str) -> dict[str, Any] | None:
        with self._lock:
            return self._row(self.db.execute("select * from runs where id = ?", (rid,)).fetchone())

    def list_runs(self, project: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if project:
                rows = self.db.execute("select * from runs where project = ? order by created desc", (project,)).fetchall()
            else:
                rows = self.db.execute("select * from runs order by created desc").fetchall()
        return [self._row(r) for r in rows]  # type: ignore[misc]

    def create_run(self, rid: str, hashes: dict[str, str], project: str | None, variant: str | None, name: str | None,
                   parent: str | None = None, ablation: Any = None) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self.db.execute(
                "insert into runs (id, project, variant, name, graph_hash, config_hash, data_hash, seed, status, parent, ablation, created, updated)"
                " values (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)",
                (rid, project, variant, name, hashes["graph"], hashes["config"], hashes["data"], int(hashes["seed"]), parent,
                 json.dumps(ablation) if ablation is not None else None, now, now),
            )
            self.db.commit()
        return self.get_run(rid)  # type: ignore[return-value]

    def update_run(self, rid: str, status: str | None = None, summary: dict[str, Any] | None = None, error: Any = ...) -> None:
        with self._lock:
            row = self.get_run(rid)
            if row is None:
                return
            sets, vals = ["updated = ?"], [time.time()]
            if status is not None:
                sets.append("status = ?")
                vals.append(status)
            if summary is not None:
                merged = {**row["summary"], **summary}
                sets.append("summary = ?")
                vals.append(json.dumps(merged, default=float))
            if error is not ...:
                sets.append("error = ?")
                vals.append(json.dumps(error) if error is not None else None)
            self.db.execute(f"update runs set {', '.join(sets)} where id = ?", (*vals, rid))
            self.db.commit()

    def delete_run(self, rid: str) -> None:
        with self._lock:
            self.db.execute("delete from runs where id = ?", (rid,))
            self.db.commit()
        shutil.rmtree(self.run_dir(rid), ignore_errors=True)

    def mark_interrupted(self) -> None:
        """On daemon start: anything that was live is now resumable from its last checkpoint."""
        with self._lock:
            self.db.execute("update runs set status = 'interrupted' where status in ('running', 'paused', 'preparing')")
            self.db.execute("update runs set status = 'stopped' where status = 'queued'")
            self.db.commit()

    # --------------------------------------------------------------- events

    def events(self, rid: str, since: int = 0, types: set[str] | None = None) -> list[dict[str, Any]]:
        path = self.run_dir(rid) / "events.jsonl"
        if not path.exists():
            return []
        out = []
        with open(path) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev["seq"] > since and (types is None or ev["type"] in types):
                    out.append(ev)
        return out

    def latest_checkpoint(self, rid: str) -> Path | None:
        ckpts = [p for p in (self.run_dir(rid) / "checkpoints").glob("step_*.pt") if not p.stem.endswith(".adapters")]
        if not ckpts:
            return None
        return max(ckpts, key=lambda p: int(p.stem.split("_")[1]))
