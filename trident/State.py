from __future__ import annotations

import json
import os
import time
import hashlib
import uuid
from typing import Any, Dict, Optional


def _now_iso() -> str:
    # ISO-ish without importing datetime everywhere; good enough for logs.
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _stable_slide_hash(slide_path: str) -> str:
    return hashlib.sha1(slide_path.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _trident_version() -> str:
    try:
        from importlib.metadata import version as pkg_version

        return pkg_version("trident")
    except Exception:
        return "unknown"


def _is_probable_path(value: Any) -> bool:
    return isinstance(value, str) and ("/" in value or value.startswith(".") or value.startswith("~"))


def _stat_path(path: str) -> Dict[str, Any]:
    try:
        st = os.stat(path)
        return {"path": path, "exists": True, "bytes": int(st.st_size)}
    except Exception:
        return {"path": path, "exists": False, "bytes": None}


def _prune_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _normalize_outputs(outputs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in outputs.items():
        if _is_probable_path(v):
            out[k] = _stat_path(os.path.expanduser(v))
        else:
            out[k] = v
    return out


def _attempt_id() -> str:
    return uuid.uuid4().hex[:12]


def _ensure_task(tasks: Dict[str, Any], task: str) -> Dict[str, Any]:
    t = tasks.setdefault(
        task,
        {
            "status": "not_started",
            "reason": None,
            "message": None,
            "attempts": [],
            "outputs": {},
            "_active_attempt_id": None,
        },
    )
    # Backward compatibility if older state is loaded:
    t.setdefault("attempts", [])
    t.setdefault("outputs", {})
    t.setdefault("_active_attempt_id", None)
    return t


def _merge_attempt(t: Dict[str, Any], attempt_event: Dict[str, Any], final_status: str) -> None:
    """
    Convert event-style attempt updates into a single human-readable attempt record.

    We keep one attempt object with started_at/finished_at/duration_s/result/error.
    """
    event = attempt_event.get("event")
    ts = attempt_event.get("ts")
    ts_epoch = attempt_event.get("ts_epoch")
    err = attempt_event.get("error")

    if event == "started":
        aid = _attempt_id()
        t["_active_attempt_id"] = aid
        t["attempts"].append(
            {
                "attempt_id": aid,
                "started_at": ts,
                "started_epoch": ts_epoch,
                "finished_at": None,
                "finished_epoch": None,
                "duration_s": None,
                "result": "running",
                "error": None,
            }
        )
        return

    if event in {"finished", "error"}:
        aid = t.get("_active_attempt_id")
        target = None
        if aid:
            for a in reversed(t.get("attempts", [])):
                if a.get("attempt_id") == aid:
                    target = a
                    break
        if target is None:
            # If we missed the start event (e.g., legacy or crash), create a synthetic attempt.
            aid = _attempt_id()
            target = {
                "attempt_id": aid,
                "started_at": None,
                "started_epoch": None,
                "finished_at": None,
                "finished_epoch": None,
                "duration_s": None,
                "result": "running",
                "error": None,
            }
            t["attempts"].append(target)

        target["finished_at"] = ts
        target["finished_epoch"] = ts_epoch
        if target.get("started_epoch") is not None and ts_epoch is not None:
            target["duration_s"] = float(ts_epoch) - float(target["started_epoch"])
        target["result"] = final_status
        if err is not None:
            target["error"] = err
        t["_active_attempt_id"] = None
        return


def _recompute_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    tasks = state.get("tasks", {}) or {}
    summary: Dict[str, Any] = {}
    patch_feats: Dict[str, str] = {}
    slide_feats: Dict[str, str] = {}

    for task_name, t in tasks.items():
        status = t.get("status")
        reason = t.get("reason")
        label = status if not reason else f"{status} ({reason})"
        if task_name.startswith("patch_features:"):
            patch_feats[task_name.split(":", 1)[1]] = label
        elif task_name.startswith("slide_features:"):
            slide_feats[task_name.split(":", 1)[1]] = label
        else:
            summary[task_name] = label

    if patch_feats:
        summary["patch_features"] = patch_feats
    if slide_feats:
        summary["slide_features"] = slide_feats

    state["summary"] = summary
    return summary


def _update_resume(
    state: Dict[str, Any],
    task: str,
    status: str,
    reason: Optional[str],
    message: Optional[str],
    attempt: Optional[Dict[str, Any]],
) -> None:
    resume = state.setdefault("resume", {})
    resume["last_task"] = task
    resume["last_status"] = status
    resume["last_updated_at"] = state.get("updated_at")
    if status == "error":
        resume["last_error"] = {
            "task": task,
            "at": state.get("updated_at"),
            "reason": reason,
            "message": message,
            "error": (attempt or {}).get("error") if attempt else message,
        }


def ensure_states_dir(job_dir: str) -> str:
    states_dir = os.path.join(job_dir, "wsi_states")
    os.makedirs(states_dir, exist_ok=True)
    return states_dir


def state_path(job_dir: str, wsi_name: str, slide_path: str) -> str:
    """
    Compute a per-slide state file path.

    Uses `wsi.name` for readability and a short hash of `slide_path` for disambiguation.
    """
    states_dir = ensure_states_dir(job_dir)
    h = _stable_slide_hash(slide_path)
    fname = f"{wsi_name}__{h}.json"
    return os.path.join(states_dir, fname)


def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def load_state(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_states(job_dir: str) -> Dict[str, Dict[str, Any]]:
    states_dir = os.path.join(job_dir, "wsi_states")
    if not os.path.isdir(states_dir):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for fn in os.listdir(states_dir):
        if not fn.endswith(".json"):
            continue
        fp = os.path.join(states_dir, fn)
        try:
            state = load_state(fp)
        except Exception:
            continue
        slide_id = state.get("slide", {}).get("id") or fn
        out[slide_id] = state
    return out


def make_slide_ref(
    *,
    name: str,
    ext: str,
    slide_path: str,
    rel_path: Optional[str] = None,
    reader_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a lightweight slide reference used for state tracking.

    We avoid dataclasses here to keep this module dependency-free and simple.
    """
    slide_path_str = str(slide_path)
    return {
        "name": name,
        "ext": ext,
        "slide_path": slide_path_str,
        "rel_path": rel_path,
        "reader_type": reader_type,
        "id": f"{name}{ext}__{_stable_slide_hash(slide_path_str)}",
    }


def make_attempt(event: str, *, error: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a human-readable attempt/event record.

    We store both a readable timestamp and the epoch time for easy sorting.
    """
    rec: Dict[str, Any] = {
        "event": event,
        "ts": _now_iso(),
        "ts_epoch": time.time(),
    }
    if error is not None:
        rec["error"] = error
    if extra:
        rec.update(extra)
    return rec


def _base_state(slide: Dict[str, Any], wsi_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "trident_version": _trident_version(),
        "slide": {
            "id": slide["id"],
            "name": slide["name"],
            "ext": slide["ext"],
            "abs_path": slide["slide_path"],
            "rel_path": slide.get("rel_path"),
            "reader_type": slide.get("reader_type"),
        },
        "meta": _prune_none(wsi_meta or {}),
        "tasks": {},
        "summary": {},
        "resume": {},
        "updated_at": _now_iso(),
    }


def update_task_state(
    job_dir: str,
    slide: Dict[str, Any],
    task: str,
    status: str,
    *,
    reason: Optional[str] = None,
    message: Optional[str] = None,
    outputs: Optional[Dict[str, Any]] = None,
    attempt: Optional[Dict[str, Any]] = None,
    wsi_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Best-effort update of per-slide state.

    This must never break the pipeline: callers should wrap it in try/except.
    """
    fp = state_path(job_dir, slide["name"], slide["slide_path"])
    if os.path.exists(fp):
        try:
            state = load_state(fp)
        except Exception:
            state = _base_state(slide, wsi_meta=wsi_meta)
    else:
        state = _base_state(slide, wsi_meta=wsi_meta)

    state["updated_at"] = _now_iso()
    # Ensure schema markers exist if we loaded an older file.
    if int(state.get("schema_version", 0) or 0) < 2:
        state["schema_version"] = 2
    state.setdefault("trident_version", _trident_version())
    state.setdefault("summary", {})
    state.setdefault("resume", {})

    # Refresh slide fields if they become available later.
    try:
        slide_rec = state.setdefault("slide", {})
        if slide.get("slide_path"):
            slide_rec["abs_path"] = str(slide.get("slide_path"))
        if slide.get("rel_path") is not None:
            slide_rec["rel_path"] = slide.get("rel_path")
        if slide.get("reader_type") is not None:
            slide_rec["reader_type"] = slide.get("reader_type")
    except Exception:
        pass

    if wsi_meta is not None:
        # Refresh snapshot if provided, but never overwrite existing values with None.
        meta = state.get("meta", {}) if isinstance(state.get("meta"), dict) else {}
        meta = dict(meta)
        for k, v in (wsi_meta or {}).items():
            if v is not None:
                meta[k] = v
        state["meta"] = _prune_none(meta)

    tasks = state.setdefault("tasks", {})
    t = _ensure_task(tasks, task)
    t["status"] = status
    if reason is not None:
        t["reason"] = reason
    if message is not None:
        t["message"] = message
    else:
        # Avoid stale "started" messages when a task ends up skipped/completed/error later.
        if status in {"skipped", "completed", "error"} and t.get("message") == "started":
            t["message"] = None
    if outputs:
        norm = _normalize_outputs(outputs)
        t["outputs"] = {**t.get("outputs", {}), **norm}
    if attempt:
        _merge_attempt(t, attempt, final_status=status)

    _recompute_summary(state)
    _update_resume(state, task, status, reason, message, attempt)

    atomic_write_json(fp, state)

