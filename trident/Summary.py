from __future__ import annotations

import json
import os
import socket
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from trident.State import load_all_states, atomic_write_json


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _trident_version() -> str:
    try:
        from importlib.metadata import version as pkg_version

        return pkg_version("trident")
    except Exception:
        return "unknown"


def _ensure_runs_dir(job_dir: str) -> str:
    runs_dir = os.path.join(job_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    return runs_dir


def _atomic_write_text(path: str, content: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")
    os.replace(tmp, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def start_run(
    job_dir: str,
    *,
    tool: str,
    args: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a lightweight run manifest under `job_dir/runs/` and return its run_id.

    This is designed to be called once per CLI invocation / "run".
    """
    run_id = uuid.uuid4().hex[:12]
    runs_dir = _ensure_runs_dir(job_dir)
    manifest_path = os.path.join(runs_dir, f"{run_id}.json")

    args_safe = _json_safe(args or {})
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "tool": tool,
        "started_at": _now_iso(),
        "finished_at": None,
        "status": "running",
        "error": None,
        "trident_version": _trident_version(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "job_dir": os.path.abspath(job_dir),
        "args": args_safe,
    }
    atomic_write_json(manifest_path, manifest)
    return run_id


def _task_label(status: Any, reason: Any) -> str:
    if not status:
        return "unknown"
    if reason:
        return f"{status} ({reason})"
    return str(status)


def _aggregate_states(job_dir: str) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
        (num_slides, task_counts, errors)
    """
    states = load_all_states(job_dir)
    num_slides = len(states)
    task_counts: Dict[str, Any] = {}
    errors: Dict[str, Any] = {}

    for _, st in states.items():
        slide = st.get("slide", {}) or {}
        slide_name = slide.get("name") or slide.get("id") or "unknown_slide"
        tasks = (st.get("tasks") or {}) if isinstance(st.get("tasks"), dict) else {}

        # Collect last error per slide (if any)
        resume = st.get("resume") or {}
        last_error = resume.get("last_error") if isinstance(resume, dict) else None
        if last_error and isinstance(last_error, dict):
            err_msg = last_error.get("error") or last_error.get("message")
            if err_msg:
                errors[slide_name] = str(err_msg)

        for task_name, t in tasks.items():
            if not isinstance(t, dict):
                continue
            status = t.get("status")
            reason = t.get("reason")
            label = _task_label(status, reason)

            if task_name.startswith("patch_features:"):
                enc = task_name.split(":", 1)[1]
                group = task_counts.setdefault("patch_features", {})
                enc_counts = group.setdefault(enc, {})
                enc_counts[label] = int(enc_counts.get(label, 0)) + 1
            elif task_name.startswith("slide_features:"):
                enc = task_name.split(":", 1)[1]
                group = task_counts.setdefault("slide_features", {})
                enc_counts = group.setdefault(enc, {})
                enc_counts[label] = int(enc_counts.get(label, 0)) + 1
            else:
                counts = task_counts.setdefault(task_name, {})
                counts[label] = int(counts.get(label, 0)) + 1

    return num_slides, task_counts, errors


def _render_counts(counts: Dict[str, int]) -> str:
    # Stable ordering: completed/skipped/error/running/not_started/unknown then alpha for the rest.
    preferred = [
        "completed",
        "skipped",
        "error",
        "running",
        "not_started",
        "unknown",
    ]

    def key(k: str) -> Tuple[int, str]:
        base = k.split(" ", 1)[0]
        try:
            return (preferred.index(base), k)
        except ValueError:
            return (len(preferred), k)

    parts = [f"{k}: {counts[k]}" for k in sorted(counts.keys(), key=key)]
    return ", ".join(parts) if parts else "no recorded tasks"


def _render_run_section(
    *,
    run_id: str,
    tool: str,
    manifest: Dict[str, Any],
    num_slides: int,
    task_counts: Dict[str, Any],
    errors: Dict[str, Any],
) -> str:
    started_at = manifest.get("started_at") or "unknown"
    finished_at = manifest.get("finished_at") or "unfinished"
    status = manifest.get("status") or "unknown"
    trident_version = manifest.get("trident_version") or "unknown"

    lines = []
    lines.append(f"## Run {started_at} (trident {trident_version}) — run_id={run_id}")
    lines.append(f"- Tool: `{tool}`")
    lines.append(f"- Status: **{status}**")
    lines.append(f"- Finished: `{finished_at}`")
    lines.append(f"- Slides with state: {num_slides}")

    args = manifest.get("args") or {}
    if isinstance(args, dict) and args:
        # Keep args compact; avoid dumping huge objects.
        keep = {}
        for k in sorted(args.keys()):
            v = args[k]
            if isinstance(v, (str, int, float, bool)) or v is None:
                keep[k] = v
        if keep:
            lines.append(f"- Args: `{json.dumps(keep, sort_keys=True)}`")

    if num_slides == 0:
        lines.append("")
        lines.append("> No `wsi_states/*.json` found yet for this job dir, so this summary only contains run metadata.")
        lines.append("")
        return "\n".join(lines)

    # Top-level tasks
    for task_name in sorted(k for k in task_counts.keys() if k not in {"patch_features", "slide_features"}):
        lines.append(f"- {task_name}: {_render_counts(task_counts[task_name])}")

    # Grouped features
    if "patch_features" in task_counts:
        lines.append("- Patch features:")
        for enc in sorted(task_counts["patch_features"].keys()):
            lines.append(f"  - {enc}: {_render_counts(task_counts['patch_features'][enc])}")
    if "slide_features" in task_counts:
        lines.append("- Slide features:")
        for enc in sorted(task_counts["slide_features"].keys()):
            lines.append(f"  - {enc}: {_render_counts(task_counts['slide_features'][enc])}")

    if errors:
        lines.append(f"- Errors ({len(errors)}):")
        for slide_name in sorted(errors.keys()):
            msg = errors[slide_name].replace("\n", " ").strip()
            if len(msg) > 200:
                msg = msg[:200] + "…"
            lines.append(f"  - {slide_name}: {msg}")

    lines.append("")
    return "\n".join(lines)


def finalize_run(
    job_dir: str,
    run_id: str,
    *,
    status: str,
    error: Optional[str] = None,
) -> None:
    """
    Finalize a run manifest and update `job_dir/summary.md`.
    """
    runs_dir = _ensure_runs_dir(job_dir)
    manifest_path = os.path.join(runs_dir, f"{run_id}.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        manifest = {"run_id": run_id, "tool": "unknown", "args": {}}

    manifest["finished_at"] = _now_iso()
    manifest["status"] = status
    manifest["error"] = error
    manifest.setdefault("trident_version", _trident_version())
    atomic_write_json(manifest_path, manifest)

    # Render and append a new section to summary.md
    num_slides, task_counts, errors = _aggregate_states(job_dir)
    section = _render_run_section(
        run_id=run_id,
        tool=str(manifest.get("tool") or "unknown"),
        manifest=manifest,
        num_slides=num_slides,
        task_counts=task_counts,
        errors=errors,
    )

    summary_path = os.path.join(job_dir, "summary.md")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            existing = f.read().rstrip() + "\n\n"
    else:
        existing = (
            "# TRIDENT job summary\n\n"
            "This file is updated once per run and summarizes what TRIDENT has done in this `job_dir`.\n\n"
            "- Per-slide machine-readable state lives in `wsi_states/*.json`.\n"
            "- Per-run manifests live in `runs/*.json`.\n\n"
        )
    _atomic_write_text(summary_path, existing + section)

