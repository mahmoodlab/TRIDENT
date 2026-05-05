import os
from pathlib import Path


def _run_batch(
    monkeypatch,
    *,
    tmp_path: Path,
    patch_encoder: str,
    gpus: list[int] | None,
) -> dict[str, bytes]:
    """
    Run `run_batch_of_slides.main()` in a fully mocked mode and return a mapping
    of relative output file paths -> bytes.
    """
    import run_batch_of_slides as rbs

    job_dir = tmp_path / "job"
    wsi_dir = tmp_path / "wsis"
    job_dir.mkdir(parents=True, exist_ok=True)
    wsi_dir.mkdir(parents=True, exist_ok=True)

    # Dummy slides (the code only uses basename/stem in our mocked run).
    slide_paths = []
    for stem in ["s1", "s2", "s3", "s4", "s5"]:
        p = wsi_dir / f"{stem}.svs"
        p.write_text("dummy")
        slide_paths.append(str(p))

    # Avoid touching real discovery and any real Processor logic.
    monkeypatch.setattr(rbs, "collect_valid_slides", lambda **_kwargs: list(slide_paths))

    # Make the run bookkeeping deterministic and out of the equation.
    monkeypatch.setattr(rbs, "start_run", lambda *_args, **_kwargs: "test-run-id")
    monkeypatch.setattr(rbs, "finalize_run", lambda *_args, **_kwargs: None)

    # Avoid removing `.lock` files or wiping caches in our tmp tree.
    monkeypatch.setattr(rbs, "cleanup_cache", lambda *_args, **_kwargs: None)

    # Mock worker to write deterministic outputs independent of GPU.
    def fake_worker_entrypoint(args):
        mag_str = f"{float(args.mag):g}"
        coords_dir = args.coords_dir or f"{mag_str}x_{args.patch_size}px_{args.overlap}px_overlap"
        out_dir = Path(args.job_dir) / coords_dir / f"features_{args.patch_encoder}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for slide_path in list(getattr(args, "selected_wsi_paths", []) or []):
            stem = os.path.splitext(os.path.basename(slide_path))[0]
            out_fp = out_dir / f"{stem}.h5"
            out_fp.write_bytes(f"encoder={args.patch_encoder}\nslide={stem}\n".encode("utf-8"))

    monkeypatch.setattr(rbs, "worker_entrypoint", fake_worker_entrypoint)

    # Make "multiprocessing" deterministic and in-process for multi-GPU mode.
    class _InlineProcess:
        def __init__(self, target, args=()):
            self._target = target
            self._args = args
            self.exitcode = None

        def start(self):
            try:
                self._target(*self._args)
                self.exitcode = 0
            except Exception:
                self.exitcode = 1
                raise

        def join(self):
            return

    class _InlineContext:
        def Process(self, target, args=()):
            return _InlineProcess(target=target, args=args)

    monkeypatch.setattr(rbs.mp, "get_context", lambda *_args, **_kwargs: _InlineContext())

    argv = [
        "run_batch_of_slides.py",
        "--task",
        "feat",
        "--job_dir",
        str(job_dir),
        "--wsi_dir",
        str(wsi_dir),
        "--patch_encoder",
        patch_encoder,
        "--mag",
        "20",
        "--patch_size",
        "256",
        "--overlap",
        "0",
        "--coords_dir",
        "coords",
        "--max_workers",
        "0",
    ]
    if gpus is not None:
        argv += ["--gpus", *[str(x) for x in gpus]]
    else:
        argv += ["--gpu", "0"]

    monkeypatch.setattr(rbs.sys, "argv", argv)
    rbs.main()

    # Collect outputs relative to job_dir for stable comparison.
    outputs: dict[str, bytes] = {}
    for fp in job_dir.rglob("*"):
        if fp.is_file():
            rel = str(fp.relative_to(job_dir))
            outputs[rel] = fp.read_bytes()
    return outputs


def test_single_vs_multi_gpu_equivalent_outputs_uni_v1(monkeypatch, tmp_path):
    single = _run_batch(monkeypatch, tmp_path=tmp_path / "single", patch_encoder="uni_v1", gpus=None)
    multi = _run_batch(monkeypatch, tmp_path=tmp_path / "multi", patch_encoder="uni_v1", gpus=[0, 1])
    assert single == multi


def test_single_vs_multi_gpu_equivalent_outputs_conch_v1(monkeypatch, tmp_path):
    single = _run_batch(monkeypatch, tmp_path=tmp_path / "single", patch_encoder="conch_v1", gpus=None)
    multi = _run_batch(monkeypatch, tmp_path=tmp_path / "multi", patch_encoder="conch_v1", gpus=[0, 1])
    assert single == multi

