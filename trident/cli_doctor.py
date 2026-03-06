from __future__ import annotations

import argparse
import ctypes.util
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CheckResult:
    status: str  # PASS, WARN, FAIL
    name: str
    message: str
    fix: str = ""


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _check_module(module_name: str, message: str, fix: str) -> CheckResult:
    if _has_module(module_name):
        return CheckResult("PASS", message, f"Module `{module_name}` is available.")
    return CheckResult("FAIL", message, f"Module `{module_name}` is missing.", fix)


def _check_file(path: Path, message: str, fix: str) -> CheckResult:
    if path.exists():
        return CheckResult("PASS", message, f"Found `{path}`.")
    return CheckResult("FAIL", message, f"Missing `{path}`.", fix)


def _check_hf_token() -> CheckResult:
    token = None

    # Preferred detection path: use huggingface_hub's resolver, which checks
    # env vars and local login cache (e.g. `huggingface-cli login`).
    try:
        from huggingface_hub import get_token  # type: ignore

        token = get_token()
    except Exception:
        token = None

    # Fallback for environments where huggingface_hub is unavailable
    # or token resolution fails for any reason.
    if not token:
        token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HF_HUB_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )

    if token:
        return CheckResult("PASS", "Hugging Face token", "HF token detected in environment.")
    return CheckResult(
        "WARN",
        "Hugging Face token",
        "No HF token detected. Public models work, gated models may fail.",
        "Run: huggingface-cli login",
    )


def _check_hf_repo_access(model_name: str, repo_id: str, repo_type: Optional[str] = "model") -> CheckResult:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import GatedRepoError
    except Exception:
        return CheckResult(
            "WARN",
            f"{model_name} gated access",
            "Cannot check gated access because `huggingface_hub` import failed.",
            "Install `huggingface_hub`.",
        )

    try:
        if repo_type == "dataset":
            HfApi().dataset_info(repo_id=repo_id)
        else:
            HfApi().model_info(repo_id=repo_id)
        return CheckResult("PASS", f"{model_name} gated access", f"Access check succeeded for `{repo_id}`.")
    except GatedRepoError:
        return CheckResult(
            "WARN",
            f"{model_name} gated access",
            f"No access to `{repo_id}`.",
            f"Request access to `{repo_id}` on Hugging Face.",
        )
    except Exception as exc:
        return CheckResult(
            "WARN",
            f"{model_name} gated access",
            f"Could not verify access ({type(exc).__name__}).",
            "Check internet connectivity, then rerun `trident-doctor --check-gated`.",
        )


def _check_chief_repo_root(repo_root: Path) -> CheckResult:
    slide_ckpts = repo_root / "trident" / "slide_encoder_models" / "local_ckpts.json"
    if not slide_ckpts.exists():
        return CheckResult(
            "WARN",
            "CHIEF local path",
            "Slide encoder checkpoint config file was not found.",
            "Ensure `trident/slide_encoder_models/local_ckpts.json` exists.",
        )

    try:
        data = json.loads(slide_ckpts.read_text())
    except Exception:
        return CheckResult(
            "WARN",
            "CHIEF local path",
            "Could not parse `local_ckpts.json`.",
            "Fix invalid JSON in `trident/slide_encoder_models/local_ckpts.json`.",
        )

    chief_path = data.get("chief")
    if not chief_path:
        return CheckResult(
            "WARN",
            "CHIEF local path",
            "No CHIEF path configured in `local_ckpts.json`.",
            "Set `chief` to your local CHIEF repo path in `trident/slide_encoder_models/local_ckpts.json`.",
        )

    chief_abs = (repo_root / chief_path).resolve()
    if chief_abs.exists():
        return CheckResult("PASS", "CHIEF local path", f"Found CHIEF repo at `{chief_abs}`.")
    return CheckResult(
        "WARN",
        "CHIEF local path",
        f"Configured CHIEF path does not exist: `{chief_abs}`.",
        "Clone CHIEF and update `trident/slide_encoder_models/local_ckpts.json`.",
    )


def _check_libvips_runtime() -> CheckResult:
    """
    Verify that pyvips can load system libvips.
    """
    try:
        import pyvips
    except Exception:
        return CheckResult(
            "FAIL",
            "pyvips Python package",
            "Module `pyvips` is missing.",
            "Install with: pip install pyvips",
        )

    try:
        # This call fails when libvips is not available at runtime.
        _ = pyvips.version(0)
        return CheckResult("PASS", "libvips runtime", "libvips is available to pyvips.")
    except Exception:
        hint = ctypes.util.find_library("vips")
        extra = "" if hint else " (no `libvips` found on linker path)"
        return CheckResult(
            "FAIL",
            "libvips runtime",
            f"pyvips cannot load libvips{extra}.",
            "Install system package: sudo apt-get update && sudo apt-get install -y libvips libvips-dev",
        )


def _check_openslide_runtime() -> CheckResult:
    """
    Verify that openslide-python can load system OpenSlide library.
    """
    try:
        import openslide
        _ = openslide.__version__
        return CheckResult("PASS", "OpenSlide runtime", "OpenSlide library is available.")
    except Exception:
        return CheckResult(
            "WARN",
            "OpenSlide runtime",
            "OpenSlide runtime is not available.",
            "Install system package: sudo apt-get update && sudo apt-get install -y libopenslide0 libopenslide-dev",
        )


def run_checks(profile: str, check_gated: bool) -> List[CheckResult]:
    repo_root = Path(__file__).resolve().parents[1]
    results: List[CheckResult] = []

    # Always useful checks
    results.append(
        _check_file(
            repo_root / "trident" / "patch_encoder_models" / "local_ckpts.json",
            "Patch checkpoint config",
            "Reinstall TRIDENT or restore the missing file.",
        )
    )
    results.append(
        _check_file(
            repo_root / "trident" / "slide_encoder_models" / "local_ckpts.json",
            "Slide checkpoint config",
            "Reinstall TRIDENT or restore the missing file.",
        )
    )
    results.append(_check_openslide_runtime())

    patch_gated_repos = [
        ("CONCH v1", "MahmoodLab/conch", "model"),
        ("CONCH v1.5", "MahmoodLab/conchv1_5", "model"),
        ("UNI", "MahmoodLab/uni", "model"),
        ("UNI2-h", "MahmoodLab/UNI2-h", "model"),
        ("Virchow", "paige-ai/Virchow", "model"),
        ("Virchow2", "paige-ai/Virchow2", "model"),
        ("H-optimus-0", "bioptimus/H-optimus-0", "model"),
        ("H-optimus-1", "bioptimus/H-optimus-1", "model"),
        ("H0-mini", "bioptimus/H0-mini", "model"),
        ("Phikon", "owkin/phikon", "model"),
        ("Phikon-v2", "owkin/phikon-v2", "model"),
        ("Hibou-L", "histai/hibou-L", "model"),
        ("Prov-GigaPath", "prov-gigapath/prov-gigapath", "model"),
        ("Midnight", "kaiko-ai/midnight", "model"),
        ("OpenMidnight", "SophontAI/OpenMidnight", "model"),
        ("GPFM", "majiabo/GPFM", "model"),
        ("Lunit vits8", "1aurent/vit_small_patch8_224.lunit_dino", "model"),
        ("Kaiko vit-small-8", "1aurent/vit_small_patch8_224.kaiko_ai_towards_large_pathology_fms", "model"),
        ("Kaiko vit-small-16", "1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms", "model"),
        ("Kaiko vit-base-8", "1aurent/vit_base_patch8_224.kaiko_ai_towards_large_pathology_fms", "model"),
        ("Kaiko vit-base-16", "1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms", "model"),
        ("Kaiko vit-large-14", "1aurent/vit_large_patch14_reg4_dinov2.kaiko_ai_towards_large_pathology_fms", "model"),
        ("ResNet50 (timm)", "timm/resnet50.tv_in1k", "model"),
        ("CTransPath weights", "MahmoodLab/hest-bench", "dataset"),
    ]

    slide_gated_repos = [
        ("PRISM", "paige-ai/Prism", "model"),
        ("Titan", "MahmoodLab/TITAN", "model"),
        ("Feather", "MahmoodLab/abmil.base.conch_v15.pc108-24k", "model"),
    ]

    if profile in {"patch-encoders", "full"}:
        results.extend(
            [
                _check_module(
                    "conch",
                    "CONCH dependency",
                    "Install with: pip install git+https://github.com/Mahmoodlab/CONCH.git",
                ),
                _check_module(
                    "musk",
                    "MUSK dependency",
                    "Install with: pip install fairscale git+https://github.com/lilab-stanford/MUSK",
                ),
                _check_module(
                    "timm_ctp",
                    "CTransPath dependency",
                    "Install with: pip install timm_ctp",
                ),
                _check_hf_token(),
            ]
        )
        if check_gated:
            results.extend(_check_hf_repo_access(name, repo, repo_type=repo_type) for name, repo, repo_type in patch_gated_repos)

    if profile in {"slide-encoders", "full"}:
        results.extend(
            [
                _check_module(
                    "environs",
                    "PRISM dependency",
                    "Install with: pip install environs==11.0.0 transformers==4.42.4 sacremoses==0.1.1",
                ),
                _check_module(
                    "gigapath",
                    "GigaPath dependency",
                    "Install with: pip install fairscale git+https://github.com/prov-gigapath/prov-gigapath.git",
                ),
                _check_module(
                    "madeleine",
                    "Madeleine dependency",
                    "Install with: pip install git+https://github.com/mahmoodlab/MADELEINE.git",
                ),
                _check_chief_repo_root(repo_root),
            ]
        )
        if check_gated:
            results.extend(_check_hf_repo_access(name, repo, repo_type=repo_type) for name, repo, repo_type in slide_gated_repos)

    if profile in {"convert", "full"}:
        results.extend(
            [
                _check_module(
                    "aicsimageio",
                    "AICSImageIO dependency",
                    "Install with: pip install aicsimageio",
                ),
                _check_libvips_runtime(),
            ]
        )
        # Optional: only required for CZI inputs.
        if not _has_module("pylibCZIrw"):
            results.append(
                CheckResult(
                    "WARN",
                    "CZI optional dependency",
                    "Module `pylibCZIrw` is not installed (only needed for CZI conversion).",
                    "Install with: pip install pylibCZIrw",
                )
            )

    return results


def _status_order(status: str) -> int:
    return {"FAIL": 0, "WARN": 1, "PASS": 2}.get(status, 3)


def _summarize(results: List[CheckResult]) -> dict:
    failed = sum(1 for x in results if x.status == "FAIL")
    warned = sum(1 for x in results if x.status == "WARN")
    passed = sum(1 for x in results if x.status == "PASS")
    return {
        "total": len(results),
        "failed": failed,
        "warnings": warned,
        "passed": passed,
    }


def _print_text_results(results: List[CheckResult], profile: str) -> int:
    summary = _summarize(results)
    sorted_results = sorted(results, key=lambda x: (_status_order(x.status), x.name.lower()))
    grouped = {"FAIL": [], "WARN": [], "PASS": []}
    for res in sorted_results:
        grouped.setdefault(res.status, []).append(res)

    print("=" * 72)
    print("TRIDENT DOCTOR REPORT")
    print("=" * 72)
    print(f"Profile: {profile}")
    print(
        f"Checks: {summary['total']} | "
        f"PASS: {summary['passed']} | "
        f"WARN: {summary['warnings']} | "
        f"FAIL: {summary['failed']}"
    )
    print("")

    for section in ["FAIL", "WARN", "PASS"]:
        entries = grouped.get(section, [])
        if not entries:
            continue
        print(f"[{section}]")
        for idx, res in enumerate(entries, start=1):
            print(f"  {idx}. {res.name}")
            print(f"     - {res.message}")
            if res.fix:
                print(f"     - Fix: {res.fix}")
        print("")

    actionable = [x.fix for x in sorted_results if x.status in {"FAIL", "WARN"} and x.fix]
    # Keep order stable while removing duplicates.
    actionable = list(dict.fromkeys(actionable))
    if actionable:
        print("Suggested next steps:")
        for idx, fix in enumerate(actionable, start=1):
            print(f"  {idx}. {fix}")
        print("")

    return 1 if summary["failed"] else 0


def _print_json_results(results: List[CheckResult], profile: str) -> int:
    summary = _summarize(results)
    payload = {
        "profile": profile,
        "summary": summary,
        "checks": [
            {
                "status": x.status,
                "name": x.name,
                "message": x.message,
                "fix": x.fix,
            }
            for x in sorted(results, key=lambda y: (_status_order(y.status), y.name.lower()))
        ],
    }
    print(json.dumps(payload, indent=2))
    return 1 if summary["failed"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trident-doctor",
        description="Preflight checks for TRIDENT optional dependencies and configuration.",
    )
    parser.add_argument(
        "--profile",
        choices=["base", "patch-encoders", "slide-encoders", "convert", "full"],
        default="base",
        help="Dependency profile to validate.",
    )
    parser.add_argument(
        "--check-gated",
        action="store_true",
        help="Attempt to verify access to gated Hugging Face models (network required).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    args = parser.parse_args()

    results = run_checks(profile=args.profile, check_gated=args.check_gated)
    if args.format == "json":
        raise SystemExit(_print_json_results(results, profile=args.profile))
    raise SystemExit(_print_text_results(results, profile=args.profile))


if __name__ == "__main__":
    main()
