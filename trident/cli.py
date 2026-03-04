from __future__ import annotations

import argparse
import sys

from trident import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trident",
        description="TRIDENT command-line interface.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"trident {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    batch = subparsers.add_parser(
        "batch",
        help="Run batch slide processing (wrapper around run_batch_of_slides).",
    )
    batch.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to run_batch_of_slides.")

    single = subparsers.add_parser(
        "single",
        help="Run single slide processing (wrapper around run_single_slide).",
    )
    single.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to run_single_slide.")

    doctor = subparsers.add_parser(
        "doctor",
        help="Run environment diagnostics (wrapper around trident-doctor).",
    )
    doctor.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to trident-doctor.")

    convert = subparsers.add_parser(
        "convert",
        help="Convert images/WSIs to pyramidal TIFF with AnyToTiffConverter.",
    )
    convert.add_argument("--input_dir", type=str, required=True, help="Directory containing files to convert.")
    convert.add_argument("--mpp_csv", type=str, required=True, help="Required CSV with columns: wsi,mpp.")
    convert.add_argument("--job_dir", type=str, required=True, help="Output directory for converted TIFF files.")
    convert.add_argument("--downscale_by", type=int, default=1, help="Downscale factor (>=1).")
    convert.add_argument("--num_workers", type=int, default=1, help="Workers: 1=sequential, 0=all CPUs.")
    convert.add_argument("--bigtiff", action="store_true", default=False, help="Enable BigTIFF output.")

    parsed = parser.parse_args()

    # Support `trident <cmd> -- ...` by dropping the separator before forwarding.
    forwarded = []
    if hasattr(parsed, "args"):
        forwarded = parsed.args[1:] if parsed.args and parsed.args[0] == "--" else parsed.args

    if parsed.command == "batch":
        from run_batch_of_slides import main as batch_main

        sys.argv = ["run_batch_of_slides", *forwarded]
        batch_main()
        return

    if parsed.command == "single":
        from run_single_slide import main as single_main

        sys.argv = ["run_single_slide", *forwarded]
        single_main()
        return

    if parsed.command == "doctor":
        from trident.cli_doctor import main as doctor_main

        sys.argv = ["trident-doctor", *forwarded]
        doctor_main()
        return

    if parsed.command == "convert":
        from trident.Converter import AnyToTiffConverter

        converter = AnyToTiffConverter(job_dir=parsed.job_dir, bigtiff=parsed.bigtiff)
        converter.process_all(
            input_dir=parsed.input_dir,
            mpp_csv=parsed.mpp_csv,
            downscale_by=parsed.downscale_by,
            num_workers=parsed.num_workers,
        )
        return

    parser.error(f"Unknown command: {parsed.command}")


if __name__ == "__main__":
    main()
