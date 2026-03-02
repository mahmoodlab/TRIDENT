from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trident",
        description="TRIDENT command-line interface.",
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

    parsed = parser.parse_args()

    # Support `trident <cmd> -- ...` by dropping the separator before forwarding.
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

    parser.error(f"Unknown command: {parsed.command}")


if __name__ == "__main__":
    main()
