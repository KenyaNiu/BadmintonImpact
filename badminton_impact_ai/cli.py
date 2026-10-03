"""Single command-line entry point for the complete research workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="badminton-impact")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Build canonical labels and pose descriptors")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--overwrite", action="store_true")

    run = commands.add_parser("run", help="Validate or run a frozen LOSO experiment")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--check-only", action="store_true", help="Validate cohort and splits without training")

    analyze = commands.add_parser("analyze", help="Analyze one completed run")
    analyze.add_argument("--run-dir", type=Path, required=True)

    artifacts = commands.add_parser("artifacts", help="Generate paper-facing artifacts from one analyzed run")
    artifacts.add_argument("--run-dir", type=Path, required=True)
    artifacts.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        from .data.preparation import prepare_dataset

        print(prepare_dataset(args.config, overwrite=args.overwrite))
    elif args.command == "run":
        from .experiment.config import load_experiment_config
        from .experiment.runner import run_experiment

        run_experiment(
            load_experiment_config(args.config),
            args.run_dir,
            resume=args.resume,
            check_only=args.check_only,
        )
    elif args.command == "analyze":
        from .experiment.analysis import analyze_run

        print(analyze_run(args.run_dir))
    else:
        from .experiment.artifacts import make_paper_artifacts

        print(make_paper_artifacts(args.run_dir, args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
