"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys

from .pipeline import load_pipeline, source_from_arg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ensemble-voice")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the pipeline on an audio file")
    run.add_argument("audio", help="path to audio file (wav/flac/mp3)")
    run.add_argument("--config", "-c", required=True, help="path to YAML config")
    run.add_argument("--language", "-l", default=None, help="language hint (e.g. en, ja)")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        pipeline = load_pipeline(args.config)
        source = source_from_arg(args.audio)
        pipeline.run(source, language_hint=args.language)
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
