from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from free_agent import __version__
from free_agent.cli.app import run
from free_agent.config import Settings


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="free-agent",
        description=(
            "Terminal chat with a local agent built on LangChain deepagents. "
            "Reads .env, free-agent.yaml, and ./free_agent_tools/*.py from the cwd."
        ),
    )
    p.add_argument(
        "-w",
        "--writable",
        action="store_true",
        help=(
            "Allow the agent to read/write files in the current directory "
            "(scoped — paths cannot escape via .. / ~ / absolute outside cwd). "
            "Overrides FREE_AGENT_WRITABLE for this run."
        ),
    )
    p.add_argument(
        "-c",
        "--config",
        metavar="PATH",
        help="Path to an agent profile YAML (defaults to ./free-agent.yaml if present).",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"free-agent {__version__}",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    load_dotenv()
    try:
        settings = Settings()
    except Exception as exc:
        sys.stderr.write(f"config error: {exc}\n")
        return 2

    if args.writable:
        settings.writable = True

    config_override = Path(args.config).expanduser() if args.config else None

    logging.basicConfig(
        level=settings.log_level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        return asyncio.run(run(settings, config_override=config_override))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
