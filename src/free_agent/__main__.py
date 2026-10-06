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
            "Profile + tools + skills come from the active workspace under "
            "~/.config/free-agent/workspaces/. Switch with /ws use <name>."
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
        help=(
            "Path to an agent profile YAML — overrides whatever the active "
            "workspace ships."
        ),
    )
    p.add_argument(
        "--workspace",
        "--ws",
        metavar="NAME",
        dest="workspace",
        help=(
            "Use this workspace for the session (default: the persisted "
            "active workspace, or auto-created `default` on first run)."
        ),
    )
    p.add_argument(
        "-m",
        "--mode",
        metavar="NAME",
        help=(
            "Start in a preset mode: code, science, security, finance (or off). "
            "Overrides the persisted mode for this run."
        ),
    )
    p.add_argument(
        "--provider",
        choices=["fireworks", "ollama", "anthropic"],
        help="Model provider for this run (default: fireworks, or the persisted choice).",
    )
    p.add_argument(
        "--model",
        metavar="NAME",
        help="Model id for this run, e.g. `glm-5p3` (Fireworks) or `qwen3:8b` (Ollama).",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"free-agent {__version__}",
    )
    return p


def _load_settings(overrides: dict[str, str]) -> Settings:
    """Build Settings; on a first run without a Fireworks key, offer to paste one.

    The key is saved to ~/.config/free-agent/secrets.json (mode 0600) so the
    prompt only ever appears once.
    """
    try:
        return Settings(**overrides)
    except Exception as exc:
        if "FIREWORKS_API_KEY" not in str(exc) or not sys.stdin.isatty():
            raise
    import getpass

    from free_agent.config import save_secret_api_key

    sys.stderr.write(
        "free-agent uses Fireworks AI as its standard model provider.\n"
        "Paste your API key (https://fireworks.ai/account/api-keys), or press\n"
        "Enter to run locally with Ollama instead.\n"
    )
    key = getpass.getpass("FIREWORKS_API_KEY: ").strip()
    if not key:
        return Settings(**{**overrides, "provider": "ollama"})
    path = save_secret_api_key(key, "fireworks_api_key")
    sys.stderr.write(f"saved to {path}\n")
    return Settings(**overrides)


def main() -> int:
    args = _build_parser().parse_args()

    load_dotenv()
    overrides: dict[str, str] = {}
    if args.provider:
        overrides["provider"] = args.provider
    try:
        settings = _load_settings(overrides)
    except Exception as exc:
        sys.stderr.write(f"config error: {exc}\n")
        return 2

    if args.model:
        from free_agent.config import normalize_fireworks_model

        if settings.provider == "fireworks":
            settings.fireworks_model = normalize_fireworks_model(args.model)
        elif settings.provider == "ollama":
            settings.ollama_model = args.model
        else:
            settings.anthropic_model = args.model

    if args.mode is not None:
        from free_agent.modes import get_mode

        try:
            mode = get_mode(args.mode)
        except ValueError as exc:
            sys.stderr.write(f"config error: {exc}\n")
            return 2
        settings.mode = mode.name if mode else ""

    if args.writable:
        settings.writable = True

    config_override = Path(args.config).expanduser() if args.config else None

    logging.basicConfig(
        level=settings.log_level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        return asyncio.run(
            run(
                settings,
                config_override=config_override,
                workspace_override=args.workspace,
            )
        )
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
