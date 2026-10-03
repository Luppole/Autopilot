"""The `free-router` command."""

import argparse
import logging
import sys
from collections.abc import Sequence

from free_router import __version__
from free_router.config import APP_TITLE, settings
from free_router.errors import RouterError
from free_router.sync import SyncResult

LOCAL_HOSTS = {"127.0.0.1", "localhost"}


def print_sync_report(result: SyncResult) -> None:
    print(f"Free models: {len(result.models)}")
    print(f"Added:       {len(result.added)}")
    print(f"Removed:     {len(result.removed)}")

    for model in result.added:
        print(f"  + {model['id']}")

    for model in result.removed:
        print(f"  - {model['id']}")

    print(f"Saved to {settings.models_file}")


def run_sync() -> None:
    from free_router.sync import sync_free_models

    print("Fetching OpenRouter models...")
    print_sync_report(sync_free_models())


def run_query(prompt: str, sync_first: bool) -> None:
    from free_router.query import ask

    if sync_first:
        run_sync()
        print()

    answer = ask(prompt)

    skipped = f" (skipped {answer.skipped})" if answer.skipped else ""
    print(f"Answered by {answer.model}{skipped}")
    print()
    print(answer.text)


def run_serve(host: str, port: int, reload: bool) -> None:
    import uvicorn

    if host not in LOCAL_HOSTS and host not in settings.allowed_hosts:
        logging.getLogger(__name__).warning(
            "Serving on %s, but only these Host headers are accepted: %s. "
            "Set FREE_ROUTER_ALLOWED_HOSTS to the addresses clients will use.",
            host,
            ", ".join(settings.allowed_hosts),
        )

    uvicorn.run("free_router.api.app:app", host=host, port=port, reload=reload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="free-router", description=APP_TITLE)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="log every attempt"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("sync", help="refresh the list of free models")

    query = commands.add_parser("query", help="ask whichever free model answers first")
    query.add_argument("prompt")
    query.add_argument(
        "--no-sync", action="store_true", help="skip refreshing the model list first"
    )

    serve = commands.add_parser("serve", help="run the router and dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="restart on code changes")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO
        if args.verbose or args.command == "serve"
        else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.command == "sync":
            run_sync()
        elif args.command == "query":
            run_query(args.prompt, sync_first=not args.no_sync)
        elif args.command == "serve":
            run_serve(args.host, args.port, args.reload)
    except RouterError as error:
        print(f"error: {error.message}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
