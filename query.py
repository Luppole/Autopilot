"""Sync, then ask a random free model a sample question.

Same as `python -m free_router query "..."`.
"""

from free_router.cli import main


if __name__ == "__main__":
    main(["query", "Explain what an API is in simple terms."])
