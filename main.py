"""Refresh the list of free OpenRouter models.

Same as `python -m free_router sync`.
"""

from free_router.cli import main


if __name__ == "__main__":
    main(["sync"])
