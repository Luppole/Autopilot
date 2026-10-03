"""Ask a single prompt from Python code or the CLI."""

from dataclasses import dataclass

from free_router.routing import route


@dataclass(frozen=True)
class Answer:
    model: str
    text: str
    skipped: int


def ask(prompt: str) -> Answer:
    result = route({"messages": [{"role": "user", "content": prompt}]})
    assert result.data is not None

    message = result.data["choices"][0].get("message") or {}

    return Answer(
        model=result.model,
        text=message.get("content") or "",
        skipped=len(result.attempts),
    )
