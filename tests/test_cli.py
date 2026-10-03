import pytest
from conftest import FakeOpenRouter, FakeResponse, completion

from free_router.cli import main


def test_query_prints_answer(
    openrouter: FakeOpenRouter, capsys: pytest.CaptureFixture[str]
) -> None:
    openrouter.reply(
        FakeResponse(429, {}), FakeResponse(200, completion("Hello there"))
    )

    main(["query", "Hi", "--no-sync"])

    out = capsys.readouterr().out
    assert "(skipped 1)" in out
    assert "Hello there" in out


def test_errors_exit_cleanly(
    openrouter: FakeOpenRouter, capsys: pytest.CaptureFixture[str]
) -> None:
    openrouter.reply(FakeResponse(429, {}))

    with pytest.raises(SystemExit) as exit_info:
        main(["query", "Hi", "--no-sync"])

    assert exit_info.value.code == 1
    assert capsys.readouterr().err.startswith("error: All 3 free models failed")
