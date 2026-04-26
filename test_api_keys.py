"""
Lightweight API key checks for the Deep Trading System V2.

Loads keys from .env, then validates:
    - OPENAI_API_KEY
    - FINNHUB_API_KEY
    - TAVILY_API_KEY
    - LANGSMITH_API_KEY

The script never prints secret values.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _require_env(var_name: str) -> str:
    value = os.environ.get(var_name, "").strip()
    if not value:
        raise RuntimeError(f"{var_name} is missing or blank")
    return value


def _run_check(name: str, check: Callable[[], str]) -> CheckResult:
    try:
        detail = check()
    except Exception as exc:  # noqa: BLE001 - this is a CLI diagnostics script.
        return CheckResult(name=name, ok=False, detail=f"{type(exc).__name__}: {exc}")
    return CheckResult(name=name, ok=True, detail=detail)


def check_openai() -> str:
    from openai import OpenAI

    api_key = _require_env("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    models = client.models.list()
    model_count = len(getattr(models, "data", []) or [])
    return (
        f"Authenticated successfully; models endpoint returned "
        f"{model_count} model(s)."
    )


def check_finnhub() -> str:
    import finnhub

    api_key = _require_env("FINNHUB_API_KEY")
    client = finnhub.Client(api_key=api_key)
    quote = client.quote("AAPL")
    current_price = quote.get("c")
    if current_price is None:
        raise RuntimeError(f"Unexpected quote response: {quote}")
    return (
        f"Authenticated successfully; AAPL current price returned as "
        f"{current_price}."
    )


def check_tavily() -> str:
    from tavily import TavilyClient

    api_key = _require_env("TAVILY_API_KEY")
    client = TavilyClient(api_key=api_key)
    response = client.search(
        query="OpenAI API key test",
        max_results=1,
        search_depth="basic",
        include_answer=False,
    )
    results = response.get("results", [])
    return f"Authenticated successfully; search returned {len(results)} result(s)."


def check_langsmith() -> str:
    from langsmith import Client

    api_key = _require_env("LANGSMITH_API_KEY")
    api_url = (
        os.environ.get("LANGSMITH_ENDPOINT")
        or os.environ.get("LANGCHAIN_ENDPOINT")
        or "https://api.smith.langchain.com"
    )
    client = Client(api_key=api_key, api_url=api_url)
    first_project = next(iter(client.list_projects()), None)
    if first_project is None:
        return "Authenticated successfully; no LangSmith projects found yet."
    project_name = getattr(first_project, "name", "unnamed project")
    return f"Authenticated successfully; found LangSmith project: {project_name}."


def main() -> int:
    load_dotenv()

    checks = [
        ("OpenAI", check_openai),
        ("Finnhub", check_finnhub),
        ("Tavily", check_tavily),
        ("LangSmith", check_langsmith),
    ]

    print("Testing API keys from .env\n")
    results = [_run_check(name, check) for name, check in checks]

    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")

    failed = [result.name for result in results if not result.ok]
    if failed:
        print(f"\nFailed check(s): {', '.join(failed)}")
        return 1

    print("\nAll API key checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
