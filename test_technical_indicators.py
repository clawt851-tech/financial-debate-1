
"""
Smoke test for get_technical_indicators().

Run:
    python test_technical_indicators.py
    python test_technical_indicators.py AAPL 2025-01-01 2025-12-31
"""

from __future__ import annotations

import argparse
import io
import sys


DEFAULT_SYMBOL = "NVDA"
DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = "2026-04-28"
EXPECTED_COLUMNS = {
    "macd",
    "rsi_14",
    "boll",
    "boll_ub",
    "boll_lb",
    "close_50_sma",
    "close_200_sma",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and validate stockstats technical indicators."
    )
    parser.add_argument("symbol", nargs="?", default=DEFAULT_SYMBOL)
    parser.add_argument("start_date", nargs="?", default=DEFAULT_START_DATE)
    parser.add_argument("end_date", nargs="?", default=DEFAULT_END_DATE)
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def load_dependencies():
    try:
        import pandas as pd
        from tools import get_technical_indicators
    except ModuleNotFoundError as exc:
        print(
            "Missing Python package: "
            f"{exc.name}\n\n"
            "Install this project's dependencies into the interpreter you are "
            "using, or run the test with the repo's configured Python.\n\n"
            f"Current interpreter:\n  {sys.executable}\n\n"
            "Install command:\n"
            f"  \"{sys.executable}\" -m pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    return pd, get_technical_indicators


def main() -> int:
    pd, get_technical_indicators = load_dependencies()
    args = parse_args()
    result = get_technical_indicators.invoke(
        {
            "symbol": args.symbol,
            "start_date": args.start_date,
            "end_date": args.end_date,
        }
    )

    if result.startswith("Error computing stockstats indicators:"):
        return fail(result)
    if result == "No data available to compute indicators.":
        return fail(result)

    try:
        indicators = pd.read_csv(io.StringIO(result), index_col=0)
    except Exception as exc:
        return fail(f"Could not parse indicator CSV: {exc}\n\nRaw output:\n{result}")

    missing_columns = EXPECTED_COLUMNS.difference(indicators.columns)
    if missing_columns:
        return fail(
            "Missing expected indicator columns: "
            + ", ".join(sorted(missing_columns))
            + f"\n\nRaw output:\n{result}"
        )

    if indicators.empty:
        return fail("Indicator CSV parsed successfully but contained no rows.")

    parsed_dates = pd.to_datetime(indicators.index, errors="coerce")
    invalid_rows = indicators.index[parsed_dates.isna()].tolist()
    if invalid_rows:
        return fail(
            "Indicator output contains non-date rows: "
            + ", ".join(str(row) for row in invalid_rows)
            + f"\n\nRaw output:\n{result}"
        )

    numeric_indicators = indicators[list(EXPECTED_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if numeric_indicators.dropna(how="all").empty:
        return fail(f"Indicator columns did not contain numeric values.\n\n{result}")

    print(
        f"PASS: fetched {len(indicators)} indicator rows for "
        f"{args.symbol.upper()} from {args.start_date} to {args.end_date}."
    )
    print()
    print(indicators.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
