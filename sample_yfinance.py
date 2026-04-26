"""
Sample: fetch the most recent 250 trading days of OHLCV data for a ticker.

Run:
    python sample_yfinance.py            # default ticker NVDA
    python sample_yfinance.py AAPL       # custom ticker
"""

import os
import sys

import pandas as pd
from dotenv import load_dotenv

from tools import get_recent_yfinance_prices


# Defaults from the user's spec for this V2 system.
TICKER = "NVDA"
DEFAULT_RECENT_TRADING_DAYS = 250


def main(symbol: str = TICKER, num_days: int = DEFAULT_RECENT_TRADING_DAYS) -> None:
    # 1. Load API keys from .env (OPENAI_API_KEY, FINNHUB_API_KEY,
    #    TAVILY_API_KEY, LANGSMITH_API_KEY). yfinance itself needs no key,
    #    but this is how the rest of the toolkit gets credentials.
    load_dotenv()
    for key in (
        "OPENAI_API_KEY",
        "FINNHUB_API_KEY",
        "TAVILY_API_KEY",
        "LANGSMITH_API_KEY",
    ):
        status = "OK" if os.getenv(key) else "MISSING"
        print(f"  {key}: {status}")

    # 2. The helper builds a date window wide enough to cover `num_days`
    #    trading days even when there are weekends/holidays in between.
    try:
        last_n, start_str, end_str = get_recent_yfinance_prices(
            symbol=symbol,
            num_days=num_days,
        )
    except RuntimeError as exc:
        print(exc)
        return

    latest_date = pd.to_datetime(
        last_n.index,
        errors="coerce",
        utc=True,
    ).dropna().max().date().isoformat()

    print(f"\nFetching {symbol} from {start_str} to {end_str} ...")
    print(f"Latest available trading date: {latest_date}")
    print(f"\nMost recent {num_days} trading days for {symbol}:\n")
    print(last_n.to_string())


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else TICKER
    main(ticker)
