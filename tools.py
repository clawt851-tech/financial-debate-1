"""
Toolkit the agents use to reach the outside world.

Every function is wrapped with @tool + Annotated type hints so the LLM
can read its docstring/signature and decide when to call it.

Tools:
    - get_yfinance_data          (raw OHLCV from Yahoo Finance)
    - get_recent_yfinance_data   (latest N OHLCV rows from Yahoo Finance)
    - get_technical_indicators   (MACD / RSI / BBands / SMAs via stockstats)
    - get_finnhub_news           (company-specific headlines)
    - get_social_media_sentiment (Tavily web search)
    - get_fundamental_analysis   (Tavily web search)
    - get_macroeconomic_news     (Tavily web search)
"""

import os
import datetime
import io
from functools import lru_cache
from typing import Annotated

import pandas as pd
import yfinance as yf
import finnhub
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from stockstats import wrap as stockstats_wrap

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


DEFAULT_RECENT_TRADING_DAYS = 250
MIN_TECHNICAL_TRADING_DAYS = 250


def _load_env_file() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _inclusive_yfinance_end_date(end_date: str) -> str:
    """Convert a user-facing inclusive end date to yfinance's exclusive end."""
    try:
        parsed = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return end_date
    return (parsed + datetime.timedelta(days=1)).isoformat()


def _calendar_lookback_for_trading_days(trading_days: int) -> int:
    """Estimate a calendar window wide enough to contain N trading sessions."""
    if trading_days <= 0:
        raise ValueError("num_days must be a positive integer")
    return max(14, int(trading_days * 365 / 252) + 30)


def _expanded_indicator_start_date(
    start_date: str,
    end_date: str,
    min_trading_days: int = MIN_TECHNICAL_TRADING_DAYS,
) -> str:
    """Move short indicator windows back far enough for long SMAs."""
    try:
        parsed_end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return start_date

    warm_start = parsed_end - datetime.timedelta(
        days=_calendar_lookback_for_trading_days(min_trading_days)
    )
    try:
        parsed_start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    except ValueError:
        return warm_start.isoformat()

    return min(parsed_start, warm_start).isoformat()


# ---------------------------------------------------------------------------
# Price + technicals
# ---------------------------------------------------------------------------
@tool
def get_yfinance_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "start date, format yyyy-mm-dd"],
    end_date: Annotated[str, "end date, format yyyy-mm-dd"],
) -> str:
    """Fetch historical price data for a ticker from Yahoo Finance.

    The public tool treats `end_date` as inclusive, even though yfinance's
    `end` argument is exclusive.
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        data = ticker.history(
            start=start_date,
            end=_inclusive_yfinance_end_date(end_date),
        )
        if data.empty:
            return (
                f"No data found for symbol '{symbol}' between "
                f"{start_date} and {end_date}"
            )
        return data.to_csv()
    except Exception as e:
        return f"Error fetching Yahoo Finance data: {e}"


@tool
def get_recent_yfinance_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    num_days: Annotated[
        int, "number of recent trading days to return"
    ] = DEFAULT_RECENT_TRADING_DAYS,
) -> str:
    """Fetch the latest N trading days of OHLCV data from Yahoo Finance."""
    try:
        recent_prices, _, _ = get_recent_yfinance_prices(
            symbol=symbol,
            num_days=num_days,
        )
        return recent_prices.to_csv()
    except Exception as e:
        return f"Error fetching recent Yahoo Finance data: {e}"


def get_recent_yfinance_prices(
    symbol: str,
    num_days: int = DEFAULT_RECENT_TRADING_DAYS,
    lookback_days: int | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Fetch recent yfinance rows using a calendar window wide enough for gaps."""
    if lookback_days is None:
        lookback_days = _calendar_lookback_for_trading_days(num_days)
    else:
        lookback_days = max(lookback_days, _calendar_lookback_for_trading_days(num_days))

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=lookback_days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    csv_text = get_yfinance_data.invoke(
        {
            "symbol": symbol,
            "start_date": start_str,
            "end_date": end_str,
        }
    )

    if csv_text.startswith("Error") or csv_text.startswith("No data"):
        raise RuntimeError(csv_text)

    df = pd.read_csv(io.StringIO(csv_text), index_col=0, parse_dates=True)
    if df.empty:
        raise RuntimeError(
            f"No yfinance rows found for {symbol} between {start_str} and {end_str}"
        )

    return df.tail(num_days), start_str, end_str


def get_latest_available_trade_date(symbol: str, lookback_days: int = 14) -> str:
    """Resolve the latest available trading date from recent yfinance data."""
    recent_prices, _, _ = get_recent_yfinance_prices(
        symbol=symbol,
        num_days=5,
        lookback_days=lookback_days,
    )
    dates = pd.to_datetime(recent_prices.index, errors="coerce", utc=True).dropna()
    if dates.empty:
        raise RuntimeError(f"Could not parse yfinance trading dates for {symbol}")
    return dates.max().date().isoformat()


def _single_ticker_download_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Reduce yfinance's optional MultiIndex columns to one OHLCV table."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    upper_symbol = symbol.upper()
    for level in range(df.columns.nlevels):
        matches = [
            value
            for value in df.columns.get_level_values(level).unique()
            if str(value).upper() == upper_symbol
        ]
        if matches:
            return df.xs(matches[0], axis=1, level=level, drop_level=True)

    if df.columns.nlevels == 2:
        last_level = df.columns.get_level_values(-1)
        if len(last_level.unique()) == 1:
            return df.droplevel(-1, axis=1)

    flattened = df.copy()
    flattened.columns = [
        "_".join(str(part) for part in column if str(part))
        for column in flattened.columns
    ]
    return flattened


@tool
def get_technical_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "start date, format yyyy-mm-dd"],
    end_date: Annotated[str, "end date, format yyyy-mm-dd"],
) -> str:
    """Compute key technical indicators for a stock using stockstats.

    The public tool treats `end_date` as inclusive, even though yfinance's
    `end` argument is exclusive.
    """
    try:
        indicator_start_date = _expanded_indicator_start_date(start_date, end_date)
        df = yf.download(
            symbol,
            start=indicator_start_date,
            end=_inclusive_yfinance_end_date(end_date),
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            return "No data available to compute indicators."
        df = _single_ticker_download_frame(df, symbol)
        stock_df = stockstats_wrap(df)
        indicators = stock_df[
            ["macd", "rsi_14", "boll", "boll_ub", "boll_lb", "close_50_sma", "close_200_sma"]
        ]
        # tail() keeps the LLM context tight
        return indicators.tail().to_csv()
    except Exception as e:
        return f"Error computing stockstats indicators: {e}"


# ---------------------------------------------------------------------------
# Company news (Finnhub)
# ---------------------------------------------------------------------------
@tool
def get_finnhub_news(
    ticker: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "start date, format yyyy-mm-dd"],
    end_date: Annotated[str, "end date, format yyyy-mm-dd"],
) -> str:
    """Fetch company news from Finnhub for the given date range."""
    try:
        _load_env_file()
        finnhub_client = finnhub.Client(api_key=os.environ["FINNHUB_API_KEY"])
        news_list = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
        news_items = []
        for news in news_list[:5]:
            news_items.append(f"Headline: {news['headline']}\nSummary: {news['summary']}")
        return "\n\n".join(news_items) if news_items else "No Finnhub news found."
    except Exception as e:
        return f"Error fetching Finnhub news: {e}"


# ---------------------------------------------------------------------------
# Web search via Tavily (one shared engine, three specialized tools)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_tavily_tool() -> TavilySearchResults:
    _load_env_file()
    return TavilySearchResults(max_results=3)


@tool
def get_social_media_sentiment(
    ticker: Annotated[str, "ticker symbol of the company"],
    trade_date: Annotated[str, "trade date, format yyyy-mm-dd"],
) -> str:
    """Live web search for social-media sentiment on a stock."""
    query = (
        f"social media sentiment and discussion about {ticker} stock "
        f"around {trade_date}"
    )
    return _get_tavily_tool().invoke({"query": query})


@tool
def get_fundamental_analysis(
    ticker: Annotated[str, "ticker symbol of the company"],
    trade_date: Annotated[str, "trade date, format yyyy-mm-dd"],
) -> str:
    """Live web search for recent fundamental analysis on a stock."""
    query = (
        f"fundamental analysis and key financial metrics for {ticker} stock "
        f"published around {trade_date}"
    )
    return _get_tavily_tool().invoke({"query": query})


@tool
def get_macroeconomic_news(
    trade_date: Annotated[str, "trade date, format yyyy-mm-dd"],
) -> str:
    """Live web search for macroeconomic news affecting the stock market."""
    query = (
        f"macroeconomic news and market trends affecting the stock market "
        f"on {trade_date}"
    )
    return _get_tavily_tool().invoke({"query": query})


# ---------------------------------------------------------------------------
# Toolkit aggregator
# ---------------------------------------------------------------------------
class Toolkit:
    """Bundles every tool behind one tidy object."""

    def __init__(self, config):
        self.config = config
        self.get_yfinance_data = get_yfinance_data
        self.get_recent_yfinance_data = get_recent_yfinance_data
        self.get_technical_indicators = get_technical_indicators
        self.get_finnhub_news = get_finnhub_news
        self.get_social_media_sentiment = get_social_media_sentiment
        self.get_fundamental_analysis = get_fundamental_analysis
        self.get_macroeconomic_news = get_macroeconomic_news

    def all_tools(self):
        """Return every tool callable on this toolkit (useful for ToolNode)."""
        return [
            getattr(self, name)
            for name in dir(self)
            if callable(getattr(self, name))
            and not name.startswith("_")
            and name != "all_tools"
        ]
