"""Resolve company names or ticker-like input to a Yahoo Finance symbol."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import yfinance as yf


_DIRECT_TICKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,14}$")
_US_EXCHANGES = {"NMS", "NGM", "NCM", "NYQ", "ASE", "PCX", "BTS", "PNK", "NQB"}
_PREFERRED_QUOTE_TYPES = {"EQUITY", "ETF"}


@dataclass(frozen=True)
class TickerMatch:
    """A candidate Yahoo Finance quote match."""

    symbol: str
    name: str
    exchange: str
    quote_type: str
    score: float
    source: str

    @property
    def display_name(self) -> str:
        parts = [self.symbol]
        if self.name and self.name.upper() != self.symbol.upper():
            parts.append(self.name)
        if self.exchange:
            parts.append(self.exchange)
        return " - ".join(parts)


class TickerResolutionError(ValueError):
    """Raised when a company name or ticker cannot be resolved."""


def _clean_query(query: str) -> str:
    cleaned = (query or "").strip().strip("$")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        raise TickerResolutionError("Enter a company name or ticker.")
    return cleaned


@lru_cache(maxsize=512)
def _has_recent_prices(symbol: str) -> bool:
    try:
        history = yf.Ticker(symbol).history(period="5d")
        return not history.empty
    except Exception:
        return False


def _direct_symbol_variants(query: str) -> list[str]:
    if not _DIRECT_TICKER_RE.match(query):
        return []

    symbol = query.upper()
    variants = [symbol]
    if "." in symbol:
        variants.append(symbol.replace(".", "-"))
    return list(dict.fromkeys(variants))


def _quote_to_match(quote: dict, source: str) -> TickerMatch | None:
    symbol = str(quote.get("symbol") or "").upper().strip()
    if not symbol:
        return None

    quote_type = str(quote.get("quoteType") or quote.get("typeDisp") or "").upper()
    name = str(
        quote.get("longname")
        or quote.get("shortname")
        or quote.get("name")
        or symbol
    ).strip()
    exchange = str(quote.get("exchange") or quote.get("exchDisp") or "").strip()
    try:
        score = float(quote.get("score") or 0)
    except (TypeError, ValueError):
        score = 0.0

    return TickerMatch(
        symbol=symbol,
        name=name,
        exchange=exchange,
        quote_type=quote_type,
        score=score,
        source=source,
    )


def _sort_key(match: TickerMatch) -> tuple[int, int, float, int]:
    is_preferred_type = match.quote_type in _PREFERRED_QUOTE_TYPES
    is_us_exchange = match.exchange in _US_EXCHANGES
    return (
        1 if is_preferred_type else 0,
        1 if is_us_exchange else 0,
        match.score,
        -len(match.symbol),
    )


@lru_cache(maxsize=256)
def search_tickers(query: str, max_results: int = 8) -> tuple[TickerMatch, ...]:
    """Return ranked Yahoo Finance quote matches for a name or symbol."""
    cleaned = _clean_query(query)
    try:
        search = yf.Search(
            cleaned,
            max_results=max_results,
            news_count=0,
            lists_count=0,
            include_cb=False,
            include_research=False,
            enable_fuzzy_query=True,
            raise_errors=True,
        )
    except Exception as exc:
        raise TickerResolutionError(f"Ticker lookup failed: {exc}") from exc

    matches: list[TickerMatch] = []
    seen_symbols: set[str] = set()
    for quote in search.quotes or []:
        match = _quote_to_match(quote, source="Yahoo Finance search")
        if match is None or match.symbol in seen_symbols:
            continue
        seen_symbols.add(match.symbol)
        matches.append(match)

    return tuple(sorted(matches, key=_sort_key, reverse=True))


@lru_cache(maxsize=256)
def resolve_ticker(query: str) -> TickerMatch:
    """Resolve user input to the best available Yahoo Finance ticker."""
    cleaned = _clean_query(query)

    for symbol in _direct_symbol_variants(cleaned):
        if _has_recent_prices(symbol):
            return TickerMatch(
                symbol=symbol,
                name=symbol,
                exchange="",
                quote_type="EQUITY",
                score=0.0,
                source="Direct ticker",
            )

    matches = search_tickers(cleaned)
    if not matches:
        raise TickerResolutionError(f"No ticker found for '{cleaned}'.")

    for match in matches:
        if match.quote_type in _PREFERRED_QUOTE_TYPES and _has_recent_prices(
            match.symbol
        ):
            return match

    for match in matches:
        if _has_recent_prices(match.symbol):
            return match

    return matches[0]
