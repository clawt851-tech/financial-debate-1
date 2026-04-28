"""Streamlit UI for the Deep Trading System.

Run locally:
    streamlit run app.py

Run as a server:
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from config import REQUIRED_KEYS
from streamlit_manager import (
    DISCLAIMERS,
    REPORT_TITLES,
    DeepTradingResult,
    StreamlitTradingManager,
)
from streamlit_printer import StreamlitProgressPrinter

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


APP_DIR = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(APP_DIR / ".env")


LANGUAGE_OPTIONS = {
    "English": "english",
    "中文": "chinese",
}


st.set_page_config(
    page_title="Deep Trading System",
    page_icon=None,
    layout="wide",
)

st.title("Deep Trading System")


def render_disclaimer(language: str, sidebar: bool = False) -> None:
    title = REPORT_TITLES[language]["disclaimer"]
    body = DISCLAIMERS[language]
    if sidebar:
        st.markdown(f"**{title}**")
        st.caption(body)
        return

    st.divider()
    st.markdown(f"#### {title}")
    st.caption(body)


with st.sidebar:
    st.header("Run")
    company_query = st.text_input("Company name or ticker", value="NVDA")
    language_label = st.selectbox("Output language", list(LANGUAGE_OPTIONS.keys()))
    selected_language = LANGUAGE_OPTIONS[language_label]
    run_clicked = st.button(
        "Run analysis",
        type="primary",
        disabled=not company_query.strip(),
        use_container_width=True,
    )

    st.divider()
    render_disclaimer(selected_language, sidebar=True)

    st.divider()
    st.header("API keys")
    missing_keys = [key for key in REQUIRED_KEYS if not os.environ.get(key)]
    for key in REQUIRED_KEYS:
        status = "OK" if os.environ.get(key) else "Missing"
        st.write(f"{key}: {status}")


def render_result(result: DeepTradingResult) -> None:
    titles = REPORT_TITLES[result.language]
    state = result.state

    st.subheader(result.match.display_name)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Ticker", result.match.symbol)
    metric_cols[1].metric("Trade date", result.trade_date)
    metric_cols[2].metric(titles["signal"], result.signal)
    metric_cols[3].metric("Exchange", result.match.exchange or "-")

    tab_summary, tab_reports, tab_debates, tab_download = st.tabs(
        ["Summary", "Reports", "Debates", "Downloads"]
    )

    with tab_summary:
        st.markdown(state.get("final_trade_decision", ""))
        render_disclaimer(result.language)

    with tab_reports:
        with st.expander(titles["market"], expanded=True):
            st.markdown(state.get("market_report", ""))
        with st.expander(titles["sentiment"]):
            st.markdown(state.get("sentiment_report", ""))
        with st.expander(titles["news"]):
            st.markdown(state.get("news_report", ""))
        with st.expander(titles["fundamentals"]):
            st.markdown(state.get("fundamentals_report", ""))
        with st.expander(titles["investment_plan"]):
            st.markdown(state.get("investment_plan", ""))
        with st.expander(titles["trader"]):
            st.markdown(state.get("trader_investment_plan", ""))
        render_disclaimer(result.language)

    with tab_debates:
        with st.expander(titles["investment_debate"], expanded=True):
            st.markdown(state["investment_debate_state"].get("history", ""))
        with st.expander(titles["risk_debate"], expanded=True):
            st.markdown(state["risk_debate_state"].get("history", ""))

        if result.alternatives:
            st.subheader("Ticker matches")
            st.dataframe(
                [
                    {
                        "symbol": match.symbol,
                        "name": match.name,
                        "exchange": match.exchange,
                        "type": match.quote_type,
                    }
                    for match in result.alternatives
                ],
                hide_index=True,
                use_container_width=True,
            )
        render_disclaimer(result.language)

    with tab_download:
        st.markdown(result.markdown_report)
        st.download_button(
            "Download Markdown",
            data=result.markdown_report.encode("utf-8"),
            file_name=f"deep_trading_report_{result.match.symbol}.md",
            mime="text/markdown",
        )


if run_clicked:
    if missing_keys:
        st.error("Missing required API keys: " + ", ".join(missing_keys))
        st.stop()

    progress_container = st.container(border=True)
    progress_container.subheader("Progress")
    printer = StreamlitProgressPrinter(progress_container)
    manager = StreamlitTradingManager(printer)

    with st.spinner("Running analysis..."):
        try:
            result = manager.run(
                query=company_query,
                language=selected_language,
            )
        except Exception as exc:
            st.exception(exc)
            st.stop()

    st.session_state["last_result"] = result
    st.success("Analysis complete.")


if "last_result" in st.session_state:
    render_result(st.session_state["last_result"])
else:
    st.info("Ready.")
