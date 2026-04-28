"""
End-to-end runner for the Deep Trading System V2.

Pipeline:
    1. Analysts (market, social, news, fundamentals) -> reports
    2. Bull / Bear researcher debate                 -> debate transcript
    3. Research manager synthesis                    -> investment_plan
    4. Trader                                        -> trader_investment_plan
                                                       (FINAL TRANSACTION tag)
    5. Risk team debate                              -> risk_debate_state
    6. Portfolio manager                             -> final_trade_decision
    7. Signal extraction                             -> BUY / SELL / HOLD

Defaults:
    TICKER     = "NVDA"
    TRADE_DATE = latest available yfinance trading date
"""

from __future__ import annotations

import argparse
import sys

from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown

from config import config, setup_environment, build_llms
from state import AgentState, InvestDebateState, RiskDebateState
from memory import build_memories
from tools import Toolkit, get_latest_available_trade_date
from agents import (
    build_analyst_nodes,
    run_analyst,
    build_researchers,
    create_research_manager,
    build_trader,
    build_risk_debaters,
    create_portfolio_manager,
    extract_signal,
)


# ---------------------------------------------------------------------------
# Defaults from the user's spec
# ---------------------------------------------------------------------------
TICKER = "NVDA"


# ---------------------------------------------------------------------------
# Console + UTF-8 setup
# ---------------------------------------------------------------------------
def configure_output_encoding() -> None:
    """Prefer UTF-8 console output so Chinese labels render on Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


configure_output_encoding()
console = Console()


# ---------------------------------------------------------------------------
# Bilingual labels
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES = {
    "en": "english",
    "eng": "english",
    "english": "english",
    "zh": "chinese",
    "cn": "chinese",
    "chinese": "chinese",
    "中文": "chinese",
}

TEXT = {
    "english": {
        "initial_prompt":
            "Analyze the trading opportunity for {ticker} on {trade_date}. "
            "Respond in English.",
        "market_running": "Running Market Analyst...",
        "market_report": "----- Market Analyst Report -----",
        "social_running": "\nRunning Social Media Analyst...",
        "social_report": "----- Social Media Analyst Report -----",
        "news_running": "\nRunning News Analyst...",
        "news_report": "----- News Analyst Report -----",
        "fundamentals_running": "\nRunning Fundamentals Analyst...",
        "fundamentals_report": "----- Fundamentals Analyst Report -----",
        "debate_round": "--- Investment Debate Round {round} ---",
        "bull_argument": "\n**Bull's Argument:**",
        "bear_rebuttal": "\n**Bear's Rebuttal:**",
        "research_running": "Running Research Manager...",
        "investment_plan": "\n----- Research Manager's Investment Plan -----",
        "trader_running": "\nRunning Trader...",
        "trader_proposal": "\n----- Trader's Proposal -----",
        "risk_debate": "--- Risk Management Debate ---",
        "risky": "\n**Aggressive Strategist:**",
        "safe": "\n**Defensive Strategist:**",
        "neutral": "\n**Balanced Strategist:**",
        "portfolio_running": "\nRunning Portfolio Manager...",
        "final_decision": "\n----- Portfolio Manager's Final Decision -----",
        "signal": "\n[bold green]EXTRACTED SIGNAL: {signal}[/bold green]",
        "failed_report": "Failed to generate report.",
    },
    "chinese": {
        "initial_prompt":
            "分析 {ticker} 在 {trade_date} 的交易机会。请用简体中文回答。",
        "market_running": "正在运行市场分析师...",
        "market_report": "----- 市场分析报告 -----",
        "social_running": "\n正在运行社交媒体分析师...",
        "social_report": "----- 社交媒体情绪报告 -----",
        "news_running": "\n正在运行新闻分析师...",
        "news_report": "----- 新闻分析报告 -----",
        "fundamentals_running": "\n正在运行基本面分析师...",
        "fundamentals_report": "----- 基本面分析报告 -----",
        "debate_round": "--- 投资辩论第 {round} 轮 ---",
        "bull_argument": "\n**多头观点：**",
        "bear_rebuttal": "\n**空头反驳：**",
        "research_running": "正在运行研究经理...",
        "investment_plan": "\n----- 研究经理投资计划 -----",
        "trader_running": "\n正在运行交易员...",
        "trader_proposal": "\n----- 交易员方案 -----",
        "risk_debate": "--- 风险管理辩论 ---",
        "risky": "\n**激进风险分析师：**",
        "safe": "\n**保守风险分析师：**",
        "neutral": "\n**中性风险分析师：**",
        "portfolio_running": "\n正在运行投资组合经理...",
        "final_decision": "\n----- 投资组合经理最终决策 -----",
        "signal": "\n[bold green]提取的交易信号: {signal}[/bold green]",
        "failed_report": "报告生成失败。",
    },
}


def normalize_language(language: str) -> str:
    """Normalize user language input to 'english' or 'chinese'."""
    normalized = SUPPORTED_LANGUAGES.get((language or "english").strip().lower())
    if normalized is None:
        valid = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"Unsupported language '{language}'. Use one of: {valid}")
    return normalized


# ---------------------------------------------------------------------------
# Initial AgentState
# ---------------------------------------------------------------------------
def make_initial_state(
    ticker: str, trade_date: str, language: str = "english"
) -> AgentState:
    """Build an empty AgentState seeded with the user's ticker + date."""
    language = normalize_language(language)
    labels = TEXT[language]
    return AgentState(
        messages=[
            HumanMessage(
                content=labels["initial_prompt"].format(
                    ticker=ticker, trade_date=trade_date
                )
            )
        ],
        company_of_interest=ticker,
        trade_date=trade_date,
        output_language=language,
        sender="",
        market_report="",
        sentiment_report="",
        news_report="",
        fundamentals_report="",
        investment_plan="",
        trader_investment_plan="",
        final_trade_decision="",
        investment_debate_state=InvestDebateState(
            history="",
            current_response="",
            count=0,
            bull_history="",
            bear_history="",
            judge_decision="",
        ),
        risk_debate_state=RiskDebateState(
            history="",
            latest_speaker="",
            current_risky_response="",
            current_safe_response="",
            current_neutral_response="",
            count=0,
            risky_history="",
            safe_history="",
            neutral_history="",
            judge_decision="",
        ),
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    ticker: str = TICKER,
    language: str = "english",
):
    language = normalize_language(language)
    labels = TEXT[language]

    # 1. Setup environment + LLMs + toolkit + memories
    setup_environment()
    trade_date = get_latest_available_trade_date(ticker)
    print(f"\nUsing latest available yfinance trading date: {trade_date}")

    deep_thinking_llm, quick_thinking_llm = build_llms()
    toolkit = Toolkit(config)
    memories = build_memories(config)

    state = make_initial_state(ticker, trade_date, language)

    # ------------------------------------------------------------------
    # 2. Analyst team (each runs its own ReAct loop)
    # ------------------------------------------------------------------
    analysts = build_analyst_nodes(quick_thinking_llm, toolkit)

    print(labels["market_running"])
    res = run_analyst(analysts["market"], state, toolkit)
    state["market_report"] = res.get("market_report", labels["failed_report"])
    console.print(labels["market_report"])
    console.print(Markdown(state["market_report"]))

    print(labels["social_running"])
    res = run_analyst(analysts["social"], state, toolkit)
    state["sentiment_report"] = res.get("sentiment_report", labels["failed_report"])
    console.print(labels["social_report"])
    console.print(Markdown(state["sentiment_report"]))

    print(labels["news_running"])
    res = run_analyst(analysts["news"], state, toolkit)
    state["news_report"] = res.get("news_report", labels["failed_report"])
    console.print(labels["news_report"])
    console.print(Markdown(state["news_report"]))

    print(labels["fundamentals_running"])
    res = run_analyst(analysts["fundamentals"], state, toolkit)
    state["fundamentals_report"] = res.get(
        "fundamentals_report", labels["failed_report"]
    )
    console.print(labels["fundamentals_report"])
    console.print(Markdown(state["fundamentals_report"]))

    # ------------------------------------------------------------------
    # 3. Bull vs. Bear debate
    # ------------------------------------------------------------------
    bull_node, bear_node = build_researchers(quick_thinking_llm, memories)

    for i in range(config["max_debate_rounds"]):
        print(labels["debate_round"].format(round=i + 1))

        bull_result = bull_node(state)
        state["investment_debate_state"] = bull_result["investment_debate_state"]
        console.print(labels["bull_argument"])
        console.print(
            Markdown(
                state["investment_debate_state"]["current_response"].replace(
                    "Bull Analyst (Optimistic Analyst): ", ""
                )
            )
        )

        bear_result = bear_node(state)
        state["investment_debate_state"] = bear_result["investment_debate_state"]
        console.print(labels["bear_rebuttal"])
        console.print(
            Markdown(
                state["investment_debate_state"]["current_response"].replace(
                    "Bear Analyst (Skeptic Analyst): ", ""
                )
            )
        )
        print()

    # ------------------------------------------------------------------
    # 4. Research manager synthesis
    # ------------------------------------------------------------------
    print(labels["research_running"])
    research_manager_node = create_research_manager(
        deep_thinking_llm, memories["invest_judge"]
    )
    manager_result = research_manager_node(state)
    state["investment_plan"] = manager_result["investment_plan"]
    console.print(labels["investment_plan"])
    console.print(Markdown(state["investment_plan"]))

    # ------------------------------------------------------------------
    # 5. Trader
    # ------------------------------------------------------------------
    print(labels["trader_running"])
    trader_node = build_trader(quick_thinking_llm, memories)
    trader_result = trader_node(state)
    state["trader_investment_plan"] = trader_result["trader_investment_plan"]
    console.print(labels["trader_proposal"])
    console.print(Markdown(state["trader_investment_plan"]))

    # ------------------------------------------------------------------
    # 6. Risk team debate
    # ------------------------------------------------------------------
    risky_node, safe_node, neutral_node = build_risk_debaters(quick_thinking_llm)

    for _ in range(config["max_risk_discuss_rounds"]):
        print(labels["risk_debate"])

        risky_result = risky_node(state)
        state["risk_debate_state"] = risky_result["risk_debate_state"]
        console.print(labels["risky"])
        console.print(Markdown(state["risk_debate_state"]["current_risky_response"]))

        safe_result = safe_node(state)
        state["risk_debate_state"] = safe_result["risk_debate_state"]
        console.print(labels["safe"])
        console.print(Markdown(state["risk_debate_state"]["current_safe_response"]))

        neutral_result = neutral_node(state)
        state["risk_debate_state"] = neutral_result["risk_debate_state"]
        console.print(labels["neutral"])
        console.print(Markdown(state["risk_debate_state"]["current_neutral_response"]))

    # ------------------------------------------------------------------
    # 7. Portfolio manager + signal extraction
    # ------------------------------------------------------------------
    print(labels["portfolio_running"])
    portfolio_manager_node = create_portfolio_manager(
        deep_thinking_llm, memories["risk_manager"]
    )
    pm_result = portfolio_manager_node(state)
    state["final_trade_decision"] = pm_result["final_trade_decision"]
    console.print(labels["final_decision"])
    console.print(Markdown(state["final_trade_decision"]))

    signal = extract_signal(state["final_trade_decision"])
    console.print(labels["signal"].format(signal=signal))

    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Deep Trading System V2.")
    parser.add_argument(
        "--ticker",
        default=TICKER,
        help=f"Ticker symbol to analyze (default: {TICKER}).",
    )
    parser.add_argument(
        "--language",
        default="english",
        help="Output language: english/chinese, en/zh, or 中文.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        ticker=args.ticker,
        language=args.language,
    )
