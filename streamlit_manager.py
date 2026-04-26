"""Streamlit-aware runner for the Deep Trading System pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from agents import (
    build_analyst_nodes,
    build_researchers,
    build_risk_debaters,
    build_trader,
    create_portfolio_manager,
    create_research_manager,
    extract_signal,
    run_analyst,
)
from config import build_llms, config, setup_environment
from main import TEXT, make_initial_state, normalize_language
from memory import build_memories
from ticker_resolver import (
    TickerMatch,
    TickerResolutionError,
    resolve_ticker,
    search_tickers,
)
from tools import Toolkit, get_latest_available_trade_date


@dataclass
class DeepTradingResult:
    query: str
    match: TickerMatch
    alternatives: tuple[TickerMatch, ...]
    trade_date: str
    language: str
    signal: str
    state: dict
    markdown_report: str


REPORT_TITLES = {
    "english": {
        "title": "Deep Trading System Report",
        "resolved": "Resolved ticker",
        "signal": "Final signal",
        "market": "Market Analyst Report",
        "sentiment": "Social Sentiment Report",
        "news": "News Analyst Report",
        "fundamentals": "Fundamentals Analyst Report",
        "investment_debate": "Investment Debate",
        "investment_plan": "Research Manager Investment Plan",
        "trader": "Trader Proposal",
        "risk_debate": "Risk Debate",
        "final": "Portfolio Manager Final Decision",
        "empty": "No content generated.",
    },
    "chinese": {
        "title": "深度交易系统报告",
        "resolved": "已识别股票代码",
        "signal": "最终信号",
        "market": "市场分析报告",
        "sentiment": "社交情绪报告",
        "news": "新闻分析报告",
        "fundamentals": "基本面分析报告",
        "investment_debate": "投资辩论",
        "investment_plan": "研究经理投资计划",
        "trader": "交易员方案",
        "risk_debate": "风险辩论",
        "final": "投资组合经理最终决策",
        "empty": "未生成内容。",
    },
}


class StreamlitTradingManager:
    """Runs the existing agents while emitting progress updates for Streamlit."""

    def __init__(self, printer) -> None:
        self.printer = printer

    def run(self, query: str, language: str) -> DeepTradingResult:
        language = normalize_language(language)
        labels = TEXT[language]

        self.printer.update_item("resolve", "Resolving ticker...")
        match = resolve_ticker(query)
        try:
            alternatives = search_tickers(query)
        except TickerResolutionError:
            alternatives = (match,)
        self.printer.mark_item_done(
            "resolve", f"Resolved {query.strip()} to {match.symbol}"
        )

        self.printer.update_item("environment", "Checking API keys and tracing...")
        setup_environment(verbose=False, interactive=False)
        self.printer.mark_item_done("environment")

        self.printer.update_item("date", "Finding latest available trade date...")
        trade_date = get_latest_available_trade_date(match.symbol)
        self.printer.mark_item_done("date", f"Using trade date {trade_date}")

        self.printer.update_item("setup", "Preparing models, tools, and memories...")
        deep_thinking_llm, quick_thinking_llm = build_llms()
        toolkit = Toolkit(config)
        memories = build_memories(config)
        state = make_initial_state(match.symbol, trade_date, language)
        self.printer.mark_item_done("setup")

        analysts = build_analyst_nodes(quick_thinking_llm, toolkit)

        self.printer.update_item("market", labels["market_running"].strip())
        res = run_analyst(analysts["market"], state, toolkit)
        state["market_report"] = res.get("market_report", labels["failed_report"])
        self.printer.mark_item_done("market")

        self.printer.update_item("social", labels["social_running"].strip())
        res = run_analyst(analysts["social"], state, toolkit)
        state["sentiment_report"] = res.get(
            "sentiment_report", labels["failed_report"]
        )
        self.printer.mark_item_done("social")

        self.printer.update_item("news", labels["news_running"].strip())
        res = run_analyst(analysts["news"], state, toolkit)
        state["news_report"] = res.get("news_report", labels["failed_report"])
        self.printer.mark_item_done("news")

        self.printer.update_item(
            "fundamentals", labels["fundamentals_running"].strip()
        )
        res = run_analyst(analysts["fundamentals"], state, toolkit)
        state["fundamentals_report"] = res.get(
            "fundamentals_report", labels["failed_report"]
        )
        self.printer.mark_item_done("fundamentals")

        bull_node, bear_node = build_researchers(quick_thinking_llm, memories)
        for i in range(config["max_debate_rounds"]):
            item_id = f"debate_{i}"
            self.printer.update_item(
                item_id, labels["debate_round"].format(round=i + 1)
            )
            bull_result = bull_node(state)
            state["investment_debate_state"] = bull_result["investment_debate_state"]
            bear_result = bear_node(state)
            state["investment_debate_state"] = bear_result["investment_debate_state"]
            self.printer.mark_item_done(item_id)

        self.printer.update_item("research", labels["research_running"])
        research_manager_node = create_research_manager(
            deep_thinking_llm, memories["invest_judge"]
        )
        manager_result = research_manager_node(state)
        state["investment_plan"] = manager_result["investment_plan"]
        self.printer.mark_item_done("research")

        self.printer.update_item("trader", labels["trader_running"].strip())
        trader_node = build_trader(quick_thinking_llm, memories)
        trader_result = trader_node(state)
        state["trader_investment_plan"] = trader_result["trader_investment_plan"]
        self.printer.mark_item_done("trader")

        risky_node, safe_node, neutral_node = build_risk_debaters(quick_thinking_llm)
        for i in range(config["max_risk_discuss_rounds"]):
            item_id = f"risk_{i}"
            self.printer.update_item(item_id, labels["risk_debate"])
            risky_result = risky_node(state)
            state["risk_debate_state"] = risky_result["risk_debate_state"]
            safe_result = safe_node(state)
            state["risk_debate_state"] = safe_result["risk_debate_state"]
            neutral_result = neutral_node(state)
            state["risk_debate_state"] = neutral_result["risk_debate_state"]
            self.printer.mark_item_done(item_id)

        self.printer.update_item("portfolio", labels["portfolio_running"].strip())
        portfolio_manager_node = create_portfolio_manager(
            deep_thinking_llm, memories["risk_manager"]
        )
        pm_result = portfolio_manager_node(state)
        state["final_trade_decision"] = pm_result["final_trade_decision"]
        signal = extract_signal(state["final_trade_decision"])
        self.printer.mark_item_done("portfolio", f"Final signal: {signal}")

        markdown_report = build_markdown_report(
            query=query,
            match=match,
            trade_date=trade_date,
            language=language,
            signal=signal,
            state=state,
        )

        return DeepTradingResult(
            query=query,
            match=match,
            alternatives=alternatives,
            trade_date=trade_date,
            language=language,
            signal=signal,
            state=state,
            markdown_report=markdown_report,
        )


def _section(title: str, body: str, empty_text: str) -> str:
    return f"## {title}\n\n{body.strip() if body else empty_text}\n"


def build_markdown_report(
    query: str,
    match: TickerMatch,
    trade_date: str,
    language: str,
    signal: str,
    state: dict,
) -> str:
    titles = REPORT_TITLES[language]
    debate = state["investment_debate_state"].get("history", "")
    risk_debate = state["risk_debate_state"].get("history", "")

    lines = [
        f"# {titles['title']}: {match.symbol}",
        "",
        f"**{titles['resolved']}:** {match.display_name}",
        f"**Query:** {query.strip()}",
        f"**Trade date:** {trade_date}",
        f"**{titles['signal']}:** {signal}",
        "",
        _section(titles["market"], state.get("market_report", ""), titles["empty"]),
        _section(
            titles["sentiment"], state.get("sentiment_report", ""), titles["empty"]
        ),
        _section(titles["news"], state.get("news_report", ""), titles["empty"]),
        _section(
            titles["fundamentals"],
            state.get("fundamentals_report", ""),
            titles["empty"],
        ),
        _section(titles["investment_debate"], debate, titles["empty"]),
        _section(
            titles["investment_plan"], state.get("investment_plan", ""), titles["empty"]
        ),
        _section(
            titles["trader"],
            state.get("trader_investment_plan", ""),
            titles["empty"],
        ),
        _section(titles["risk_debate"], risk_debate, titles["empty"]),
        _section(
            titles["final"],
            state.get("final_trade_decision", ""),
            titles["empty"],
        ),
    ]
    return "\n".join(lines)
