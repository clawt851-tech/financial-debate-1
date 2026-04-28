"""
Single-file, notebook-style version of the deep-trading system V2.

Pipeline:
    Analysts -> Bull/Bear debate -> Research Manager -> Trader -> Risk debate
    -> Portfolio Manager -> machine-readable signal

Defaults:
    TICKER     = "NVDA"
    TRADE_DATE = latest available yfinance trading date
"""

# ===========================================================================
# 0. Imports + (optional) install hints
# ===========================================================================
# !pip install -U langchain langgraph langchain_openai langchain_community \
#                 tavily-python yfinance finnhub-python stockstats \
#                 beautifulsoup4 chromadb rich python-dotenv pandas langsmith

from __future__ import annotations

import os
import sys
import io
import functools
import datetime
from getpass import getpass
from pprint import pprint
from typing import Annotated
from typing_extensions import TypedDict

import yfinance as yf
import finnhub
import chromadb
import pandas as pd
from chromadb.config import Settings
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from stockstats import wrap as stockstats_wrap
from rich.console import Console
from rich.markdown import Markdown

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


DEFAULT_RECENT_TRADING_DAYS = 250
MIN_TECHNICAL_TRADING_DAYS = 250


# ===========================================================================
# 1. Environment + LangSmith tracing
# ===========================================================================
LANGSMITH_TRACING = "true"
LANGSMITH_PROJECT = "Deep-Trading-System-V2"


def _set_env(var: str) -> None:
    if not os.environ.get(var):
        os.environ[var] = getpass(f"Please enter your {var}: ")


# Load .env (OPENAI_API_KEY, FINNHUB_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY).
if load_dotenv is not None:
    load_dotenv()

for key in ("OPENAI_API_KEY", "FINNHUB_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY"):
    status = "OK" if os.getenv(key) else "MISSING"
    print(f"  {key}: {status}")

_set_env("OPENAI_API_KEY")
_set_env("FINNHUB_API_KEY")
_set_env("TAVILY_API_KEY")
_set_env("LANGSMITH_API_KEY")

os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
os.environ["LANGCHAIN_TRACING_V2"] = LANGSMITH_TRACING
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
if os.environ.get("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGCHAIN_API_KEY", os.environ["LANGSMITH_API_KEY"])


# ===========================================================================
# 2. Central config dictionary
# ===========================================================================
config = {
    "results_dir": "./results",
    "langsmith_tracing": LANGSMITH_TRACING,
    "langsmith_project": LANGSMITH_PROJECT,
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.5",
    "quick_think_llm": "gpt-5.4-mini",
    "backend_url": "https://api.openai.com/v1",
    "max_debate_rounds": 2,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    "online_tools": True,
    "data_cache_dir": "./data_cache",
}
os.makedirs(config["data_cache_dir"], exist_ok=True)
os.makedirs(config["results_dir"], exist_ok=True)
print("\nConfig dictionary:")
pprint(config)


# ===========================================================================
# 3. LLMs (deep + quick)
# ===========================================================================
deep_thinking_llm = ChatOpenAI(
    model=config["deep_think_llm"],
    base_url=config["backend_url"],
    temperature=0.1,
)
quick_thinking_llm = ChatOpenAI(
    model=config["quick_think_llm"],
    base_url=config["backend_url"],
    temperature=0.1,
)


# ===========================================================================
# 4. Shared state schemas
# ===========================================================================
class InvestDebateState(TypedDict):
    bull_history: str
    bear_history: str
    history: str
    current_response: str
    judge_decision: str
    count: int


class RiskDebateState(TypedDict):
    risky_history: str
    safe_history: str
    neutral_history: str
    history: str
    latest_speaker: str
    current_risky_response: str
    current_safe_response: str
    current_neutral_response: str
    judge_decision: str
    count: int


class AgentState(MessagesState):
    company_of_interest: str
    trade_date: str
    sender: str
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str
    investment_debate_state: InvestDebateState
    investment_plan: str
    trader_investment_plan: str
    risk_debate_state: RiskDebateState
    final_trade_decision: str


# ===========================================================================
# 5. Toolkit (yfinance / stockstats / Finnhub / Tavily)
# ===========================================================================
def _inclusive_yfinance_end_date(end_date: str) -> str:
    try:
        parsed = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return end_date
    return (parsed + datetime.timedelta(days=1)).isoformat()


def _calendar_lookback_for_trading_days(trading_days: int) -> int:
    if trading_days <= 0:
        raise ValueError("num_days must be a positive integer")
    return max(14, int(trading_days * 365 / 252) + 30)


def _expanded_indicator_start_date(
    start_date: str,
    end_date: str,
    min_trading_days: int = MIN_TECHNICAL_TRADING_DAYS,
) -> str:
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
    recent_prices, _, _ = get_recent_yfinance_prices(
        symbol=symbol,
        num_days=5,
        lookback_days=lookback_days,
    )
    dates = pd.to_datetime(recent_prices.index, errors="coerce", utc=True).dropna()
    if dates.empty:
        raise RuntimeError(f"Could not parse yfinance trading dates for {symbol}")
    return dates.max().date().isoformat()


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
        stock_df = stockstats_wrap(df)
        indicators = stock_df[
            ["macd", "rsi_14", "boll", "boll_ub", "boll_lb", "close_50_sma", "close_200_sma"]
        ]
        return indicators.tail().to_csv()
    except Exception as e:
        return f"Error computing stockstats indicators: {e}"


@tool
def get_finnhub_news(ticker: str, start_date: str, end_date: str) -> str:
    """Fetch company news from Finnhub for the given date range."""
    try:
        finnhub_client = finnhub.Client(api_key=os.environ["FINNHUB_API_KEY"])
        news_list = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
        items = []
        for news in news_list[:5]:
            items.append(f"Headline: {news['headline']}\nSummary: {news['summary']}")
        return "\n\n".join(items) if items else "No Finnhub news found."
    except Exception as e:
        return f"Error fetching Finnhub news: {e}"


tavily_tool = TavilySearchResults(max_results=3)


@tool
def get_social_media_sentiment(ticker: str, trade_date: str) -> str:
    """Live web search for social-media sentiment on a stock."""
    query = (
        f"social media sentiment and discussion about {ticker} stock "
        f"around {trade_date}"
    )
    return tavily_tool.invoke({"query": query})


@tool
def get_fundamental_analysis(ticker: str, trade_date: str) -> str:
    """Live web search for recent fundamental analysis on a stock."""
    query = (
        f"fundamental analysis and key financial metrics for {ticker} stock "
        f"published around {trade_date}"
    )
    return tavily_tool.invoke({"query": query})


@tool
def get_macroeconomic_news(trade_date: str) -> str:
    """Live web search for macroeconomic news affecting the stock market."""
    query = (
        f"macroeconomic news and market trends affecting the stock market "
        f"on {trade_date}"
    )
    return tavily_tool.invoke({"query": query})


class Toolkit:
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
        return [
            self.get_yfinance_data,
            self.get_recent_yfinance_data,
            self.get_technical_indicators,
            self.get_finnhub_news,
            self.get_social_media_sentiment,
            self.get_fundamental_analysis,
            self.get_macroeconomic_news,
        ]


toolkit = Toolkit(config)


# ===========================================================================
# 6. Long-term memory (ChromaDB) per learning agent
# ===========================================================================
class FinancialSituationMemory:
    def __init__(self, name: str, config: dict):
        self.embedding_model = "text-embedding-3-small"
        self.client = OpenAI(base_url=config["backend_url"])
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def get_embedding(self, text: str):
        response = self.client.embeddings.create(
            model=self.embedding_model, input=text
        )
        return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        if not situations_and_advice:
            return
        offset = self.situation_collection.count()
        ids = [str(offset + i) for i, _ in enumerate(situations_and_advice)]
        situations = [s for s, _ in situations_and_advice]
        recommendations = [r for _, r in situations_and_advice]
        embeddings = [self.get_embedding(s) for s in situations]
        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": r} for r in recommendations],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation: str, n_matches: int = 1):
        if self.situation_collection.count() == 0:
            return []
        q = self.get_embedding(current_situation)
        results = self.situation_collection.query(
            query_embeddings=[q],
            n_results=min(n_matches, self.situation_collection.count()),
            include=["metadatas"],
        )
        return [
            {"recommendation": meta["recommendation"]}
            for meta in results["metadatas"][0]
        ]


bull_memory = FinancialSituationMemory("bull_memory", config)
bear_memory = FinancialSituationMemory("bear_memory", config)
trader_memory = FinancialSituationMemory("trader_memory", config)
invest_judge_memory = FinancialSituationMemory("invest_judge_memory", config)
risk_manager_memory = FinancialSituationMemory("risk_manager_memory", config)


# ===========================================================================
# 7. Analyst factory + four concrete analysts
# ===========================================================================
def create_analyst_node(llm, toolkit, system_message, tools, output_field):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant collaborating with other assistants. "
                "Use the provided tools to step through the question. It is fine if "
                "you cannot fully answer; another assistant with different tools will "
                "continue from where you stop. Do as much useful work as you can. "
                "You may use these tools: {tool_names}.\n{system_message}"
                "\nFor reference, today's date is {current_date}. "
                "We are looking at the ticker {ticker}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))

    def analyst_node(state):
        prompt_with_data = prompt.partial(
            current_date=state["trade_date"],
            ticker=state["company_of_interest"],
        )
        result = (prompt_with_data | llm.bind_tools(tools)).invoke(state["messages"])
        report = "" if result.tool_calls else result.content
        return {"messages": [result], output_field: report}

    return analyst_node


def run_analyst(analyst_node, initial_state, max_steps: int = 5):
    state = initial_state
    tool_node = ToolNode(toolkit.all_tools())
    for _ in range(max_steps):
        result = analyst_node(state)
        merged_messages = state["messages"] + result["messages"]
        state = {**state, **result, "messages": merged_messages}

        if tools_condition({"messages": merged_messages}) == "tools":
            tool_result = tool_node.invoke({"messages": merged_messages})
            state = {**state, "messages": merged_messages + tool_result["messages"]}
        else:
            break
    return state


market_analyst_node = create_analyst_node(
    quick_thinking_llm,
    toolkit,
    "You are a trading assistant specialized in financial markets. Pick the most "
    "relevant technical indicators to analyze the stock's price action, momentum "
    "and volatility. Use the last 250 trading days of OHLCV price data and "
    "technical indicators. In the price table, show the latest 250 available "
    "trading days, ending on today's reference date when data is available.",
    [toolkit.get_recent_yfinance_data, toolkit.get_technical_indicators],
    "market_report",
)
social_analyst_node = create_analyst_node(
    quick_thinking_llm,
    toolkit,
    "You are a social media analyst. Analyze social media posts and public "
    "sentiment about the company over the past week. Use your tools to find "
    "relevant discussion and write a comprehensive report with insights, "
    "implications for traders and a summary table.",
    [toolkit.get_social_media_sentiment],
    "sentiment_report",
)
news_analyst_node = create_analyst_node(
    quick_thinking_llm,
    toolkit,
    "You are a news researcher analyzing the latest news and trends from the "
    "past week. Write a comprehensive report on the current state of the world "
    "as it relates to trading and the macroeconomy. Use your tools and include "
    "a summary table.",
    [toolkit.get_finnhub_news, toolkit.get_macroeconomic_news],
    "news_report",
)
fundamentals_analyst_node = create_analyst_node(
    quick_thinking_llm,
    toolkit,
    "You are a researcher analyzing a company's fundamentals. Write a "
    "comprehensive report on its financial state, insider sentiment and trading "
    "activity to give a full picture of fundamental health. Include a summary "
    "table.",
    [toolkit.get_fundamental_analysis],
    "fundamentals_report",
)


# ===========================================================================
# 8. Bull / Bear researchers + Research Manager
# ===========================================================================
def create_researcher_node(llm, memory, role_prompt, agent_name):
    def researcher_node(state):
        situation_summary = (
            f"Market Report: {state['market_report']}\n"
            f"Sentiment Report: {state['sentiment_report']}\n"
            f"News Report: {state['news_report']}\n"
            f"Fundamentals Report: {state['fundamentals_report']}"
        )
        past_memories = memory.get_memories(situation_summary)
        past_memory_str = "\n".join(m["recommendation"] for m in past_memories)
        prompt = (
            f"{role_prompt}\n"
            f"Here is the current analysis state:\n{situation_summary}\n"
            f"Conversation so far: {state['investment_debate_state']['history']}\n"
            f"Latest argument from the opposing side: "
            f"{state['investment_debate_state']['current_response']}\n"
            f"Reflections on similar past situations: "
            f"{past_memory_str or 'No past memories found.'}\n"
            "Based on all of the above, lay out your argument conversationally."
        )
        response = llm.invoke(prompt)
        argument = f"{agent_name}: {response.content}"

        debate_state = state["investment_debate_state"].copy()
        debate_state["history"] += "\n" + argument
        if agent_name == "Bull Analyst (Optimistic Analyst)":
            debate_state["bull_history"] += "\n" + argument
        else:
            debate_state["bear_history"] += "\n" + argument
        debate_state["current_response"] = argument
        debate_state["count"] += 1
        return {"investment_debate_state": debate_state}

    return researcher_node


bull_prompt = (
    "You are a bullish analyst. Your goal is to argue the case for investing in "
    "this stock. Focus on growth potential, competitive advantages and positive "
    "indicators in the reports. Counter the bear's points effectively."
)
bear_prompt = (
    "You are a bearish analyst. Your goal is to argue against investing in this "
    "stock. Focus on the risks, challenges and negative indicators. Counter the "
    "bull's points effectively."
)
bull_researcher_node = create_researcher_node(
    quick_thinking_llm, bull_memory, bull_prompt, "Bull Analyst (Optimistic Analyst)"
)
bear_researcher_node = create_researcher_node(
    quick_thinking_llm, bear_memory, bear_prompt, "Bear Analyst (Skeptic Analyst)"
)


def create_research_manager(llm, memory):
    def research_manager_node(state):
        prompt = (
            "As the Research Manager, your job is to critically evaluate the "
            "debate between the Bull and Bear analysts and make a final "
            "decision. Summarize the key arguments, then issue a clear "
            "recommendation: BUY, SELL or HOLD. Lay out a detailed investment "
            "plan for the trader, including your reasoning and strategic "
            f"actions.\n\nDebate history:\n{state['investment_debate_state']['history']}"
        )
        response = llm.invoke(prompt)
        return {"investment_plan": response.content}
    return research_manager_node


research_manager_node = create_research_manager(deep_thinking_llm, invest_judge_memory)


# ===========================================================================
# 9. Trader + Risk debaters + Portfolio Manager
# ===========================================================================
def create_trader(llm, memory):
    def trader_node(state, name):
        prompt = (
            "You are a trading agent. Based on the provided investment plan, "
            "create a concise trade proposal. Your response must end with "
            "'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'.\n\n"
            f"Proposed investment plan: {state['investment_plan']}"
        )
        result = llm.invoke(prompt)
        return {"trader_investment_plan": result.content, "sender": name}
    return trader_node


trader_node_func = create_trader(quick_thinking_llm, trader_memory)
trader_node = functools.partial(trader_node_func, name="Trader")


def create_risk_debator(llm, role_prompt, agent_name):
    def risk_debator_node(state):
        risk_state = state["risk_debate_state"]
        opponents_args = []
        if agent_name != "Aggressive Strategist" and risk_state["current_risky_response"]:
            opponents_args.append(
                f"Aggressive Strategist: {risk_state['current_risky_response']}"
            )
        if agent_name != "Defensive Strategist" and risk_state["current_safe_response"]:
            opponents_args.append(
                f"Defensive Strategist: {risk_state['current_safe_response']}"
            )
        if agent_name != "Balanced Strategist" and risk_state["current_neutral_response"]:
            opponents_args.append(
                f"Balanced Strategist: {risk_state['current_neutral_response']}"
            )

        prompt = (
            f"{role_prompt}\n"
            f"Here is the trader's plan: {state['trader_investment_plan']}\n"
            f"Debate history: {risk_state['history']}\n"
            f"Latest arguments from opponents:\n{chr(10).join(opponents_args)}\n"
            "Critique or support the plan from your perspective."
        )
        response = llm.invoke(prompt).content

        new_risk_state = risk_state.copy()
        new_risk_state["history"] += f"\n{agent_name}: {response}"
        new_risk_state["latest_speaker"] = agent_name
        if agent_name == "Aggressive Strategist":
            new_risk_state["current_risky_response"] = response
        elif agent_name == "Defensive Strategist":
            new_risk_state["current_safe_response"] = response
        else:
            new_risk_state["current_neutral_response"] = response
        new_risk_state["count"] += 1
        return {"risk_debate_state": new_risk_state}
    return risk_debator_node


risky_node = create_risk_debator(
    quick_thinking_llm,
    "You are an aggressive risk analyst. You advocate for high-return "
    "opportunities and bold strategies.",
    "Aggressive Strategist",
)
safe_node = create_risk_debator(
    quick_thinking_llm,
    "You are a conservative risk analyst. You prioritize capital preservation "
    "and minimizing volatility.",
    "Defensive Strategist",
)
neutral_node = create_risk_debator(
    quick_thinking_llm,
    "You are a neutral risk analyst. You provide a balanced perspective, "
    "weighing both reward and risk.",
    "Balanced Strategist",
)


def create_portfolio_manager(llm, memory):
    def portfolio_manager_node(state):
        prompt = (
            "You are the Portfolio Manager and the final approver. Weigh the "
            "trader's plan against the risk debate and issue the FINAL "
            "decision. Your response must end with "
            "'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'.\n\n"
            f"Trader's plan:\n{state['trader_investment_plan']}\n\n"
            f"Risk debate transcript:\n{state['risk_debate_state']['history']}"
        )
        response = llm.invoke(prompt)
        return {"final_trade_decision": response.content}
    return portfolio_manager_node


portfolio_manager_node = create_portfolio_manager(deep_thinking_llm, risk_manager_memory)


def extract_signal(text: str) -> str:
    text = (text or "").upper()
    if "FINAL TRANSACTION PROPOSAL" in text:
        tail = text.split("FINAL TRANSACTION PROPOSAL", 1)[-1]
        for s in ("BUY", "SELL", "HOLD"):
            if s in tail:
                return s
    for s in ("BUY", "SELL", "HOLD"):
        if s in text:
            return s
    return "UNKNOWN"


# ===========================================================================
# 10. Run the pipeline end-to-end
# ===========================================================================
console = Console()
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


TICKER = "NVDA"
TRADE_DATE = get_latest_available_trade_date(TICKER)
print(f"\nUsing latest available yfinance trading date: {TRADE_DATE}")

initial_state = AgentState(
    messages=[HumanMessage(content=f"Analyze the trading opportunity for {TICKER} on {TRADE_DATE}")],
    company_of_interest=TICKER,
    trade_date=TRADE_DATE,
    sender="",
    market_report="",
    sentiment_report="",
    news_report="",
    fundamentals_report="",
    investment_plan="",
    trader_investment_plan="",
    final_trade_decision="",
    investment_debate_state=InvestDebateState(
        history="", current_response="", count=0,
        bull_history="", bear_history="", judge_decision="",
    ),
    risk_debate_state=RiskDebateState(
        history="", latest_speaker="",
        current_risky_response="", current_safe_response="", current_neutral_response="",
        count=0, risky_history="", safe_history="", neutral_history="", judge_decision="",
    ),
)


def main() -> AgentState:
    state = initial_state

    # ----- Analyst team -----------------------------------------------------
    print("Running Market Analyst...")
    res = run_analyst(market_analyst_node, state)
    state["market_report"] = res.get("market_report", "Failed to generate report.")
    console.print("----- Market Analyst Report -----")
    console.print(Markdown(state["market_report"]))

    print("\nRunning Social Media Analyst...")
    res = run_analyst(social_analyst_node, state)
    state["sentiment_report"] = res.get("sentiment_report", "Failed to generate report.")
    console.print("----- Social Media Analyst Report -----")
    console.print(Markdown(state["sentiment_report"]))

    print("\nRunning News Analyst...")
    res = run_analyst(news_analyst_node, state)
    state["news_report"] = res.get("news_report", "Failed to generate report.")
    console.print("----- News Analyst Report -----")
    console.print(Markdown(state["news_report"]))

    print("\nRunning Fundamentals Analyst...")
    res = run_analyst(fundamentals_analyst_node, state)
    state["fundamentals_report"] = res.get("fundamentals_report", "Failed to generate report.")
    console.print("----- Fundamentals Analyst Report -----")
    console.print(Markdown(state["fundamentals_report"]))

    # ----- Bull vs. Bear debate --------------------------------------------
    for i in range(config["max_debate_rounds"]):
        print(f"--- Investment Debate Round {i + 1} ---")
        bull_result = bull_researcher_node(state)
        state["investment_debate_state"] = bull_result["investment_debate_state"]
        console.print("\n**Bull's Argument:**")
        console.print(Markdown(
            state["investment_debate_state"]["current_response"].replace("Bull Analyst (Optimistic Analyst): ", "")
        ))

        bear_result = bear_researcher_node(state)
        state["investment_debate_state"] = bear_result["investment_debate_state"]
        console.print("\n**Bear's Rebuttal:**")
        console.print(Markdown(
            state["investment_debate_state"]["current_response"].replace("Bear Analyst (Skeptic Analyst): ", "")
        ))
        print()

    # ----- Research manager synthesis --------------------------------------
    print("Running Research Manager...")
    manager_result = research_manager_node(state)
    state["investment_plan"] = manager_result["investment_plan"]
    console.print("\n----- Research Manager's Investment Plan -----")
    console.print(Markdown(state["investment_plan"]))

    # ----- Trader ----------------------------------------------------------
    print("\nRunning Trader...")
    trader_result = trader_node(state)
    state["trader_investment_plan"] = trader_result["trader_investment_plan"]
    console.print("\n----- Trader's Proposal -----")
    console.print(Markdown(state["trader_investment_plan"]))

    # ----- Risk debate -----------------------------------------------------
    for _ in range(config["max_risk_discuss_rounds"]):
        print("--- Risk Management Debate ---")

        risky_result = risky_node(state)
        state["risk_debate_state"] = risky_result["risk_debate_state"]
        console.print("\n**Aggressive Strategist:**")
        console.print(Markdown(state["risk_debate_state"]["current_risky_response"]))

        safe_result = safe_node(state)
        state["risk_debate_state"] = safe_result["risk_debate_state"]
        console.print("\n**Defensive Strategist:**")
        console.print(Markdown(state["risk_debate_state"]["current_safe_response"]))

        neutral_result = neutral_node(state)
        state["risk_debate_state"] = neutral_result["risk_debate_state"]
        console.print("\n**Balanced Strategist:**")
        console.print(Markdown(state["risk_debate_state"]["current_neutral_response"]))

    # ----- Portfolio manager -----------------------------------------------
    print("\nRunning Portfolio Manager...")
    pm_result = portfolio_manager_node(state)
    state["final_trade_decision"] = pm_result["final_trade_decision"]
    console.print("\n----- Portfolio Manager's Final Decision -----")
    console.print(Markdown(state["final_trade_decision"]))

    signal = extract_signal(state["final_trade_decision"])
    console.print(f"\n[bold green]EXTRACTED SIGNAL: {signal}[/bold green]")

    return state


if __name__ == "__main__":
    main()
