"""
Every agent factory used by the deep-trading system.

    Analysts        create_analyst_node       -> market / social / news / fundamentals
    Researchers     create_researcher_node    -> bull / bear (debate, consult memory)
    Research mgr    create_research_manager   -> synthesizes investment_plan
    Trader          create_trader             -> turns plan into actionable proposal
    Risk debaters   create_risk_debator       -> risky / safe / neutral
    Portfolio mgr   create_portfolio_manager  -> final_trade_decision
    Reflection      create_reflector          -> writes lessons back into memory

A small `run_analyst()` helper drives the ReAct loop for analyst agents
outside of a full LangGraph run (matches the article's notebook style).
"""

import functools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode, tools_condition

# ---------------------------------------------------------------------------
# Language helper
# ---------------------------------------------------------------------------
def _language_instruction(state) -> str:
    """Return a prompt instruction for the selected output language."""
    language = state.get("output_language", "english")
    if language == "chinese":
        return (
            "Write the response in Simplified Chinese. Keep ticker symbols, "
            "API/tool names, and the final tag exactly in English when present: "
            "'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'."
        )
    return (
        "Write the response in English. Keep the final tag exactly in English "
        "when present: 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'."
    )

# ===========================================================================
# Analyst factory (ReAct loop: reason -> call tool -> observe -> repeat)
# ===========================================================================
def create_analyst_node(llm, toolkit, system_message, tools, output_field):
    """Build a LangGraph node for one type of analyst."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant collaborating with other assistants. "
                "Use the provided tools to step through the question. "
                "It is fine if you cannot fully answer; another assistant with "
                "different tools will continue from where you stop. Do as much "
                "useful work as you can. You may use these tools: {tool_names}.\n"
                "{system_message}\n{language_instruction}\n"
                "For reference, today's date is {current_date}. "
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
            language_instruction=_language_instruction(state),
        )
        result = (prompt_with_data | llm.bind_tools(tools)).invoke(state["messages"])
        report = ""
        if not result.tool_calls:
            report = result.content
        return {"messages": [result], output_field: report}

    return analyst_node


def run_analyst(analyst_node, initial_state, toolkit, max_steps: int = 5):
    """
    Drive a single analyst's ReAct loop outside a full LangGraph run.
    FIXED: Added config to tool_node.invoke to prevent ValueError 'N/A'.
    """
    state = initial_state
    tool_node = ToolNode(toolkit.all_tools())

    for _ in range(max_steps):
        result = analyst_node(state)
        merged_messages = state["messages"] + result["messages"]
        state = {**state, **result, "messages": merged_messages}

        if tools_condition({"messages": merged_messages}) == "tools":
            # FIX: We pass a config dict to avoid "Missing required config key 'N/A'"
            tool_result = tool_node.invoke(
                {"messages": merged_messages},
                config={"configurable": {"thread_id": "analyst_loop"}}
            )
            state = {
                **state,
                "messages": merged_messages + tool_result["messages"],
            }
        else:
            break
    return state

# ===========================================================================
# Concrete analyst nodes
# ===========================================================================
def build_analyst_nodes(quick_thinking_llm, toolkit):
    market_msg = (
        "You are a trading assistant specialized in financial markets. "
        "Pick the most relevant technical indicators to analyze the stock's "
        "price action, momentum and volatility."
    )
    market_analyst_node = create_analyst_node(
        quick_thinking_llm,
        toolkit,
        market_msg,
        [toolkit.get_recent_yfinance_data, toolkit.get_technical_indicators],
        "market_report",
    )

    social_msg = ("Analyze social media posts and public sentiment about the company.")
    social_analyst_node = create_analyst_node(
        quick_thinking_llm,
        toolkit,
        social_msg,
        [toolkit.get_social_media_sentiment],
        "sentiment_report",
    )

    news_msg = ("Analyze the latest news and trends from the past week.")
    news_analyst_node = create_analyst_node(
        quick_thinking_llm,
        toolkit,
        news_msg,
        [toolkit.get_finnhub_news, toolkit.get_macroeconomic_news],
        "news_report",
    )

    fundamentals_msg = ("Analyze a company's fundamentals and financial state.")
    fundamentals_analyst_node = create_analyst_node(
        quick_thinking_llm,
        toolkit,
        fundamentals_msg,
        [toolkit.get_fundamental_analysis],
        "fundamentals_report",
    )

    return {
        "market": market_analyst_node,
        "social": social_analyst_node,
        "news": news_analyst_node,
        "fundamentals": fundamentals_analyst_node,
    }

# ===========================================================================
# Researcher factory (bull / bear)
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
            f"{_language_instruction(state)}\n"
            f"Analysis state:\n{situation_summary}\n"
            f"Conversation so far: {state['investment_debate_state']['history']}\n"
            f"Opponent's argument: {state['investment_debate_state']['current_response']}\n"
            f"Past reflections: {past_memory_str or 'None.'}\n"
            "Lay out your argument conversationally."
        )

        response = llm.invoke(prompt)
        argument = f"{agent_name}: {response.content}"

        debate_state = state["investment_debate_state"].copy()
        debate_state["history"] += "\n" + argument
        if agent_name == "Bull Analyst":
            debate_state["bull_history"] += "\n" + argument
        else:
            debate_state["bear_history"] += "\n" + argument
        debate_state["current_response"] = argument
        debate_state["count"] += 1
        return {"investment_debate_state": debate_state}

    return researcher_node

def build_researchers(quick_thinking_llm, memories):
    bull = create_researcher_node(quick_thinking_llm, memories["bull"], "You are bullish.", "Bull Analyst")
    bear = create_researcher_node(quick_thinking_llm, memories["bear"], "You are bearish.", "Bear Analyst")
    return bull, bear

# ===========================================================================
# Research manager, Trader, Risk Debaters, etc.
# ===========================================================================
def create_research_manager(llm, memory):
    def research_manager_node(state):
        prompt = (
            "Evaluate the debate. Issue BUY, SELL or HOLD recommendation and a plan.\n\n"
            f"{_language_instruction(state)}\n\n"
            f"History:\n{state['investment_debate_state']['history']}"
        )
        response = llm.invoke(prompt)
        return {"investment_plan": response.content}
    return research_manager_node

def create_trader(llm, memory):
    def trader_node(state, name):
        prompt = (
            "Create a trade proposal. End with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'.\n\n"
            f"{_language_instruction(state)}\n\n"
            f"Plan: {state['investment_plan']}"
        )
        result = llm.invoke(prompt)
        return {"trader_investment_plan": result.content, "sender": name}
    return trader_node

def build_trader(quick_thinking_llm, memories):
    trader_node_func = create_trader(quick_thinking_llm, memories["trader"])
    return functools.partial(trader_node_func, name="Trader")

def create_risk_debator(llm, role_prompt, agent_name):
    def risk_debator_node(state):
        risk_state = state["risk_debate_state"]
        prompt = (
            f"{role_prompt}\n{_language_instruction(state)}\n"
            f"Trader's plan: {state['trader_investment_plan']}\n"
            f"Debate history: {risk_state['history']}\n"
            "Critique or support the plan."
        )
        response = llm.invoke(prompt).content
        new_risk_state = risk_state.copy()
        new_risk_state["history"] += f"\n{agent_name}: {response}"
        new_risk_state["count"] += 1
        return {"risk_debate_state": new_risk_state}
    return risk_debator_node

def build_risk_debaters(quick_thinking_llm):
    risky = create_risk_debator(quick_thinking_llm, "You are aggressive.", "Risky Analyst")
    safe = create_risk_debator(quick_thinking_llm, "You are conservative.", "Safe Analyst")
    neutral = create_risk_debator(quick_thinking_llm, "You are neutral.", "Neutral Analyst")
    return risky, safe, neutral

def create_portfolio_manager(llm, memory):
    def portfolio_manager_node(state):
        prompt = (
            "Final approval. End with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'.\n\n"
            f"{_language_instruction(state)}\n\n"
            f"Plan: {state['trader_investment_plan']}\n"
            f"Risk Debate: {state['risk_debate_state']['history']}"
        )
        response = llm.invoke(prompt)
        return {"final_trade_decision": response.content}
    return portfolio_manager_node

def extract_signal(final_trade_decision: str) -> str:
    text = (final_trade_decision or "").upper()
    if "FINAL TRANSACTION PROPOSAL" in text:
        tail = text.split("FINAL TRANSACTION PROPOSAL", 1)[-1]
        for signal in ("BUY", "SELL", "HOLD"):
            if signal in tail: return signal
    for signal in ("BUY", "SELL", "HOLD"):
        if signal in text: return signal
    return "UNKNOWN"

def create_reflector(llm):
    def reflect(state, outcome_summary: str, memory):
        situation = f"Ticker: {state.get('company_of_interest')}\nOutcome: {outcome_summary}"
        prompt = f"Write one actionable lesson for this situation:\n{situation}"
        lesson = llm.invoke(prompt).content.strip()
        memory.add_situations([(situation, lesson)])
        return lesson
    return reflect
