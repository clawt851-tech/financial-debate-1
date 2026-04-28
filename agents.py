"""
Every agent factory used by the deep-trading system.

    Analysts       create_analyst_node       -> market / social / news / fundamentals
    Researchers    create_researcher_node    -> bull / bear (debate, consult memory)
    Research mgr   create_research_manager   -> synthesizes investment_plan
    Trader         create_trader             -> turns plan into actionable proposal
    Risk debaters  create_risk_debator       -> risky / safe / neutral
    Portfolio mgr  create_portfolio_manager  -> final_trade_decision
    Reflection     create_reflector          -> writes lessons back into memory

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
# Analyst factory  (ReAct loop: reason -> call tool -> observe -> repeat)
# ===========================================================================
def create_analyst_node(llm, toolkit, system_message, tools, output_field):
    """
    Build a LangGraph node for one type of analyst.

    Args:
        llm:            the model to use
        toolkit:        the shared Toolkit instance
        system_message: role/persona prompt for this analyst
        tools:          subset of toolkit functions this analyst can call
        output_field:   key on AgentState where the final report is stored
    """
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

def run_analyst(analyst_node, initial_state, tools, max_steps: int = 5):
    state = initial_state
    tools_by_name = {tool.name: tool for tool in tools}

    for _ in range(max_steps):
        result = analyst_node(state)

        merged_messages = state["messages"] + result["messages"]
        state = {**state, **result, "messages": merged_messages}

        ai_msg = merged_messages[-1]

        if not getattr(ai_msg, "tool_calls", None):
            break

        tool_messages = []
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]

            if tool_name not in tools_by_name:
                raise KeyError(
                    f"Tool '{tool_name}' was called, but available tools are: "
                    f"{list(tools_by_name.keys())}"
                )

            tool = tools_by_name[tool_name]
            tool_msg = tool.invoke(tool_call)
            tool_messages.append(tool_msg)

        state["messages"] = state["messages"] + tool_messages

    return state
# ===========================================================================
# Concrete analyst nodes
# ===========================================================================
def build_analyst_nodes(quick_thinking_llm, toolkit):
    market_msg = (
        "You are a trading assistant specialized in financial markets. "
        "Pick the most relevant technical indicators to analyze the stock's "
        "price action, momentum and volatility. Use the last 250 trading days "
        "of OHLCV price data and technical indicators. In the price table, "
        "show the latest 250 available trading days, ending on today's "
        "reference date when data is available."
    )
    market_analyst_node = create_analyst_node(
        quick_thinking_llm,
        toolkit,
        market_msg,
        [toolkit.get_recent_yfinance_data, toolkit.get_technical_indicators],
        "market_report",
    )

    social_msg = (
        "You are a social media analyst. Analyze social media posts and public "
        "sentiment about the company over the past week. Use your tools to find "
        "relevant discussion and write a comprehensive report with insights, "
        "implications for traders and a summary table."
    )
    social_analyst_node = create_analyst_node(
        quick_thinking_llm,
        toolkit,
        social_msg,
        [toolkit.get_social_media_sentiment],
        "sentiment_report",
    )

    news_msg = (
        "You are a news researcher analyzing the latest news and trends from "
        "the past week. Write a comprehensive report on the current state of "
        "the world as it relates to trading and the macroeconomy. Use your "
        "tools and include a summary table."
    )
    news_analyst_node = create_analyst_node(
        quick_thinking_llm,
        toolkit,
        news_msg,
        [toolkit.get_finnhub_news, toolkit.get_macroeconomic_news],
        "news_report",
    )

    fundamentals_msg = (
        "You are a researcher analyzing a company's fundamentals. Write a "
        "comprehensive report on its financial state, insider sentiment and "
        "trading activity to give a full picture of fundamental health. "
        "Include a summary table."
    )
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
    """
    Build a debate-style researcher node that:
        1. summarizes all four analyst reports as fixed context
        2. retrieves relevant past lessons from its private memory
        3. argues conversationally (in character) and rebuts the opponent
    """
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


def build_researchers(quick_thinking_llm, memories):
    bull_prompt = (
        "You are a bullish analyst. Your goal is to argue the case for "
        "investing in this stock. Focus on growth potential, competitive "
        "advantages and positive indicators in the reports. Counter the bear's "
        "points effectively."
    )
    bear_prompt = (
        "You are a bearish analyst. Your goal is to argue against investing "
        "in this stock. Focus on the risks, challenges and negative "
        "indicators. Counter the bull's points effectively."
    )
    bull = create_researcher_node(
        quick_thinking_llm, memories["bull"], bull_prompt, "Bull Analyst (Optimistic Analyst)"
    )
    bear = create_researcher_node(
        quick_thinking_llm, memories["bear"], bear_prompt, "Bear Analyst (Skeptic Analyst)"
    )
    return bull, bear


# ===========================================================================
# Research manager (judge + synthesizer)
# ===========================================================================
def create_research_manager(llm, memory):
    """Pick a side AND lay out a concrete investment plan for the trader."""
    def research_manager_node(state):
        prompt = (
            "As the Research Manager, your job is to critically evaluate the "
            "debate between the Bull and Bear analysts and make a final "
            "decision. Summarize the key arguments, then issue a clear "
            "recommendation: BUY, SELL or HOLD. Lay out a detailed investment "
            "plan for the trader, including your reasoning and strategic "
            "actions.\n\n"
            f"{_language_instruction(state)}\n\n"
            f"Debate history:\n{state['investment_debate_state']['history']}"
        )
        response = llm.invoke(prompt)
        return {"investment_plan": response.content}
    return research_manager_node


# ===========================================================================
# Trader (turns plan into actionable proposal with FINAL TRANSACTION tag)
# ===========================================================================
def create_trader(llm, memory):
    def trader_node(state, name):
        prompt = (
            "You are a trading agent. Based on the provided investment plan, "
            "create a concise trade proposal. Your response must end with "
            "'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'.\n\n"
            f"{_language_instruction(state)}\n\n"
            f"Proposed investment plan: {state['investment_plan']}"
        )
        result = llm.invoke(prompt)
        return {"trader_investment_plan": result.content, "sender": name}
    return trader_node


def build_trader(quick_thinking_llm, memories):
    trader_node_func = create_trader(quick_thinking_llm, memories["trader"])
    return functools.partial(trader_node_func, name="Trader")


# ===========================================================================
# Risk debaters (Aggressive / Defensive / Balanced)
# ===========================================================================
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
            f"{_language_instruction(state)}\n"
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


def build_risk_debaters(quick_thinking_llm):
    risky_prompt = (
        "You are an aggressive risk analyst. You advocate for high-return "
        "opportunities and bold strategies."
    )
    safe_prompt = (
        "You are a conservative risk analyst. You prioritize capital "
        "preservation and minimizing volatility."
    )
    neutral_prompt = (
        "You are a neutral risk analyst. You provide a balanced perspective, "
        "weighing both reward and risk."
    )
    risky = create_risk_debator(quick_thinking_llm, risky_prompt, "Aggressive Strategist")
    safe = create_risk_debator(quick_thinking_llm, safe_prompt, "Defensive Strategist")
    neutral = create_risk_debator(quick_thinking_llm, neutral_prompt, "Balanced Strategist")
    return risky, safe, neutral


# ===========================================================================
# Portfolio manager + signal extractor
# ===========================================================================
def create_portfolio_manager(llm, memory):
    """
    Final approver. Reads the trader's plan + the risk debate transcript and
    issues final_trade_decision. Output must end with the FINAL TRANSACTION
    PROPOSAL tag so a downstream extractor can parse a machine-readable
    BUY/SELL/HOLD signal.
    """
    def portfolio_manager_node(state):
        prompt = (
            "You are the Portfolio Manager and the final approver. Weigh the "
            "trader's plan against the risk debate and issue the FINAL "
            "decision. Your response must end with "
            "'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'.\n\n"
            f"{_language_instruction(state)}\n\n"
            f"Trader's plan:\n{state['trader_investment_plan']}\n\n"
            f"Risk debate transcript:\n{state['risk_debate_state']['history']}"
        )
        response = llm.invoke(prompt)
        return {"final_trade_decision": response.content}
    return portfolio_manager_node


def extract_signal(final_trade_decision: str) -> str:
    """Pull a machine-readable BUY / SELL / HOLD signal from the manager's text."""
    text = (final_trade_decision or "").upper()
    # Prefer the explicit tag when present
    if "FINAL TRANSACTION PROPOSAL" in text:
        tail = text.split("FINAL TRANSACTION PROPOSAL", 1)[-1]
        for signal in ("BUY", "SELL", "HOLD"):
            if signal in tail:
                return signal
    for signal in ("BUY", "SELL", "HOLD"):
        if signal in text:
            return signal
    return "UNKNOWN"


# ===========================================================================
# Reflection - feedback loop that turns outcomes into lessons in memory
# ===========================================================================
def create_reflector(llm):
    """
    Build a reflector that distills (situation, outcome) pairs into a lesson
    string that gets written back into the relevant memory store.
    """
    def reflect(state, outcome_summary: str, memory):
        situation = (
            f"Ticker: {state.get('company_of_interest')}\n"
            f"Date: {state.get('trade_date')}\n"
            f"Market Report: {state.get('market_report', '')}\n"
            f"Sentiment Report: {state.get('sentiment_report', '')}\n"
            f"News Report: {state.get('news_report', '')}\n"
            f"Fundamentals Report: {state.get('fundamentals_report', '')}\n"
            f"Investment Plan: {state.get('investment_plan', '')}\n"
            f"Trader Plan: {state.get('trader_investment_plan', '')}\n"
            f"Final Decision: {state.get('final_trade_decision', '')}"
        )
        prompt = (
            "You are a reflective trading coach. Given the situation below and "
            "the realized outcome, write ONE concise actionable lesson "
            "(1-3 sentences) that would help an agent make a better decision "
            "next time. Avoid restating the situation; output only the lesson.\n\n"
            f"Situation:\n{situation}\n\n"
            f"Outcome:\n{outcome_summary}"
        )
        lesson = llm.invoke(prompt).content.strip()
        memory.add_situations([(situation, lesson)])
        return lesson
    return reflect
