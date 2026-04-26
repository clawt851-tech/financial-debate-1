"""
Every agent factory used by the deep-trading system.

    Analysts        create_analyst_node       -> market / social / news / fundamentals
    Researchers     create_researcher_node    -> bull / bear (debate, consult memory)
    Research mgr    create_research_manager   -> synthesizes investment_plan
    Trader          create_trader             -> turns plan into actionable proposal
    Risk debaters   create_risk_debator       -> risky / safe / neutral
    Portfolio mgr   create_portfolio_manager  -> final_trade_decision
    Reflection      create_reflector          -> writes lessons back into memory
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
                "You may use these tools: {tool_names}.\n"
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
    FIXED: Added config to tool_node.invoke to prevent ValueError.
    """
    state = initial_state
    tool_node = ToolNode(toolkit.all_tools())

    for _ in range(max_steps):
        result = analyst_node(state)
        merged_messages = state["messages"] + result["messages"]
        state = {**state, **result, "messages": merged_messages}

        if tools_condition({"messages": merged_messages}) == "tools":
            # FIX: We pass a config dict to satisfy the validation check
            tool_result = tool_node.invoke(
                {"messages": merged_messages},
                config={"configurable": {"thread_id": "analyst_react_loop"}}
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
    return {
        "market": create_analyst_node(
            quick_thinking_llm, toolkit, 
            "Analyze technical indicators and price action.",
            [toolkit.get_recent_yfinance_data, toolkit.get_technical_indicators],
            "market_report"
        ),
        "social": create_analyst_node(
            quick_thinking_llm, toolkit,
            "Analyze social media posts and public sentiment.",
            [toolkit.get_social_media_sentiment],
            "sentiment_report"
        ),
        "news": create_analyst_node(
            quick_thinking_llm, toolkit,
            "Analyze latest news and macroeconomic trends.",
            [toolkit.get_finnhub_news, toolkit.get_macroeconomic_news],
            "news_report"
        ),
        "fundamentals": create_analyst_node(
            quick_thinking_llm, toolkit,
            "Analyze fundamental financial health and insider sentiment.",
            [toolkit.get_fundamental_analysis],
            "fundamentals_report"
        ),
    }

# ===========================================================================
# Researcher, Manager, Trader, Risk, and Reflection nodes
# ===========================================================================
def create_researcher_node(llm, memory, role_prompt, agent_name):
    def researcher_node(state):
        situation = (f"Market: {state['market_report']}\n"
                     f"Sentiment: {state['sentiment_report']}\n"
                     f"News: {state['news_report']}\n"
                     f"Fundamentals: {state['fundamentals_report']}")
        past_memories = "\n".join(m["recommendation"] for m in memory.get_memories(situation))
        
        prompt = (
            f"{role_prompt}\n{_language_instruction(state)}\n"
            f"State:\n{situation}\n"
            f"History: {state['investment_debate_state']['history']}\n"
            f"Memories: {past_memories or 'None'}\n"
            "Formulate your argument."
        )
        response = llm.invoke(prompt)
        argument = f"{agent_name}: {response.content}"
        
        debate_state = state["investment_debate_state"].copy()
        debate_state["history"] += f"\n{argument}"
        debate_state["current_response"] = argument
        debate_state["count"] += 1
        return {"investment_debate_state": debate_state}
    return researcher_node

def build_researchers(quick_thinking_llm, memories):
    bull = create_researcher_node(quick_thinking_llm, memories["bull"], "You are a Bullish analyst.", "Bull Analyst")
    bear = create_researcher_node(quick_thinking_llm, memories["bear"], "You are a Bearish analyst.", "Bear Analyst")
    return bull, bear

def create_research_manager(llm, memory):
    def research_manager_node(state):
        prompt = f"Synthesize the debate and issue BUY/SELL/HOLD.\n{_language_instruction(state)}\nDebate: {state['investment_debate_state']['history']}"
        return {"investment_plan": llm.invoke(prompt).content}
    return research_manager_node

def create_trader(llm, memory):
    def trader_node(state, name):
        prompt = f"Create a proposal ending with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'.\nPlan: {state['investment_plan']}"
        return {"trader_investment_plan": llm.invoke(prompt).content, "sender": name}
    return trader_node

def build_trader(quick_thinking_llm, memories):
    return functools.partial(create_trader(quick_thinking_llm, memories["trader"]), name="Trader")

def create_risk_debator(llm, role_prompt, agent_name):
    def risk_debator_node(state):
        risk_state = state["risk_debate_state"].copy()
        prompt = f"{role_prompt}\nEvaluate: {state['trader_investment_plan']}\nHistory: {risk_state['history']}"
        response = llm.invoke(prompt).content
        risk_state["history"] += f"\n{agent_name}: {response}"
        risk_state["count"] += 1
        return {"risk_debate_state": risk_state}
    return risk_debator_node

def build_risk_debaters(quick_thinking_llm):
    return (
        create_risk_debator(quick_thinking_llm, "Prioritize high returns.", "Risky Analyst"),
        create_risk_debator(quick_thinking_llm, "Prioritize capital preservation.", "Safe Analyst"),
        create_risk_debator(quick_thinking_llm, "Provide a balanced view.", "Neutral Analyst")
    )

def create_portfolio_manager(llm, memory):
    def portfolio_manager_node(state):
        prompt = f"Final Decision. Must end with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'.\nPlan: {state['trader_investment_plan']}\nRisk Debate: {state['risk_debate_state']['history']}"
        return {"final_trade_decision": llm.invoke(prompt).content}
    return portfolio_manager_node
