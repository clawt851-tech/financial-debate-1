"""
Shared memory / state for the multi-agent system.

AgentState is the central object that flows through every node.
Sub-states (InvestDebateState, RiskDebateState) are dedicated scratchpads
for the bull/bear research debate and the risky/safe/neutral risk debate.
"""

from typing_extensions import TypedDict
from langgraph.graph import MessagesState


# ---------------------------------------------------------------------------
# Bull vs. Bear research debate
# ---------------------------------------------------------------------------
class InvestDebateState(TypedDict):
    bull_history: str       # arguments raised by the bull agent
    bear_history: str       # arguments raised by the bear agent
    history: str            # full transcript of the debate
    current_response: str   # most recent argument
    judge_decision: str     # final decision by the research manager
    count: int              # debate round counter


# ---------------------------------------------------------------------------
# Aggressive / Defensive / Balanced risk-management debate
# ---------------------------------------------------------------------------
class RiskDebateState(TypedDict):
    risky_history: str
    safe_history: str
    neutral_history: str
    history: str
    latest_speaker: str
    current_risky_response: str
    current_safe_response: str
    current_neutral_response: str
    judge_decision: str     # final decision by the portfolio manager
    count: int


# ---------------------------------------------------------------------------
# Master AgentState that flows through the entire graph
# ---------------------------------------------------------------------------
class AgentState(MessagesState):
    company_of_interest: str       # ticker being analyzed
    trade_date: str                # analysis date (yyyy-mm-dd)
    output_language: str           # "english" or "chinese"
    sender: str                    # agent that last touched the state

    # Per-analyst report fields
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str

    # Nested debate states
    investment_debate_state: InvestDebateState
    investment_plan: str           # research-manager's synthesized plan
    trader_investment_plan: str    # trader's actionable proposal
    risk_debate_state: RiskDebateState
    final_trade_decision: str      # portfolio manager's final call
