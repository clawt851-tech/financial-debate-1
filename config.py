"""
Environment, API keys, LangSmith tracing and the central config dictionary.

Two-tier LLM design:
    deep_think_llm  (gpt-4o)      -> heavy reasoning + final decisions
    quick_think_llm (gpt-4o-mini) -> routine data processing + first-pass

Loads keys from `.env`:
    OPENAI_API_KEY, FINNHUB_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY
"""

import os
from getpass import getpass

from langchain_openai import ChatOpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# ---------------------------------------------------------------------------
# 1. API keys + LangSmith tracing
# ---------------------------------------------------------------------------
REQUIRED_KEYS = (
    "OPENAI_API_KEY",
    "FINNHUB_API_KEY",
    "TAVILY_API_KEY",
    "LANGSMITH_API_KEY",
)

LANGSMITH_TRACING = "true"
LANGSMITH_PROJECT = "Deep-Trading-System-V2"


def _set_env(var: str, interactive: bool = True) -> None:
    """Prompt for an env var if it isn't already set in the process or .env."""
    if not os.environ.get(var):
        if not interactive:
            raise EnvironmentError(f"Missing required environment variable: {var}")
        os.environ[var] = getpass(f"Please enter your {var}: ")


def setup_environment(verbose: bool = True, interactive: bool = True) -> None:
    """Load .env, ensure all required keys are present, enable LangSmith tracing."""
    if load_dotenv is not None:
        load_dotenv()

    if verbose:
        for key in REQUIRED_KEYS:
            status = "OK" if os.getenv(key) else "MISSING"
            print(f"  {key}: {status}")

    missing_keys = [key for key in REQUIRED_KEYS if not os.environ.get(key)]
    if missing_keys and not interactive:
        missing = ", ".join(missing_keys)
        raise EnvironmentError(f"Missing required environment variables: {missing}")

    for key in REQUIRED_KEYS:
        _set_env(key, interactive=interactive)

    # Enable LangSmith tracing for full observability. These are code-owned
    # defaults so `.env` only needs to contain secret API keys.
    os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    # LangChain also looks at LANGCHAIN_* variables in some versions
    os.environ["LANGCHAIN_TRACING_V2"] = LANGSMITH_TRACING
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    if os.environ.get("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_API_KEY", os.environ["LANGSMITH_API_KEY"])


# ---------------------------------------------------------------------------
# 2. Central config dictionary (control panel)
# ---------------------------------------------------------------------------
config = {
    "results_dir": "./results",
    "langsmith_tracing": LANGSMITH_TRACING,
    "langsmith_project": LANGSMITH_PROJECT,

    # LLM settings - which model handles which cognitive task
    "llm_provider": "openai",
    "deep_think_llm": "gpt-4o",        # heavy reasoning + final decisions
    "quick_think_llm": "gpt-4o-mini",  # fast / cheap data processing
    "backend_url": "https://api.openai.com/v1",

    # Debate / discussion settings
    "max_debate_rounds": 2,            # bull vs. bear rounds
    "max_risk_discuss_rounds": 1,      # risk team rounds
    "max_recur_limit": 100,            # safety cap for agent ReAct loops

    # Tool settings
    "online_tools": True,              # True = live API; False = cached data
    "data_cache_dir": "./data_cache",
}

os.makedirs(config["data_cache_dir"], exist_ok=True)
os.makedirs(config["results_dir"], exist_ok=True)


# ---------------------------------------------------------------------------
# 3. LLM factories
# ---------------------------------------------------------------------------
def build_llms():
    """
    Build the two LLMs the system uses.

    temperature=0.1 keeps financial analysis deterministic and fact-based.
    """
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
    return deep_thinking_llm, quick_thinking_llm
