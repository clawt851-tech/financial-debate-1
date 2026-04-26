# Deep Trading System V2

Multi-agent trading research pipeline that analyzes a ticker, debates the bull
and bear cases, evaluates risk, and extracts a final `BUY`, `SELL`, or `HOLD`
signal.

This project defaults to:

- Ticker: `NVDA`
- Trade date: the latest available trading date returned by yfinance
- LangSmith project: `Deep-Trading-System-V2`

> This is an educational research tool, not financial advice. Always validate
> outputs against current market data and your own risk process before trading.

## What It Runs

The full pipeline in `main.py` follows this flow:

1. Market, social, news, and fundamentals analysts gather reports.
2. Bull and bear researchers debate the investment case.
3. A research manager synthesizes the debate into an investment plan.
4. A trader turns the plan into a trade proposal.
5. Risk analysts debate risky, safe, and neutral perspectives.
6. A portfolio manager makes the final decision.
7. The system extracts a final `BUY`, `SELL`, or `HOLD` signal.

There is also a single-file version in `notebook_style.py` that mirrors the
same end-to-end workflow.

## Project Layout

```text
deep_trading_system_v_2/
|-- .env.example          # API key template
|-- .gitignore
|-- requirements.txt
|-- config.py             # environment setup, config, and LLM factories
|-- state.py              # LangGraph-style agent state definitions
|-- tools.py              # yfinance, stockstats, Finnhub, and Tavily tools
|-- memory.py             # ChromaDB-backed financial situation memory
|-- agents.py             # analyst, debate, trader, risk, PM, and reflector nodes
|-- main.py               # full CLI pipeline runner
|-- app.py                # Streamlit server UI
|-- streamlit_manager.py  # Streamlit-aware pipeline runner
|-- streamlit_printer.py  # Progress renderer for Streamlit
|-- ticker_resolver.py    # company-name / ticker resolver
|-- run.bat               # Windows Streamlit server launcher
|-- run.sh                # Unix Streamlit server launcher
|-- sample_yfinance.py    # quick OHLCV smoke test
|-- test_api_keys.py      # live API credential checks
`-- notebook_style.py     # single-file pipeline version
```

## Requirements

- Python 3.10+
- API keys for:
  - OpenAI
  - Finnhub
  - Tavily
  - LangSmith

The default models are configured in `config.py`:

- `gpt-5.5` for deeper reasoning and final decisions
- `gpt-5.4-mini` for faster analyst and data-processing steps

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create your local environment file:

```powershell
Copy-Item .env.example .env
```

Fill in `.env`:

```env
OPENAI_API_KEY=your_openai_key
FINNHUB_API_KEY=your_finnhub_key
TAVILY_API_KEY=your_tavily_key
LANGSMITH_API_KEY=your_langsmith_key
```

LangSmith tracing is configured in code with project
`Deep-Trading-System-V2`, so `.env` only needs the secret keys.

## Quick Checks

Fetch the most recent 250 trading days for the default ticker:

```powershell
python sample_yfinance.py
```

Use a different ticker:

```powershell
python sample_yfinance.py AAPL
```

Validate all required API keys against live services:

```powershell
python test_api_keys.py
```

## Run The Pipeline

Run the full default pipeline:

```powershell
python main.py
```

Override the ticker or output language:

```powershell
python main.py --ticker AAPL --language english
python main.py --ticker NVDA --language chinese
```

Run the notebook-style single-file version:

```powershell
python notebook_style.py
```

## Run The Streamlit Server

The browser UI accepts either a company name or a ticker, resolves it through
Yahoo Finance, runs the same agent pipeline, and renders the final report in
English or Simplified Chinese.

Local only:

```powershell
streamlit run app.py
```

Server mode:

```powershell
.\run.bat 8501
```

Equivalent command:

```powershell
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then open `http://<host>:8501`.

## Configuration

Main settings live in `config.py`:

- `max_debate_rounds`: bull/bear debate rounds
- `max_risk_discuss_rounds`: risk discussion rounds
- `max_recur_limit`: safety cap for agent tool loops
- `results_dir`: output folder
- `data_cache_dir`: cache folder
- `online_tools`: live API mode flag

## Data Sources

The tools in `tools.py` use:

- `yfinance` for OHLCV market data
- `stockstats` for technical indicators
- Finnhub for company news
- Tavily for web search around sentiment, fundamentals, and macro news
- ChromaDB plus OpenAI embeddings for role-specific financial memory

## Notes

- `.env`, `results/`, `data_cache/`, virtual environments, and Python cache files
  are ignored by git.
- The API check script never prints secret values.
- LangSmith tracing is enabled by default so you can inspect agent runs in the
  `Deep-Trading-System-V2` project.
