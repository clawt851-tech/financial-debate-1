#!/usr/bin/env bash
# Launch the Deep Trading System Streamlit app as a server.
# Usage: ./run.sh [port]
set -euo pipefail

PORT="${1:-8501}"
cd "$(dirname "$0")"

exec streamlit run app.py --server.address 0.0.0.0 --server.port "$PORT"
