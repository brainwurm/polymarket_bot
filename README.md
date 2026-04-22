# Polymarket Multi-Agent Trading Bot

An advanced, asynchronous multi-agent trading bot for [Polymarket](https://polymarket.com), featuring LLM-driven analysis, statistical arbitrage, and real-time news trading.

## 🧠 Architecture & Agents

The bot runs on an asynchronous event loop and uses several specialized agents that operate concurrently:

1. **News-Edge Agent (LLM)**: Consumes headlines via NewsAPI and uses Anthropic's Claude to find mispriced markets based on breaking news. Focuses on Crypto, Tech/AI, and Niche News.
2. **Arb-Scanner Agent (LLM)**: Analyzes open markets to find logical inconsistencies and cross-market arbitrage opportunities (e.g., conflicting dates, mutually exclusive outcomes summing to >1).
3. **Crypto-Edge Agent (Quant)**: Fetches Binance OHLCV data to compute composite directional signals (SMA crossovers + momentum) and maps them to correlated Polymarket crypto markets. No LLM required.
4. **Spread-Farmer Agent (Quant)**: Runs on a tight high-frequency loop (every 3s) scanning for wide spreads in niche volume markets to fade back to fair value.

### Key Features

*   **Asynchronous Execution**: LLM calls and API requests never block the main event loop (`asyncio.gather`).
*   **Kelly Criterion Sizing**: Bet sizes are automatically calculated based on the agent's confidence, edge, and your current bankroll.
*   **Hard Stop-Loss**: The bot will automatically halt all trading if the bankroll drops by a configurable percentage (default 20%).
*   **Dry Run Mode**: Safely test agents and signals without risking real funds. All actions are logged to a local SQLite database.
*   **Local Database**: Every signal, fill, and bankroll update is logged to `polymarket_bot.db` for backtesting and analysis.

## 📂 Project Structure

*   `orchestrator.py`: The main entry point. Runs the agent loops, handles CLOB execution, and manages the SQLite database.
*   `binance_data.py`: Provides the quantitative data and signal generation for the `crypto-edge` agent.
*   `wallet_analyzer.py`: Utilities for analyzing historical wallet performance.
*   `backtest.py`: Tools to backtest signal effectiveness from the local SQLite database.
*   `dashboard/`: A React + Vite web dashboard for monitoring the bot's performance, current signals, and bankroll.

## 🚀 Getting Started

### Prerequisites

*   Python 3.10+
*   Node.js & npm (for the dashboard)
*   Required Python packages (see imports in `orchestrator.py`: `anthropic`, `aiohttp`, `py-clob-client`, `pandas`, `numpy`)

### Environment Setup

Create a `.env` file in the root directory with the following keys:

```env
# Required
ANTHROPIC_API_KEY=your_anthropic_key
WALLET_PRIVATE_KEY=your_wallet_private_key_0x_prefixed  # (Required for live trading)
WALLET_ADDRESS=your_wallet_address_0x_prefixed          # (Required for live trading)
NEWSAPI_KEY=your_news_api_key

# Optional Configuration
BANK_ROLL=25.0
LLM_LOOP_INTERVAL=10.0
SPREAD_LOOP_INTERVAL=3.0
MAX_KELLY_FRAC=0.15
MIN_CONFIDENCE=0.60
STOP_LOSS_PCT=0.20
DRY_RUN=1 # Set to 0 to enable live trading
```

### Running the Bot

1.  **Start the Orchestrator**:
    ```bash
    python orchestrator.py
    ```

2.  **Start the Dashboard (Optional)**:
    ```bash
    cd dashboard
    npm install
    npm run dev
    ```

## ⚠️ Disclaimer

This software is for educational purposes only. Do not trade with money you cannot afford to lose. The cryptocurrency and prediction markets are highly volatile. Use `DRY_RUN=1` to thoroughly test the bot before enabling live trading.
