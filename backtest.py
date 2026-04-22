"""
backtest.py — Vectorized Backtest for Polymarket Bot Strategies
===============================================================
Adapted from @sopersone's 12-part series (Articles 3-5).
Applied to Polymarket historical data instead of BTC/USDT candles.

What this does:
  1. Fetches historical Polymarket trade data for a market
  2. Reconstructs a price series (YES probability over time)
  3. Applies our spread-farmer and mean-reversion logic as signals
  4. Runs vectorized backtest: calculates return, Sharpe, max drawdown, win rate
  5. Compares against a "hold YES" baseline (like buy-and-hold)

Run this BEFORE going live to validate your strategy has edge.

Usage:
    python backtest.py
"""

import json
import time
import logging
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — saves chart to file instead of showing window
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("backtest")

GAMMA_BASE = "https://gamma-api.polymarket.com"
DATA_API   = "https://data-api.polymarket.com"

# ── Fetch historical trades for a market ─────────────────────────────────────
def fetch_market_trades(market_id: str, limit: int = 500) -> list[dict]:
    """Pull raw trade history for a Polymarket market."""
    try:
        resp = requests.get(
            f"{GAMMA_BASE}/trades",
            params={"market": market_id, "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.error(f"Failed to fetch trades for {market_id}: {e}")
        return []

def fetch_active_markets(limit: int = 20) -> list[dict]:
    """Get active markets with enough volume to backtest."""
    try:
        resp = requests.get(
            f"{GAMMA_BASE}/markets",
            params={"active": "true", "limit": limit, "sortBy": "volume24hr"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        log.error(f"Failed to fetch markets: {e}")
        return []

# ── Build price series from trades ───────────────────────────────────────────
def trades_to_price_series(trades: list[dict]) -> Optional[pd.DataFrame]:
    """
    Convert raw trades into a time-indexed DataFrame of YES prices.
    This is the Polymarket equivalent of OHLCV candles.
    """
    if not trades:
        return None

    rows = []
    for t in trades:
        try:
            ts    = float(t.get("timestamp") or t.get("createdAt") or 0)
            price = float(t.get("price") or 0)
            size  = float(t.get("usdcSize") or t.get("size") or 0)
            side  = t.get("side", "BUY").upper()
            if price <= 0 or price >= 1:
                continue
            rows.append({"ts": ts, "price": price, "size": size, "side": side})
        except Exception:
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["ts"], unit="s")
    df = df.set_index("datetime").sort_index()

    # Resample to hourly candles (like the series uses daily BTC candles)
    # Use volume-weighted average price per period
    def vwap(group):
        if group["size"].sum() == 0:
            return group["price"].mean()
        return (group["price"] * group["size"]).sum() / group["size"].sum()

    price_series = df.groupby(pd.Grouper(freq="1h")).apply(vwap).dropna()
    price_series.name = "price"

    result = pd.DataFrame(price_series)

    # Log returns (from Article 3 — additive, ideal for backtesting)
    result["log_return"] = np.log(result["price"] / result["price"].shift(1))
    result = result.dropna()

    return result

# ── Strategy 1: Spread-Farmer (our existing agent, vectorized) ────────────────
def spread_farmer_signal(df: pd.DataFrame, spread_threshold: float = 0.08) -> pd.DataFrame:
    """
    Buy YES when price is significantly below 0.5 (wide spread opportunity).
    Sell when price returns to 0.5.
    This is the vectorized version of our spread_farmer_agent().
    """
    df = df.copy()

    # Signal: 1 when price is far below 0.5 (buy YES), 0 otherwise
    df["signal"] = np.where(df["price"] < (0.5 - spread_threshold), 1, 0)

    # Shift by 1 period to avoid look-ahead bias (Article 4 key insight)
    df["position"] = df["signal"].shift(1).fillna(0)

    return df

# ── Strategy 2: Mean Reversion on YES probability (from Article 4) ────────────
def mean_reversion_signal(df: pd.DataFrame, window: int = 24, threshold: float = -1.5) -> pd.DataFrame:
    """
    Buy YES when the z-score of price drops below threshold (price "too low").
    Exit when price reverts toward the rolling mean.
    Directly adapted from mean_reversion_strategy() in the series.
    """
    df = df.copy()

    df["rolling_mean"] = df["price"].rolling(window=window).mean()
    df["rolling_std"]  = df["price"].rolling(window=window).std()
    df["zscore"]       = (df["price"] - df["rolling_mean"]) / df["rolling_std"]

    # Buy when significantly below mean, exit when z-score recovers above 0
    df["signal"]   = np.where(df["zscore"] < threshold, 1, 0)
    df["position"] = df["signal"].shift(1).fillna(0)

    return df.dropna()

# ── Strategy 3: Momentum on YES probability (from Article 4) ──────────────────
def momentum_signal(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    Buy YES when the cumulative log return over the last N hours is positive
    (market trending toward YES). Exit when momentum turns negative.
    """
    df = df.copy()

    df["momentum"] = np.log(df["price"] / df["price"].shift(window))
    df["signal"]   = np.where(df["momentum"] > 0, 1, 0)
    df["position"] = df["signal"].shift(1).fillna(0)

    return df.dropna()

# ── Vectorized Backtest Engine (from Article 5) ───────────────────────────────
@dataclass
class BacktestResult:
    strategy_name:  str
    total_return:   float
    baseline_return: float   # "buy and hold YES"
    annual_return:  float
    annual_vol:     float
    sharpe:         float
    max_drawdown:   float
    win_rate:       float
    trading_periods: int

def backtest(df: pd.DataFrame, strategy_name: str = "Strategy") -> BacktestResult:
    """
    Universal vectorized backtest — directly adapted from the series' backtest() function.
    Works on any DataFrame with 'log_return' and 'position' columns.

    Key insight from Article 5:
        strategy_return[t] = log_return[t] * position[t]
    If position=1 we get the market return. If position=0 we're out.
    """
    result = df.copy().dropna(subset=["position", "log_return"])

    # Strategy return = market return multiplied by position
    result["strategy_return"] = result["log_return"] * result["position"]

    # Cumulative return (log returns sum to total return)
    result["cumret_strategy"]  = result["strategy_return"].cumsum().apply(np.exp)
    result["cumret_baseline"]  = result["log_return"].cumsum().apply(np.exp)

    # Metrics
    total_return    = result["cumret_strategy"].iloc[-1] - 1
    baseline_return = result["cumret_baseline"].iloc[-1] - 1

    n_periods    = len(result)
    periods_per_year = 24 * 365   # hourly data

    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    annual_vol    = result["strategy_return"].std() * np.sqrt(periods_per_year)
    sharpe        = annual_return / annual_vol if annual_vol > 0 else 0

    # Maximum drawdown (Article 5)
    rolling_max  = result["cumret_strategy"].cummax()
    drawdown     = (result["cumret_strategy"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Win rate
    trading_periods = int((result["position"] == 1).sum())
    winning_periods = int((result["strategy_return"] > 0).sum())
    win_rate        = winning_periods / trading_periods if trading_periods > 0 else 0

    return BacktestResult(
        strategy_name=strategy_name,
        total_return=total_return,
        baseline_return=baseline_return,
        annual_return=annual_return,
        annual_vol=annual_vol,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        trading_periods=trading_periods,
    )

# ── Print metrics (from Article 5) ───────────────────────────────────────────
def print_metrics(r: BacktestResult):
    print(f"\n{'='*50}")
    print(f"  {r.strategy_name}")
    print(f"{'='*50}")
    print(f"  Strategy return:   {r.total_return:>+8.1%}")
    print(f"  Baseline return:   {r.baseline_return:>+8.1%}")
    print(f"  Annual return:     {r.annual_return:>+8.1%}")
    print(f"  Annual volatility: {r.annual_vol:>8.1%}")
    print(f"  Sharpe ratio:      {r.sharpe:>8.2f}")
    print(f"  Max drawdown:      {r.max_drawdown:>+8.1%}")
    print(f"  Win rate:          {r.win_rate:>8.1%}")
    print(f"  Periods in market: {r.trading_periods:>8}")

# ── Plot equity curves (from Article 5) ──────────────────────────────────────
def plot_equity_curves(df_list: list[tuple[str, pd.DataFrame]], output_path: str = "backtest_results.png"):
    """Compare equity curves of all strategies vs baseline."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    colors = ["#00ff88", "#00cfff", "#ffaa00"]

    # Plot baseline on top chart
    first_df = df_list[0][1]
    axes[0].plot(first_df.index, first_df["cumret_baseline"],
                 linewidth=1.2, color="white", linestyle="--",
                 label="Hold YES (baseline)", alpha=0.5)

    # Plot each strategy
    for (name, df), color in zip(df_list, colors):
        if "cumret_strategy" not in df.columns:
            continue
        axes[0].plot(df.index, df["cumret_strategy"],
                     linewidth=1.2, color=color, label=name)

        # Drawdown on bottom chart
        rolling_max = df["cumret_strategy"].cummax()
        drawdown    = (df["cumret_strategy"] - rolling_max) / rolling_max
        axes[1].fill_between(df.index, 0, drawdown * 100,
                             alpha=0.4, color=color, label=name)

    axes[0].set_title("Equity Curves: Strategies vs Baseline (Hold YES)",
                       color="white")
    axes[0].set_ylabel("Cumulative Return (start = 1)", color="white")
    axes[0].legend(facecolor="#111", labelcolor="white")
    axes[0].set_facecolor("#0a0a0a")
    axes[0].tick_params(colors="white")

    axes[1].set_ylabel("Drawdown (%)", color="white")
    axes[1].set_xlabel("Date", color="white")
    axes[1].legend(facecolor="#111", labelcolor="white", fontsize=8)
    axes[1].set_facecolor("#0a0a0a")
    axes[1].tick_params(colors="white")

    fig.patch.set_facecolor("#000")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#000")
    log.info(f"Chart saved to {output_path}")

# ── Run full backtest on top markets ─────────────────────────────────────────
def run_full_backtest():
    log.info("Fetching top active markets...")
    markets = fetch_active_markets(limit=10)

    if not markets:
        log.error("No markets fetched — check your connection")
        return

    all_results = []

    for market in markets[:5]:  # Test on top 5 markets
        market_id = market.get("conditionId") or market.get("id")
        question  = market.get("question", "Unknown")[:60]
        volume    = market.get("volume", 0)

        log.info(f"\nMarket: {question}")
        log.info(f"Volume: ${volume:,.0f} | ID: {market_id}")

        trades = fetch_market_trades(market_id, limit=500)
        df     = trades_to_price_series(trades)

        if df is None or len(df) < 50:
            log.warning(f"  Not enough data ({len(df) if df is not None else 0} periods) — skipping")
            continue

        log.info(f"  Price series: {len(df)} hourly periods")
        log.info(f"  Price range: {df['price'].min():.3f} — {df['price'].max():.3f}")

        # Apply all 3 strategies
        df_spread = spread_farmer_signal(df.copy(), spread_threshold=0.08)
        df_mr     = mean_reversion_signal(df.copy(), window=24, threshold=-1.5)
        df_mom    = momentum_signal(df.copy(), window=12)

        # Run backtests
        strategies_to_test = [
            ("Spread Farmer",    df_spread),
            ("Mean Reversion",   df_mr),
            ("Momentum",         df_mom),
        ]

        market_results = []
        plot_data      = []

        for name, df_strat in strategies_to_test:
            if "position" not in df_strat.columns:
                continue
            # Add cumret columns for plotting
            df_strat["strategy_return"]  = df_strat["log_return"] * df_strat["position"]
            df_strat["cumret_strategy"]  = df_strat["strategy_return"].cumsum().apply(np.exp)
            df_strat["cumret_baseline"]  = df_strat["log_return"].cumsum().apply(np.exp)

            r = backtest(df_strat, strategy_name=name)
            print_metrics(r)
            market_results.append(r)
            plot_data.append((name, df_strat))

        all_results.extend(market_results)

        # Save chart for this market
        safe_name = "".join(c if c.isalnum() else "_" for c in question[:30])
        plot_equity_curves(plot_data, output_path=f"backtest_{safe_name}.png")

        time.sleep(0.5)   # polite rate limiting

    # Summary table across all markets
    if all_results:
        print(f"\n{'='*70}")
        print("  SUMMARY — All Markets")
        print(f"{'='*70}")
        print(f"  {'Strategy':<20} {'Avg Return':>12} {'Avg Sharpe':>12} {'Avg WinRate':>12}")
        print(f"  {'-'*56}")

        by_strategy: dict[str, list] = {}
        for r in all_results:
            by_strategy.setdefault(r.strategy_name, []).append(r)

        for name, results in by_strategy.items():
            avg_ret    = statistics.mean(r.total_return for r in results)
            avg_sharpe = statistics.mean(r.sharpe for r in results)
            avg_wr     = statistics.mean(r.win_rate for r in results)
            print(f"  {name:<20} {avg_ret:>+11.1%} {avg_sharpe:>12.2f} {avg_wr:>11.1%}")

        # Kelly calibration from backtest results
        print(f"\n{'='*70}")
        print("  KELLY CALIBRATION (use these in orchestrator.py)")
        print(f"{'='*70}")
        for name, results in by_strategy.items():
            avg_wr   = statistics.mean(r.win_rate for r in results)
            avg_odds = 1.0  # binary markets, approximate
            b = avg_odds
            p = avg_wr
            q = 1 - p
            kelly = max(0, (b * p - q) / b)
            half_kelly = kelly * 0.5
            print(f"  {name:<20}: win_rate={avg_wr:.1%} → full Kelly={kelly:.1%} → recommended half-Kelly={half_kelly:.1%}")

if __name__ == "__main__":
    run_full_backtest()