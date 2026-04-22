"""
binance_data.py — Free Binance Market Data + Signal Generation
==============================================================
Adapted from @sopersone's series (Article 3) — fetching, processing,
and storing financial data from Binance's public API.

No API key required for historical data.
Binance public endpoints are open without authentication.

Signals generated here feed into the crypto-edge agent in orchestrator.py.

Usage (standalone):
    python3 binance_data.py

Usage (as module):
    from binance_data import get_crypto_signals
    signals = get_crypto_signals()
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("binance")

BINANCE_BASE = "https://api.binance.com/api/v3"

# ── Symbols we track ──────────────────────────────────────────────────────────
TRACKED_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
]

# ── Interval constants (mirrors python-binance naming) ────────────────────────
INTERVAL_1M  = "1m"
INTERVAL_5M  = "5m"
INTERVAL_15M = "15m"
INTERVAL_1H  = "1h"
INTERVAL_4H  = "4h"
INTERVAL_1D  = "1d"

# ── Fetch OHLCV candles ───────────────────────────────────────────────────────
def fetch_candles(
    symbol: str,
    interval: str = INTERVAL_1H,
    limit: int = 200,
) -> Optional[pd.DataFrame]:
    """
    Fetch candlestick (OHLCV) data from Binance public API.
    No API key required.

    Returns a DataFrame with columns: open, high, low, close, volume
    and datetime index — exactly the format from Article 3.
    """
    try:
        resp = requests.get(
            f"{BINANCE_BASE}/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        log.error(f"Binance fetch failed for {symbol}: {e}")
        return None

    if not raw:
        return None

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=columns)

    # Convert types (Article 3 cleaning step)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df.set_index("open_time")
    df.index.name = "date"
    df = df[["open", "high", "low", "close", "volume"]]

    # Log returns (Article 3 — additive, ideal for backtesting)
    df["log_return"]  = np.log(df["close"] / df["close"].shift(1))
    df["simple_return"] = df["close"].pct_change()

    return df.dropna()

def fetch_current_price(symbol: str) -> Optional[float]:
    """Get the current price of a symbol."""
    try:
        resp = requests.get(
            f"{BINANCE_BASE}/ticker/price",
            params={"symbol": symbol},
            timeout=5,
        )
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception as e:
        log.error(f"Price fetch failed for {symbol}: {e}")
        return None

def fetch_24h_stats(symbol: str) -> Optional[dict]:
    """Get 24h price change statistics."""
    try:
        resp = requests.get(
            f"{BINANCE_BASE}/ticker/24hr",
            params={"symbol": symbol},
            timeout=5,
        )
        resp.raise_for_status()
        d = resp.json()
        return {
            "symbol":        symbol,
            "price":         float(d["lastPrice"]),
            "change_pct":    float(d["priceChangePercent"]),
            "high_24h":      float(d["highPrice"]),
            "low_24h":       float(d["lowPrice"]),
            "volume_24h":    float(d["volume"]),
            "quote_volume":  float(d["quoteVolume"]),
        }
    except Exception as e:
        log.error(f"24h stats failed for {symbol}: {e}")
        return None

# ── Technical indicators (Article 4 strategy building blocks) ─────────────────
def add_sma(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """SMA Crossover signals — from Article 4."""
    df = df.copy()
    df[f"SMA_{fast}"] = df["close"].rolling(window=fast).mean()
    df[f"SMA_{slow}"] = df["close"].rolling(window=slow).mean()
    df["sma_signal"]  = np.where(df[f"SMA_{fast}"] > df[f"SMA_{slow}"], 1, -1)
    return df

def add_momentum(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Momentum signal — from Article 4."""
    df = df.copy()
    df["momentum"]        = np.log(df["close"] / df["close"].shift(window))
    df["momentum_signal"] = np.where(df["momentum"] > 0, 1, -1)
    return df

def add_mean_reversion(df: pd.DataFrame, window: int = 20, threshold: float = -1.0) -> pd.DataFrame:
    """Mean reversion z-score signal — from Article 4."""
    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window).mean()
    df["rolling_std"]  = df["close"].rolling(window=window).std()
    df["zscore"]       = (df["close"] - df["rolling_mean"]) / df["rolling_std"]
    # 1 = oversold (buy), -1 = overbought (sell), 0 = neutral
    df["mr_signal"] = np.where(
        df["zscore"] < threshold, 1,
        np.where(df["zscore"] > -threshold, -1, 0)
    )
    return df

def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling volatility — useful for Kelly sizing adjustments."""
    df = df.copy()
    df["volatility"]      = df["log_return"].rolling(window=window).std() * np.sqrt(24 * 365)
    df["high_vol"]        = df["volatility"] > df["volatility"].rolling(window=window*2).mean()
    return df

# ── Composite signal for a symbol ─────────────────────────────────────────────
@dataclass
class CryptoSignal:
    symbol:          str
    price:           float
    change_24h_pct:  float
    momentum_20h:    float    # 20-period log return
    momentum_signal: int      # 1=bullish, -1=bearish
    sma_signal:      int      # 1=fast>slow (bullish), -1=bearish
    mr_signal:       int      # 1=oversold, -1=overbought, 0=neutral
    zscore:          float
    volatility:      float
    composite:       float    # weighted composite: -1 (very bearish) to +1 (very bullish)
    timestamp:       float = field(default_factory=time.time)

    @property
    def direction(self) -> str:
        if self.composite > 0.3:   return "BULLISH"
        if self.composite < -0.3:  return "BEARISH"
        return "NEUTRAL"

    @property
    def strength(self) -> str:
        abs_c = abs(self.composite)
        if abs_c > 0.7:  return "STRONG"
        if abs_c > 0.4:  return "MODERATE"
        return "WEAK"

def compute_signal(symbol: str) -> Optional[CryptoSignal]:
    """
    Fetch Binance data for a symbol and compute a composite directional signal.
    Uses SMA crossover + momentum + mean reversion (all three Article 4 strategies).
    """
    df = fetch_candles(symbol, interval=INTERVAL_1H, limit=100)
    if df is None or len(df) < 50:
        return None

    df = add_sma(df, fast=10, slow=30)
    df = add_momentum(df, window=20)
    df = add_mean_reversion(df, window=20, threshold=-1.0)
    df = add_volatility(df, window=20)

    latest = df.iloc[-1]

    # Weighted composite: momentum gets highest weight since it's most predictive
    # for short-term Polymarket market movements
    composite = (
        latest["momentum_signal"] * 0.40 +
        latest["sma_signal"]      * 0.35 +
        latest["mr_signal"]       * 0.25
    )

    stats = fetch_24h_stats(symbol)
    change_24h = stats["change_pct"] if stats else 0.0

    return CryptoSignal(
        symbol=symbol,
        price=float(latest["close"]),
        change_24h_pct=change_24h,
        momentum_20h=float(latest["momentum"]),
        momentum_signal=int(latest["momentum_signal"]),
        sma_signal=int(latest["sma_signal"]),
        mr_signal=int(latest["mr_signal"]),
        zscore=float(latest["zscore"]),
        volatility=float(latest["volatility"]),
        composite=float(composite),
    )

# ── Get all crypto signals ────────────────────────────────────────────────────
def get_crypto_signals() -> list[CryptoSignal]:
    """
    Fetch signals for all tracked symbols.
    Called by crypto-edge agent in orchestrator.py.
    """
    signals = []
    for symbol in TRACKED_SYMBOLS:
        sig = compute_signal(symbol)
        if sig:
            signals.append(sig)
            log.info(
                f"[{symbol}] ${sig.price:,.2f} | "
                f"{sig.change_24h_pct:+.1f}% 24h | "
                f"composite={sig.composite:+.2f} ({sig.direction} {sig.strength})"
            )
        time.sleep(0.2)   # polite rate limiting
    return signals

# ── Map crypto signals to Polymarket questions ────────────────────────────────
CRYPTO_MARKET_KEYWORDS = {
    "BTCUSDT": ["bitcoin", "btc", "crypto", "digital asset", "satoshi"],
    "ETHUSDT": ["ethereum", "eth", "ether", "defi", "smart contract"],
    "SOLUSDT": ["solana", "sol"],
    "BNBUSDT": ["binance", "bnb"],
}

def find_relevant_markets(
    crypto_signal: CryptoSignal,
    markets: list[dict],
) -> list[tuple[dict, float]]:
    """
    Match a crypto signal to Polymarket questions that are likely
    correlated with that asset's price direction.

    Returns list of (market, relevance_score) tuples.
    """
    keywords = CRYPTO_MARKET_KEYWORDS.get(crypto_signal.symbol, [])
    results  = []

    for market in markets:
        question = market.get("question", "").lower()
        # Score by keyword matches
        score = sum(1 for kw in keywords if kw in question)
        if score > 0:
            results.append((market, score))

    # Sort by relevance
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def signal_to_market_bias(
    crypto_signal: CryptoSignal,
    market: dict,
) -> Optional[tuple[str, float, str]]:
    """
    Given a crypto signal and a Polymarket market, return:
        (outcome, fair_value_estimate, rationale)

    Logic:
    - If BTC is BULLISH and market asks "Will BTC exceed $X?" → lean YES
    - If BTC is BEARISH and market asks "Will BTC exceed $X?" → lean NO
    - Strength of signal maps to how far we push fair value from current price
    """
    question = market.get("question", "").lower()
    yes_price = float((market.get("outcomePrices") or ["0.5"])[0])

    # Direction of push based on composite signal
    if crypto_signal.composite > 0:
        # Bullish → lean YES on upside questions
        outcome    = "YES"
        push       = abs(crypto_signal.composite) * 0.12   # max 12% push
        fair_value = min(0.95, yes_price + push)
    else:
        # Bearish → lean NO (i.e. YES is overpriced)
        outcome    = "NO"
        push       = abs(crypto_signal.composite) * 0.12
        fair_value = max(0.05, yes_price - push)

    rationale = (
        f"{crypto_signal.symbol} {crypto_signal.direction} {crypto_signal.strength} "
        f"(composite={crypto_signal.composite:+.2f}, "
        f"mom={crypto_signal.momentum_20h:+.3f}, "
        f"z={crypto_signal.zscore:+.2f})"
    )

    return outcome, fair_value, rationale

# ── Standalone diagnostic run ─────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print("\n" + "="*60)
    print("  Binance Signal Diagnostic")
    print("="*60)

    signals = get_crypto_signals()

    print(f"\n{'Symbol':<10} {'Price':>10} {'24h%':>8} {'Momentum':>10} {'Z-Score':>9} {'Composite':>11} {'Direction'}")
    print("-" * 75)
    for s in signals:
        print(
            f"{s.symbol:<10} "
            f"${s.price:>9,.2f} "
            f"{s.change_24h_pct:>+7.1f}% "
            f"{s.momentum_20h:>+10.4f} "
            f"{s.zscore:>+9.2f} "
            f"{s.composite:>+10.2f}  "
            f"{s.direction} {s.strength}"
        )

    print("\n✓ Binance data module ready — import get_crypto_signals() in orchestrator.py")