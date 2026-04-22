"""
wallet_analyzer.py
==================
Scrapes top Polymarket wallets, computes win rate, sizing consistency,
and market diversity to identify traders with genuine edge (not luck).

Run once before deploying the bot to calibrate your Kelly fractions.
"""

import json
import time
import logging
import requests
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
import statistics

log = logging.getLogger("wallet-analyzer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

GAMMA_BASE = "https://gamma-api.polymarket.com"
DATA_API   = "https://data-api.polymarket.com"

# ── Data structures ─────────────────────────────────────────────────────────
@dataclass
class TraderProfile:
    address: str
    total_trades: int = 0
    winning_trades: int = 0
    total_volume: float = 0.0
    pnl: float = 0.0
    markets_traded: set = field(default_factory=set)
    bet_sizes: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def sizing_consistency(self) -> float:
        """Low CV = consistent sizing = disciplined trader."""
        if len(self.bet_sizes) < 3:
            return 0.0
        mean = statistics.mean(self.bet_sizes)
        if mean == 0:
            return 0.0
        cv = statistics.stdev(self.bet_sizes) / mean
        return max(0.0, 1.0 - cv)  # invert so higher = more consistent

    @property
    def market_diversity(self) -> int:
        return len(self.markets_traded)

    @property
    def edge_score(self) -> float:
        """
        Composite score separating skill from luck.
        High score = high win rate + consistent sizing + diverse markets.
        """
        if self.total_trades < 15:  # minimum sample size
            return 0.0
        return (
            self.win_rate * 0.5 +
            self.sizing_consistency * 0.3 +
            min(self.market_diversity / 20, 1.0) * 0.2
        )

    def is_skilled(self) -> bool:
        return (
            self.win_rate >= 0.58 and
            self.total_trades >= 20 and
            self.market_diversity >= 5 and
            self.edge_score >= 0.55
        )

# ── Fetch trades ────────────────────────────────────────────────────────────
def fetch_trades_for_address(address: str, limit: int = 200) -> list[dict]:
    """Pull trade history for a single wallet."""
    try:
        resp = requests.get(
            f"{DATA_API}/activity",
            params={"user": address, "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning(f"Failed to fetch trades for {address}: {e}")
        return []

def fetch_top_wallets(limit: int = 100) -> list[str]:
    """
    Get addresses of top traders by volume/PnL from Polymarket leaderboard.
    Falls back to scraping recent large trades if leaderboard API is unavailable.
    """
    try:
        resp = requests.get(
            f"{DATA_API}/portfolio-users",
            params={"limit": limit, "sortBy": "pnl", "order": "DESC"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        addresses = [u.get("proxyWallet") or u.get("address") for u in data if u.get("proxyWallet") or u.get("address")]
        log.info(f"Got {len(addresses)} wallets from leaderboard")
        return addresses[:limit]
    except Exception as e:
        log.warning(f"Leaderboard API failed ({e}), falling back to large-trade scrape")
        return _scrape_large_trades(limit)

def _scrape_large_trades(limit: int) -> list[str]:
    """Collect unique addresses from recent high-volume trades."""
    addresses = set()
    try:
        resp = requests.get(
            f"{GAMMA_BASE}/trades",
            params={"limit": 500, "sortBy": "usdcSize", "order": "DESC"},
            timeout=15,
        )
        resp.raise_for_status()
        trades = resp.json()
        for t in trades:
            addr = t.get("maker") or t.get("transactorAddress")
            if addr:
                addresses.add(addr)
        log.info(f"Scraped {len(addresses)} unique addresses from large trades")
    except Exception as e:
        log.error(f"Large-trade scrape failed: {e}")
    return list(addresses)[:limit]

# ── Build profiles ──────────────────────────────────────────────────────────
def build_profile(address: str) -> Optional[TraderProfile]:
    trades = fetch_trades_for_address(address)
    if not trades:
        return None

    profile = TraderProfile(address=address)

    for trade in trades:
        size   = float(trade.get("usdcSize") or trade.get("size") or 0)
        side   = trade.get("side", "").upper()
        market = trade.get("market") or trade.get("conditionId") or ""
        outcome_price = float(trade.get("price") or 0.5)

        profile.total_trades += 1
        profile.total_volume += size
        profile.markets_traded.add(market)
        if size > 0:
            profile.bet_sizes.append(size)

        # A trade is "winning" if they bought YES and price > 0.5,
        # or bought NO and price < 0.5 (rough heuristic without resolution data)
        resolved = trade.get("resolved")
        if resolved is not None:
            if resolved:
                profile.winning_trades += 1
        else:
            # Use price momentum as proxy (imperfect but better than nothing)
            if side == "BUY" and outcome_price >= 0.55:
                profile.winning_trades += 1
            elif side == "SELL" and outcome_price <= 0.45:
                profile.winning_trades += 1

    return profile

# ── Main analysis ───────────────────────────────────────────────────────────
def analyze_wallets(n_wallets: int = 200) -> list[TraderProfile]:
    log.info(f"Fetching top {n_wallets} wallets...")
    addresses = fetch_top_wallets(n_wallets)

    profiles = []
    for i, addr in enumerate(addresses):
        log.info(f"Analyzing wallet {i+1}/{len(addresses)}: {addr[:10]}...")
        profile = build_profile(addr)
        if profile:
            profiles.append(profile)
        time.sleep(0.3)  # polite rate limiting

    skilled = [p for p in profiles if p.is_skilled()]
    log.info(f"\n{'='*60}")
    log.info(f"Analyzed {len(profiles)} wallets | Skilled traders found: {len(skilled)}")

    skilled.sort(key=lambda p: p.edge_score, reverse=True)

    log.info("\n🏆 TOP TRADERS WITH GENUINE EDGE:")
    for p in skilled[:10]:
        log.info(
            f"  {p.address[:12]}... | "
            f"WR={p.win_rate:.1%} | "
            f"Trades={p.total_trades} | "
            f"Markets={p.market_diversity} | "
            f"EdgeScore={p.edge_score:.3f}"
        )

    # Save results
    output = {
        "analyzed": len(profiles),
        "skilled_count": len(skilled),
        "skilled_traders": [
            {
                "address": p.address,
                "win_rate": round(p.win_rate, 4),
                "total_trades": p.total_trades,
                "market_diversity": p.market_diversity,
                "sizing_consistency": round(p.sizing_consistency, 4),
                "edge_score": round(p.edge_score, 4),
                "total_volume": round(p.total_volume, 2),
            }
            for p in skilled
        ]
    }

    with open("wallet_analysis.json", "w") as f:
        json.dump(output, f, indent=2)
    log.info("\nSaved to wallet_analysis.json")

    return skilled

# ── Kelly calibration from top wallets ─────────────────────────────────────
def derive_kelly_params(skilled_traders: list[TraderProfile]) -> dict:
    """
    Derive Kelly parameters from the sizing patterns of skilled traders.
    Returns recommended win_prob and max_kelly_fraction for your bot.
    """
    if not skilled_traders:
        return {"win_prob": 0.55, "max_kelly": 0.10}

    avg_win_rate    = statistics.mean(p.win_rate for p in skilled_traders)
    avg_consistency = statistics.mean(p.sizing_consistency for p in skilled_traders)

    # Conservative: half-Kelly based on observed edge
    recommended_max = avg_consistency * 0.15  # cap based on how consistent pros are

    params = {
        "avg_win_rate_of_skilled": round(avg_win_rate, 4),
        "recommended_min_confidence": round(avg_win_rate * 0.95, 4),
        "recommended_max_kelly_fraction": round(recommended_max, 4),
        "notes": "Use half-Kelly (divide max by 2) for safety during initial deployment."
    }

    log.info(f"\n📊 KELLY CALIBRATION:")
    for k, v in params.items():
        log.info(f"  {k}: {v}")

    return params

# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    skilled = analyze_wallets(n_wallets=200)
    kelly_params = derive_kelly_params(skilled)

    with open("kelly_params.json", "w") as f:
        json.dump(kelly_params, f, indent=2)
    log.info("Kelly params saved to kelly_params.json")