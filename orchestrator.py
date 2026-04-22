from dotenv import load_dotenv
load_dotenv()

"""
orchestrator.py  —  Polymarket Multi-Agent Trading Orchestrator
===============================================================
Architecture
------------
  • AsyncAnthropic client — LLM calls never block the event loop
  • asyncio.gather()     — news-edge + arb-scanner run in parallel
  • Separate tight loop  — spread-farmer runs every 3 s (no LLM)
  • py-clob-client       — real Polygon/CLOB order execution
  • SQLite               — every signal + fill logged for backtesting
  • Hard stop-loss       — halts bot if bankroll drops > 20 %

Environment variables required
-------------------------------
  ANTHROPIC_API_KEY
  WALLET_PRIVATE_KEY   (0x-prefixed)
  WALLET_ADDRESS       (0x-prefixed)
  NEWSAPI_KEY          (free at newsapi.org)

Optional
--------
  BANK_ROLL            default 25.0  (USDC)
  LLM_LOOP_INTERVAL    default 10    (seconds between LLM cycles)
  SPREAD_LOOP_INTERVAL default 3     (seconds between spread-farmer cycles)
  MAX_KELLY_FRAC       default 0.15
  MIN_CONFIDENCE       default 0.60
  STOP_LOSS_PCT        default 0.20  (halt if bankroll falls this fraction)
  DRY_RUN              default 1     (set to 0 to enable live trading)
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import aiohttp
import anthropic

# ── Lazy import of binance_data (graceful if numpy/pandas not installed) ──────
try:
    from binance_data import get_crypto_signals, find_relevant_markets, signal_to_market_bias
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    log_placeholder = logging.getLogger("orchestrator")
    log_placeholder.warning("binance_data not available — install numpy pandas to enable crypto-edge agent")

# ── Lazy import of py-clob-client so the file is still useful without it ──
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.constants import BUY, SELL
    CLOB_AVAILABLE = True
except ImportError:
    CLOB_AVAILABLE = False
    BUY = SELL = None

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-14s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orchestrator")

# ── Config (all overridable via env) ─────────────────────────────────────────
ANTHROPIC_API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
WALLET_PRIVATE_KEY   = os.environ.get("WALLET_PRIVATE_KEY", "")
WALLET_ADDRESS       = os.environ.get("WALLET_ADDRESS", "")
NEWSAPI_KEY          = os.environ.get("NEWSAPI_KEY", "")

BANK_ROLL            = float(os.environ.get("BANK_ROLL",            25.0))
LLM_LOOP_INTERVAL    = float(os.environ.get("LLM_LOOP_INTERVAL",    10.0))
SPREAD_LOOP_INTERVAL = float(os.environ.get("SPREAD_LOOP_INTERVAL",  3.0))
MAX_KELLY_FRAC       = float(os.environ.get("MAX_KELLY_FRAC",        0.15))
MIN_CONFIDENCE       = float(os.environ.get("MIN_CONFIDENCE",        0.60))
STOP_LOSS_PCT        = float(os.environ.get("STOP_LOSS_PCT",         0.20))
DRY_RUN              = os.environ.get("DRY_RUN", "1") != "0"

CLOB_BASE  = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"

# ── Anthropic async client ────────────────────────────────────────────────────
ai = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# ── CLOB client (singleton) ───────────────────────────────────────────────────
_clob: Optional["ClobClient"] = None

def get_clob() -> Optional["ClobClient"]:
    global _clob
    if _clob is None and CLOB_AVAILABLE and WALLET_PRIVATE_KEY:
        try:
            _clob = ClobClient(
                host=CLOB_BASE,
                key=WALLET_PRIVATE_KEY,
                chain_id=137,
                signature_type=2,
                funder=WALLET_ADDRESS,
            )
            log.info("CLOB client initialised ✓")
        except Exception as e:
            log.error(f"CLOB client init failed: {e}")
    return _clob

# ── SQLite setup ──────────────────────────────────────────────────────────────
DB_PATH = "polymarket_bot.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL,
            agent       TEXT,
            market_id   TEXT,
            question    TEXT,
            outcome     TEXT,
            price       REAL,
            fair_value  REAL,
            edge        REAL,
            confidence  REAL,
            rationale   TEXT,
            bet_usdc    REAL,
            status      TEXT DEFAULT 'pending'
        );
        CREATE TABLE IF NOT EXISTS fills (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id   INTEGER REFERENCES signals(id),
            ts          REAL,
            market_id   TEXT,
            outcome     TEXT,
            size_usdc   REAL,
            order_id    TEXT,
            raw_response TEXT
        );
        CREATE TABLE IF NOT EXISTS bankroll_log (
            ts       REAL,
            bankroll REAL,
            note     TEXT
        );
    """)
    con.commit()
    con.close()
    log.info(f"SQLite DB ready: {DB_PATH}")

def db_insert_signal(sig: "Signal", bet: float, status: str = "pending") -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.execute(
        """INSERT INTO signals
           (ts, agent, market_id, question, outcome, price, fair_value,
            edge, confidence, rationale, bet_usdc, status)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (sig.timestamp, sig.agent, sig.market_id, sig.question, sig.outcome,
         sig.current_price, sig.fair_value, sig.edge, sig.confidence,
         sig.rationale, bet, status),
    )
    row_id = cur.lastrowid
    con.commit(); con.close()
    return row_id

def db_insert_fill(signal_id: int, sig: "Signal", size: float, order_id: str, raw: str):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO fills (signal_id, ts, market_id, outcome, size_usdc, order_id, raw_response)
           VALUES (?,?,?,?,?,?,?)""",
        (signal_id, time.time(), sig.market_id, sig.outcome, size, order_id, raw),
    )
    con.commit(); con.close()

def db_log_bankroll(bankroll: float, note: str = ""):
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO bankroll_log (ts, bankroll, note) VALUES (?,?,?)",
                (time.time(), bankroll, note))
    con.commit(); con.close()

# ── Signal dataclass ──────────────────────────────────────────────────────────
@dataclass
class Signal:
    agent:         str
    market_id:     str
    question:      str
    outcome:       str          # "YES" | "NO"
    current_price: float        # 0–1
    fair_value:    float        # agent's estimate
    confidence:    float        # 0–1
    rationale:     str
    timestamp:     float = field(default_factory=time.time)

    @property
    def edge(self) -> float:
        return self.fair_value - self.current_price

# ── Kelly sizing ──────────────────────────────────────────────────────────────
def kelly_fraction(win_prob: float, price: float) -> float:
    if price <= 0 or price >= 1:
        return 0.0
    odds = 1 / price          # decimal odds
    b    = odds - 1
    p    = win_prob
    q    = 1 - p
    f    = (b * p - q) / b
    return max(0.0, min(f, MAX_KELLY_FRAC))

def size_bet(bankroll: float, sig: Signal) -> float:
    price = sig.current_price if sig.outcome == "YES" else (1 - sig.current_price)
    f     = kelly_fraction(sig.fair_value, price)
    return round(bankroll * f, 2)

# ── Async HTTP helpers ────────────────────────────────────────────────────────
async def async_get(session: aiohttp.ClientSession, url: str, params: dict = None) -> Optional[dict | list]:
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
            r.raise_for_status()
            return await r.json(content_type=None)
    except Exception as e:
        log.warning(f"GET {url} failed: {e}")
        return None

# ── Market focus: keywords per category ──────────────────────────────────────
# We only trade 3 categories: crypto, tech/AI, niche news.
# Avoids liquid political markets where we have no edge.
CATEGORY_KEYWORDS = {
    "crypto": [
        "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto",
        "blockchain", "defi", "nft", "altcoin", "binance", "bnb", "xrp",
        "ripple", "dogecoin", "doge", "coinbase", "stablecoin", "halving",
        "mining", "wallet", "web3", "layer 2", "polygon", "avalanche",
        "chainlink", "uniswap", "token", "memecoin",
    ],
    "tech_ai": [
        "openai", "chatgpt", "gpt", "anthropic", "claude", "gemini", "llm",
        "artificial intelligence", "machine learning", "nvidia", "apple",
        "google", "meta", "microsoft", "amazon", "aws", "tesla", "spacex",
        "elon musk", "sam altman", "iphone", "ipo", "earnings", "acquisition",
        "merger", "valuation", "startup", "ai model", "chipmaker", "semiconductor",
        "deepmind", "mistral", "groq", "perplexity", "regulation",
    ],
    "niche_news": [
        "weather", "hurricane", "earthquake", "flood", "storm",
        "science", "nasa", "spaceflight", "rocket", "launch",
        "sports", "nba", "nfl", "nhl", "mlb", "soccer", "fifa",
        "oscars", "grammy", "emmy", "award", "celebrity",
        "lawsuit", "trial", "verdict", "arrest", "indictment",
        "local", "mayor", "governor", "referendum", "ballot",
        "disease", "outbreak", "fda", "drug", "approval",
    ],
}

BLACKLIST_KEYWORDS = [
    "president", "presidential", "congress", "senate",
    "republican", "democrat", "trump", "biden", "harris",
    "federal reserve", "fed rate", "interest rate",
    "gdp", "inflation", "recession", "s&p 500", "dow jones",
]

def categorize_market(question: str):
    q = question.lower()
    if any(kw in q for kw in BLACKLIST_KEYWORDS):
        return None
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return category
    return None

def filter_target_markets(markets: list[dict]) -> list[dict]:
    filtered = []
    for m in markets:
        category = categorize_market(m.get("question", ""))
        if category:
            m["_category"] = category
            filtered.append(m)
    by_cat: dict[str, int] = {}
    for m in filtered:
        cat = m.get("_category", "unknown")
        by_cat[cat] = by_cat.get(cat, 0) + 1
    log.info(f"Market filter: {len(filtered)}/{len(markets)} kept — {by_cat}")
    return filtered

async def fetch_markets(session: aiohttp.ClientSession, limit: int = 100) -> list[dict]:
    """Fetch active markets and filter to our 3 target categories only."""
    data = await async_get(session, f"{GAMMA_BASE}/markets",
                           {"active": "true", "limit": limit, "sortBy": "volume24hr"})
    if not isinstance(data, list):
        return []
    return filter_target_markets(data)

async def fetch_headlines(session: aiohttp.ClientSession) -> list[str]:
    if not NEWSAPI_KEY:
        return []
    data = await async_get(session,
        "https://newsapi.org/v2/top-headlines",
        {"language": "en", "pageSize": 20, "apiKey": NEWSAPI_KEY},
    )
    if not data:
        return []
    return [a["title"] for a in data.get("articles", []) if a.get("title")]

# ── Agent 1: news-edge (async, LLM) ──────────────────────────────────────────
async def news_edge_agent(session: aiohttp.ClientSession, markets: list[dict]) -> list[Signal]:
    headlines = await fetch_headlines(session)
    if not headlines or not markets:
        return []

    market_summaries = [
        {"id": m.get("conditionId", m.get("id", "")),
         "question": m.get("question", ""),
         "yes_price": m.get("outcomePrices", ["0.5"])[0]}
        for m in markets[:20]
    ]

    prompt = f"""You are a prediction market quant analyst specializing in 3 categories:
1. CRYPTO — Bitcoin, Ethereum, altcoins, DeFi, NFTs, exchange events
2. TECH/AI — AI model releases, Big Tech earnings, product launches, IPOs, M&A
3. NICHE NEWS — local politics, sports, science, weather, awards, legal verdicts

You ONLY trade these 3 categories. You ignore political/macro markets entirely.

RECENT HEADLINES:
{json.dumps(headlines, indent=2)}

OPEN MARKETS (pre-filtered to our 3 categories):
{json.dumps(market_summaries, indent=2)}

Your edge comes from:
- Knowing crypto price action (BTC/ETH momentum, on-chain events)
- Understanding AI/tech news before the crowd prices it in
- Catching niche markets with wide spreads and thin liquidity

Find markets where these headlines create genuine mispricing in OUR categories only.
Respond ONLY with a JSON array — no markdown, no preamble. Each element:
{{
  "market_id": "...",
  "question": "...",
  "category": "crypto" or "tech_ai" or "niche_news",
  "outcome": "YES" or "NO",
  "current_price": <float 0-1>,
  "fair_value": <your estimate, float 0-1>,
  "confidence": <float 0-1>,
  "rationale": "<one concise sentence explaining the edge>"
}}
Only include markets where |fair_value - current_price| > 0.05 and confidence >= {MIN_CONFIDENCE}.
Prefer niche/illiquid markets over high-volume ones — that is where our edge lives.
Return [] if nothing qualifies."""

    try:
        resp = await ai.messages.create(
            model="claude-opus-4-5",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(resp.content[0].text.strip())
        signals = []
        for item in data:
            s = Signal(
                agent="news-edge",
                market_id=str(item["market_id"]),
                question=str(item["question"]),
                outcome=str(item["outcome"]),
                current_price=float(item["current_price"]),
                fair_value=float(item["fair_value"]),
                confidence=float(item["confidence"]),
                rationale=str(item["rationale"]),
            )
            if s.edge > 0.05 and s.confidence >= MIN_CONFIDENCE:
                signals.append(s)
        log.info(f"[news-edge]    {len(signals)} signals")
        return signals
    except Exception as e:
        log.error(f"[news-edge] error: {e}")
        return []

# ── Agent 2: arb-scanner (async, LLM) ────────────────────────────────────────
async def arb_scanner_agent(session: aiohttp.ClientSession, markets: list[dict]) -> list[Signal]:
    if not markets:
        return []

    market_data = [
        {"id": m.get("conditionId", m.get("id", "")),
         "question": m.get("question", ""),
         "yes_price": m.get("outcomePrices", ["0.5"])[0],
         "volume": m.get("volume", 0)}
        for m in markets[:30]
    ]

    prompt = f"""You are a quantitative arbitrage specialist in prediction markets.
You focus exclusively on 3 categories: CRYPTO, TECH/AI, and NICHE NEWS.

MARKETS (pre-filtered to our categories):
{json.dumps(market_data, indent=2)}

Find cross-market arbitrage: logically correlated markets that are priced inconsistently.

Category-specific arb patterns to look for:

CRYPTO arb:
- "Will BTC hit ?" markets at different strike prices that are inconsistently ordered
- Token launch markets vs parent chain markets (e.g. ETH ecosystem)
- Exchange listing markets vs price target markets for the same asset

TECH/AI arb:
- "Will X release product Y by date A?" priced higher than "by date B" (date A < date B)
- Competing product markets that are mutually exclusive but sum > 1
- Earnings surprise markets vs stock price target markets for same company

NICHE arb:
- Regional sub-events priced higher than aggregate (e.g. "Team wins game 3" > "Team wins series")
- Mutually exclusive award nominations summing > 1
- Sequential events where early must precede late

Respond ONLY with a JSON array. Each element:
{{
  "market_id": "...",
  "question": "...",
  "category": "crypto" or "tech_ai" or "niche_news",
  "outcome": "YES" or "NO",
  "current_price": <float>,
  "fair_value": <derived fair value>,
  "confidence": <float 0-1>,
  "rationale": "<explain the specific arb logic>"
}}
Return [] if no clear arbitrage found."""

    try:
        resp = await ai.messages.create(
            model="claude-opus-4-5",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(resp.content[0].text.strip())
        signals = []
        for item in data:
            s = Signal(
                agent="arb-scanner",
                market_id=str(item["market_id"]),
                question=str(item["question"]),
                outcome=str(item["outcome"]),
                current_price=float(item["current_price"]),
                fair_value=float(item["fair_value"]),
                confidence=float(item["confidence"]),
                rationale=str(item["rationale"]),
            )
            if s.confidence >= MIN_CONFIDENCE:
                signals.append(s)
        log.info(f"[arb-scanner]  {len(signals)} signals")
        return signals
    except Exception as e:
        log.error(f"[arb-scanner] error: {e}")
        return []

# ── Agent 3: crypto-edge (Binance signals → Polymarket markets, no LLM) ──────
async def crypto_edge_agent(markets: list[dict]) -> list[Signal]:
    """
    Fetches free Binance OHLCV data, computes composite directional signals
    (SMA crossover + momentum + mean reversion from Articles 4-5),
    then maps those signals to correlated Polymarket markets.

    No LLM call needed — pure quantitative signal generation.
    Runs as part of the LLM loop cycle.
    """
    if not BINANCE_AVAILABLE:
        return []

    try:
        # Run in executor so blocking requests don't block event loop
        loop = asyncio.get_event_loop()
        crypto_signals = await loop.run_in_executor(None, get_crypto_signals)
    except Exception as e:
        log.error(f"[crypto-edge] Binance fetch error: {e}")
        return []

    signals: list[Signal] = []

    for cs in crypto_signals:
        # Skip neutral or weak signals
        if abs(cs.composite) < 0.3:
            continue

        # Find Polymarket markets correlated with this asset
        relevant = find_relevant_markets(cs, markets)
        if not relevant:
            continue

        for market, relevance_score in relevant[:3]:   # top 3 matches per asset
            result = signal_to_market_bias(cs, market)
            if result is None:
                continue

            outcome, fair_value, rationale = result
            yes_price = float((market.get("outcomePrices") or ["0.5"])[0])
            current_price = yes_price if outcome == "YES" else (1 - yes_price)

            edge = abs(fair_value - current_price)
            if edge < 0.05:
                continue

            # Confidence scales with signal strength and relevance
            confidence = min(0.85, 0.60 + abs(cs.composite) * 0.2 + relevance_score * 0.05)

            signals.append(Signal(
                agent="crypto-edge",
                market_id=str(market.get("conditionId", market.get("id", ""))),
                question=str(market.get("question", "")),
                outcome=outcome,
                current_price=current_price,
                fair_value=fair_value,
                confidence=confidence,
                rationale=rationale,
            ))

    log.info(f"[crypto-edge]  {len(signals)} signals")
    return signals

# ── Agent 4: spread-farmer (pure Python, no LLM — runs on tight loop) ────────
def spread_farmer_agent(markets: list[dict]) -> list[Signal]:
    signals = []
    for m in markets:
        try:
            prices = m.get("outcomePrices", [])
            if len(prices) < 2:
                continue
            yes_p   = float(prices[0])
            no_p    = float(prices[1])
            spread  = abs(1.0 - (yes_p + no_p))
            volume  = float(m.get("volume", 0))

            # Sweet spot: wide spread + niche volume range
            # Avoid very high volume (efficiently priced) and near-zero (no exit)
            category = m.get("_category")
            niche_volume = 500 < volume < 50000
            if spread > 0.08 and niche_volume and category:
                outcome    = "YES" if yes_p < 0.5 else "NO"
                fair_value = 0.5
                # Slightly higher confidence for crypto where Binance data gives context
                conf = 0.68 if category == "crypto" else 0.63
                signals.append(Signal(
                    agent="spread-farmer",
                    market_id=str(m.get("conditionId", m.get("id", ""))),
                    question=str(m.get("question", "")),
                    outcome=outcome,
                    current_price=yes_p if outcome == "YES" else no_p,
                    fair_value=fair_value,
                    confidence=conf,
                    rationale=f"[{category}] spread={spread:.2f} vol=${volume:,.0f} — fading to fair value",
                ))
        except Exception:
            continue
    log.info(f"[spread-farmer] {len(signals)} signals")
    return signals

# ── Order execution ───────────────────────────────────────────────────────────
async def execute_trade(sig: Signal, size_usdc: float) -> dict:
    """
    DRY_RUN=1  → logs only, no real orders
    DRY_RUN=0  → places real FOK market order via py-clob-client
    """
    if DRY_RUN:
        result = {"status": "dry_run", "market_id": sig.market_id,
                  "outcome": sig.outcome, "size": size_usdc}
        log.info(f"[DRY RUN] {sig.agent} | {sig.outcome} ${size_usdc:.2f} | "
                 f"edge={sig.edge:+.3f} conf={sig.confidence:.0%} | {sig.question[:55]}")
        return result

    clob = get_clob()
    if not clob:
        log.error("CLOB client unavailable — order skipped")
        return {"status": "error", "reason": "no_clob_client"}

    try:
        side      = BUY if sig.outcome == "YES" else SELL
        order_args = OrderArgs(
            token_id=sig.market_id,
            price=sig.current_price,
            size=size_usdc,
            side=side,
        )
        order  = clob.create_market_order(order_args)
        result = clob.post_order(order, OrderType.FOK)   # Fill-or-Kill
        log.info(f"[TRADE] {sig.agent} | {sig.outcome} ${size_usdc:.2f} | "
                 f"order={result.get('orderID','?')} status={result.get('status','?')}")
        return result
    except Exception as e:
        log.error(f"[TRADE] execution failed: {e}")
        return {"status": "error", "reason": str(e)}

# ── Stop-loss guard ───────────────────────────────────────────────────────────
def check_stop_loss(bankroll: float, initial: float) -> bool:
    """Returns True if we should halt trading."""
    loss_pct = (initial - bankroll) / initial
    if loss_pct >= STOP_LOSS_PCT:
        log.critical(
            f"🛑 STOP-LOSS TRIGGERED — bankroll ${bankroll:.2f} "
            f"({loss_pct:.1%} loss from ${initial:.2f}). Halting."
        )
        return True
    return False

# ── Signal deduplication ──────────────────────────────────────────────────────
def deduplicate(signals: list[Signal]) -> list[Signal]:
    """Per market_id, keep the signal with highest confidence."""
    best: dict[str, Signal] = {}
    for s in signals:
        if s.market_id not in best or s.confidence > best[s.market_id].confidence:
            best[s.market_id] = s
    return list(best.values())

# ── LLM agent loop (news-edge + arb-scanner in parallel, every 10 s) ─────────
async def llm_loop(bankroll_ref: list[float]):
    initial = bankroll_ref[0]
    log.info(f"LLM loop started (interval={LLM_LOOP_INTERVAL}s)")

    async with aiohttp.ClientSession() as session:
        while True:
            cycle_start = time.monotonic()

            if check_stop_loss(bankroll_ref[0], initial):
                return

            try:
                markets = await fetch_markets(session, limit=50)
                if not markets:
                    log.warning("No markets fetched — skipping LLM cycle")
                else:
                    # Run all LLM + quant agents in parallel
                    # crypto-edge uses Binance (free, no LLM) but runs concurrently
                    results = await asyncio.gather(
                        news_edge_agent(session, markets),
                        arb_scanner_agent(session, markets),
                        crypto_edge_agent(markets),
                        return_exceptions=True,
                    )

                    all_signals: list[Signal] = []
                    for r in results:
                        if isinstance(r, list):
                            all_signals.extend(r)
                        elif isinstance(r, Exception):
                            log.error(f"Agent exception: {r}")

                    for sig in deduplicate(all_signals):
                        bet = size_bet(bankroll_ref[0], sig)
                        if bet < 1.0:
                            db_insert_signal(sig, bet, status="skipped_small")
                            continue

                        sig_id = db_insert_signal(sig, bet)
                        result = await execute_trade(sig, bet)

                        if result.get("status") not in ("error",):
                            db_insert_fill(sig_id, sig, bet,
                                           result.get("orderID", "dry"),
                                           json.dumps(result))

            except Exception as e:
                log.error(f"LLM loop error: {e}")

            elapsed = time.monotonic() - cycle_start
            sleep_for = max(0, LLM_LOOP_INTERVAL - elapsed)
            log.info(f"LLM cycle done in {elapsed:.1f}s — sleeping {sleep_for:.1f}s")
            await asyncio.sleep(sleep_for)

# ── Spread-farmer loop (pure Python, every 3 s) ───────────────────────────────
async def spread_loop(bankroll_ref: list[float]):
    initial = bankroll_ref[0]
    log.info(f"Spread loop started (interval={SPREAD_LOOP_INTERVAL}s)")

    async with aiohttp.ClientSession() as session:
        while True:
            if check_stop_loss(bankroll_ref[0], initial):
                return

            try:
                markets = await fetch_markets(session, limit=50)
                signals = spread_farmer_agent(markets)

                for sig in deduplicate(signals):
                    bet = size_bet(bankroll_ref[0], sig)
                    if bet < 1.0:
                        continue
                    sig_id = db_insert_signal(sig, bet)
                    result = await execute_trade(sig, bet)
                    if result.get("status") not in ("error",):
                        db_insert_fill(sig_id, sig, bet,
                                       result.get("orderID", "dry"),
                                       json.dumps(result))
            except Exception as e:
                log.error(f"Spread loop error: {e}")

            await asyncio.sleep(SPREAD_LOOP_INTERVAL)

# ── Bankroll logger (every 60 s) ──────────────────────────────────────────────
async def bankroll_logger(bankroll_ref: list[float]):
    while True:
        db_log_bankroll(bankroll_ref[0])
        await asyncio.sleep(60)

# ── Entrypoint ────────────────────────────────────────────────────────────────
async def main():
    log.info("=" * 60)
    log.info("  Polymarket Multi-Agent Orchestrator")
    log.info(f"  Mode      : {'DRY RUN 🟡' if DRY_RUN else 'LIVE TRADING 🔴'}")
    log.info(f"  Bankroll  : ${BANK_ROLL:.2f} USDC")
    log.info(f"  Stop-loss : {STOP_LOSS_PCT:.0%}")
    log.info(f"  LLM loop  : every {LLM_LOOP_INTERVAL}s")
    log.info(f"  Spread lp : every {SPREAD_LOOP_INTERVAL}s")
    log.info(f"  Max Kelly : {MAX_KELLY_FRAC:.0%}")
    log.info(f"  Binance   : {'enabled (crypto-edge agent active)' if BINANCE_AVAILABLE else 'disabled (pip install numpy pandas)'}")
    log.info("=" * 60)

    if not ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY not set — exiting"); return
    if not DRY_RUN and not WALLET_PRIVATE_KEY:
        log.error("WALLET_PRIVATE_KEY required for live trading — exiting"); return
    if not DRY_RUN and not CLOB_AVAILABLE:
        log.error("py-clob-client not installed — run: pip install py-clob-client"); return

    init_db()
    get_clob()   # warm up CLOB connection

    # Shared mutable bankroll (passed by reference via list)
    bankroll_ref = [BANK_ROLL]

    await asyncio.gather(
        llm_loop(bankroll_ref),
        spread_loop(bankroll_ref),
        bankroll_logger(bankroll_ref),
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Orchestrator stopped by user.")