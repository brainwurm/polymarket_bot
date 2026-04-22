import { useState, useEffect, useRef } from "react";

// ── Config ───────────────────────────────────────────────────────────────────
const AGENTS = [
    { name: "news-edge", color: "#00ff88", baseP: 1053 },
    { name: "arb-scanner", color: "#00cfff", baseP: 828 },
    { name: "spread-farmer", color: "#ffaa00", baseP: 652 },
    { name: "copy-trader", color: "#cc88ff", baseP: 457 },
    { name: "whale-tracker", color: "#00cfff", baseP: 386 },
    { name: "pump-sniper", color: "#ffaa00", baseP: 218 },
    { name: "liq-hunter", color: "#ff88aa", baseP: 227 },
    { name: "dex-rebal", color: "#ff4444", baseP: -68 },
];

const AGENT_COLORS = Object.fromEntries(AGENTS.map(a => [a.name, a.color]));
const EXCHANGES = ["Polymarket", "Binance", "Phantom", "CoinMktCap", "Helios", "Done", "Browser"];
const ACTIONS = [
    "monitorWallet(0xa8) → $126 detected",
    "signTx(swap 2.3 SOL) → confirmed",
    "getFillrate(BNB 0.1) → crowded",
    "getQuote(ETH) → $3,047.21",
    "screenshot(polymarket) → captured",
    "getFundings() → 8.13% crowded",
    "printOrder(BNB 0.2) → filled qty.58",
    "getCandles(BTC 5m) → 48 candles",
    "run_query(last 30 news)",
    "scrapePools() → 3 signals found",
    "signTx(swap 2.3 SOL) → confirmed",
];
const POS_DELTAS = ["+$44", "+$18", "+$8", "+$58", "+$22", "+$18", "+$28", "+$8", "+$30", "+$12"];
const NEG_DELTAS = ["-$12", "-$38", "-$32", "-$58", "-$4", "-$8", "-$18"];
const FILLS_INIT = [
    { agent: "pump-sniper", side: "SELL", size: 57, price: 35.18, delta: "+$35" },
    { agent: "news-edge", side: "BUY", size: 109, price: 62.01, delta: "+$62" },
    { agent: "pump-sniper", side: "BUY", size: 88, price: 31.12, delta: "+$31" },
    { agent: "news-edge", side: "SELL", size: 177, price: 7.65, delta: "+$7" },
    { agent: "arb-scanner", side: "BUY", size: 66, price: 72.77, delta: "+$72" },
    { agent: "pump-sniper", side: "SELL", size: 113, price: 58.35, delta: "-$58" },
    { agent: "arb-scanner", side: "BUY", size: 88, price: 25.75, delta: "+$25" },
    { agent: "spread-farmer", side: "BUY", size: 123, price: 75.93, delta: "+$75" },
    { agent: "whale-tracker", side: "BUY", size: 44, price: 3.02, delta: "+$3" },
    { agent: "arb-scanner", side: "SELL", size: 150, price: 38.64, delta: "-$38" },
];

// ── Helpers ──────────────────────────────────────────────────────────────────
const rnd = (lo, hi) => Math.random() * (hi - lo) + lo;
const pick = arr => arr[Math.floor(Math.random() * arr.length)];

function buildPnLCurve(n = 80) {
    const pts = [0];
    for (let i = 1; i < n; i++) {
        const drift = (i / n) * 90;
        pts.push(Math.max(0, pts[i - 1] + drift + rnd(-18, 30)));
    }
    return pts;
}

function nowStr(offset = 0) {
    const d = new Date(Date.now() + offset * 1000);
    return `11:47:${String(d.getSeconds()).padStart(2, "0")}`;
}

function genLog(i) {
    const agent = pick(AGENTS).name;
    const pos = Math.random() > 0.28;
    return { time: nowStr(i), agent, exch: pick(EXCHANGES), action: pick(ACTIONS), delta: pos ? pick(POS_DELTAS) : pick(NEG_DELTAS), pos };
}

function genChain(i) {
    const types = ["TRANSFER", "PROFIT", "SWAP"];
    const t = pick(types);
    return { time: nowStr(i), type: t, detail: `$${rnd(50, 500).toFixed(2)}`, pos: t === "PROFIT" };
}

// ── Blink cursor ─────────────────────────────────────────────────────────────
function Blink({ color = "#00ff88" }) {
    const [on, setOn] = useState(true);
    useEffect(() => { const t = setInterval(() => setOn(v => !v), 650); return () => clearInterval(t); }, []);
    return <span style={{ color, opacity: on ? 1 : 0 }}>█</span>;
}

// ── PnL Sparkline ─────────────────────────────────────────────────────────────
function PnLChart({ data, width, height }) {
    if (data.length < 2) return null;
    const max = Math.max(...data, 10);
    const pts = data.map((v, i) => {
        const x = (i / (data.length - 1)) * width;
        const y = height - (v / max) * height * 0.92;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    const lx = width, ly = height - (data[data.length - 1] / max) * height * 0.92;
    const fill = `0,${height} ${pts} ${width},${height}`;
    return (
        <svg width={width} height={height} style={{ display: "block", overflow: "visible" }}>
            <defs>
                <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#00ff88" stopOpacity="0.28" />
                    <stop offset="100%" stopColor="#00ff88" stopOpacity="0.01" />
                </linearGradient>
            </defs>
            <polygon points={fill} fill="url(#g)" />
            <polyline points={pts} fill="none" stroke="#00ff88" strokeWidth="1.5" />
            <circle cx={lx} cy={ly} r="3" fill="#00ff88" />
        </svg>
    );
}

// ── Agent sidebar row ─────────────────────────────────────────────────────────
function AgentRow({ name, color, pnl, trades, wr }) {
    const pos = pnl >= 0;
    const bar = Math.min(100, Math.abs(pnl) / 14);
    return (
        <div style={{ marginBottom: 9 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                <span style={{ color, fontSize: 8, letterSpacing: 0.5, fontWeight: 700 }}>● {name}</span>
                <span style={{ color: pos ? "#00ff88" : "#ff4444", fontSize: 8, fontWeight: 700 }}>
                    {pos ? "+" : ""}${Math.abs(pnl).toFixed(2)}
                </span>
            </div>
            <div style={{ background: "#0a0a0a", height: 3, borderRadius: 2, marginBottom: 2 }}>
                <div style={{ height: "100%", borderRadius: 2, width: `${bar}%`, background: pos ? color : "#ff4444", maxWidth: "100%" }} />
            </div>
            <div style={{ fontSize: 7, color: "#2a2a2a" }}>{trades} trades · {(wr * 100).toFixed(0)}% wr · active</div>
        </div>
    );
}

// ── Stat row ──────────────────────────────────────────────────────────────────
function SR({ label, value, color = "#777" }) {
    return (
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span style={{ fontSize: 8, color: "#333", letterSpacing: 1 }}>{label}</span>
            <span style={{ fontSize: 8, color, fontWeight: 700 }}>{value}</span>
        </div>
    );
}

// ── Main ──────────────────────────────────────────────────────────────────────
export default function App() {
    const [tick, setTick] = useState(0);
    const [live, setLive] = useState(true);
    const [pnl, setPnl] = useState(4348.04);
    const [trades, setTrades] = useState(242);
    const [calls, setCalls] = useState(242);
    const [pnlCurve, setPnlCurve] = useState(buildPnLCurve(80));
    const [logs, setLogs] = useState(() => Array.from({ length: 14 }, (_, i) => genLog(i)));
    const [chain, setChain] = useState(() => Array.from({ length: 8 }, (_, i) => genChain(i)));
    const [fills, setFills] = useState(FILLS_INIT);
    const [agents, setAgents] = useState(() => AGENTS.map(a => ({
        ...a, pnl: a.baseP + rnd(-10, 10), trades: Math.floor(rnd(28, 70)), wr: rnd(0.55, 0.73),
    })));

    useEffect(() => {
        if (!live) return;
        const t = setInterval(() => {
            setTick(v => v + 1);
            setPnl(v => +(v + rnd(0, 75)).toFixed(2));
            setTrades(v => v + (Math.random() > 0.55 ? 1 : 0));
            setCalls(v => v + Math.floor(rnd(0, 3)));
            setPnlCurve(prev => { const last = prev[prev.length - 1]; return [...prev, Math.max(0, last + rnd(0, 55))].slice(-120); });
            setLogs(prev => [genLog(tick), ...prev].slice(0, 20));
            setChain(prev => [genChain(tick), ...prev].slice(0, 14));
            setFills(prev => {
                const f = { agent: pick(AGENTS).name, side: Math.random() > 0.5 ? "BUY" : "SELL", size: Math.floor(rnd(20, 200)), price: +rnd(2, 100).toFixed(2), delta: (Math.random() > 0.25 ? "+" : "-") + "$" + Math.floor(rnd(2, 80)) };
                return [f, ...prev].slice(0, 12);
            });
            setAgents(prev => prev.map(a => ({ ...a, pnl: a.name === "dex-rebal" ? Math.min(-5, a.pnl + rnd(-4, 1)) : a.pnl + rnd(0, 12), trades: a.trades + (Math.random() > 0.7 ? 1 : 0) })));
        }, 1900);
        return () => clearInterval(t);
    }, [live, tick]);

    const ret = ((pnl / 25) * 100).toFixed(1);
    const best = agents.reduce((b, a) => a.pnl > b.pnl ? a : b).name;

    return (
        <div style={{ background: "#000", minHeight: "100vh", color: "#ccc", padding: 10, fontFamily: "'JetBrains Mono','Fira Code','Courier New',monospace", fontSize: 10, boxSizing: "border-box" }}>

            {/* ── Top status bar ── */}
            <div style={{ display: "flex", alignItems: "center", gap: 20, marginBottom: 8, borderBottom: "1px solid #111", paddingBottom: 6 }}>
                <div>
                    <div style={{ fontSize: 7, color: "#2a2a2a", letterSpacing: 3 }}>MI AGENT ORCHESTRATOR</div>
                    <div style={{ fontSize: 7, color: "#333" }}>AGENT P&L // SESSION · <span style={{ color: "#555" }}>+LIVE</span></div>
                </div>
                <div style={{ fontSize: 7, color: "#222" }}>POLYMARKET · LIVE AGENT FRAMEWORK<br />IN-RUN ANALYZER</div>
                <div style={{ marginLeft: "auto", display: "flex", gap: 24, alignItems: "center" }}>
                    {[
                        ["AGENTS", <><Blink />8</>, "#00ff88"],
                        ["TOOLS", calls, "#aaa"],
                        ["CALLS", trades, "#aaa"],
                        ["WIN RATE", "60.4%", "#00ff88"],
                        ["SHARPE", "2.45K", "#aaa"],
                    ].map(([label, val, color]) => (
                        <div key={label} style={{ textAlign: "center" }}>
                            <div style={{ fontSize: 6, color: "#222", letterSpacing: 2 }}>{label}</div>
                            <div style={{ fontSize: 11, color, fontWeight: 700, lineHeight: 1.2 }}>{val}</div>
                        </div>
                    ))}
                    <div style={{ width: 1, background: "#111", height: 24 }} />
                    <div style={{ fontSize: 9, color: "#00ff88" }}>11:47:37</div>
                    <button onClick={() => setLive(v => !v)} style={{ background: "none", border: "1px solid #1a1a1a", color: "#444", padding: "3px 10px", cursor: "pointer", fontSize: 7, fontFamily: "inherit", letterSpacing: 2 }}>
                        {live ? "PAUSE" : "RESUME"}
                    </button>
                </div>
            </div>

            {/* ── Main grid ── */}
            <div style={{ display: "grid", gridTemplateColumns: "165px 1fr 185px", gap: 8, marginBottom: 8 }}>

                {/* LEFT: Agent sidebar */}
                <div>
                    <div style={{ fontSize: 7, color: "#222", letterSpacing: 3, marginBottom: 8 }}>AGENT P&L // SESSION</div>
                    {agents.map(a => <AgentRow key={a.name} {...a} />)}
                </div>

                {/* CENTER: PnL chart */}
                <div style={{ border: "1px solid #0d0d0d", padding: "10px 12px" }}>
                    <div style={{ fontSize: 7, color: "#222", letterSpacing: 3 }}>CUMULATIVE P&L // ALL AGENTS</div>
                    <div style={{ fontSize: 7, color: "#333", marginBottom: 6 }}>STARTED 24h · OVERNIGHT RUN · {trades} TRADES</div>
                    <div style={{ fontSize: 26, color: "#00ff88", fontWeight: 700, marginBottom: 8, lineHeight: 1 }}>
                        +${pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                    <PnLChart data={pnlCurve} width={380} height={130} />
                </div>

                {/* RIGHT: Stats */}
                <div style={{ border: "1px solid #0d0d0d", padding: "10px 12px" }}>
                    <div style={{ fontSize: 7, color: "#222", letterSpacing: 3, marginBottom: 6 }}>PERFORMANCE STATS</div>
                    <div style={{ fontSize: 14, color: "#00ff88", fontWeight: 700, marginBottom: 10 }}>+${pnl.toLocaleString()}</div>
                    <SR label="TOTAL P&L" value={`+$${pnl.toLocaleString()}`} color="#00ff88" />
                    <SR label="SEED CAPITAL" value="$25.00" color="#777" />
                    <SR label="RETURN" value={`+${ret}%`} color="#00ff88" />
                    <SR label="TOTAL TRADES" value={trades} color="#777" />
                    <SR label="WIN RATE" value="60.4%" color="#00ff88" />
                    <SR label="SHARPE RATIO" value="2.41" color="#777" />
                    <SR label="MAX DRAWDOWN" value="-$48.20" color="#ff4444" />
                    <div style={{ borderTop: "1px solid #0d0d0d", margin: "8px 0" }} />
                    <SR label="BEST AGENT" value={best} color="#00ff88" />
                    <SR label="WALLETS ANALYZED" value="10,043" color="#777" />
                    <SR label="TOKENS/SEC" value="12,760" color="#777" />
                    <SR label="FILL RATE" value="97.3%" color="#00ff88" />
                    <SR label="UPTIME" value="08:24:38" color="#777" />
                </div>
            </div>

            {/* ── Bottom grid ── */}
            <div style={{ display: "grid", gridTemplateColumns: "165px 1fr 185px", gap: 8 }}>

                {/* LEFT: Top wallets */}
                <div>
                    <div style={{ fontSize: 7, color: "#222", letterSpacing: 3, marginBottom: 6 }}>TOP WALLETS // COPIED</div>
                    {[
                        { addr: "0x8b27bc92bf8...", val: "$225,950", trades: 36, score: 90 },
                        { addr: "0x6fa33e0023a...", val: "$110,440", trades: 42, score: 88 },
                        { addr: "0x2e21fc1b0a...", val: "$84,200", trades: 28, score: 82 },
                        { addr: "0x2e21fc1b0a...", val: "$61,800", trades: 19, score: 76 },
                        { addr: "0xf6a8d221fc0...", val: "$44,380", trades: 31, score: 74 },
                    ].map((w, i) => (
                        <div key={i} style={{ marginBottom: 7, paddingBottom: 5, borderBottom: "1px solid #080808" }}>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                                <span style={{ color: "#00cfff", fontSize: 8 }}>{w.addr}</span>
                                <span style={{ color: "#00ff88", fontSize: 8 }}>{w.val}</span>
                            </div>
                            <div style={{ fontSize: 7, color: "#252525", marginBottom: 3 }}>{w.trades} trades · score {w.score}</div>
                            <div style={{ background: "#080808", height: 2 }}>
                                <div style={{ height: "100%", width: `${w.score}%`, background: "#00cfff20" }} />
                            </div>
                        </div>
                    ))}
                </div>

                {/* CENTER: Execution log */}
                <div style={{ border: "1px solid #0d0d0d" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 10px", borderBottom: "1px solid #0d0d0d" }}>
                        <span style={{ fontSize: 7, color: "#222", letterSpacing: 3 }}>EXECUTION LOG // TOOL CALLS</span>
                        <span style={{ fontSize: 7, color: "#333" }}>{calls} CALLS</span>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "52px 88px 76px 1fr 48px", gap: 4, padding: "4px 8px", borderBottom: "1px solid #080808" }}>
                        {["TIME", "AGENT", "EXCHANGE", "ACTION", "RESULT"].map(h => (
                            <span key={h} style={{ fontSize: 6, color: "#1c1c1c", letterSpacing: 2 }}>{h}</span>
                        ))}
                    </div>
                    <div style={{ height: 158, overflowY: "auto" }}>
                        {logs.map((l, i) => (
                            <div key={i} style={{ display: "grid", gridTemplateColumns: "52px 88px 76px 1fr 48px", gap: 4, padding: "2px 8px", background: i % 2 === 0 ? "#030303" : "#000", borderBottom: "1px solid #080808" }}>
                                <span style={{ fontSize: 7, color: "#2a2a2a" }}>{l.time}</span>
                                <span style={{ fontSize: 7, color: AGENT_COLORS[l.agent] || "#aaa" }}>{l.agent}</span>
                                <span style={{ fontSize: 7, color: "#444" }}>{l.exch}</span>
                                <span style={{ fontSize: 7, color: "#383838", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{l.action}</span>
                                <span style={{ fontSize: 7, color: l.pos ? "#00ff88" : "#ff4444", textAlign: "right" }}>{l.delta}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* RIGHT: Fills + On-chain */}
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {/* Fills */}
                    <div style={{ border: "1px solid #0d0d0d", flex: 1 }}>
                        <div style={{ fontSize: 7, color: "#222", letterSpacing: 3, padding: "5px 8px", borderBottom: "1px solid #0d0d0d" }}>RECENT FILLS</div>
                        <div style={{ display: "grid", gridTemplateColumns: "76px 32px 36px 44px 38px", gap: 2, padding: "3px 8px", borderBottom: "1px solid #080808" }}>
                            {["AGENT", "SIDE", "SIZE", "PRICE", "Δ"].map(h => (
                                <span key={h} style={{ fontSize: 6, color: "#1c1c1c", letterSpacing: 1 }}>{h}</span>
                            ))}
                        </div>
                        <div style={{ overflowY: "auto", maxHeight: 100 }}>
                            {fills.map((f, i) => (
                                <div key={i} style={{ display: "grid", gridTemplateColumns: "76px 32px 36px 44px 38px", gap: 2, padding: "2px 8px", background: i % 2 === 0 ? "#030303" : "#000" }}>
                                    <span style={{ fontSize: 7, color: AGENT_COLORS[f.agent] || "#aaa", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f.agent}</span>
                                    <span style={{ fontSize: 7, color: f.side === "BUY" ? "#00ff88" : "#ff4444" }}>{f.side}</span>
                                    <span style={{ fontSize: 7, color: "#444" }}>${f.size}</span>
                                    <span style={{ fontSize: 7, color: "#383838" }}>${f.price}</span>
                                    <span style={{ fontSize: 7, color: f.delta.startsWith("+") ? "#00ff88" : "#ff4444" }}>{f.delta}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* On-chain feed */}
                    <div style={{ border: "1px solid #0d0d0d" }}>
                        <div style={{ fontSize: 7, color: "#222", letterSpacing: 3, padding: "5px 8px", borderBottom: "1px solid #0d0d0d" }}>ON-CHAIN FEED</div>
                        <div style={{ overflowY: "auto", maxHeight: 92 }}>
                            {chain.map((l, i) => (
                                <div key={i} style={{ display: "flex", gap: 6, padding: "2px 8px", background: i % 2 === 0 ? "#030303" : "#000" }}>
                                    <span style={{ fontSize: 7, color: "#252525", flexShrink: 0 }}>{l.time}</span>
                                    <span style={{ fontSize: 7, fontWeight: 700, flexShrink: 0, color: l.type === "PROFIT" ? "#00ff88" : l.type === "TRANSFER" ? "#00cfff" : "#ffaa00" }}>{l.type}</span>
                                    <span style={{ fontSize: 7, color: l.pos ? "#00ff88" : "#ff4444" }}>{l.detail}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* ── Status bar ── */}
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8, paddingTop: 5, borderTop: "1px solid #0d0d0d", fontSize: 7, color: "#1c1c1c", letterSpacing: 2 }}>
                <span>ETH/USD 0.52 +0.01</span>
                <span>BTC/USD 0.41 +0.05</span>
                <span>SOL/USD 0.30 +0.08</span>
                <span>BNB/USD 0.71 +0.01</span>
            </div>
        </div>
    );
}