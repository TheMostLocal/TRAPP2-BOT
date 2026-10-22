#!/usr/bin/env python3
"""
bot_runner.py — autonomous daily trading for the Valuatio bot.

Runs as a GitHub Action so the bot makes trades WITHOUT the app being open. Each
run it:

  1. READS state         — bot_training_data.json from this repo (bankroll, open
                           positions, learned weights, equity curve, full history).
  2. READS the market    — master.json (universe + latest prices) from TRAPP2 /
                           TRAPP2-2, and daily history from data/history/<T>.json.
  3. MANAGES positions   — marks open positions to the latest price; closes any
                           that hit their stop, target, or horizon, freezing
                           realized P&L from shares (exactly like the app's z35).
  4. SCORES + TRADES     — scores candidates with a faithful subset of the app's
                           signal engine (trend, momentum, mean-reversion, grade),
                           weighted by the bot's LEARNED weights, then opens the
                           best few within risk + cash limits. Never re-buys a
                           name already held.
  5. WRITES state        — recomputes performance + equity point, writes
                           bot_training_data.json back (the workflow commits it).
  6. Supabase            — the workflow's existing sync step mirrors trades after.

HONEST SCOPE
------------
The scoring is a faithful subset of the app's brain that now includes: trend
(SMA20/60), momentum (~3-month), mean-reversion (RSI), PEER-SECTOR GRADE (sector-
relative grade percentile blended with sector-ETF momentum), and REGIME (SPY
trend + realized vol → risk-on/off/choppy, which both tilts the signal weights
and applies a sector/beta regime-grade signal) — all weighted by the bot's
LEARNED weights. Still not ported: options IV, explicit cross-asset, the macro-
tab "quad", and Fed expectations. The runner remains long-only by default (no
shorts/options/leverage) unless turned on. With peer-grade + regime in, the
runner's picks now converge much closer to the app's; it stays deliberately
conservative on sizing and trade count. Everything is recorded in the same schema
the app reads.

SAFETY KNOBS (env vars, all optional)
  RUNNER_MAX_NEW_TRADES   default 2    new positions opened per run
  RUNNER_MAX_POSITIONS    default 15   total open positions allowed
  RUNNER_POSITION_PCT     default 5    % of bankroll per new position
  RUNNER_CASH_RESERVE_PCT default 20   % of bankroll kept as cash, never deployed
  RUNNER_MIN_SCORE        default 0.35 minimum signed score to open a long
  RUNNER_DRY_RUN          default off  set to "1" to compute + log but NOT write

Stdlib only (urllib + json + math). No pip installs.
"""
import json
import math
import os
import trade_sync as ts
import market_hours as mh   # z121: the NYSE regular session (ET) - the only time the bot trades
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alt_research  # noqa: E402  research-driven gate for crypto & futures
import trade_book as tb  # noqa: E402  z138: horizons (short / medium / long), options, catalysts, position reviews

# Repo owner — resolved at runtime so the pipeline follows the repos to any
# GitHub account. Actions sets GITHUB_REPOSITORY_OWNER automatically;
# VALUATIO_OWNER (repo variable/env) overrides; TheMostLocal is the fallback.
_GH_OWNER = (__import__("os").environ.get("VALUATIO_OWNER")
             or __import__("os").environ.get("GITHUB_REPOSITORY_OWNER")
             or "TheMostLocal").strip()

ROOT = Path(__file__).resolve().parent.parent
# Canonical state lives in data/. The runner previously read the ROOT copy, which
# was stale (bankroll reset to $100k, only a couple of trades) — so it ignored the
# real book. Read data/ first; migrate from the legacy root file if data/ is absent.
STATE_FILE = ROOT / "data" / "bot_training_data.json"
_LEGACY_STATE = ROOT / "bot_training_data.json"

RAW = f"https://raw.githubusercontent.com/{_GH_OWNER}"
# Repos the bot SCANS for new ideas (strategy unchanged) ...
UNIVERSE_SOURCES = [
    f"{RAW}/TRAPP2/main/data/master.json",
    f"{RAW}/TRAPP2-2/main/data/master.json",
]
# ... and repos it only PRICES, so a position opened from any book (the app's
# bot trades foreign listings too, e.g. 000660.KS) is marked and managed.
PRICE_ONLY_SOURCES = [
    f"{RAW}/TRAPP2-1/main/data/master.json",
    f"{RAW}/TRAPP2-3/main/data/master.json",
]
HISTORY_BASE = {
    "TRAPP2": f"{RAW}/TRAPP2/main/data/history",
    "TRAPP2-2": f"{RAW}/TRAPP2-2/main/data/history",
    "TRAPP2-1": f"{RAW}/TRAPP2-1/main/data/history",
    "TRAPP2-3": f"{RAW}/TRAPP2-3/main/data/history",
}
FX_URL = f"{RAW}/TRAPP2-1/main/data/fx/rates.json"
GRADES_URL = f"{RAW}/TRAPP2-ANALYTICS/main/data/research_grades.json"
# Sector ETFs live in TRAPP2-1's history; used for sector-momentum + regime.
ETF_HISTORY_BASE = f"{RAW}/TRAPP2-1/main/data/history"
SPY_HISTORY_URL = f"{RAW}/TRAPP2-1/main/data/history/SPY.json"
# The REAL FRED-derived quad, written by TRAPP2-1's macro-quad workflow. If
# present the runner uses it; otherwise it falls back to the market-implied proxy.
MACRO_QUAD_URL = f"{RAW}/TRAPP2-1/main/data/macro/quad.json"
# Option chains live in TRAPP2 (data/options/<TICKER>.json + manifest.json). The
# runner reads them for the same forward-looking IV/skew/term signals the app
# uses, plus a market-wide options read (index IV → vol regime + market tilt).
OPTIONS_BASE = f"{RAW}/TRAPP2-OPTIONS/main/data/options"          # z138: the options repo (46 chains, Greeks); legacy TRAPP2 as fallback
OPTIONS_BASE_LEGACY = f"{RAW}/TRAPP2/main/data/options"
EARNINGS_CAL_URL = f"{RAW}/TRAPP2-ANALYTICS/main/data/earnings/calendar.json"   # z138: catalysts
OPTIONS_MANIFEST_URL = f"{OPTIONS_BASE}/manifest.json"
# Index proxies whose ATM IV defines the market's expected move (VIX-like).
MKT_OPT_INDEXES = ["SPY", "QQQ", "IWM"]

# Cross-asset baskets (mirrors extractCrossAssetSignals). The runner reads the
# latest daily return of each from TRAPP2-1 history.
XASSET_BASKETS = {
    "dollar":     ["UUP", "DX=F"],
    "safeHaven":  ["GLD", "TLT", "^VIX"],     # gold + bonds + VIX (flight to safety)
    "industrial": ["CPER", "XLB", "PICK"],    # copper / materials / mining
    "energy":     ["USO", "XLE", "AMLP"],
    "indices":    ["^GSPC", "^IXIC", "^FTSE", "^N225", "^GDAXI", "^HSI"],
}

# SECTOR_ETFS quad favoring (mirrors the app): which quad (1-4) each sector ETF
# favors / is hurt by. Quads: 1=Growth↑Infl↓ (Goldilocks), 2=Growth↑Infl↑
# (Reflation), 3=Growth↓Infl↑ (Stagflation), 4=Growth↓Infl↓ (Deflation).
SECTOR_ETF_QUAD = {
    "XLK":  {"favors": [1, 2, 3], "hurts": [4]},
    "XLY":  {"favors": [1, 2],    "hurts": [3, 4]},
    "XLI":  {"favors": [1, 2],    "hurts": [3]},
    "XLF":  {"favors": [2],       "hurts": [3, 4]},
    "XLE":  {"favors": [2, 3],    "hurts": [1]},
    "XLP":  {"favors": [4],       "hurts": [1]},
    "XLV":  {"favors": [4],       "hurts": []},
    "XLU":  {"favors": [3, 4],    "hurts": [1]},
    "XLRE": {"favors": [3, 4],    "hurts": []},
    "XLB":  {"favors": [2, 3],    "hurts": [4]},
    "XLC":  {"favors": [1, 2],    "hurts": [4]},
}

# Equity sector name → sector ETF (mirrors the app's SECTOR_TO_ETF).
SECTOR_TO_ETF = {
    "Technology": "XLK",
    "Consumer Cyclical": "XLY", "Consumer Discretionary": "XLY",
    "Industrials": "XLI", "Industrial": "XLI",
    "Financial Services": "XLF", "Financials": "XLF", "Financial": "XLF",
    "Energy": "XLE",
    "Consumer Defensive": "XLP", "Consumer Staples": "XLP",
    "Healthcare": "XLV", "Health Care": "XLV",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB", "Materials": "XLB",
    "Communication Services": "XLC",
}

STARTING_BANKROLL = 100000.0

# ---- safety knobs ----
def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)

# ---- trading MODE -------------------------------------------------------------
# explore   (default now): QUANTITY over quality - more, smaller, still-analysed
#           trades to collect outcome data the bot learns from. Bad trades are
#           expected and useful; nothing opens without the full signal blend +
#           a fired trigger (crypto/futures also need their research case).
# precision (later): the original stricter, fewer/larger-trade settings.
# Any RUNNER_* repo Variable still overrides the mode's default.
RUNNER_MODE = (os.environ.get("RUNNER_MODE") or "explore").strip().lower()
if RUNNER_MODE not in ("explore", "precision"):
    RUNNER_MODE = "explore"
_MODE = {
    "explore":   {"max_new": 6, "max_pos": 40, "pos_pct": 3, "reserve": 15, "min_score": 0.25,
                  "min_size_pct": 0.5, "pattern_nudge": 0.05, "same_primary": 2},
    "precision": {"max_new": 2, "max_pos": 25, "pos_pct": 5, "reserve": 20, "min_score": 0.35,
                  "min_size_pct": 1.0, "pattern_nudge": 0.0, "same_primary": 99},
}[RUNNER_MODE]
MAX_NEW_TRADES = int(_envf("RUNNER_MAX_NEW_TRADES", _MODE["max_new"]))
MAX_POSITIONS = int(_envf("RUNNER_MAX_POSITIONS", _MODE["max_pos"]))
MIN_POSITION_PCT = _envf("RUNNER_MIN_POSITION_PCT", _MODE["min_size_pct"]) / 100.0
PATTERN_NUDGE = _envf("RUNNER_PATTERN_NUDGE", _MODE["pattern_nudge"])   # see open_new_trades
MAX_SAME_PRIMARY = int(_envf("RUNNER_MAX_SAME_PRIMARY", _MODE["same_primary"]))
FACTOR_CAP = _envf("RUNNER_FACTOR_CAP", 0.25)        # max share of the blend any one signal can carry

# ---- confidence, triggers, and horizon-aware exits -------------------------
# The runner classifies every position as SWING (momentum, tight exits) or CORE
# (long-term / value-like, ride it). Which one is decided by CONFIDENCE (signal
# agreement + strength + breadth) and whether the blend leans on slow, fundamental
# signals. Exit rules then differ by class — a long-term winner is TRIMMED, not
# force-closed, and is never dumped on a fixed calendar.
CORE_CONF      = _envf("RUNNER_CORE_CONF", 0.62)        # min confidence to hold as long-term/core
TRIGGER_LEVEL  = _envf("RUNNER_TRIGGER_LEVEL", 0.33)    # an aligned signal this strong "fires"
MIN_TRIGGERS   = int(_envf("RUNNER_MIN_TRIGGERS", 1))   # concrete triggers needed to open

SWING_STOP     = _envf("RUNNER_SWING_STOP", 8) / 100.0      # -8%  swing stop
SWING_TARGET   = _envf("RUNNER_SWING_TARGET", 16) / 100.0   # +16% swing target
SWING_HORIZON  = int(_envf("RUNNER_SWING_HORIZON", 21))     # base swing horizon (days), conf-scaled

CORE_HARD_STOP = _envf("RUNNER_CORE_STOP", 25) / 100.0      # -25% disaster stop for core
CORE_TRAIL     = _envf("RUNNER_CORE_TRAIL", 18) / 100.0     # 18% trailing from peak (once in profit)
CORE_EXIT_SCORE= _envf("RUNNER_CORE_EXIT_SCORE", 0.12)      # thesis-break: signal flips this far against
CORE_TRIM_TIERS= [0.25, 0.50, 1.00]                         # +25/50/100% → trim tiers (long only)
CORE_TRIM_FRAC = _envf("RUNNER_CORE_TRIM_FRAC", 25) / 100.0 # trim this fraction of shares per tier

LONG_TERM_SIGNALS = ("fundamentals", "researchGrade", "health", "peerGrade", "regimeGrade", "fed", "crossAsset")
SWING_SIGNALS     = ("trend", "momentum", "meanReversion", "optionsIV", "optionsMarket")
POSITION_PCT = _envf("RUNNER_POSITION_PCT", _MODE["pos_pct"]) / 100.0
CASH_RESERVE_PCT = _envf("RUNNER_CASH_RESERVE_PCT", _MODE["reserve"]) / 100.0
MIN_SCORE = _envf("RUNNER_MIN_SCORE", _MODE["min_score"])
DRY_RUN = os.environ.get("RUNNER_DRY_RUN", "") in ("1", "true", "yes")
# Optional: allow the runner to SHORT on strongly-negative scores (off by default;
# the runner is long-only unless you flip this). Options/leverage remain off.
ALLOW_SHORTS = os.environ.get("RUNNER_ALLOW_SHORTS", "") in ("1", "true", "yes")
SHORT_SCORE = -MIN_SCORE - 0.10   # a bit more conviction to short
FEE_BPS = 0.0005  # ~5bps/side, matches the app's estimate


def log(*a):
    print("[bot_runner]", *a, flush=True)


def fetch_json(url, timeout=45):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "valuatio-bot-runner"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            log(f"  ! HTTP {e.code} for {url}")
        return None
    except Exception as e:
        log(f"  ! fetch failed {url}: {e}")
        return None


# ----------------------------- state -----------------------------------------
def load_state():
    for path in (STATE_FILE, _LEGACY_STATE):
        if path.exists():
            try:
                d = json.loads(path.read_text())
                if isinstance(d, dict):
                    if path is _LEGACY_STATE:
                        log(f"  migrated bot state from legacy {path.name} "
                            f"(bankroll ${d.get('bankroll', 0):,.0f}, {len(d.get('trades') or [])} trades)")
                    return d
            except Exception as e:
                log(f"  ! state parse failed ({path.name}): {e}")
    # fresh
    return {
        "schema": "valuatio-bot-training/v1",
        "bankroll": STARTING_BANKROLL,
        "startingBankroll": STARTING_BANKROLL,
        "trades": [],
        "openPositions": [],
        "equityCurve": [],
        "learnedWeights": {},
        "benchmarkStart": None,
    }


def trades_list(state):
    """The full per-trade records live in 'trades'; that's our working list."""
    t = state.get("trades")
    return t if isinstance(t, list) else []


# ----------------------------- market data ------------------------------------
# ----------------------------- USD basis ---------------------------------------
# Every price the bot trades or marks is in USD - the same basis the app's bot
# uses (normalizeRowToUSD), so entries, stops, targets, P&L, the equity tape and
# the app's display all agree. master.json quotes in LOCAL units: foreign
# listings in their currency, CBOT/ICE softs in US CENTS (USX), London in PENCE.
# Before z80 the runner used those raw - a corn future opened at "$497" (the app
# then marked it at $4.97 = -99%), and foreign holdings were never managed.
_SUBUNIT = {"USX": ("USD", 0.01), "USd": ("USD", 0.01),
            "GBp": ("GBP", 0.01), "GBX": ("GBP", 0.01),
            "ZAc": ("ZAR", 0.01), "ZAC": ("ZAR", 0.01),
            "ILA": ("ILS", 0.01), "ILa": ("ILS", 0.01)}


def load_fx():
    """USD-per-unit map from TRAPP2-1/data/fx/rates.json (the file the app uses)."""
    d = fetch_json(FX_URL)
    rates = (d.get("rates") if isinstance(d, dict) else {}) or {}
    out = {"USD": 1.0}
    for ccy, v in rates.items():
        try:
            up = float(v.get("usdPer")) if isinstance(v, dict) else None
        except (TypeError, ValueError):
            up = None
        if up and up > 0 and math.isfinite(up):
            out[str(ccy).upper()] = up
    return out


def usd_factor(currency, fx):
    """Multiplier local-quote -> USD, or None when the currency has no rate."""
    raw = (currency or "USD").strip() or "USD"
    base, sub = _SUBUNIT.get(raw, (raw.upper(), 1.0))
    rate = fx.get(base)
    return None if rate is None else rate * sub


def load_universe():
    """ticker -> {price (USD), priceLocal, currency, fxRate, sector, name, repo,
    scan, ...} from master.json files. `scan` marks the repos the bot may open new
    trades from; the rest are priced so held positions are always managed."""
    fx = load_fx()
    rows, no_rate = {}, {}
    for url, scan in [(u, True) for u in UNIVERSE_SOURCES] + [(u, False) for u in PRICE_ONLY_SOURCES]:
        data = fetch_json(url)
        if not isinstance(data, list):
            continue
        repo = url.split(f"{RAW}/", 1)[1].split("/", 1)[0]
        for r in data:
            t = (r.get("ticker") or r.get("symbol") or "").upper()
            if not t or t in rows:
                continue
            price, fld = None, None
            for fld in ("price", "close", "closeyest"):   # last close when there's no live quote
                v = _f(r.get(fld))
                if v is not None and math.isfinite(v) and v > 0:
                    price = v
                    break
            if price is None:
                continue
            ccy = (r.get("currency") or "USD")
            f = usd_factor(ccy, fx)
            if f is None:
                no_rate[t] = ccy
                continue
            rows[t] = {
                "ticker": t, "price": round(price * f, 6),
                "priceLocal": price, "currency": ccy, "fxRate": f, "scan": scan,
                "asOf": r.get("fetched_at") if fld == "price" else None,   # z121: a close is never a live quote
                "assetClass": r.get("asset_class"),
                "sector": r.get("sector") or "Unknown",
                "name": r.get("name") or t,
                "repo": repo,
                "changepct": _f(r.get("changepct")),
                "beta": _f(r.get("beta")),
                # Fundamentals carried through for the ported reasoning-chain /
                # health engine (same fields the app's stockbook rows expose).
                "marketcap": _f(r.get("marketcap")),
                "eps": _f(r.get("eps")),
                "profitMargin": _f(r.get("profitMargin")),
                "operatingMargin": _f(r.get("operatingMargin")),
                "freeCashFlow": _f(r.get("freeCashFlow")),
                "netIncome": _f(r.get("netIncome")),
                "revenue": _f(r.get("revenue")),
                "revenueGrowth": _f(r.get("revenueGrowth")),
                "earningsGrowth": _f(r.get("earningsGrowth")),
                "returnOnEquity": _f(r.get("returnOnEquity")),
                "returnOnAssets": _f(r.get("returnOnAssets")),
                "debtToEquity": _f(r.get("debtToEquity")),
                "currentRatio": _f(r.get("currentRatio")),
                "cash": _f(r.get("cash")),
                "totalDebt": _f(r.get("totalDebt")),
                "totalEquity": _f(r.get("totalEquity")),
                "pe": _f(r.get("pe")),
                "priceToBook": _f(r.get("priceToBook")),
                "dividend_yield": _f(r.get("dividend_yield")),
                "payoutRatio": _f(r.get("payoutRatio")),
            }
    n_scan = sum(1 for v in rows.values() if v["scan"])
    n_conv = sum(1 for v in rows.values() if v["fxRate"] != 1.0)
    log(f"universe: {len(rows)} priced tickers in USD ({n_scan} scannable, {len(rows) - n_scan} price-only; "
        f"{n_conv} converted from local units)")
    if no_rate:
        log(f"  {len(no_rate)} skipped - no FX rate: "
            + ", ".join(f"{t}({c})" for t, c in sorted(no_rate.items())[:10]))
    return rows


def load_grades():
    """ticker -> gradeScore (0-100) from research_grades.json. {} if unavailable."""
    data = fetch_json(GRADES_URL)
    out = {}
    if isinstance(data, dict):
        bt = data.get("byTicker") or {}
        for t, g in bt.items():
            gs = g.get("gradeScore")
            if isinstance(gs, (int, float)):
                out[t.upper()] = {"gradeScore": float(gs), "sector": g.get("sector")}
    log(f"grades: {len(out)} tickers")
    return out


# Cache sector-ETF 3-month momentum so we fetch each ETF once per run.
_ETF_MOM_CACHE = {}

def sector_etf_momentum(sector):
    """3-month return of the sector's ETF, mapped to a 0-100 score (50 = flat).
    Mirrors the app: −15%..+15% over ~63 trading days → 0..100."""
    etf = SECTOR_TO_ETF.get(sector)
    if not etf:
        return None
    if etf in _ETF_MOM_CACHE:
        return _ETF_MOM_CACHE[etf]
    data = fetch_json(f"{ETF_HISTORY_BASE}/{etf}.json", timeout=30)
    score = None
    if isinstance(data, list) and len(data) >= 60:
        closes = [(_f(b.get("close")) or _f(b.get("price"))) for b in data]
        closes = [c for c in closes if c and c > 0]
        if len(closes) >= 60:
            last = closes[-1]
            ago = closes[max(0, len(closes) - 63)]
            ret = (last - ago) / ago if ago > 0 else 0
            score = max(0.0, min(100.0, 50 + (ret / 0.15) * 50))
    _ETF_MOM_CACHE[etf] = score
    return score


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_history(ticker, repo):
    """Daily closes [{date, close}] for a ticker, newest last. None if absent."""
    base = HISTORY_BASE.get(repo)
    if not base:
        return None
    data = fetch_json(f"{base}/{ticker}.json", timeout=30)
    if not isinstance(data, list) or len(data) < 30:
        return None
    closes = []
    for bar in data:
        c = bar.get("close")
        if c is None:
            c = bar.get("price")
        try:
            c = float(c)
        except (TypeError, ValueError):
            continue
        if c and c > 0:
            closes.append(c)
    return closes if len(closes) >= 30 else None


# ----------------------------- signals ----------------------------------------
def sma(vals, n):
    if len(vals) < n:
        return None
    return sum(vals[-n:]) / n


def rsi(vals, n=14):
    if len(vals) < n + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(-n, 0):
        ch = vals[i] - vals[i - 1]
        if ch >= 0:
            gains += ch
        else:
            losses -= ch
    if losses == 0:
        return 100.0
    rs = (gains / n) / (losses / n)
    return 100 - (100 / (1 + rs))


def score_ticker(closes, weights):
    """A faithful SUBSET of the app's engine → signed score in [-1, +1].

    Components (each signed, then weighted by the bot's learned weights):
      trend         SMA20 vs SMA60 alignment + price above/below
      momentum      3-month return, squashed
      meanReversion RSI extremes (oversold = mild long signal)
    Returns (signed_score, components_dict).
    """
    last = closes[-1]
    s20, s60 = sma(closes, 20), sma(closes, 60)
    comps = {}

    # Trend: +1 strong uptrend, -1 strong downtrend.
    if s20 and s60:
        if last > s20 > s60:
            comps["trend"] = min(1.0, (last - s60) / s60 * 4)
        elif last < s20 < s60:
            comps["trend"] = max(-1.0, (last - s60) / s60 * 4)
        else:
            comps["trend"] = (last - s60) / s60 * 1.5
            comps["trend"] = max(-1.0, min(1.0, comps["trend"]))

    # Momentum: ~3-month (63 trading days) return, squashed with tanh.
    if len(closes) >= 64:
        mom = (last - closes[-64]) / closes[-64]
        comps["momentum"] = math.tanh(mom * 3)

    # Mean reversion: RSI. Oversold (<30) → mild long; overbought (>70) → mild short.
    r = rsi(closes, 14)
    if r is not None:
        if r < 30:
            comps["meanReversion"] = (30 - r) / 30 * 0.6      # up to +0.6 when very oversold
        elif r > 70:
            comps["meanReversion"] = -(r - 70) / 30 * 0.6     # down to -0.6 when very overbought
        else:
            comps["meanReversion"] = 0.0
    return comps


def compute_regime(spy_closes, opt_market=None):
    """Market regime from SPY trend + realized vol → mode + weight modifiers.
    Faithful to the app's botAssessRegime / _regimeProfile. When an options-market
    read is supplied, rich index IV ('fear'/'stressed') de-risks the regime and
    genuine 'calm' can confirm risk-on — the same forward-looking overlay the app
    applies."""
    profiles = {
        "risk-on":  {"weightMods": {"trend": 1.3, "momentum": 1.3, "meanReversion": 0.7,
                                    "peerGrade": 1.2, "regimeGrade": 1.4, "crossAsset": 1.1,
                                    "optionsIV": 0.8, "optionsMarket": 0.9, "fed": 1.0}, "longBar": 0.0},
        "risk-off": {"weightMods": {"trend": 0.7, "momentum": 0.8, "meanReversion": 1.0,
                                    "peerGrade": 1.3, "regimeGrade": 1.5, "crossAsset": 1.2,
                                    "optionsIV": 1.4, "optionsMarket": 1.5, "fed": 1.2}, "longBar": 0.10},
        "choppy":   {"weightMods": {"trend": 0.7, "momentum": 0.8, "meanReversion": 1.3,
                                    "peerGrade": 1.1, "regimeGrade": 1.2, "crossAsset": 1.0,
                                    "optionsIV": 1.3, "optionsMarket": 1.3, "fed": 1.1}, "longBar": 0.05},
    }
    mode = "choppy"
    trend_down = False
    if spy_closes and len(spy_closes) >= 60:
        last = spy_closes[-1]
        s20, s60 = sma(spy_closes, 20), sma(spy_closes, 60)
        rets = [(spy_closes[i] - spy_closes[i - 1]) / spy_closes[i - 1]
                for i in range(len(spy_closes) - 20, len(spy_closes))]
        mean = sum(rets) / len(rets)
        vol20 = math.sqrt(sum((x - mean) ** 2 for x in rets) / len(rets))
        trend_up = last > s20 > s60
        trend_down = last < s20 < s60
        vol_high = vol20 > 0.018
        if trend_down or (vol_high and not trend_up):
            mode = "risk-off"
        elif trend_up and not vol_high:
            mode = "risk-on"

    # Options overlay: the market's chains price in forward risk price history
    # hasn't shown. Fear/stress de-risks; calm confirms.
    opt_note = ""
    if opt_market:
        vr = opt_market.get("volRegime")
        opt_note = f" · opt IV {opt_market.get('indexIV', 0)*100:.0f}% ({vr})"
        if vr in ("fear", "stressed") and mode != "risk-off":
            mode = "choppy" if mode == "risk-on" else "risk-off"
            opt_note += " → de-risked"
        elif vr == "calm" and mode == "choppy" and not trend_down:
            mode = "risk-on"
            opt_note += " → calm confirms"

    prof = profiles[mode]
    prof["mode"] = mode
    prof["optNote"] = opt_note
    return prof


def peer_rank_table(grades):
    """For each ticker, its percentile WITHIN its sector by gradeScore (0-100).
    Mirrors computePeerGrade's peer-rank component."""
    by_sector = {}
    for t, g in grades.items():
        sec = g.get("sector")
        if sec and g.get("gradeScore") is not None:
            by_sector.setdefault(sec, []).append((t, g["gradeScore"]))
    out = {}
    for sec, lst in by_sector.items():
        if len(lst) < 3:
            continue
        lst.sort(key=lambda kv: kv[1], reverse=True)
        n = len(lst)
        for idx, (t, _gs) in enumerate(lst):
            out[t] = round((1 - idx / (n - 1)) * 100) if n > 1 else 50
    return out


def peer_grade_signal(ticker, sector, peer_ranks):
    """Blend sector-relative grade percentile (65%) + sector ETF momentum (35%),
    return signed [-1,+1]. Faithful to computePeerGrade."""
    peer_rank = peer_ranks.get(ticker)
    etf_mom = sector_etf_momentum(sector)
    if peer_rank is not None and etf_mom is not None:
        score = peer_rank * 0.65 + etf_mom * 0.35
    elif peer_rank is not None:
        score = peer_rank
    elif etf_mom is not None:
        score = etf_mom
    else:
        return None
    return (score - 50) / 50.0


def regime_grade_signal(sector, beta, mode, quad=None):
    """Full computeRegimeGrade: quad-favoring (±22) + risk-on/off sector/beta tilt
    (±14/16), signed [-1,+1]. quad is the market-implied proxy when present."""
    if not sector:
        return None
    b = beta if (beta and isfinite_num(beta)) else 1.0
    etf = SECTOR_TO_ETF.get(sector)
    qdef = SECTOR_ETF_QUAD.get(etf) if etf else None
    score = 50.0
    # Quad favoring (the piece newly ported from the macro layer).
    if qdef and quad is not None:
        if quad in qdef.get("favors", []):
            score += 22
        elif quad in qdef.get("hurts", []):
            score -= 22
    # Risk regime tilt by sector/beta.
    is_defensive = any(k in sector.lower() for k in ("utilit", "staple", "consumer defensive", "health"))
    is_cyclical = any(k in sector.lower() for k in ("tech", "consumer cyclical", "discretionary",
                                                    "financ", "industri", "material", "energy"))
    if mode == "risk-on":
        if is_cyclical or b > 1.1:
            score += 14
        elif is_defensive:
            score -= 8
    elif mode == "risk-off":
        if is_defensive or b < 0.9:
            score += 14
        elif is_cyclical or b > 1.1:
            score -= 16
    score = max(0.0, min(100.0, score))
    return (score - 50) / 50.0


def isfinite_num(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# --- Cross-asset + macro-quad (computed once per run) ---
_XASSET_LAST_RET = {}

def _latest_daily_return(ticker):
    """Last daily % return for a TRAPP2-1 ticker, as a percent (e.g. +0.8)."""
    if ticker in _XASSET_LAST_RET:
        return _XASSET_LAST_RET[ticker]
    data = fetch_json(f"{ETF_HISTORY_BASE}/{ticker}.json", timeout=25)
    ret = None
    if isinstance(data, list) and len(data) >= 2:
        closes = [(_f(b.get("close")) or _f(b.get("price"))) for b in data]
        closes = [c for c in closes if c and c > 0]
        if len(closes) >= 2 and closes[-2]:
            ret = (closes[-1] - closes[-2]) / closes[-2] * 100
    _XASSET_LAST_RET[ticker] = ret
    return ret


def _basket_avg(tickers):
    vals = [r for r in (_latest_daily_return(t) for t in tickers) if r is not None]
    return (sum(vals) / len(vals)) if vals else None


def compute_cross_asset():
    """Latest cross-asset basket moves → signed signals in [-1,+1], faithful to
    extractCrossAssetSignals (using daily returns since the runner is EOD)."""
    out = {}
    d = _basket_avg(XASSET_BASKETS["dollar"])
    out["dollar"] = max(-1, min(1, d / 2)) if d is not None else None
    # safe haven: gold+bonds avg + 0.3×VIX. Risk-OFF when high → invert for risk appetite.
    gold = _latest_daily_return("GLD")
    bonds = _latest_daily_return("TLT")
    vix = _latest_daily_return("^VIX")
    sh_vals = [v for v in (gold, bonds) if v is not None]
    if sh_vals:
        sh = sum(sh_vals) / len(sh_vals) + (vix * 0.3 if vix is not None else 0)
        out["safeHaven"] = max(-1, min(1, sh / 2))
    else:
        out["safeHaven"] = None
    ind = _basket_avg(XASSET_BASKETS["industrial"])
    out["industrial"] = max(-1, min(1, ind / 1.5)) if ind is not None else None
    en = _basket_avg(XASSET_BASKETS["energy"])
    out["energy"] = max(-1, min(1, en / 1.5)) if en is not None else None
    idx = _basket_avg(XASSET_BASKETS["indices"])
    out["indices"] = max(-1, min(1, idx / 1.5)) if idx is not None else None
    return out


def cross_asset_for_sector(sector, xa):
    """Pick the most relevant cross-asset signal for a sector (mirrors the app)."""
    if not sector or not xa:
        return None
    s = sector.lower()
    if any(k in s for k in ("material", "metal", "chem")):
        return xa.get("industrial")
    if "energy" in s:
        return xa.get("energy")
    if any(k in s for k in ("financ", "tech", "communication")):
        return xa.get("indices")
    # default macro alignment = global indices minus safe-haven pull
    idx = xa.get("indices")
    sh = xa.get("safeHaven")
    if idx is None and sh is None:
        return None
    return max(-1, min(1, (idx or 0) - 0.5 * (sh or 0)))


def compute_quad_proxy(xa, regime_mode):
    """MARKET-IMPLIED quad proxy (1-4) when the FRED growth/inflation RoC isn't
    available to the runner. Growth proxy = industrial-metals + index breadth;
    inflation proxy = energy + dollar-inverse. This is an honest market proxy of
    the app's FRED-based classifyQuad, not the macro tab's exact quad.

    Returns (quad:int|None, note:str)."""
    if not xa:
        return None, "no cross-asset data"
    growth_bits = [xa.get("industrial"), xa.get("indices")]
    growth_bits = [b for b in growth_bits if b is not None]
    infl_bits = []
    if xa.get("energy") is not None:
        infl_bits.append(xa["energy"])
    if xa.get("dollar") is not None:
        infl_bits.append(-xa["dollar"])      # strong dollar ↔ disinflationary
    if not growth_bits or not infl_bits:
        return None, "insufficient proxy inputs"
    growth_up = (sum(growth_bits) / len(growth_bits)) > 0
    infl_up = (sum(infl_bits) / len(infl_bits)) > 0
    if growth_up and not infl_up:
        return 1, "proxy: growth↑ inflation↓ (Goldilocks)"
    if growth_up and infl_up:
        return 2, "proxy: growth↑ inflation↑ (Reflation)"
    if not growth_up and infl_up:
        return 3, "proxy: growth↓ inflation↑ (Stagflation)"
    return 4, "proxy: growth↓ inflation↓ (Deflation)"


def load_real_quad():
    """Read the FRED-derived quad written by TRAPP2-1's macro-quad workflow.
    Returns (quad:int, note:str) or (None, reason)."""
    data = fetch_json(MACRO_QUAD_URL, timeout=20)
    if isinstance(data, dict):
        cur = data.get("current") or {}
        q = cur.get("quad")
        if isinstance(q, int) and 1 <= q <= 4:
            return q, f"FRED quad {q} ({cur.get('label','')}, as of {cur.get('asOf','?')})"
    return None, "no FRED quad file"


def resolve_quad(xa, regime_mode):
    """Prefer the REAL FRED quad; fall back to the market-implied proxy."""
    q, note = load_real_quad()
    if q is not None:
        return q, note
    return compute_quad_proxy(xa, regime_mode)


# --- Options: per-ticker IV signal + market-wide options read (computed once) ---
_OPT_CACHE = {}          # ticker -> parsed chain (or None)
_CAL = {}                # z138: earnings calendar {TICKER: {next, beatRate, avgSurprise, n}} (TRAPP2-ANALYTICS)


def load_catalysts():
    global _CAL
    d = fetch_json(EARNINGS_CAL_URL, timeout=30)
    _CAL = (d or {}).get("tickers") or {} if isinstance(d, dict) else {}
    return _CAL
_OPT_MARKET = {"done": False, "data": None}

def _load_option_chain(ticker):
    if ticker in _OPT_CACHE:
        return _OPT_CACHE[ticker]
    data = fetch_json(f"{OPTIONS_BASE}/{ticker}.json", timeout=25)
    if not isinstance(data, dict):
        data = fetch_json(f"{OPTIONS_BASE_LEGACY}/{ticker}.json", timeout=25)
    _OPT_CACHE[ticker] = data if isinstance(data, dict) else None
    return _OPT_CACHE[ticker]

def _atm(lst, spot):
    if not lst:
        return None
    return sorted(lst, key=lambda c: abs((c.get("strike") or 0) - spot))[0]

def extract_options_signal(ticker):
    """Per-name forward signal from the option chain, faithful to the app's
    extractOptionsSignal: ATM IV level + put/call skew + term structure → a small
    signed signal (rich IV / heavy downside skew / backwardation lean negative).
    Returns dict or None."""
    od = _load_option_chain(ticker)
    if not od or not isinstance(od.get("expiries"), list) or not od["expiries"]:
        return None
    spot = od.get("spot")
    if not spot:
        return None
    exps = sorted(od["expiries"], key=lambda e: (e.get("dte") if e.get("dte") is not None else 9999))
    ref = next((e for e in exps if 20 <= (e.get("dte") or 0) <= 60), exps[0])
    if not ref:
        return None
    atm_call, atm_put = _atm(ref.get("calls"), spot), _atm(ref.get("puts"), spot)
    ivs = [x.get("iv") for x in (atm_call, atm_put) if x and x.get("iv") is not None and isfinite_num(x.get("iv"))]
    if not ivs:
        return None
    iv = sum(ivs) / len(ivs)
    skew = None
    if atm_put and atm_call and atm_put.get("iv") is not None and atm_call.get("iv") is not None:
        skew = atm_put["iv"] - atm_call["iv"]
    term_slope = None
    far = next((e for e in exps if (e.get("dte") or 0) >= 100), None)
    if far:
        far_atm = _atm(far.get("calls"), spot)
        if far_atm and far_atm.get("iv") is not None and isfinite_num(far_atm.get("iv")):
            term_slope = iv - far_atm["iv"]
    # Map to a signed signal (same anchors as the app).
    iv_signal = 0.0
    if iv > 0.45:
        iv_signal = -min(1.0, (iv - 0.45) / 0.45)
    elif iv < 0.22:
        iv_signal = min(0.5, (0.22 - iv) / 0.22)
    if skew is not None and skew >= 0.05:
        iv_signal -= min(0.3, (skew - 0.05) / 0.15 * 0.3 + 0.05)
    if term_slope is not None and term_slope > 0.05:
        iv_signal -= min(0.2, (term_slope - 0.05) / 0.15)
    return {"iv": round(iv, 4), "skew": round(skew, 4) if skew is not None else None,
            "termSlope": round(term_slope, 4) if term_slope is not None else None,
            "ivSignal": max(-1.0, min(1.0, round(iv_signal, 3)))}

def compute_options_market(universe_tickers):
    """Aggregate option chains across the manifest into a market-wide read:
    index-led IV → vol regime + a signed market-trend tilt. Faithful to the app's
    computeOptionsMarketSignal. Loads each chain once. Returns dict or None."""
    if _OPT_MARKET["done"]:
        return _OPT_MARKET["data"]
    man = fetch_json(OPTIONS_MANIFEST_URL, timeout=20)
    tickers = (man.get("tickers") if isinstance(man, dict) else None) or []
    rows, idx_iv = [], []
    for tk in tickers:
        sig = extract_options_signal(tk)
        if not sig or sig.get("iv") is None:
            continue
        is_index = tk in MKT_OPT_INDEXES
        rows.append({"ticker": tk, "iv": sig["iv"], "skew": sig["skew"],
                     "termSlope": sig["termSlope"], "isIndex": is_index})
        if is_index:
            idx_iv.append(sig["iv"])
    if not rows:
        _OPT_MARKET.update(done=True, data=None)
        return None
    def avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None
    breadth_iv = avg([r["iv"] for r in rows])
    index_iv = (sum(idx_iv) / len(idx_iv)) if idx_iv else breadth_iv
    avg_skew = avg([r["skew"] for r in rows])
    avg_term = avg([r["termSlope"] for r in rows])
    iv_pct = index_iv * 100
    if iv_pct >= 32:
        vol_regime = "fear"
    elif iv_pct >= 24:
        vol_regime = "stressed"
    elif iv_pct >= 17:
        vol_regime = "normal"
    else:
        vol_regime = "calm"
    market_trend = 0.0
    if iv_pct < 17:
        market_trend += min(0.5, (17 - iv_pct) / 17)
    elif iv_pct > 24:
        market_trend -= min(0.8, (iv_pct - 24) / 16)
    if avg_skew is not None and avg_skew >= 0.05:
        market_trend -= min(0.4, (avg_skew - 0.05) / 0.15 * 0.4)
    if avg_term is not None and avg_term > 0.05:
        market_trend -= min(0.3, (avg_term - 0.05) / 0.15 * 0.3)
    data = {"count": len(rows), "indexIV": round(index_iv, 4), "breadthIV": round(breadth_iv, 4),
            "avgSkew": round(avg_skew, 4) if avg_skew is not None else None,
            "volRegime": vol_regime, "marketTrend": max(-1.0, min(1.0, round(market_trend, 3)))}
    _OPT_MARKET.update(done=True, data=data)
    return data


# --- Fed rate expectations (baseline path) ---
# The app's Fed signal blends user-editable odds with seeded market odds in the
# browser; the runner can't see the user's overrides, so it replicates the SEEDED
# baseline (public-record current range + market-implied odds for the remaining
# FOMC meetings). Expected move per meeting = cut×−25 + hike×+25 bps.
FED_CURRENT_RANGE = {"low": 3.50, "high": 3.75}
FOMC_MARKET_ODDS = {
    "2026-06-17": {"cut": 0.06, "hike": 0.00},
    "2026-07-29": {"cut": 0.20, "hike": 0.02},
    "2026-09-16": {"cut": 0.42, "hike": 0.02},
    "2026-10-28": {"cut": 0.30, "hike": 0.02},
    "2026-12-09": {"cut": 0.35, "hike": 0.02},
}

def fed_expectation(today):
    """Expected rate-path bps change from now to year-end, from the seeded odds.
    Negative = net cuts expected (easing). Faithful to fedRateExpectation's
    baseline."""
    mid = (FED_CURRENT_RANGE["low"] + FED_CURRENT_RANGE["high"]) / 2
    expected = mid
    ahead = 0
    for date, odds in sorted(FOMC_MARKET_ODDS.items()):
        if date < today:
            continue
        expected += ((odds["cut"] * -25) + (odds["hike"] * 25)) / 100.0
        ahead += 1
    bps = (expected - mid) * 100
    stance = "easing" if bps < -10 else ("tightening" if bps > 10 else "on-hold")
    return {"expectedBpsChange": round(bps, 1), "stance": stance, "meetingsAhead": ahead}

def fed_signal(sector, fed):
    """Rate tailwind/headwind for a sector, signed [-1,+1]. Easing helps rate-
    sensitive longs (REITs, utilities, homebuilders) and growth; hurts banks.
    Mirrors the app engine's 'fed' component."""
    if not sector or not fed:
        return None
    easing = max(-1.0, min(1.0, -fed["expectedBpsChange"] / 75.0))   # cuts → positive
    s = sector.lower()
    if any(k in s for k in ("real estate", "reit", "utilit", "homebuild")):
        sens = 1.0
    elif any(k in s for k in ("tech", "growth", "biotech")):
        sens = 0.5
    elif any(k in s for k in ("financ", "bank")):
        sens = -0.4
    else:
        return None
    return max(-1.0, min(1.0, round(easing * sens, 3)))


# Per-signal BASE weights — copied verbatim from the in-app engine's add() calls
# (app.js). The runner previously blended every signal at equal weight (1.0),
# which made its decisions diverge from the app even with identical inputs.
# These restore the app's relative signal importance for the 9 shared signals.
BASE_WEIGHTS = {
    # Fundamental core (ported from the app's bot engine — was missing server-side).
    "fundamentals": 0.26, "researchGrade": 0.16, "health": 0.12,
    # Price / cross-sectional / macro signals.
    "trend": 0.18, "momentum": 0.14, "meanReversion": 0.06,
    "crossAsset": 0.10, "peerGrade": 0.12, "regimeGrade": 0.14,
    "optionsIV": 0.07, "optionsMarket": 0.05, "fed": 0.08,
    # Chart patterns / templates (TRAPP2-ANALYTICS patterns engine): ONE factor
    # among many - weighted like cross-asset, well below fundamentals or trend,
    # and further capped by FACTOR_CAP. It can fire a trigger; it can't decide alone.
    "pattern": 0.10,
}


def _capped(ws, cap):
    """Limit any single signal to `cap` of the total absolute weight (only when
    >= 5 signals are present - with fewer, a cap can't be satisfied). Keeps the
    blend from leaning on one factor. Exact water-filling: capped signals get
    exactly cap x the final total; the rest keep their weights."""
    if len(ws) < 5 or cap <= 0 or cap * len(ws) < 1:
        return ws
    capped = set()
    while True:
        free = sum(abs(w) for k, w in ws.items() if k not in capped)
        if not free:
            return ws
        total = free / (1 - cap * len(capped))
        over = {k for k, w in ws.items() if k not in capped and abs(w) > cap * total + 1e-12}
        if not over:
            break
        capped |= over
    return {k: ((1 if w >= 0 else -1) * cap * total if k in capped else w) for k, w in ws.items()}


def blend_score(comps, weights, weight_mods):
    """Weighted blend mirroring the app engine: base × learned × regime-mod.

    base    = BASE_WEIGHTS (the app's per-signal importance)
    learned = the bot's learnedWeights (1.0 = neutral; grows from feedback)
    mod     = the regime tilt (weightMods). Matches app.js add() exactly.
    """
    num, den = 0.0, 0.0
    ws = {}
    for k, v in comps.items():
        if v is None:
            continue
        base = BASE_WEIGHTS.get(k, 0.10)          # app per-signal base weight
        learned = weights.get(k, 1.0)             # learning layer (1.0 = neutral)
        try:
            learned = float(learned)
        except (TypeError, ValueError):
            learned = 1.0
        mod = weight_mods.get(k, 1.0)             # regime tilts the signal's influence
        ws[k] = base * learned * mod
    for k, w in _capped(ws, FACTOR_CAP).items():   # no single factor dominates
        num += comps[k] * w
        den += abs(w)
    signed = (num / den) if den else 0.0
    return max(-1.0, min(1.0, signed))


# ----------------------------- bond trend context (z94) --------------------------
# "It's the trend that matters, not the absolute level": a falling TLT (rising
# long yields) gives investors a low-risk alternative to stocks. Tested on
# 2006-2026 daily data the TLT 20/50-day trend did NOT forecast bigger 20-day
# equity drawdowns on its own (see STATUS §1c); the stock-bond link flipped from
# hedge (negative correlation, 2007-2020) to co-moving (positive, 2022+). So
# this is recorded as OBSERVED CONTEXT on every trade - trend, slope, and the
# 60-day stock/bond correlation - not as a score component. The trade journal
# then shows whether entries made during TLT downtrends do worse; only proven
# live evidence would promote it to a weighted signal.
def bond_trend_context(spy_closes):
    tlt = load_history("TLT", "TRAPP2") or load_history("TLT", "TRAPP2-1")
    if not tlt or len(tlt) < 61 or not spy_closes or len(spy_closes) < 61:
        return None
    ma = lambda a, n: sum(a[-n:]) / n
    m20, m50 = ma(tlt, 20), ma(tlt, 50)
    slope10 = tlt[-1] / tlt[-11] - 1
    trend = "down" if (m20 < m50 and slope10 < 0) else "up" if (m20 > m50 and slope10 > 0) else "mixed"
    n = 60
    rt = [tlt[i] / tlt[i - 1] - 1 for i in range(len(tlt) - n, len(tlt))]
    rs = [spy_closes[i] / spy_closes[i - 1] - 1 for i in range(len(spy_closes) - n, len(spy_closes))]
    mt, ms = sum(rt) / n, sum(rs) / n
    cov = sum((a - mt) * (b - ms) for a, b in zip(rt, rs))
    vt = math.sqrt(sum((a - mt) ** 2 for a in rt)); vs = math.sqrt(sum((b - ms) ** 2 for b in rs))
    corr = cov / (vt * vs) if vt and vs else None
    return {"trend": trend, "tlt": round(tlt[-1], 2), "ma20": round(m20, 2), "ma50": round(m50, 2),
            "slope10Pct": round(slope10 * 100, 2), "stockBondCorr60": round(corr, 2) if corr is not None else None}


# ----------------------------- Fed context (z105) ------------------------------
# QE/QT stance, net-liquidity trend and the latest FOMC statement's tone from
# TRAPP2-1's Fed monitor. Like the bond trend: OBSERVED context recorded on every
# trade (regimeContext.fed) and logged - not a scored factor - until the trade
# record shows it adds something.
def fed_context():
    m = fetch_json(f"{RAW}/TRAPP2-1/main/data/fed/monitor.json", timeout=20)
    if not isinstance(m, dict):
        return None
    bs, st = m.get("balanceSheet") or {}, m.get("statement") or {}
    nl = bs.get("netLiquidity") or []
    nl13 = round(nl[-1][1] - nl[-14][1], 1) if len(nl) >= 14 else None
    return {"stance": bs.get("stance"), "monthlyPaceBn": bs.get("monthlyPaceBn"),
            "netLiquidityBn": bs.get("netLiquidityBn"), "netLiquidity13wBn": nl13,
            "statementTone": st.get("tone"), "toneChange": st.get("toneChange"), "statementDate": st.get("date"),
            "asOf": m.get("generatedAt")}


def curve_context(now=None):
    """z126: the Treasury curve on the trade date - recorded as CONTEXT, never scored (the
    rule for the bond trend and the Fed): 2s10s, 3m10y, 10y real (DGS10 - T10YIE, FRED's own
    definition of the breakeven), 10y / 5y breakeven, the 5y5y forward, and last month's move
    in the textbook 2Y / 10Y reading (bull / bear steepener / flattener, twist, parallel;
    within +-3 bp = little changed) - the same rule as the app's Data -> Yield curve."""
    base = f"{RAW}/TRAPP2-1/main/data/macro/"

    def series(sid):
        d = fetch_json(base + sid + ".json", timeout=20)
        out = {}
        for o in ((d or {}).get("observations") or []):
            try:
                if o.get("date") and o.get("value") not in (None, "", "."):
                    out[o["date"]] = float(o["value"])
            except (TypeError, ValueError):
                pass
        return out
    s = {k: series(k) for k in ("DGS3MO", "DGS2", "DGS10", "T10YIE", "T5YIE", "T5YIFR")}
    dates = sorted(set(s["DGS2"]) & set(s["DGS10"]))
    if len(dates) < 30:
        return None
    d = dates[-1]
    y, m, day = int(d[:4]), int(d[5:7]) - 1, int(d[8:10])
    if m == 0:
        y, m = y - 1, 12
    import calendar
    target = f"{y:04d}-{m:02d}-{min(day, calendar.monthrange(y, m)[1]):02d}"
    prev = max((x for x in dates if x <= target), default=None)

    def bp(a, b):
        return None if a is None or b is None else round((a - b) * 100)
    y2, y10, m3 = s["DGS2"][d], s["DGS10"][d], s["DGS3MO"].get(d)
    move = None
    if prev:
        d2, d10 = bp(y2, s["DGS2"][prev]), bp(y10, s["DGS10"][prev])
        ds = d10 - d2
        if abs(d2) < 3 and abs(d10) < 3:
            lab = "little changed"
        elif d2 * d10 < 0:
            lab = "steepening twist" if ds > 0 else "flattening twist"
        else:
            bull = d2 + d10 < 0
            lab = ("parallel rally" if bull else "parallel sell-off") if abs(ds) < 3 else f"{'bull' if bull else 'bear'} {'steepener' if ds > 0 else 'flattener'}"
        move = {"label": lab, "d2": d2, "d10": d10, "vs": prev}
    be10 = s["T10YIE"].get(d)
    return {"date": d, "y2": y2, "y10": y10, "s2s10": bp(y10, y2), "s3m10y": bp(y10, m3),
            "real10": round(y10 - be10, 4) if be10 is not None else None, "be10": be10,
            "be5": s["T5YIE"].get(d), "fwd5y5y": s["T5YIFR"].get(d), "move1m": move}


def credit_context():
    """z132: corporate credit on the trade date - CONTEXT, never scored (the bond-trend / Fed / curve
    rule): ICE BofA high-yield, CCC and investment-grade option-adjusted spreads (bp) from TRAPP2-1's
    FRED files, their 1-month change, CCC - BB, and the junk-spread band the app shows
    (tight < 300 bp, normal < 450, wide < 650, stressed < 900, distressed)."""
    base = f"{RAW}/TRAPP2-1/main/data/macro/"

    def series(sid):
        d = fetch_json(base + sid + ".json", timeout=20)
        out = []
        for o in ((d or {}).get("observations") or []):
            try:
                if o.get("date") and o.get("value") not in (None, "", "."):
                    out.append((o["date"], float(o["value"])))
            except (TypeError, ValueError):
                pass
        return sorted(out)
    s = {k: series(k) for k in ("BAMLH0A0HYM2", "BAMLH0A3HYC", "BAMLH0A1HYBB", "BAMLC0A0CM")}
    hy = s["BAMLH0A0HYM2"]
    if len(hy) < 30:
        return None
    d = hy[-1][0]
    y, m, day = int(d[:4]), int(d[5:7]) - 1, int(d[8:10])
    if m == 0:
        y, m = y - 1, 12
    import calendar
    target = f"{y:04d}-{m:02d}-{min(day, calendar.monthrange(y, m)[1]):02d}"

    def last(ser, on=None):
        v = None
        for dd, x in ser:
            if on is None or dd <= on:
                v = x
            else:
                break
        return v

    def bp(v):
        return None if v is None else round(v * 100)
    hy_now, hy_then = last(hy), last(hy, target)
    band = None
    if hy_now is not None:
        b = hy_now * 100
        band = "tight" if b < 300 else "normal" if b < 450 else "wide" if b < 650 else "stressed" if b < 900 else "distressed"
    ccc, bb = last(s["BAMLH0A3HYC"]), last(s["BAMLH0A1HYBB"])
    return {"date": d, "hyOas": bp(hy_now), "hy1m": None if hy_then is None else round((hy_now - hy_then) * 100),
            "cccOas": bp(ccc), "cccMinusBb": None if ccc is None or bb is None else round((ccc - bb) * 100),
            "igOas": bp(last(s["BAMLC0A0CM"])), "band": band}


# ----------------------------- regime certainty (z93) ---------------------------
# Regime shifts are hard to predict. Before anything leans on the regime, ask
# TRAPP2-1's regime_certainty.json whether the call is DIALED IN (margin,
# stability, quad agreement, cross-check, track record). If it isn't - or the
# file can't be read - the bot runs REGIME-AGNOSTIC: no regime weight tilts, no
# regime entry bar, no regime-based sector grade, no regime component in the
# crypto/futures research, pattern trust from ALL regimes. Trades then rest on
# company-level and price evidence instead of a shaky macro call.
def load_regime_certainty():
    rc = fetch_json(f"{RAW}/TRAPP2-1/main/data/regime_certainty.json", timeout=20)
    return rc if isinstance(rc, dict) and "certainty" in rc else None


def regime_call_outcomes(rc):
    """date -> was that day's regime call right (21-session SPY outcome)?"""
    out = {}
    for c in ((rc or {}).get("calls") or []):
        if c.get("date") and "correct" in c:
            out[c["date"]] = bool(c["correct"])
    return out


# ----------------------------- chart-pattern factor ---------------------------
PATTERN_STATUS_W = {"confirmed": 1.0, "identified": 0.6, "open": 0.6, "timeout": 0.3,
                    "success": 0.35,       # already hit its target - less left
                    "invalidated": -0.4,   # a FAILED pattern often runs the other way
                    "failure": -0.4, "expired": 0.0}


LIVE_MIN_N = 15        # live calls judged before the live record counts at all
LIVE_PRIOR_N = 60      # live calls at which live and backtest weigh equally


def load_patterns(use_regime=True):
    """Recent identifications from the patterns engine + each template's
    backtest edge. -> {"by": {TICKER: [signal, ...]}, "rel": {template: 0..1}}"""
    base = f"{RAW}/TRAPP2-ANALYTICS/main/data/patterns"
    sig = fetch_json(f"{base}/signals.json") or {}
    st = fetch_json(f"{base}/stats.json") or {}
    rel, rel_src = {}, {}
    now = ((st.get("regimeNow") or {}).get("trend")) if use_regime else None
    rel_vol = {}
    for k, t in ((st.get("templates") or {}).items()):
        edge = ((t or {}).get("backtest") or {}).get("edgeVsBaselinePct")
        src = "all"
        # Regime-aware: use the template's record IN THE CURRENT MARKET REGIME
        # (SPY trend: up / mixed / down) once it has enough judged calls - a 52-week
        # breakout earns trust in up-trends and loses it in down-trends.
        rg = (((t or {}).get("byRegime") or {}).get("trend") or {}).get(now or "")
        if rg and rg.get("usable") and isinstance(rg.get("edgeVsBaselinePct"), (int, float)):
            edge, src = rg["edgeVsBaselinePct"], now
        # z104: backtest -> LIVE transition. The live record (calls made since the
        # live ledger started, judged only on later bars) is the honest test. Blend
        # it in by sample size against a fixed prior weight for the backtest, so
        # live evidence takes over as it accumulates (at LIVE_PRIOR_N live calls
        # it carries half the weight) and a backtest edge that doesn't hold live
        # loses trust.
        live = (t or {}).get("live") or {}
        ln, le = live.get("judged") or 0, live.get("edgeVsBaselinePct")
        w = 0.0
        if ln >= LIVE_MIN_N and isinstance(le, (int, float)) and isinstance(edge, (int, float)):
            w = ln / (ln + LIVE_PRIOR_N)
            edge = (1 - w) * edge + w * le
            src = f"{src}+live{ln}"
        # earned trust: +4pp 20-day edge -> 1.0, no edge -> 0.5, negative -> floor 0.15
        rel[k] = max(0.15, min(1.0, 0.5 + (edge / 4.0))) if isinstance(edge, (int, float)) else 0.5
        # z103: breakout-volume split (patterns-v3) - a volume-confirmed call
        # earns the confirmed record's trust, a light-volume call the other.
        for vk, vs in (((t or {}).get("byVolume") or {}).items()):
            ve = (vs or {}).get("edgeVsBaselinePct")
            if (vs or {}).get("usable") and isinstance(ve, (int, float)):
                if w:                                    # same live blend as the template trust
                    ve = (1 - w) * ve + w * le
                rel_vol.setdefault(k, {})[vk] = max(0.15, min(1.0, 0.5 + (ve / 4.0)))
        rel_src[k] = src
    by = {}
    for s in (sig.get("signals") or []):
        t = (s.get("ticker") or "").upper()
        if t:
            by.setdefault(t, []).append(s)
    log(f"patterns: {sum(len(v) for v in by.values())} recent identifications on {len(by)} tickers"
        f" (scan {sig.get('asOf', '?')}; market trend regime {now or '?'}); template trust " +
        ", ".join(f"{k} {v:.2f}{'' if rel_src.get(k) == 'all' else '*'}" for k, v in sorted(rel.items(), key=lambda x: -x[1])[:6])
        + "  (* = current-regime and/or live-blended record)")
    return {"by": by, "rel": rel, "relVol": rel_vol, "relSource": rel_src, "regime": now, "asOf": sig.get("asOf")}


def pattern_signal(tk, pats):
    """-> (value in [-1, 1] or None, detail). Direction x lifecycle weight x the
    template's earned trust, summed over the ticker's recent calls."""
    if not pats:
        return None, None
    rows = pats["by"].get((tk or "").upper()) or []
    if not rows:
        return None, None
    total, parts = 0.0, []
    for s in rows:
        d = 1 if s.get("direction") == "bullish" else -1
        w = PATTERN_STATUS_W.get(s.get("status"), 0.0)
        r = pats["rel"].get(s.get("template"), 0.5)
        vc = s.get("volConfirmed")
        rv = (pats.get("relVol") or {}).get(s.get("template")) or {}
        if vc is True and "confirmed" in rv:
            r = rv["confirmed"]
        elif vc is False and "unconfirmed" in rv:
            r = rv["unconfirmed"]
        v = d * w * r
        if v:
            total += v
            parts.append({"template": s.get("template"), "status": s.get("status"),
                          "identDate": s.get("identDate"), "trust": round(r, 2), "value": round(v, 3),
                          "trustFrom": (pats.get("relSource") or {}).get(s.get("template"), "all"),
                          "volConfirmed": vc})
    if not parts:
        return None, None
    parts.sort(key=lambda x: -abs(x["value"]))
    return max(-1.0, min(1.0, total)), parts


# ----------------------------- position management ----------------------------
# ----------------------------- app ↔ runner (z100) -----------------------------
# Both the app and this runner may trade the bot's book. The runner is the one
# that runs on a schedule; the app can still open / close / trim / add cash by
# hand. The app never edits this repo's state file (that would race the runner):
# it drops one small file per action into data/app_actions/ (creating a NEW file
# never conflicts) and dispatches bot-actions.yml. Every run - the full nightly
# scan and the 15-minute actions pass - applies those files first, in time
# order, exactly once (ids remembered in state["appliedActions"]), then deletes
# them. Accounting is the runner's: a trim credits the bankroll when it happens
# and is tracked in trims[] / realizedTrim; a close credits the remainder and
# folds realizedTrim into pnl - so the app and the runner keep one ledger.
APP_ACTIONS_DIR = ROOT / "data" / "app_actions"
ACTIONS_ONLY = ("--actions-only" in sys.argv) or os.environ.get("RUNNER_ACTIONS_ONLY") == "1"

# ---- z121: ONE trader (this runner), trading ONLY in the NYSE regular session ----
# The app no longer opens, closes or trims anything: its Run button asks this runner
# to run, and a close / trim from a device is a REQUEST executed here, in session, at
# this runner's own fresh quote. RUNNER_APP_TRADES=1 restores the z100-z118 behaviour.
RUNNER_ONLY = os.environ.get("RUNNER_APP_TRADES", "").strip().lower() not in ("1", "true", "yes")
FORCE_SESSION = os.environ.get("RUNNER_FORCE_SESSION", "").strip() == "1"          # tests only
MAX_QUOTE_AGE_MIN = float(os.environ.get("RUNNER_MAX_QUOTE_AGE_MIN") or 30)       # a trade needs a quote this fresh
ENTRY_STOP_BEFORE_CLOSE_MIN = int(os.environ.get("RUNNER_ENTRY_STOP_BEFORE_CLOSE_MIN") or 30)


def _hhmm(s, default):
    try:
        h, m = str(s).strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return default


ENTRY_AFTER_MIN = _hhmm(os.environ.get("RUNNER_ENTRY_AFTER_ET") or "10:00", 600)   # after the opening auction settles


def _now_override():
    s = (os.environ.get("RUNNER_NOW_UTC") or "").strip()                               # tests only
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


NOW_UTC = _now_override()


# ----------------------------- Supabase pull (z112) ----------------------------
# The app writes bot trades straight to Supabase (bot_trades, anon key) - that
# works on every device even without a GitHub token. So each run first PULLS
# Supabase: trades the repo doesn't have are stored here; for trades both have,
# trade_sync.winner() picks the version (a close beats open, more trims beat
# fewer, else the later updatedAt). The bankroll is then derived from the
# journal (trade_sync.ledger_bankroll), the same rule the app uses, so a trade
# that arrives by any path is accounted for exactly once.
def _sb_env():
    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE") or "").strip()
    return url, key


def _sig(t):
    return (t.get("status"), round(float(t.get("shares") or 0), 6), len(t.get("trims") or []), t.get("exitDate"))


def stamp_changes(state, before):
    """updatedAt = now on every trade this run opened / closed / trimmed, so the
    Supabase merge (trade_sync.winner) knows this is the newer version."""
    now, n = ts.now_iso(), 0
    for t in trades_list(state):
        if before.get(t.get("id")) != _sig(t):
            t["updatedAt"] = now
            n += 1
    return n


def finalize_state(state, before):
    n = stamp_changes(state, before)
    derived = ts.ledger_bankroll(state, STARTING_BANKROLL)
    if abs((state.get("bankroll") or 0) - derived) >= 0.01:
        log(f"bankroll check: running {(state.get('bankroll') or 0):,.2f} vs journal {derived:,.2f} → journal wins")
    state["bankroll"] = derived
    if n:
        log(f"stamped updatedAt on {n} changed trade(s)")


def pull_supabase_trades(state):
    url, key = _sb_env()
    if not url or not key:
        log("supabase pull: skipped (no SUPABASE_URL / service key in this job)")
        return {"skipped": True}
    import urllib.request
    rows, offset = [], 0
    try:
        while True:
            req = urllib.request.Request(f"{url}/rest/v1/bot_trades?select=id,trade,updated_at&order=id&limit=1000&offset={offset}",
                                         headers={"apikey": key, "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                page = json.loads(r.read().decode())
            rows += page
            if len(page) < 1000:
                break
            offset += 1000
    except Exception as e:
        log(f"supabase pull: failed ({type(e).__name__}: {e}) — continuing with the repo journal")
        return {"error": str(e)}
    trades = trades_list(state)
    by_id = {t.get("id"): t for t in trades if t.get("id")}
    added = updated = phantom = 0
    for row in rows:
        t = row.get("trade") if isinstance(row, dict) else None
        tid = (t or {}).get("id") or (row or {}).get("id")
        if not isinstance(t, dict) or not tid:
            continue
        t["id"] = tid
        if tid not in by_id:
            if t.get("status") == "void":
                continue
            # z116: a RUNNER-made trade the repo doesn't have comes from a run whose
            # state was never saved (the commit bug fixed in z116). Those runs each
            # restarted from the 2026-06-22 book, so their trades overlap and were
            # sized off a stale bankroll - not imported (sync marks them void).
            # RUNNER_RECOVER_SUPABASE=1 imports them, one per ticker not already open.
            if ts.is_runner_id(tid):
                if os.environ.get("RUNNER_RECOVER_SUPABASE") == "1" and t.get("status") == "open" and not any(
                        x.get("ticker") == t.get("ticker") and x.get("status") == "open" for x in trades):
                    t["_via"] = "supabase-recovered"
                    trades.append(t)
                    by_id[tid] = t
                    added += 1
                else:
                    phantom += 1
                continue
            t.setdefault("origin", "app")
            t["_via"] = "supabase"
            trades.append(t)
            by_id[tid] = t
            added += 1
            log(f"  supabase → repo: stored {t.get('status', '?')} {t.get('ticker')} ({tid})")
            continue
        loc = by_id[tid]
        if ts.winner(loc, t) == "remote":
            for k in ts.STATUS_FIELDS:
                if k in t:
                    loc[k] = t[k]
            loc["_via"] = "supabase"
            updated += 1
            log(f"  supabase → repo: updated {loc.get('ticker')} ({tid}) → {loc.get('status')}")
    log(f"supabase pull: {len(rows)} row(s) · {added} new trade(s) stored · {updated} updated from the app"
        + (f" · {phantom} runner trade(s) from unsaved runs ignored (marked void by the mirror)" if phantom else ""))
    return {"rows": len(rows), "added": added, "updated": updated, "phantom": phantom}


def apply_app_actions(state, today, session=None, fresh=None):
    if not APP_ACTIONS_DIR.exists():
        return []
    files = sorted(APP_ACTIONS_DIR.glob("*.json"))
    if not files:
        return []
    actions = []
    for f in files:
        try:
            a = json.loads(f.read_text())
            a["_file"] = f
            actions.append(a)
        except Exception as e:
            log(f"  app action {f.name} unreadable ({e}) — left in place")
    actions.sort(key=lambda a: str(a.get("ts") or ""))
    applied = state.setdefault("appliedActions", [])
    seen = {x.get("id") for x in applied}
    trades = trades_list(state)
    by_id = {t.get("id"): t for t in trades if t.get("id")}
    done, deferred = [], 0
    for a in actions:
        aid, kind = a.get("id"), a.get("type")
        if not aid or aid in seen:
            a["_file"].unlink(missing_ok=True)
            continue
        note = ""
        if kind in ("open", "close", "trim") and RUNNER_ONLY:
            # z122: the bot is autonomous - a device can't open, close or trim anything
            note = f"{kind} rejected: the bot is autonomous (z122) - only the runner trades"
            applied.append({"id": aid, "type": kind, "appliedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": note})
            seen.add(aid)
            done.append(note)
            log(f"  app action {aid} ({kind}): {note}")
            if not DRY_RUN:
                a["_file"].unlink(missing_ok=True)
            continue
        if kind in ("close", "trim") and not session:
            deferred += 1                   # z121: a close / trim waits for the NYSE session (file left in place)
            continue
        try:
            if kind == "open" and RUNNER_ONLY:
                note = "open rejected: runner-only trading (z121) - only the runner opens trades"
            elif kind == "open":
                t = dict(a.get("trade") or {})
                if t.get("id") and t["id"] not in by_id and t.get("ticker") and _f(t.get("entryPrice")):
                    t.setdefault("status", "open")
                    t["origin"] = "app"
                    t.setdefault("entryDate", today)
                    if not t.get("shares") and _f(t.get("notional")):
                        t["shares"] = round(_f(t["notional"]) / _f(t["entryPrice"]), 6)
                    trades.append(t)
                    by_id[t["id"]] = t
                    note = f"opened {t.get('direction', 'long')} {t['ticker']} {t.get('shares')} @ {t['entryPrice']}"
                else:
                    note = "open skipped (exists or incomplete)"
            elif kind in ("close", "trim"):
                t = by_id.get(a.get("tradeId"))
                req_px = px = _f(a.get("price"))
                if RUNNER_ONLY and t is not None:
                    # z121: a request, executed at this runner's own fresh quote - never the device's price
                    u = (fresh or {}).get((t.get("ticker") or "").upper())
                    px = _f(u.get("price")) if u else None
                    if t.get("status") == "open" and not px:
                        deferred += 1               # no fresh quote this pass - the next pass tries again
                        continue
                if not t or t.get("status") != "open" or not px:
                    note = (f"{kind} skipped: {a.get('tradeId')} is not in the journal (z118: the app holds these "
                            f"and sends the whole trade instead - 'adopt')" if not t
                            else f"{kind} skipped (trade not open here or no price)")
                elif kind == "trim":
                    sh = _f(a.get("shares")) or 0
                    cur = _f(t.get("shares")) or 0
                    if 0 < sh < cur:
                        entry = _f(t.get("entryPrice"))
                        d = -1 if t.get("direction") == "short" else 1
                        realized = round(d * (px - entry) * sh, 2)
                        t["shares"] = round(cur - sh, 6)
                        t["notional"] = round(t["shares"] * entry, 2)
                        t.setdefault("trims", []).append({"date": today if RUNNER_ONLY else (a.get("date") or today),
                                                          "shares": sh, "price": px, "realized": realized, "origin": "app",
                                                          "requestedPrice": req_px,
                                                          "at": (NOW_UTC or datetime.now(timezone.utc)).isoformat(timespec="seconds")})
                        t["realizedTrim"] = round((t.get("realizedTrim") or 0) + realized, 2)
                        state["bankroll"] = round(state.get("bankroll", STARTING_BANKROLL) + realized, 2)
                        note = f"trimmed {t['ticker']} {sh} @ {px} → {realized:+.2f}"
                    else:
                        note = "trim skipped (shares out of range)"
                else:
                    r = realized_pnl(t, px)
                    t.update(status="closed", exitDate=today if RUNNER_ONLY else (a.get("date") or today), exitPrice=px,
                             exitReason=a.get("reason") or "app-manual", exitRequestedPrice=req_px,
                             exitAt=(NOW_UTC or datetime.now(timezone.utc)).isoformat(timespec="seconds"))
                    total = round(r + (t.get("realizedTrim") or 0), 2)
                    t["realizedPL"] = t["pnl"] = total
                    t["won"] = total > 0
                    state["bankroll"] = round(state.get("bankroll", STARTING_BANKROLL) + r, 2)
                    note = f"closed {t['ticker']} @ {px} → {total:+.2f}"
            elif kind == "adopt":
                # z118: a trade the app holds that this journal never had (opened on a
                # device before the outbox existed, or missed by it). The snapshot is
                # the device's CURRENT version - its trims / close included - so it is
                # stored as is, never re-applied. One the journal already has merges by
                # the shared winner rule, exactly like a Supabase pull.
                t = dict(a.get("trade") or {})
                tid = t.get("id")
                if not tid or not t.get("ticker") or ts.is_runner_id(tid) or t.get("status") == "void":
                    note = "adopt skipped (no id / ticker, runner id or void)"
                elif tid not in by_id:
                    t["origin"] = t.get("origin") or "app"
                    t["_via"] = "app-adopt"
                    t["updatedAt"] = t.get("updatedAt") or a.get("ts") or ts.now_iso()
                    trades.append(t)
                    by_id[tid] = t
                    note = f"adopted {t.get('status', 'open')} {t.get('direction', 'long')} {t['ticker']} ({tid})"
                elif ts.winner(by_id[tid], t) == "remote":
                    loc = by_id[tid]
                    for k in ts.STATUS_FIELDS:
                        if k in t:
                            loc[k] = t[k]
                    loc["_via"] = "app-adopt"
                    note = f"adopt: the app's version of {loc.get('ticker')} ({tid}) wins -> {loc.get('status')}"
                else:
                    note = f"adopt: the journal already has {tid} (journal version kept)"
            elif kind in ("deposit", "withdrawal"):
                amt = _f(a.get("amount")) or 0
                state["bankroll"] = round(state.get("bankroll", STARTING_BANKROLL) + amt, 2)
                state["cashAdded"] = round((state.get("cashAdded") or 0) + amt, 2)
                state.setdefault("transactions", []).append({"id": aid, "ts": a.get("ts"), "type": kind,
                                                             "amount": amt, "origin": "app"})
                note = f"{kind} {amt:+,.2f}"
            else:
                note = f"unknown action type {kind!r} — ignored"
        except Exception as e:
            note = f"failed: {type(e).__name__}: {e}"
        applied.append({"id": aid, "type": kind, "appliedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "note": note})
        seen.add(aid)
        done.append(note)
        log(f"  app action {aid} ({kind}): {note}")
        if not DRY_RUN:
            a["_file"].unlink(missing_ok=True)
    if deferred:
        log(f"  {deferred} close/trim request(s) wait for the NYSE session or a fresh quote (left in the inbox)")
    state["appliedActions"] = applied[-2000:]
    return done


def realized_pnl(pos, exit_price):
    """Shares-based realized P&L — identical to the app's z35 freeze (long-only here)."""
    shares = pos.get("shares")
    entry = pos.get("entryPrice")
    if not shares and entry:
        shares = (pos.get("notional") or pos.get("dollars") or 0) / entry
    fees = pos.get("fees") or 0
    direction = -1 if pos.get("direction") == "short" else 1
    lev = pos.get("leverage") or 1
    return round(direction * (exit_price - entry) * shares * lev - fees, 2)


def _norm_fund(u):
    """Map master.json fields to the engine's names + derive missing ratios.
    Faithful to the app's normalizeEngineFields."""
    def n(v):
        try:
            x = float(v)
            return x if (x == x and x not in (float("inf"), float("-inf"))) else None
        except (TypeError, ValueError):
            return None
    f = {
        "marketCap": n(u.get("marketcap")) if n(u.get("marketcap")) is not None else n(u.get("marketCap")),
        "eps": n(u.get("eps")),
        "netMargin": n(u.get("profitMargin")),
        "freeCashFlow": n(u.get("freeCashFlow")),
        "roe": n(u.get("returnOnEquity")),
        "debtToEquity": n(u.get("debtToEquity")),
        "currentRatio": n(u.get("currentRatio")),
        "cash": n(u.get("cash")),
        "totalDebt": n(u.get("totalDebt")),
        "totalEquity": n(u.get("totalEquity")),
        "pe": n(u.get("pe")),
        "priceToBook": n(u.get("priceToBook")),
        "revenueGrowth": n(u.get("revenueGrowth")),
        "payoutRatio": n(u.get("payoutRatio")),
    }
    if f["debtToEquity"] is None:
        d, e = n(u.get("totalDebt")), n(u.get("totalEquity"))
        if d is not None and e and e > 0:
            f["debtToEquity"] = d / e
    ps = n(u.get("priceToSales"))
    if ps is None:
        mc, rev = f["marketCap"], n(u.get("revenue"))
        if mc is not None and rev and rev > 0:
            ps = mc / rev
    f["priceToSales"] = ps
    # master.json emits dividend_yield as a PERCENT (0.49 = 0.49%) → decimal.
    dy = n(u.get("dividend_yield"))
    if dy is None:
        dy = n(u.get("dividendYield"))
    if dy is None:
        f["dividendYield"] = None
    elif dy == 0:
        f["dividendYield"] = 0.0
    else:
        f["dividendYield"] = (dy / 100) if dy <= 40 else None
    return f


def _fund_engine(u):
    """Port of financialReasoningChain (-> conviction) + computeCompanyHealth's
    dimension composite (-> health), both 0-100. One pass over the same fundamental
    step points powers both signals, exactly as the app does."""
    f = _norm_fund(u)
    steps = {}
    mc = f["marketCap"]
    if mc is None:
        steps[1] = None
    elif mc >= 1e12: steps[1] = 100
    elif mc >= 200e9: steps[1] = 92
    elif mc >= 10e9: steps[1] = 80
    elif mc >= 2e9: steps[1] = 62
    elif mc >= 1e9: steps[1] = 52
    elif mc >= 300e6: steps[1] = 38
    elif mc >= 50e6: steps[1] = 20
    else: steps[1] = 10
    over1b = mc is not None and mc >= 1e9

    eps, nm, fcf, roe = f["eps"], f["netMargin"], f["freeCashFlow"], f["roe"]
    if eps is not None or nm is not None or fcf is not None:
        profitable = (eps is not None and eps > 0) or (nm is not None and nm > 0) or (fcf is not None and fcf > 0)
        if profitable:
            steps[2] = min(100.0, 55 + (min(30.0, nm * 150) if nm is not None else 0) + (10 if (roe is not None and roe > 0.15) else 0))
        else:
            steps[2] = max(0.0, 35 - (20 if (nm is not None and nm < -0.1) else 0))
    else:
        steps[2] = None

    de, curr, cash, debt, eq = f["debtToEquity"], f["currentRatio"], f["cash"], f["totalDebt"], f["totalEquity"]
    if de is not None or curr is not None or (cash is not None and debt is not None):
        pts = 50.0
        if de is not None: pts += 18 if de < 0.5 else 8 if de < 1 else -5 if de < 2 else -20
        if curr is not None: pts += 12 if curr > 2 else 4 if curr > 1 else -18
        if cash is not None and debt is not None and cash > debt: pts += 12
        if eq is not None and eq < 0: pts -= 25
        steps[3] = max(0.0, min(100.0, pts))
    else:
        steps[3] = None

    pe, pb, ps, rg = f["pe"], f["priceToBook"], f["priceToSales"], f["revenueGrowth"]
    if pe is not None or pb is not None or ps is not None:
        pts = 50.0
        quality = (roe is not None and roe > 0.15) or (rg is not None and rg > 0.15)
        if pe is not None and pe > 0:
            fair = 35 if quality else 20
            if pe < fair * 0.6: pts += 18
            elif pe < fair: pts += 6
            elif pe < fair * 1.7: pts -= 6
            else: pts -= 18
        if ps is not None:
            if ps < 2: pts += 8
            elif ps > 12: pts -= 10
        if pb is not None and pb < 1: pts += 10
        steps[4] = max(0.0, min(100.0, pts))
    else:
        steps[4] = None

    dy, payout = f["dividendYield"], f["payoutRatio"]
    if dy is not None and dy > 0:
        pts = 50 + min(25.0, dy * 500)
        if payout is not None:
            if payout > 0.9: pts -= 20
            elif payout < 0.6: pts += 10
        steps[5] = max(0.0, min(100.0, pts))
    elif dy == 0:
        steps[5] = 50.0
    else:
        steps[5] = None

    steps[6] = max(0.0, min(100.0, 50 + rg * 150)) if rg is not None else None

    cw = ({1: 0.12, 2: 0.22, 3: 0.20, 4: 0.22, 5: 0.10, 6: 0.14} if over1b
          else {1: 0.10, 2: 0.24, 3: 0.28, 4: 0.18, 5: 0.04, 6: 0.16})
    ssum = wsum = 0.0
    for k, p in steps.items():
        if p is not None:
            ssum += p * cw[k]; wsum += cw[k]
    conviction = (ssum / wsum) if wsum > 0 else None

    # Health = 60% dimension composite + 40% chain, dims mapped to the same step
    # points (scale/profit/strength/value/income/growth), stage-weighted.
    is_growth = (rg or 0) > 0.15 and (dy or 0) < 0.01
    is_div = (dy or 0) > 0.02
    if is_growth:
        hw = {1: 0.08, 2: 0.10, 4: 0.16, 3: 0.14, 6: 0.24, 5: 0.04}
    elif is_div:
        hw = {1: 0.08, 2: 0.18, 4: 0.13, 3: 0.18, 6: 0.08, 5: 0.17}
    else:
        hw = {1: 0.10, 2: 0.18, 4: 0.15, 3: 0.13, 6: 0.12, 5: 0.08}
    hsum = hwsum = 0.0
    for k, w in hw.items():
        p = steps.get(k)
        if p is not None:
            hsum += p * w; hwsum += w
    dim_comp = (hsum / hwsum) if hwsum > 0 else None
    if dim_comp is not None and conviction is not None:
        health = dim_comp * 0.6 + conviction * 0.4
    elif conviction is not None:
        health = conviction
    else:
        health = dim_comp
    return {"conviction": conviction, "health": health}


def score_symbol(tk, u, ctx):
    """Full signal blend for one name — the SAME pipeline open_new uses, so a held
    position is re-scored identically for thesis checks."""
    closes = load_history(tk, u["repo"])
    if not closes:
        return None
    comps = score_ticker(closes, ctx["weights"])
    # Fundamental core (reasoning-chain conviction + health) + research grade.
    fe = _fund_engine(u)
    if fe.get("conviction") is not None:
        comps["fundamentals"] = max(-1.0, min(1.0, (fe["conviction"] - 50) / 50))
    if fe.get("health") is not None:
        comps["health"] = max(-1.0, min(1.0, (fe["health"] - 50) / 50))
    g = (ctx.get("grades") or {}).get(tk)
    if g and g.get("gradeScore") is not None:
        comps["researchGrade"] = max(-1.0, min(1.0, (g["gradeScore"] - 50) / 50))
    pg = peer_grade_signal(tk, u.get("sector"), ctx["peer_ranks"])
    if pg is not None:
        comps["peerGrade"] = pg
    rg = None if ctx.get("regime_agnostic") else regime_grade_signal(u.get("sector"), u.get("beta"), ctx["mode"], ctx["quad"])
    if rg is not None:
        comps["regimeGrade"] = rg
    ca = cross_asset_for_sector(u.get("sector"), ctx["xa"])
    if ca is not None:
        comps["crossAsset"] = ca
    osig = extract_options_signal(tk)
    if osig and osig.get("ivSignal") is not None and abs(osig["ivSignal"]) > 0.05:
        comps["optionsIV"] = osig["ivSignal"]
    if ctx.get("opt_market") and ctx["opt_market"].get("marketTrend") is not None \
            and abs(ctx["opt_market"]["marketTrend"]) > 0.03:
        comps["optionsMarket"] = ctx["opt_market"]["marketTrend"]
    if ctx.get("fed"):
        fs = fed_signal(u.get("sector"), ctx["fed"])
        if fs is not None and abs(fs) > 0.02:
            comps["fed"] = fs
    pv, _ = pattern_signal(tk, ctx.get("patterns"))
    if pv is not None:
        comps["pattern"] = pv
    # z138: the catalyst - the next earnings report and how this name has reported (beat rate,
    # average surprise). A signal in the blend at the default weight; the retrain moves it.
    cs = tb.catalyst_signal(tb.catalyst_info(_CAL, tk, ctx.get("today") or datetime.now(timezone.utc).strftime("%Y-%m-%d")))
    if cs is not None and abs(cs) > 0.03:
        comps["catalyst"] = cs
    return {"comps": comps, "signed": blend_score(comps, ctx["weights"], ctx["weight_mods"])}


def compute_confidence(comps, signed):
    """0..1 — how much the signals AGREE, how STRONG the blend is, how BROAD the
    support. Distinct from raw score: a +0.4 driven by one signal is less certain
    than a +0.4 with six aligned signals."""
    active = [v for v in comps.values() if isinstance(v, (int, float)) and abs(v) > 0.05]
    if not active:
        return 0.0
    same = sum(1 for v in active if (v > 0) == (signed >= 0))
    agreement = same / len(active)
    strength = min(1.0, abs(signed) / 0.6)
    breadth = min(1.0, len(active) / 6.0)
    return round(0.5 * agreement + 0.35 * strength + 0.15 * breadth, 3)


def compute_triggers(comps, signed, level):
    """The signals that actually FIRED to make this trade attractive: aligned with
    the direction and at least `level` strong. This is what makes a trade real vs a
    marginal blend — and it's stored so the reasoning is auditable."""
    dir_pos = signed >= 0
    fired = [{"signal": k, "value": round(v, 3)}
             for k, v in comps.items()
             if isinstance(v, (int, float)) and abs(v) >= level and (v > 0) == dir_pos]
    fired.sort(key=lambda x: abs(x["value"]), reverse=True)
    return fired


def classify_horizon(comps, confidence):
    """SWING vs CORE. CORE = high confidence AND the blend leans on slow, fundamental
    / macro signals (peer grade, regime, Fed, cross-asset). Those get ridden and
    trimmed; momentum names stay swing with tight exits."""
    lt = sum(abs(comps.get(k, 0) or 0) for k in LONG_TERM_SIGNALS)
    sw = sum(abs(comps.get(k, 0) or 0) for k in SWING_SIGNALS)
    lean = lt / (lt + sw + 1e-9)
    return ("core" if (confidence >= CORE_CONF and lean >= 0.45) else "swing"), round(lean, 3)


def _review_context(state, universe, today):
    """z138 - the scoring context for the after-close review (the same signals the session uses,
    built without the entry-only pieces)."""
    grades = load_grades()
    spy_data = fetch_json(SPY_HISTORY_URL, timeout=30)
    spy_closes = [c for c in ((_f(b.get("close")) or _f(b.get("price"))) for b in spy_data) if c and c > 0] if isinstance(spy_data, list) else None
    xa = compute_cross_asset()
    opt_market = compute_options_market(list(universe.keys()))
    regime = compute_regime(spy_closes, opt_market)
    mq = fetch_json(f"{RAW}/TRAPP2-1/main/data/macro_quad.json", timeout=20)
    quad = mq["quad"] if isinstance(mq, dict) and isinstance(mq.get("quad"), int) else resolve_quad(xa, regime.get("mode"))[0]
    rc = load_regime_certainty()
    agnostic = not (rc and rc.get("dialedIn"))
    if agnostic:
        regime, quad = dict(regime, weightMods={}, longBar=0.0), None
    weights = state.get("learnedWeights", {}) if isinstance(state.get("learnedWeights"), dict) else {}
    return {"weights": weights, "weight_mods": regime.get("weightMods", {}), "mode": regime.get("mode", "choppy"),
            "peer_ranks": peer_rank_table(grades), "xa": xa, "quad": quad, "opt_market": opt_market, "fed": fed_expectation(today),
            "grades": grades, "today": today, "patterns": load_patterns(use_regime=not agnostic), "regime_agnostic": agnostic,
            "research": alt_research.ResearchData(fetch_json, RAW, universe, today)}


def _rotation_pool(state, today):
    """z140: open SHARE positions this pass re-scored below the entry bar, past their minimum hold
    (legacy positions: 5 sessions) - weakest first, as (aligned score, position)."""
    out = []
    for p in trades_list(state):
        if p.get("status") != "open" or p.get("instrument", "shares") != "shares":
            continue
        r = p.get("lastReview") or {}
        if r.get("at") != today or r.get("scoreNow") is None or r.get("decision") not in ("hold", None):
            continue
        aligned = r["scoreNow"] * (-1 if p.get("direction") == "short" else 1)
        held = r.get("daysHeld") if r.get("daysHeld") is not None else tb.sessions_between(p.get("entryDate"), today, mh.day_status)
        if aligned < MIN_SCORE and held >= max(int(p.get("minHoldDays") or 0), 5):
            out.append((round(aligned, 3), p))
    return sorted(out, key=lambda x: x[0])


def _rotate_out(state, pos, px, today, why):
    """Close a share position at px for a rotation (same accounting as any exit). Returns the cash freed."""
    entry = pos.get("entryPrice") or px
    shares = pos.get("shares") or ((pos.get("notional") or 0) / entry if entry else 0)
    r = realized_pnl(pos, px)
    total_r = round(r + (pos.get("realizedTrim") or 0), 2)
    freed = (pos.get("notional") or 0) + r
    pos.update({"status": "closed", "exitDate": today, "exitPrice": px, "exitReason": "rotation", "rotationNote": why,
                "realizedPL": total_r, "pnl": total_r, "won": total_r > 0, "sharesAtExit": round(shares or 0, 4),
                "returnPct": round((r / (shares * entry)) * 100, 2) if shares and entry else 0})
    state["bankroll"] = round(state.get("bankroll", STARTING_BANKROLL) + r, 2)
    return freed


def manage_open_positions(state, universe, today, ctx=None, review_only=False):
    """Mark to market, then apply CONFIDENCE- and HORIZON-aware management:
      • Every held name is re-scored and RE-CLASSIFIED each run (the bot reassesses
        what it holds), so a position that has grown into a high-conviction, macro-
        driven idea is treated as CORE even if it was opened as a swing.
      • SWING: tight -8% stop / +16% target; horizon exit ONLY once the signal that
        justified it has faded (no arbitrary calendar dump).
      • CORE : no fixed horizon. Disaster stop at -25%; a trailing stop protects big
        gains once in profit; a decisive signal flip closes on thesis-break. A long
        winner is TRIMMED in tiers (+25/50/100%) rather than force-sold — held like a
        value position."""
    closed_now, trimmed_now = [], []
    review_lines = []                                     # z138: every open position, every pass
    for pos in trades_list(state):
        if pos.get("status") != "open":
            continue
        u = universe.get((pos.get("ticker") or "").upper())
        if not u:
            # z138: still reviewed - no fresh quote means no action this pass, and the log says so
            review_lines.append(f"{pos.get('ticker')} {pos.get('direction')} · {tb.horizon_class(pos)} → WAIT: no fresh quote this pass "
                                f"(last ${pos.get('lastPrice') or pos.get('entryPrice')}) - managed at the next fresh quote")
            continue
        px = u["price"]                     # USD
        entry0 = pos.get("entryPrice")
        # Guard: an entry recorded in LOCAL units (e.g. cents) against a USD mark
        # shows as a 100x move. Never stop/target/close on that - flag and skip.
        if entry0 and (px / entry0 > 10 or px / entry0 < 0.1):
            pos["markSuspect"] = {"date": today, "markUsd": px, "entry": entry0,
                                  "currency": u.get("currency")}
            log(f"  ! {pos.get('ticker')}: mark ${px:,.4f} vs entry {entry0:,.4f} - unit mismatch "
                f"({u.get('currency')}); position left untouched")
            continue
        pos.pop("markSuspect", None)
        pos["lastPrice"] = px
        pos["lastPriceLocal"] = u.get("priceLocal")
        direction = -1 if pos.get("direction") == "short" else 1
        entry = pos.get("entryPrice") or px
        shares = pos.get("shares") or ((pos.get("notional") or 0) / entry if entry else 0)
        pos["pnl"] = round(direction * (px - entry) * shares, 2)
        ret = (direction * (px - entry) / entry) if entry else 0.0

        # favorable peak, for the core trailing stop
        peak = pos.get("peakPrice")
        peak = entry if peak is None else peak
        peak = max(peak, px) if direction == 1 else min(peak, px)
        pos["peakPrice"] = round(peak, 4)
        peak_ret = (direction * (peak - entry) / entry) if entry else 0.0

        # Re-score + RE-CLASSIFY this holding (the bot reassesses what it owns).
        signed_now = None
        htype = pos.get("horizonType") or "swing"
        is_alt = alt_research.asset_class(pos.get("ticker"), u) is not None
        if ctx is not None:
            sc = score_symbol(pos["ticker"], u, ctx)
            if sc:
                signed_now = sc["signed"]
                conf_now = compute_confidence(sc["comps"], signed_now)
                htype_now, lean_now = classify_horizon(sc["comps"], conf_now)
                pos["scoreNow"] = round(signed_now, 3)
                pos["confidence"] = conf_now
                pos["longTermLean"] = lean_now
                # Crypto / futures are tactical, research-driven trades: they keep
                # their swing stop/target and are never re-classified to CORE.
                if not is_alt and not pos.get("horizonClass"):     # z138: a planned horizon stays the plan
                    htype = htype_now
                    pos["horizonType"] = htype

        aligned_now = (signed_now * direction) if signed_now is not None else None
        exit_reason = None
        # z138: the position's plan - sessions held, the minimum hold, the catalyst ahead
        days_held = tb.sessions_between(pos.get("entryDate"), today, mh.day_status)
        min_hold = int(pos.get("minHoldDays") or 0)
        in_hold = days_held < min_hold
        hc = tb.horizon_class(pos)
        cat = tb.catalyst_info(_CAL, pos.get("ticker"), today)
        if pos.get("instrument") == "option":
            chain = _load_option_chain(pos["ticker"])
            mark, bid, src = tb.option_mark(pos, chain, px, today)
            ed = tb._d(pos.get("optionExpiry"))
            dte = (ed - tb._d(today)).days if ed else None
            if mark is None:
                review_lines.append(tb.review(pos, today, price=px, ret=0.0, days_held=days_held, min_hold=min_hold, decision="hold",
                                              reason="no option mark this pass (chain and model unavailable)", score_now=signed_now, cat=cat, dte=dte))
                continue
            prem = pos.get("optionPremium") or 0
            oret = (mark / prem - 1) if prem else 0.0
            pos["optionMark"], pos["optionMarkAt"], pos["optionMarkSource"] = round(mark, 4), today, src
            pos["pnl"] = tb.option_pnl(pos, mark)
            pos["returnPct"] = round(oret * 100, 2)
            hz = tb.HZ[hc]
            if dte is not None and dte <= 0:
                exit_reason = "expiry"
            elif not in_hold:
                if oret <= -tb.OPT_STOP:
                    exit_reason = "option-stop"
                elif oret >= hz["opt_tp"]:
                    exit_reason = "option-target"
                elif dte is not None and dte <= hz["exit_dte"]:
                    exit_reason = "time (expiry near)"
                elif aligned_now is not None and aligned_now <= -CORE_EXIT_SCORE:
                    exit_reason = "thesis-break"
            if exit_reason and review_only:
                review_lines.append(tb.review(pos, today, price=px, ret=oret, days_held=days_held, min_hold=min_hold, decision="flag",
                                              reason=f"{exit_reason} - acts at the next session's first pass", score_now=signed_now, cat=cat,
                                              mark=mark, mark_src=src, dte=dte))
                continue
            if exit_reason:
                exit_px = mark if exit_reason == "expiry" else (bid if bid else mark)
                r = tb.option_pnl(pos, exit_px, fee_sides=2)          # entry + exit commissions
                pos.update({"status": "closed", "exitDate": today, "exitPrice": px, "optionExit": round(exit_px, 4),
                            "exitReason": exit_reason, "realizedPL": r, "pnl": r, "won": r > 0,
                            "returnPct": round(r / (pos.get("notional") or 1) * 100, 2)})
                state["bankroll"] = round(state.get("bankroll", STARTING_BANKROLL) + r, 2)
                closed_now.append((pos["ticker"], exit_reason, r))
                review_lines.append(tb.review(pos, today, price=px, ret=oret, days_held=days_held, min_hold=min_hold, decision="close",
                                              reason=f"{exit_reason} - sold at ${exit_px:.2f} ({src})", score_now=signed_now, cat=cat,
                                              mark=mark, mark_src=src, dte=dte))
            else:
                why = (f"minimum hold ({days_held}/{min_hold} sessions) - only expiry can close it" if in_hold else
                       f"inside the plan: stop at −{tb.OPT_STOP:.0%}, take profit at +{hz['opt_tp']:.0%}, roll off at {hz['exit_dte']} DTE")
                review_lines.append(tb.review(pos, today, price=px, ret=oret, days_held=days_held, min_hold=min_hold, decision="hold",
                                              reason=why, score_now=signed_now, cat=cat, mark=mark, mark_src=src, dte=dte))
            continue
        hold_disaster = tb.HZ[hc]["disaster"]

        if htype == "core":
            if ret <= -CORE_HARD_STOP:
                exit_reason = "hard-stop"
            elif in_hold:
                pass                                       # z138: inside the minimum hold only the hard stop can close it
            elif aligned_now is not None and aligned_now <= -CORE_EXIT_SCORE:
                exit_reason = "thesis-break"
            elif peak_ret >= 0.10 and direction == 1 and px <= peak * (1 - CORE_TRAIL):
                exit_reason = "trail-stop"
            elif peak_ret >= 0.10 and direction == -1 and px >= peak * (1 + CORE_TRAIL):
                exit_reason = "trail-stop"
            else:
                # TRIM winners in tiers instead of a hard target exit (long only).
                tier = int(pos.get("trimTier") or 0)
                if not review_only and direction == 1 and tier < len(CORE_TRIM_TIERS) and ret >= CORE_TRIM_TIERS[tier]:
                    trim_shares = round(shares * CORE_TRIM_FRAC, 4)
                    if trim_shares > 0 and (shares - trim_shares) > 0:
                        realized = round((px - entry) * trim_shares - abs(px * trim_shares) * FEE_BPS, 2)
                        new_shares = round(shares - trim_shares, 4)
                        pos["shares"] = new_shares
                        pos["notional"] = round(new_shares * entry, 2)  # cost basis of remainder
                        pos["trimTier"] = tier + 1
                        pos.setdefault("trims", []).append(
                            {"date": today, "shares": trim_shares, "price": px,
                             "realized": realized, "tier": tier + 1, "atReturnPct": round(ret * 100, 2)})
                        pos["realizedTrim"] = round((pos.get("realizedTrim") or 0) + realized, 2)
                        state["bankroll"] = round(state.get("bankroll", STARTING_BANKROLL) + realized, 2)
                        trimmed_now.append((pos["ticker"], trim_shares, realized, tier + 1))
                # else: HOLD — ride it; no calendar exit
        else:
            stop, target = pos.get("stopPrice"), pos.get("targetPrice")
            if in_hold:
                if ret <= -hold_disaster:                  # z138: inside the minimum hold only a disaster move closes it
                    exit_reason = "disaster-stop"
            elif stop and ((direction == 1 and px <= stop) or (direction == -1 and px >= stop)):
                exit_reason = "stop-loss"
            elif target and ((direction == 1 and px >= target) or (direction == -1 and px <= target)):
                exit_reason = "target"
            elif aligned_now is not None and aligned_now <= -CORE_EXIT_SCORE:
                exit_reason = "thesis-break"
            elif is_alt and ctx is not None and ctx.get("research") is not None and \
                    _alt_research_against(pos, u, ctx, today):
                exit_reason = "research-break"
            else:
                # Horizon exit ONLY when the reason to hold (the signal) has faded.
                ed = pos.get("entryDate")
                if pos.get("horizonClass"):
                    # z138: the plan is in SESSIONS, and the signal must actually be re-scored as faded
                    planned = int(pos.get("horizonDaysPlanned") or tb.HZ[hc]["days"])
                    if aligned_now is not None and aligned_now < MIN_SCORE and days_held >= planned:
                        exit_reason = "horizon"
                    ed = None
                hd = pos.get("horizonDays") or SWING_HORIZON
                signal_gone = (aligned_now is None) or (aligned_now < MIN_SCORE)
                if ed and hd and signal_gone:
                    try:
                        age = (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(ed[:10], "%Y-%m-%d")).days
                        if age >= hd:
                            exit_reason = "horizon"
                    except ValueError:
                        pass

        if exit_reason and review_only:
            review_lines.append(tb.review(pos, today, price=px, ret=ret, days_held=days_held, min_hold=min_hold, decision="flag",
                                          reason=f"{exit_reason} at ${px:,.2f} - acts at the next session's first pass", score_now=signed_now, cat=cat))
            continue
        if exit_reason:
            r = realized_pnl(pos, px)
            pos["status"] = "closed"
            pos["exitDate"] = today
            pos["exitPrice"] = px
            pos["exitReason"] = exit_reason
            total_r = round(r + (pos.get("realizedTrim") or 0), 2)  # include locked trim gains
            pos["realizedPL"] = total_r
            pos["pnl"] = total_r
            pos["won"] = total_r > 0
            entry_cap = (shares or 0) * entry
            pos["returnPct"] = round((r / entry_cap) * 100, 2) if entry_cap else 0
            pos["sharesAtExit"] = round(shares or 0, 4)
            state["bankroll"] = round(state.get("bankroll", STARTING_BANKROLL) + r, 2)
            closed_now.append((pos["ticker"], exit_reason, total_r))
        # z138: the review - what it is, how it's doing, what's ahead, what the bot decided and why
        if exit_reason:
            decision, why = "close", f"{exit_reason} at ${px:,.2f}"
        elif trimmed_now and trimmed_now[-1][0] == pos.get("ticker") and pos.get("trims") and pos["trims"][-1].get("date") == today:
            decision, why = "trim", f"tier {pos.get('trimTier')} at {ret * 100:+.1f}% - riding the rest"
        elif in_hold:
            decision, why = "hold", f"minimum hold ({days_held}/{min_hold} sessions) - only a −{hold_disaster:.0%} move closes it"
        elif aligned_now is not None and aligned_now < MIN_SCORE:
            decision, why = "hold", f"signal faded to {aligned_now:+.2f} - exits at the stop / target or the {pos.get('horizonDaysPlanned') or pos.get('horizonDays') or '—'}-session horizon"
        else:
            decision, why = "hold", ("thesis intact" + (f" (aligned {aligned_now:+.2f} in its direction)" if aligned_now is not None else ""))
        review_lines.append(tb.review(pos, today, price=px, ret=ret, days_held=days_held, min_hold=min_hold, decision=decision,
                                      reason=why, score_now=signed_now, cat=cat))
    for line in review_lines:
        log("  review " + line)
    state["lastPositionReview"] = {"at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "date": today,
                                   "positions": len(review_lines), "closed": len(closed_now), "trimmed": len(trimmed_now),
                                   "kind": "after-close review (no trades)" if review_only else "session pass"}
    for tk, why, r in closed_now:
        log(f"  closed {tk} ({why}) realized ${r:,.2f}")
    for tk, sh, r, tier in trimmed_now:
        log(f"  trimmed {tk} {sh:g} sh @tier{tier} -> locked ${r:,.2f}")
    return closed_now


def _alt_research_against(pos, u, ctx, today):
    """Held crypto / futures position: has its research case turned decisively
    against it? (fundamentals don't exist for these - research is the thesis)."""
    try:
        flip, ev = alt_research.research_flip(pos, u, ctx.get("quad"), ctx["research"], today)
    except Exception as e:
        log(f"  research check failed for {pos.get('ticker')}: {e}")
        return False
    if flip is None:
        return False
    pos["researchNow"] = {"date": today, "aligned": round(flip, 3),
                          "components": (ev or {}).get("components")}
    return flip <= alt_research.EXIT_FLIP


def _alt_class_group(pos):
    cls, prof = alt_research.profile_for(pos.get("ticker"), {"assetClass": pos.get("assetClass")})
    return cls, (prof or {}).get("group")


# ----------------------------- open new trades --------------------------------
def open_new_trades(state, universe, today, regime, grades, peer_ranks, xa, quad, opt_market=None, fed=None,
                    research=None, patterns=None, regime_ctx=None):
    bankroll = state.get("bankroll", STARTING_BANKROLL)
    held = {(p.get("ticker") or "").upper() for p in trades_list(state) if p.get("status") == "open"}
    n_open = len(held)
    if n_open >= MAX_POSITIONS:
        log(f"  at max positions ({n_open}/{MAX_POSITIONS}) — no new entries")
        return []

    weights = state.get("learnedWeights", {}) if isinstance(state.get("learnedWeights"), dict) else {}
    weight_mods = regime.get("weightMods", {})
    mode = regime.get("mode", "choppy")
    # The regime still raises the bar in rough markets; explore mode softens it by
    # 0.10 so data keeps flowing (precision mode keeps the full regime bar).
    soften = 0.10 if RUNNER_MODE == "explore" else 0.0
    min_score = max(MIN_SCORE, regime.get("longBar", 0.0) - soften)
    ctx = {"weights": weights, "weight_mods": weight_mods, "mode": mode,
           "peer_ranks": peer_ranks, "xa": xa, "quad": quad,
           "opt_market": opt_market, "fed": fed, "grades": grades, "research": research,
           "patterns": patterns, "regime_agnostic": not (regime_ctx or {}).get("dialedIn", False)}
    pat_seen = {"aligned": 0, "nudged": 0, "overruled": 0}
    alt_seen = {"crypto": 0, "future": 0}
    alt_rejects = {}
    alt_detail = []

    committed = sum((p.get("notional") or 0) for p in trades_list(state) if p.get("status") == "open")
    cash = bankroll - committed
    reserve = bankroll * CASH_RESERVE_PCT
    deployable = max(0.0, cash - reserve)
    # z140 (owner direction §1d - new positions are welcome, a mix of everything): a fully invested book
    # still trades. Options (defined risk) may use half of the cash reserve, and ONE faded position
    # (past its minimum hold, re-scored below the entry bar this pass) may be rotated out for a clearly
    # stronger idea (+0.15 score). Before z140 the runner just sat out whenever the book was full.
    opt_room = max(0.0, cash - reserve * 0.5)
    rot_pool = _rotation_pool(state, today)
    constrained = deployable < bankroll * 0.02
    rotations = 0
    if constrained:
        if opt_room < bankroll * tb.OPT_TRADE_PCT[0] and not rot_pool:
            log(f"  only ${deployable:,.0f} deployable after reserve, no option room, nothing faded to rotate — sitting out")
            return []
        weakest = ", ".join(str(p.get("ticker")) + f" {a:+.2f}" for a, p in rot_pool[:3]) or "none"
        log(f"  fully invested: ${deployable:,.0f} after the reserve · option room ${opt_room:,.0f} (half the reserve) · "
            f"{len(rot_pool)} faded position(s) eligible to rotate (weakest: {weakest})")

    candidates = []
    scanned = 0
    for tk, u in universe.items():
        if tk in held:
            continue
        # Index levels (^VIX, ^MOVE, ^GSPC ...) are reference series, not
        # instruments - they can't be bought, so they never become positions.
        # They still feed regime / cross-asset signals elsewhere.
        if tk.startswith("^"):
            continue
        if not u.get("scan", True):
            continue   # price-only book (TRAPP2-1 / TRAPP2-3): managed when held, never scanned
        sc = score_symbol(tk, u, ctx)
        scanned += 1
        if sc is None:
            continue
        comps, signed = sc["comps"], sc["signed"]
        conf = compute_confidence(comps, signed)
        trigs = compute_triggers(comps, signed, TRIGGER_LEVEL)
        # Crypto & futures: no filings, earnings or research grade - price alone
        # isn't a thesis. They must clear the research gate (drivers, supply
        # chain, news, regime, seasonality) at a stricter bar.
        cls = alt_research.asset_class(tk, u)
        if cls:
            alt_seen[cls] += 1
            ev = (alt_research.evaluate(tk, u, signed, quad, research, today, min_score, ALLOW_SHORTS)
                  if research is not None else None)
            if not ev or not ev.get("eligible") or not trigs:
                if ev is None:
                    why = "no research data"
                elif not ev.get("eligible"):
                    why = (ev.get("reasons") or ["not eligible"])[0]
                else:
                    why = "no technical trigger"
                key = ("below the research/tech bar" if "below the alt bar" in why else
                       "thin evidence" if ("component" in why or "no driver" in why or "no research" in why) else
                       "vetoed" if why.startswith("vetoed") else why[:40])
                alt_rejects[key] = alt_rejects.get(key, 0) + 1
                if ev:
                    alt_detail.append((signed, tk, ev))
                continue
            candidates.append({"ticker": tk, "score": ev["rankScore"], "direction": ev["direction"],
                               "components": comps, "confidence": conf, "triggers": trigs,
                               "price": u["price"], "sector": u["sector"], "name": u["name"],
                               "priceLocal": u.get("priceLocal"), "currency": u.get("currency"),
                               "fxRate": u.get("fxRate"), "alt": ev})
            continue
        # Only take a trade that actually has concrete triggers firing for it.
        pat = comps.get("pattern")
        pat_fired = pat is not None and pat >= TRIGGER_LEVEL and any(t["signal"] == "pattern" for t in trigs)
        if pat is not None and pat > 0:
            pat_seen["aligned"] += 1
        entry_reason = None
        if signed >= min_score and len(trigs) >= MIN_TRIGGERS:
            entry_reason = "blend"
        elif pat_fired and PATTERN_NUDGE > 0 and signed >= min_score - PATTERN_NUDGE:
            # A confirmed, trusted chart pattern can tip a NEAR-threshold name over
            # the line - the rest of the blend still has to be almost there.
            entry_reason = "pattern-nudge"
            pat_seen["nudged"] += 1
        elif pat is not None and pat >= TRIGGER_LEVEL:
            pat_seen["overruled"] += 1          # TA said yes, the other factors didn't
        if entry_reason:
            candidates.append({"ticker": tk, "score": signed, "direction": "long",
                               "components": comps, "confidence": conf, "triggers": trigs,
                               "price": u["price"], "sector": u["sector"], "name": u["name"],
                               "priceLocal": u.get("priceLocal"), "currency": u.get("currency"), "fxRate": u.get("fxRate"),
                               "entryReason": entry_reason})
        elif signed <= SHORT_SCORE and len(trigs) >= MIN_TRIGGERS and \
                (ALLOW_SHORTS or tb.chain_fresh(_load_option_chain(tk), today)):
            # z138: a bearish idea trades as a short sale (RUNNER_ALLOW_SHORTS) or as a put - defined risk
            candidates.append({"ticker": tk, "score": abs(signed), "direction": "short",
                               "components": comps, "confidence": conf, "triggers": trigs,
                               "price": u["price"], "sector": u["sector"], "name": u["name"],
                               "priceLocal": u.get("priceLocal"), "currency": u.get("currency"), "fxRate": u.get("fxRate"),
                               "putOnly": not ALLOW_SHORTS})
    candidates.sort(key=lambda c: c["score"], reverse=True)
    n_long = sum(1 for c in candidates if c["direction"] == "long")
    n_short = sum(1 for c in candidates if c["direction"] == "short")
    log(f"  mode={RUNNER_MODE} regime={mode} quad={quad} · scanned {scanned} · {n_long} long"
        f"{f' / {n_short} short' if ALLOW_SHORTS else ''} pass (min {min_score:.2f})")
    if patterns:
        log(f"  pattern factor: {pat_seen['aligned']} names with a bullish pattern read · "
            f"{pat_seen['nudged']} nudged over the bar · {pat_seen['overruled']} strong patterns overruled by the other factors")
    n_alt_pass = sum(1 for c in candidates if c.get("alt"))
    if sum(alt_seen.values()):
        log(f"  alt research: {alt_seen['crypto']} crypto + {alt_seen['future']} futures scanned · "
            f"{n_alt_pass} passed the research gate"
            + (" · rejected: " + ", ".join(f"{k} {v}" for k, v in sorted(alt_rejects.items(), key=lambda x: -x[1]))
               if alt_rejects else ""))
        # The strongest technical setups that research held back - so the log
        # explains WHY a hot chart wasn't bought.
        for sgn, tk, ev in sorted(alt_detail, key=lambda x: -x[0])[:5]:
            comp = ", ".join(f"{k} {v:+.2f}" for k, v in (ev.get("components") or {}).items())
            r = ev.get("research")
            log(f"    {tk:8s} tech {sgn:+.2f} · research {r:+.2f} [{comp}] · news {ev.get('newsCoverage', 0)}"
                if r is not None else f"    {tk:8s} tech {sgn:+.2f} · {', '.join(ev.get('reasons') or [])}")

    # Concurrency caps for crypto / futures (per class, and per futures complex).
    open_alt = {"crypto": 0, "future": 0}
    open_groups = {}
    for p in trades_list(state):
        if p.get("status") != "open":
            continue
        cls_p, grp_p = _alt_class_group(p)
        if cls_p:
            open_alt[cls_p] += 1
            if cls_p == "future" and grp_p:
                open_groups[grp_p] = open_groups.get(grp_p, 0) + 1

    opened = []
    slots = min(MAX_NEW_TRADES, MAX_POSITIONS - n_open)
    primary_count = {}
    # Explore: pattern-led setups rank below full-blend ones by construction, so
    # without help they'd never be sampled. Reserve the LAST slot for the best
    # candidate whose triggers include a chart pattern (it still passed the blend
    # or the near-threshold nudge above) - so TA-triggered trades get tested too.
    if RUNNER_MODE == "explore" and slots >= 2 and len(candidates) > slots - 1:
        is_pat = lambda c: any(t.get("signal") == "pattern" for t in (c.get("triggers") or []))
        if not any(is_pat(c) for c in candidates[:slots]):
            best = next((c for c in candidates[slots - 1:] if is_pat(c)), None)
            if best:
                candidates.remove(best)
                candidates.insert(slots - 1, best)
                log(f"  reserved a slot for pattern-led {best['ticker']} (score {best['score']:.2f}, "
                    f"{best.get('entryReason', 'blend')})")
    cell_count = {}                                            # z138: direction × horizon × instrument, this session
    for c in candidates:
        if len(opened) >= slots:
            break
        # Breadth: at most MAX_SAME_PRIMARY new trades per run may share the same
        # lead trigger - the data set should cover many reasons to trade, not one.
        primary = (c.get("triggers") or [{}])[0].get("signal") or "blend"
        if primary_count.get(primary, 0) >= MAX_SAME_PRIMARY:
            continue
        alt = c.get("alt")
        if alt:
            cls_c, grp_c = alt["assetClass"], alt.get("group")
            if open_alt.get(cls_c, 0) >= alt_research.MAX_OPEN[cls_c]:
                log(f"  skip {c['ticker']}: {cls_c} cap ({alt_research.MAX_OPEN[cls_c]}) reached")
                continue
            if cls_c == "future" and open_groups.get(grp_c, 0) >= alt_research.MAX_PER_GROUP:
                log(f"  skip {c['ticker']}: already hold a {grp_c} future")
                continue
        conf = c.get("confidence", 0.0)
        htype, lean = classify_horizon(c["components"], conf)
        if alt:
            htype = "swing"            # research-driven alt trades are tactical
        is_short = c.get("direction") == "short"
        # z138: horizon (short / medium / long, steered toward the book's mix), then the instrument
        cat = tb.catalyst_info(_CAL, c["ticker"], today)
        mix = tb.open_mix(trades_list(state))
        hclass, h_why, steered = tb.choose_horizon(conf, lean, cat, c.get("direction"), mix, CORE_CONF)
        if alt:
            hclass, h_why, steered = "short", "research-driven alt trade (tactical)", False
        cell = f"{c.get('direction')}|{hclass}"
        option = None
        chain = None if alt else _load_option_chain(c["ticker"])
        cat_in_window = bool(cat and cat.get("days") is not None and 1 <= cat["days"] <= tb.HZ[hclass]["days"] + 5
                             and abs(c["components"].get("catalyst") or 0) >= 0.10)
        if not alt and (conf >= tb.OPT_CONF or cat_in_window or c.get("putOnly")) and tb.chain_fresh(chain, today):
            prem_open = sum((p.get("notional") or 0) for p in trades_list(state) if p.get("status") == "open" and p.get("instrument") == "option")
            option, o_why = tb.pick_contract(chain, c.get("direction"), hclass, today, cat if cat_in_window else None)
            if option and prem_open >= bankroll * tb.OPT_BOOK_PCT:
                option, o_why = None, f"option book full ({prem_open / bankroll:.1%} of the bankroll at risk)"
            if not option:
                log(f"  {c['ticker']}: no option ({o_why})" + ("" if not c.get("putOnly") else " - bearish idea skipped (shorts off)"))
        if c.get("putOnly") and not option:
            continue
        cell = f"{cell}|{'option' if option else 'shares'}"
        if cell_count.get(cell, 0) >= 2:
            continue                       # z138: variety - at most two new trades per direction × horizon × instrument per session
        explore = not any(tb.horizon_class(p) == hclass and p.get("direction") == c.get("direction")
                          and (p.get("instrument") == "option") == bool(option)
                          for p in trades_list(state) if p.get("status") == "open")
        if option:
            # premium = 1.0 % → 2.5 % of the bankroll with confidence; the premium is the most it can lose
            budget = tb.option_budget(bankroll, conf, explore)        # 1 % → 2.5 %; deep conviction 4 %
            cap_now = min(budget, max(deployable, opt_room) if constrained else deployable)
            contracts = int(cap_now // (option["ask"] * 100))
            if contracts < 1:
                log(f"  {c['ticker']}: one contract (${option['ask'] * 100:,.0f}) is more than the ${cap_now:,.0f} available "
                    f"(budget ${budget:,.0f}{f', option room left ${opt_room:,.0f}' if constrained else ''})")
                if c.get("putOnly"):
                    continue
                option = None
        if option:
            shares = 0
            notional = round(contracts * option["ask"] * 100, 2)
            fees = round(tb.OPT_FEE * contracts, 2)
            opt_room -= notional
        else:
            if constrained and deployable < bankroll * 0.02:
                # z140: shares need cash - rotate one faded position out if this idea is clearly stronger
                if alt or rotations >= 1 or not rot_pool or c["score"] - rot_pool[0][0] < 0.15:
                    continue
                weak_al, weak = rot_pool.pop(0)
                wu = universe.get((weak.get("ticker") or "").upper())
                if not wu:
                    continue
                freed = _rotate_out(state, weak, wu["price"], today,
                                    f"rotation: faded to {weak_al:+.2f}, capital to {c['ticker']} (score {c['score']:.2f})")
                rotations += 1
                deployable += freed
                bankroll = state.get("bankroll", bankroll)
                log(f"  rotated out {weak.get('ticker')} (faded {weak_al:+.2f}, realized ${weak.get('realizedPL', 0):,.2f}) → room for {c['ticker']} (score {c['score']:.2f})")
            # Confidence-scaled sizing: 0.6x (low) → 1.4x (high), hard-capped at 1.6x base.
            size = min(bankroll * POSITION_PCT * (0.6 + 0.8 * conf),
                       deployable, bankroll * POSITION_PCT * 1.6)
            if alt:
                size *= alt["sizeMult"]    # smaller + volatility-scaled for crypto / futures
            if explore and not alt:
                size *= 0.7                # z138: a first trade in an empty cell is a smaller, exploratory one
            if size < bankroll * MIN_POSITION_PCT:
                if alt:
                    continue
                break
            shares = round(size / c["price"], 4)
            if shares <= 0:
                continue
            notional = round(shares * c["price"], 2)
            fees = round(notional * FEE_BPS, 2)
        if option:
            stop = target = None
            horizon_days = tb.HZ[hclass]["days"]
            style = f"option-{hclass}"
            htype = "core" if hclass == "long" else "swing"
        elif hclass in ("short", "medium") and not alt:
            hz = tb.HZ[hclass]
            stop = round(c["price"] * ((1 + hz["stop"]) if is_short else (1 - hz["stop"])), 4)
            target = round(c["price"] * ((1 - hz["target"]) if is_short else (1 + hz["target"])), 4)
            horizon_days = hz["days"]
            style = "catalyst" if (cat_in_window and hclass == "short") else ("momentum" if hclass == "short" else "swing")
            htype = "swing"
        elif hclass == "long" and not alt:
            htype = "core"
        if option or (hclass in ("short", "medium") and not alt):
            pass                           # z138: set above (option / short / medium shares)
        elif htype == "core":
            stop = round(c["price"] * ((1 + CORE_HARD_STOP) if is_short else (1 - CORE_HARD_STOP)), 4)
            target = None                                   # ride it; managed by trim + thesis-break
            horizon_days = None                             # no fixed-calendar exit
            style = "core"
        elif alt:
            sp, tp = alt["stopPct"], alt["targetPct"]          # volatility-based
            stop = round(c["price"] * ((1 + sp) if is_short else (1 - sp)), 6)
            target = round(c["price"] * ((1 - tp) if is_short else (1 + tp)), 6)
            horizon_days = SWING_HORIZON
            style = "research"
        else:
            stop = round(c["price"] * ((1 + SWING_STOP) if is_short else (1 - SWING_STOP)), 4)
            target = round(c["price"] * ((1 - SWING_TARGET) if is_short else (1 + SWING_TARGET)), 4)
            horizon_days = int(SWING_HORIZON * (1 + conf))  # more confidence → longer leash
            style = "momentum"
        trigs = c.get("triggers", [])
        trade = {
            "id": f"{c['ticker']}-{today}-runner-{int(datetime.now(timezone.utc).timestamp())}{'-opt' if option else ''}",
            "ticker": c["ticker"], "name": c["name"], "sector": c["sector"],
            "direction": "short" if is_short else "long", "instrument": "shares",
            "style": style, "horizonType": htype, "horizonDays": horizon_days,
            "entryDate": today, "exitDate": None,
            "entryPrice": c["price"], "exitPrice": None, "peakPrice": c["price"],
            # USD basis (z80): entryPrice/stop/target/P&L are USD; the local quote
            # and the conversion used are kept for audit.
            "priceBasis": "USD", "currency": c.get("currency") or "USD",
            "entryPriceLocal": c.get("priceLocal"), "fxAtEntry": c.get("fxRate"),
            "shares": shares, "notional": notional, "dollars": notional,
            "allocationPct": round(notional / bankroll * 100, 2), "leverage": 1, "hedge": False,
            "stopPrice": stop, "targetPrice": target, "trimTier": 0, "trims": [],
            "fees": fees, "realizedPL": 0, "pnl": 0, "returnPct": 0, "won": False,
            "exitReason": "open", "status": "open",
            "conviction": round(c["score"] if not is_short else -c["score"], 3),
            "confidence": round(conf, 3), "longTermLean": lean,
            "components": {k: round(v, 3) for k, v in c["components"].items()},
            "topDrivers": trigs,
            "signalsThatHelped": [t["signal"] for t in trigs],
            "rationale": [f"{htype.upper()} · conf {conf:.2f} · score {c['score']:.2f}. Triggers: " +
                          (", ".join(f"{t['signal']} {t['value']:+.2f}" for t in trigs) or "none")],
            "cashAfter": round(cash - notional, 2),
            "placedBy": "runner",
            # learning metadata: why this trade exists and under which policy
            "mode": RUNNER_MODE, "entryReason": c.get("entryReason") or ("research" if alt else "blend"),
            "primaryTrigger": primary,
            # what the bot believed about the regime when it entered, and whether
            # it used it - so a win can't credit a regime call that was wrong
            "regimeContext": regime_ctx,
            # z138: how long it means to hold, why, and what it's waiting for
            "horizonClass": hclass, "horizonReason": h_why, "steered": steered, "explore": explore,
            "horizonDaysPlanned": tb.HZ[hclass]["days"], "minHoldDays": tb.HZ[hclass]["min_hold"],
            "plannedExitBy": (datetime.strptime(today, "%Y-%m-%d") + timedelta(days=round(tb.HZ[hclass]["days"] * 7 / 5))).strftime("%Y-%m-%d"),
            "catalyst": cat,
        }
        if option:
            trade.update({"instrument": "option", "optionType": option["type"], "optionStrike": option["strike"],
                          "optionExpiry": option["expiry"], "optionContracts": contracts, "optionPremium": option["ask"],
                          "optionMark": option["mid"], "optionMarkAt": today, "optionDelta": option.get("delta"),
                          "optionIV": option.get("iv"), "optionChainAsOf": option.get("chainAsOf"), "multiplier": 100,
                          "maxLoss": notional, "leverage": 1})
            trade["rationale"].append(
                f"OPTION {option['type'].upper()} {option['strike']} exp {option['expiry']} ({option['dte']}d) ×{contracts} @ ${option['ask']:.2f} ask"
                f" · Δ {option.get('delta')} · premium ${notional:,.0f} = the most it can lose" +
                (" · catalyst inside the horizon" if cat_in_window else f" · conviction {conf:.2f}"))
        trade["rationale"].append(f"{hclass.upper()} horizon (~{tb.HZ[hclass]['days']} sessions, min hold {tb.HZ[hclass]['min_hold']}): {h_why}"
                                  + (" · exploratory size" if explore else "") +
                                  (f" · earnings {cat['date']} ({cat['days']}d, beat rate {cat.get('beatRate')})" if cat and cat.get("date") else ""))
        pv, pdet = pattern_signal(c["ticker"], patterns)
        if pdet:
            trade["pattern"] = {"value": round(pv, 3), "calls": pdet[:3], "scanAsOf": (patterns or {}).get("asOf")}
        if alt:
            det = alt.get("detail") or {}
            trade["assetClass"] = alt["assetClass"]
            trade["altGroup"] = alt.get("group")
            trade["research"] = {
                "composite": alt.get("research"), "components": alt.get("components"),
                "newsCoverage": alt.get("newsCoverage"), "stopPct": alt["stopPct"],
                "targetPct": alt["targetPct"], "sizeMult": alt["sizeMult"], "dailyVol": alt.get("dailyVol"),
                "drivers": det.get("drivers"), "supplyChain": det.get("supplyChain"),
                "news": det.get("news"), "seasonality": det.get("seasonality"), "regime": det.get("regime"),
            }
            comp_txt = ", ".join(f"{k} {v:+.2f}" for k, v in (alt.get("components") or {}).items())
            trade["rationale"].append(
                f"RESEARCH {alt['assetClass']}/{alt.get('group')} · composite {alt.get('research'):+.2f} "
                f"({comp_txt}) · {alt.get('newsCoverage', 0)} headline(s) · stop {alt['stopPct']*100:.1f}% / "
                f"target {alt['targetPct']*100:.1f}% · size x{alt['sizeMult']:.2f}")
            open_alt[alt["assetClass"]] = open_alt.get(alt["assetClass"], 0) + 1
            if alt["assetClass"] == "future" and alt.get("group"):
                open_groups[alt["group"]] = open_groups.get(alt["group"], 0) + 1
        # cash decrements on buy so the book stays consistent
        committed += notional
        cash -= notional
        deployable -= notional
        primary_count[primary] = primary_count.get(primary, 0) + 1
        cell_count[cell] = cell_count.get(cell, 0) + 1
        state.setdefault("trades", []).append(trade)
        opened.append((c["ticker"], notional, c["score"]))
        if not state.get("startedAt"):
            state["startedAt"] = today
    for tk, notional, sc in opened:
        t = next((x for x in reversed(trades_list(state)) if x.get("ticker") == tk and x.get("status") == "open"), {})
        what = (f"{t.get('optionType', '').upper()} {t.get('optionStrike')} {t.get('optionExpiry')} ×{t.get('optionContracts')}"
                if t.get("instrument") == "option" else f"{t.get('direction')} shares")
        log(f"  opened {tk} {what} · {t.get('horizonClass', '?')} · ${notional:,.0f} (score {sc:.2f})")
    return opened


# ----------------------------- performance + equity ---------------------------
def _perf_by_regime_mode(closed, pl):
    """z101 - did trades do better when the regime was trusted? Closed trades split
    by the regime context recorded at entry: regime-aware (dialed in) vs regime-
    agnostic vs unknown (older trades), and by whether the entry-day regime call
    later proved right. Evidence for (or against) leaning on the regime."""
    def stats(rows):
        if not rows:
            return {"n": 0}
        w = [r for r in rows if pl(r) > 0]
        return {"n": len(rows), "winRate": round(len(w) / len(rows), 3),
                "avgPnl": round(sum(pl(r) for r in rows) / len(rows), 2), "totalPnl": round(sum(pl(r) for r in rows), 2)}
    rc = lambda b: (b.get("regimeContext") or {})
    return {
        "aware": stats([b for b in closed if rc(b).get("dialedIn") is True]),
        "agnostic": stats([b for b in closed if rc(b).get("dialedIn") is False]),
        "unknown": stats([b for b in closed if "dialedIn" not in rc(b)]),
        "callRight": stats([b for b in closed if b.get("regimeCallCorrect") is True]),
        "callWrong": stats([b for b in closed if b.get("regimeCallCorrect") is False]),
    }


# ----------------------------- ledger (z117) -----------------------------------
LEDGER_VERSION = 2
# The z112 opening adjustment (ledgerVersion 1) compared the stored book's bankroll
# with a ledger that already held LATER events pulled from Supabase (the 2026-10-04
# closes, the 10-14 app trims), so it absorbed -$738.71 of real results. The stored
# book's facts come from git history and are re-derived once, as of its own date.
OPENING_FACTS = {
    "ledger-open-2026-10-16": {
        "storedBankroll": 110369.24, "asOfTs": "2026-06-22T23:14:39Z",
        "source": "TRAPP2-BOT@d306933 data/bot_training_data.json (generatedAt 2026-06-22T23:14:39.141Z, 29 trades)"},
}


def fix_v1_opening(state):
    for x in state.get("transactions") or []:
        if x.get("type") != "adjustment" or x.get("asOf"):
            continue
        f = OPENING_FACTS.get(x.get("id"))
        if not f:
            log(f"ledger: {x.get('id')} has no stored-book date on record - left as is ({x.get('amount')})")
            continue
        asof = f["asOfTs"][:10]
        new = round(f["storedBankroll"] - ts.ledger_asof(state, STARTING_BANKROLL, asof, exclude_ids={x.get("id")}), 2)
        log(f"ledger: opening adjustment {x.get('id')} re-derived as of {asof}: "
            f"{float(x.get('amount') or 0):+,.2f} -> {new:+,.2f}")
        x.update({"correctedFrom": x.get("amount"), "recordedAt": x.get("ts"), "amount": new, "ts": f["asOfTs"],
                  "asOf": asof, "storedBankroll": f["storedBankroll"], "source": f["source"],
                  "reason": "opening reconciliation: the 2026-06-22 book's bankroll not backed by trades realized "
                            "through that date (z117 re-derived; z112 had compared it with later events)"})


def reconcile_ledger(state, before, today):
    """z117 - one journal ledger, honest about time. Returns repaired trade ids.
    1. pnl that contradicts a closed trade's own prices is repaired (trade_sync.repair_pnl).
    2. A pre-ledger book (no ledgerVersion): its stored bankroll is dated (generatedAt);
       the gap vs what had been realized BY THAT DATE becomes ONE dated adjustment.
       Events after the date (e.g. a later close pulled from Supabase) keep moving the
       bankroll - they are never absorbed into the gap.
    3. A ledgerVersion-1 book (z112's undated adjustment) is re-derived once."""
    fixed = ts.repair_pnl(trades_list(state))
    for tid, was, now in fixed:
        log(f"ledger: pnl repaired {tid}: {was:+,.2f} -> {now:+,.2f} (entry/exit/shares and realizedPL agree)")
    ver = state.get("ledgerVersion") or 0
    if ver < 1:
        asof_ts = str(state.get("generatedAt") or "") or ts.now_iso()
        stored = before if isinstance(before, (int, float)) else STARTING_BANKROLL
        gap = round(stored - ts.ledger_asof(state, STARTING_BANKROLL, asof_ts[:10]), 2)
        if abs(gap) >= 0.01:
            state.setdefault("transactions", []).append({
                "id": f"ledger-open-{today}", "ts": asof_ts, "recordedAt": ts.now_iso(), "type": "adjustment",
                "amount": gap, "asOf": asof_ts[:10], "storedBankroll": stored,
                "reason": "opening reconciliation: stored bankroll not backed by trades realized through its own date"})
            log(f"ledger: stored bankroll {stored:,.2f} (as of {asof_ts[:10]}) vs trades realized by then -> "
                f"{gap:+,.2f} recorded as one dated adjustment")
    elif ver == 1:
        fix_v1_opening(state)
    state["ledgerVersion"] = LEDGER_VERSION
    return fixed


def recompute(state, universe, today):
    bets = trades_list(state)
    closed = [b for b in bets if b.get("status") == "closed"]
    open_b = [b for b in bets if b.get("status") == "open"]

    def _pl(b):
        # z117: what the trade put into the bankroll - exit P/L + its trims, the
        # number the ledger sums. realizedPL alone left out trims (a full exit by
        # partial sells read as a $0 loss) and disagreed with pnl on 21 trades.
        return round(ts.exit_realized(b) + ts.trim_total(b), 2)

    wins = [b for b in closed if _pl(b) > 0]
    losses = [b for b in closed if _pl(b) < 0]
    flat = len(closed) - len(wins) - len(losses)      # opened and sold at the same price
    total_pnl = round(sum(_pl(b) for b in closed), 2)
    win_rate = round(len(wins) / (len(wins) + len(losses)), 3) if (wins or losses) else None
    avg_win = round(sum(_pl(b) for b in wins) / len(wins), 2) if wins else None
    avg_loss = round(sum(_pl(b) for b in losses) / len(losses), 2) if losses else None
    pf = (round(abs((avg_win * len(wins)) / (avg_loss * len(losses))), 2)
          if avg_loss and wins and losses else None)

    # mark-to-market book value = bankroll + open unrealized
    open_pnl = sum((b.get("pnl") or 0) for b in open_b)
    book = round(state.get("bankroll", STARTING_BANKROLL) + open_pnl, 2)

    state["counts"] = {"total": len(bets), "open": len(open_b), "closed": len(closed),
                       "wins": len(wins), "losses": len(losses), "flat": flat}
    state["performance"] = {"winRate": win_rate, "totalPnl": total_pnl,
                            "avgWin": avg_win, "avgLoss": avg_loss, "profitFactor": pf,
                            "byRegimeMode": _perf_by_regime_mode(closed, _pl)}
    # z117: capital in / out and the dated opening adjustment are the BASE, not
    # performance - money the trade record can't explain is not a return.
    flows = sum(float(x.get("amount") or 0) for x in (state.get("transactions") or [])
                if x.get("type") in ts.CAPITAL_TYPES)
    base = STARTING_BANKROLL + flows
    state["allTimeReturnPct"] = (round((state.get("bankroll", STARTING_BANKROLL) - base) / base * 100, 2)
                                 if base > 0 else None)
    state["openPositions"] = [{"ticker": b["ticker"], "direction": b.get("direction"),
                               "instrument": b.get("instrument"), "entryDate": b.get("entryDate"),
                               "entryPrice": b.get("entryPrice"), "conviction": b.get("conviction")}
                              for b in open_b]

    # one equity-curve point per day (replace today's if it already exists)
    curve = state.get("equityCurve") if isinstance(state.get("equityCurve"), list) else []
    curve = [p for p in curve if p.get("date") != today]
    curve.append({"date": today, "value": book})
    state["equityCurve"] = curve[-400:]

    # benchmark anchor for vs-SPY (price captured once on first run)
    spy = universe.get("SPY")
    if spy and not state.get("benchmarkStart"):
        state["benchmarkStart"] = spy["price"]

    state["schema"] = "valuatio-bot-training/v1"
    state["generatedAt"] = datetime.now(timezone.utc).isoformat()
    state["startingBankroll"] = STARTING_BANKROLL
    return state


# ----------------------------- main -------------------------------------------
# ----------------------------- session gate (z121) ------------------------------
_AMERICAS = (".TO", ".V", ".NE", ".CN", ".MX", ".SA", ".BA")


def session_tradable(ticker):
    """Can a NEW position in this ticker be opened at a live price during the NYSE
    session? US listings, Americas exchanges, crypto (-USD) and futures (=F): yes.
    Asian / European listings: their home market is shut (or closing) during the
    NYSE session, so their "price" is an old close - no entries."""
    t = (ticker or "").upper()
    if not t or t.startswith("^") or t.endswith("=X"):
        return False
    if t.endswith("-USD") or t.endswith("=F"):
        return True
    return "." not in t or t.endswith(_AMERICAS)


def _age_min(asof, now):
    try:
        d = datetime.fromisoformat(str(asof).replace("Z", "+00:00"))
    except Exception:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return (now - d).total_seconds() / 60.0


def fresh_universe(universe, now):
    """Only rows whose live quote is at most MAX_QUOTE_AGE_MIN old trade this pass."""
    out = {}
    for t, u in universe.items():
        a = _age_min(u.get("asOf"), now)
        if a is not None and -5 <= a <= MAX_QUOTE_AGE_MIN:
            out[t] = u
    return out


def entry_window(sess, state):
    last = state.get("lastEntryScan") or {}
    if last.get("date") == sess["date"]:
        return False, (f"today's entry scan already ran ({str(last.get('et', ''))[11:16]} ET) - one scan per "
                       f"session; exits and requests still run every pass")
    if FORCE_SESSION and not sess["open"]:
        return True, "forced (test)"
    if sess["minute"] < ENTRY_AFTER_MIN:
        return False, f"before {ENTRY_AFTER_MIN // 60:02d}:{ENTRY_AFTER_MIN % 60:02d} ET - entries wait for the opening auction to settle"
    if sess.get("closeMin") is not None and sess["minute"] >= sess["closeMin"] - ENTRY_STOP_BEFORE_CLOSE_MIN:
        return False, f"within {ENTRY_STOP_BEFORE_CLOSE_MIN} min of the close - no new entries"
    return True, ""


def stamp_execution(state, ids_before, open_before, sess, universe, now_utc):
    """Every fill gets its UTC time and quote time, so "only in session" is auditable."""
    now = now_utc.isoformat(timespec="seconds")
    for t in trades_list(state):
        tid, tk = t.get("id"), (t.get("ticker") or "").upper()
        u = universe.get(tk) or {}
        if tid not in ids_before and not t.get("entryAt"):
            t.update(entryAt=now, entryQuoteAt=u.get("asOf"), entrySession=sess.get("label"))
        if tid in open_before and t.get("status") == "closed" and not t.get("exitAt"):
            t.update(exitAt=now, exitQuoteAt=u.get("asOf"))
            if not session_tradable(tk):
                t["exitPriceBasis"] = "last available price (home market not in the NYSE session)"


def _write_state(state, msg):
    if DRY_RUN:
        log(f"DRY RUN - not writing state ({msg}).")
        return
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, separators=(",", ":")))
    log(f"{msg} · wrote {STATE_FILE}")


def main():
    now_utc = NOW_UTC or datetime.now(timezone.utc)
    today = now_utc.strftime("%Y-%m-%d")
    log(f"=== run {today} · dry_run={DRY_RUN} ===")
    log(f"limits: maxNew={MAX_NEW_TRADES} maxPos={MAX_POSITIONS} sizePct={POSITION_PCT*100:.0f} "
        f"reservePct={CASH_RESERVE_PCT*100:.0f} minScore={MIN_SCORE}")

    state = load_state()
    log(f"state: bankroll ${state.get('bankroll', STARTING_BANKROLL):,.2f} · "
        f"{sum(1 for b in trades_list(state) if b.get('status')=='open')} open · "
        f"{len(trades_list(state))} total trades")

    sess = mh.session_now(now_utc)
    in_session = bool(sess["open"]) or FORCE_SESSION
    log(f"market: {sess['text']}" + (" · RUNNER_FORCE_SESSION=1 (test): treated as open" if FORCE_SESSION and not sess["open"] else ""))
    universe = load_universe()
    if not universe:
        log("✗ no universe data — aborting (won't write empty state)")
        sys.exit(1)
    fresh = fresh_universe(universe, now_utc) if in_session else {}
    if in_session:
        log(f"quotes: {len(fresh)}/{len(universe)} fresh (≤ {MAX_QUOTE_AGE_MIN:.0f} min old) - only these can trade this pass")
    # z121: the app no longer trades, so Supabase is this journal's mirror, not an input
    if RUNNER_ONLY:
        log("runner-only (z121): only this runner trades; Supabase mirrors the journal and is not read back")
    else:
        pull_supabase_trades(state)
    app_done = apply_app_actions(state, today, sess if in_session else None, fresh)
    before = state.get("bankroll")
    reconcile_ledger(state, before, today)       # z117: pnl repair + dated opening adjustment
    state["bankroll"] = ts.ledger_bankroll(state, STARTING_BANKROLL)
    if before is not None and abs((before or 0) - state["bankroll"]) >= 0.01:
        log(f"bankroll re-derived from the journal: {before:,.2f} → {state['bankroll']:,.2f}")
    sig_before = {t.get("id"): _sig(t) for t in trades_list(state)}
    if app_done:
        log(f"applied {len(app_done)} app action(s)")

    if not in_session:
        # z121: nothing trades outside the NYSE regular session - nights, weekends, holidays,
        # after a 1 pm early close. Bookkeeping still applies (adopting a device's trades,
        # cash in / out); close / trim requests wait in the inbox for the session.
        # z138: once per trading day, after the close, every open position gets a full review at the
        # closing prices (thesis re-scored, catalysts, stops / targets / expiries) - analysis only;
        # anything it flags acts at the next session's first pass.
        et = mh.et_now(now_utc)
        ok_day, close_min, _ = mh.day_status(et.date())
        eod = ok_day and close_min is not None and (et.hour * 60 + et.minute) >= close_min
        if eod and (state.get("lastPositionReview") or {}).get("date") != sess["date"] and any(t.get("status") == "open" for t in trades_list(state)):
            log("after-close review: every open position at the closing prices (no trades)")
            try:
                load_catalysts()
            except Exception:
                pass
            review_ctx = None
            try:
                review_ctx = _review_context(state, universe, sess["date"])
            except Exception as e:
                log(f"  review without re-scoring ({e})")
            manage_open_positions(state, universe, sess["date"], review_ctx, review_only=True)
            finalize_state(state, sig_before)
            recompute(state, universe, today)
            _write_state(state, "after-close review of open positions (no trades)")
            return
        if not app_done:
            log("market closed - no trades; journal unchanged (nothing written)")
            return
        finalize_state(state, sig_before)
        recompute(state, universe, today)
        _write_state(state, f"closed-market pass: {len(app_done)} bookkeeping action(s), no trades")
        return

    if ACTIONS_ONLY:
        # 15-minute pass: requests at fresh quotes + marks, no new-trade scan.
        finalize_state(state, sig_before)
        recompute(state, universe, today)
        _write_state(state, f"actions-only pass: {len(app_done)} applied · bankroll ${state['bankroll']:,.2f}")
        return

    # z138: catalysts (earnings calendar + beat record) for scoring, horizons and reviews
    try:
        n_cal = len(load_catalysts())
        log(f"catalysts: earnings calendar for {n_cal} tickers")
    except Exception as e:
        log(f"catalysts unavailable ({e})")
    # Macro + cross-sectional context (faithful peer-grade + regime signals).
    grades = load_grades()
    peer_ranks = peer_rank_table(grades)
    spy_data = fetch_json(SPY_HISTORY_URL, timeout=30)
    spy_closes = None
    if isinstance(spy_data, list):
        spy_closes = [(_f(b.get("close")) or _f(b.get("price"))) for b in spy_data]
        spy_closes = [c for c in spy_closes if c and c > 0]
    regime = compute_regime(spy_closes)
    xa = compute_cross_asset()
    # Market-wide options read (index IV → vol regime + market tilt). Loaded once.
    opt_market = compute_options_market(list(universe.keys()))
    regime = compute_regime(spy_closes, opt_market)
    fed = fed_expectation(today)
    # Prefer the REAL FRED quad (macro_quad.json, written by the macro-quad
    # workflow). Fall back to the market-implied proxy only if the file is absent.
    quad, quad_note = None, ""
    mq = fetch_json(f"{RAW}/TRAPP2-1/main/data/macro_quad.json", timeout=20)
    if isinstance(mq, dict) and isinstance(mq.get("quad"), int):
        quad = mq["quad"]
        quad_note = f"FRED quad {quad}: {mq.get('quadName', '')}"
    else:
        quad, quad_note = resolve_quad(xa, regime.get("mode"))
    xa_have = [k for k, v in xa.items() if v is not None]
    opt_note = ""
    if opt_market:
        opt_note = f" · options {opt_market['indexIV']*100:.0f}% IV ({opt_market['volRegime']}, {opt_market['count']} chains)"
    fed_note = f" · Fed {fed['stance']} ({fed['expectedBpsChange']:+.0f}bps/{fed['meetingsAhead']}mtg)" if fed else ""
    log(f"regime: {regime['mode']}{regime.get('optNote','')} · {quad_note} · cross-asset: {', '.join(xa_have) or 'none'} · "
        f"peer-ranked sectors cover {len(peer_ranks)} tickers{opt_note}{fed_note}")

    rc = load_regime_certainty()
    agnostic = not (rc and rc.get("dialedIn"))
    if agnostic:
        regime = dict(regime, weightMods={}, longBar=0.0)       # neutral tilts, base entry bar
        quad = None                                              # no quad-based sector grade / research regime
        log(f"regime certainty: {('%.2f' % rc['certainty']) if rc else 'unavailable'} — NOT dialed in → REGIME-AGNOSTIC "
            f"(no regime tilts, no regimeGrade, pattern trust from all regimes)"
            + (f" · {'; '.join((rc.get('why') or [])[:3])}" if rc else ""))
    else:
        log(f"regime certainty: {rc['certainty']:.2f} — dialed in ({rc.get('regime')}, quad {rc.get('quadNowcast')}) → regime-aware")
    bond = None
    try:
        bond = bond_trend_context(spy_closes)
    except Exception as e:
        log(f"bond trend context failed: {e}")
    if bond:
        log(f"bond trend (context, not scored): TLT {bond['trend']} · {bond['tlt']} vs MA20 {bond['ma20']} / MA50 {bond['ma50']} · "
            f"10d {bond['slope10Pct']:+.2f}% · stock/bond corr60 {bond['stockBondCorr60']}")
    # NB: `fed` is already the rate-expectation dict the scorer uses - this one is fed_ctx
    fed_ctx = None
    try:
        fed_ctx = fed_context()
    except Exception as e:
        log(f"fed context failed: {e}")
    if fed_ctx:
        log(f"fed (context, not scored): {fed_ctx.get('stance')} · pace {fed_ctx.get('monthlyPaceBn')}bn/mo · net liquidity "
            f"{fed_ctx.get('netLiquidityBn')}bn (13w {fed_ctx.get('netLiquidity13wBn')}) · statement tone {fed_ctx.get('statementTone')}"
            f" (Δ {fed_ctx.get('toneChange')})")
    curve = None
    try:
        curve = curve_context()
    except Exception as e:
        log(f"curve context failed: {e}")
    if curve:
        mv = curve.get("move1m") or {}
        log(f"curve (context, not scored): 2s10s {curve.get('s2s10')} bp · 3m10y {curve.get('s3m10y')} bp · "
            f"real 10y {curve.get('real10')} · breakeven 10y {curve.get('be10')} · 5y5y {curve.get('fwd5y5y')} · "
            f"last month: {mv.get('label')} (2Y {mv.get('d2')}, 10Y {mv.get('d10')} bp)")
    credit = None
    try:
        credit = credit_context()
    except Exception as e:
        log(f"credit context failed: {e}")
    if credit:
        log(f"credit (context, not scored): high yield {credit.get('hyOas')} bp ({credit.get('band')}, {credit.get('hy1m')} bp 1M) · "
            f"CCC {credit.get('cccOas')} bp · CCC-BB {credit.get('cccMinusBb')} bp · IG {credit.get('igOas')} bp")
    regime_ctx = {"regime": (rc or {}).get("regime"), "quad": (rc or {}).get("quadNowcast"),
                  "certainty": (rc or {}).get("certainty"), "dialedIn": not agnostic, "botMode": regime.get("mode"),
                  "bondTrend": bond, "fed": fed_ctx, "curve": curve, "credit": credit}
    # ex-post: mark every trade whose entry-day regime call has since been judged
    outcomes = regime_call_outcomes(rc)
    for t in trades_list(state):
        d = t.get("entryDate")
        if d in outcomes:
            t["regimeCallCorrect"] = outcomes[d]
    weights = state.get("learnedWeights", {}) if isinstance(state.get("learnedWeights"), dict) else {}
    ctx = {"weights": weights, "weight_mods": regime.get("weightMods", {}),
           "mode": regime.get("mode", "choppy"), "peer_ranks": peer_ranks,
           "xa": xa, "quad": quad, "opt_market": opt_market, "fed": fed, "grades": grades, "today": today}
    # Research layer for crypto / futures (drivers, supply chain, news, regime,
    # seasonality) - shared cache for management + new entries.
    ctx["research"] = alt_research.ResearchData(fetch_json, RAW, universe, today)
    ctx["patterns"] = load_patterns(use_regime=not agnostic)
    ctx["regime_agnostic"] = agnostic
    ids_before = {t.get("id") for t in trades_list(state)}
    open_before = {t.get("id") for t in trades_list(state) if t.get("status") == "open"}
    # z121: exits only at fresh quotes (a stale one is not managed this pass, never filled at an old price)
    closed = manage_open_positions(state, fresh, today, ctx)
    ok_entry, why = entry_window(sess, state)
    eligible = {t: u for t, u in fresh.items() if session_tradable(t)} if ok_entry else {}
    if ok_entry and eligible:
        log(f"entries: {len(eligible)} tickers with a fresh quote whose home market is open now")
        opened = open_new_trades(state, eligible, today, regime, grades, peer_ranks, xa, quad, opt_market, fed,
                                 research=ctx["research"], patterns=ctx["patterns"], regime_ctx=regime_ctx)
        state["lastEntryScan"] = {"date": sess["date"], "et": sess["et"], "opened": len(opened)}
    elif ok_entry:
        opened = []
        log("entries: no fresh, session-tradable quotes yet - the next pass scans (today's scan not used up)")
    else:
        opened = []
        log(f"entries: {why}")
    stamp_execution(state, ids_before, open_before, sess, universe, now_utc)
    finalize_state(state, sig_before)
    recompute(state, universe, today)

    log(f"summary: +{len(opened)} opened · {len(closed)} closed · "
        f"bankroll ${state['bankroll']:,.2f} · book ${state['equityCurve'][-1]['value']:,.2f}")

    if DRY_RUN:
        log("DRY RUN — not writing state.")
        return
    if not opened and not closed:
        # Still write the refreshed marks/equity point so the curve advances daily,
        # but only if something actually changed in the equity value.
        pass
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, separators=(",", ":")))
    log(f"wrote {STATE_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
