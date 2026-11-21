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
import corporate_actions as ca  # noqa: E402  z166: splits and dividends
CA_STATE = {"store": None, "fx": None}
import trade_book as tb  # noqa: E402  z138: horizons (short / medium / long), options, catalysts, position reviews
import risk_book as rb  # noqa: E402  z186: risk-budgeted sizes, sector / cluster / beta caps, drawdown throttle
import market_tape as mt  # noqa: E402  z202: the market read as known at a session's open (the record of trends)

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
    # z164: the macro book is scanned too - country ETFs (EWZ, EWW, EWJ ...), gold / oil ETFs, its bond ETFs.
    # Its foreign listings stay out (session_tradable), its indices are never instruments.
    f"{RAW}/TRAPP2-1/main/data/master.json",
]
# ... and repos it only PRICES, so a position opened from any book (the app's
# bot trades foreign listings too, e.g. 000660.KS) is marked and managed.
PRICE_ONLY_SOURCES = [
    f"{RAW}/TRAPP2-3/main/data/master.json",
    f"{RAW}/TRAPP2-4/main/data/master.json",   # z178: the NYSE equities that weren't in the app - priced, not scanned
]
HISTORY_BASE = {
    "TRAPP2": f"{RAW}/TRAPP2/main/data/history",
    "TRAPP2-2": f"{RAW}/TRAPP2-2/main/data/history",
    "TRAPP2-1": f"{RAW}/TRAPP2-1/main/data/history",
    "TRAPP2-3": f"{RAW}/TRAPP2-3/main/data/history",
    "TRAPP2-4": f"{RAW}/TRAPP2-4/main/data/history",   # z178
}
FX_URL = f"{RAW}/TRAPP2-1/main/data/fx/rates.json"
GRADES_URL = f"{RAW}/TRAPP2-ANALYTICS/main/data/research_grades.json"
RATINGS_URL = (os.environ.get("RUNNER_RATINGS_URL") or
               f"{RAW}/TRAPP2-CREDIT/main/data/ratings.json")    # z194: credit ratings (companies, funds, futures, FX)
WARNING_URL = (os.environ.get("RUNNER_WARNING_URL") or
               f"{RAW}/TRAPP2-ANALYTICS/main/data/warning/signals.json")   # z196: the early-warning read ("the digging")
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
OPTIONS_MARKET_URL = f"{OPTIONS_BASE}/market.json"      # z196: the market read, precomputed over ~2,300 chains
OPTIONS_SIGNALS_URL = f"{OPTIONS_BASE}/signals.json"    # z196: every name's options signal in one file
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

LONG_TERM_SIGNALS = ("fundamentals", "researchGrade", "health", "peerGrade", "regimeGrade", "fed", "crossAsset", "credit")
SWING_SIGNALS     = ("trend", "momentum", "meanReversion", "optionsIV", "optionsMarket")
# z192 (owner, 11-17: "the bot needs to learn from every trade"; leveraged / inverse ETPs stay tradable as long as
# signals open and close them). A daily-reset product holds only while its signal does: re-scored in its direction
# below this, past its minimum hold, it is closed ("signal-faded") - every session held costs the volatility drag.
ETP_HOLD_SCORE = _envf("RUNNER_ETP_HOLD_SCORE", 0.10)
COMPANY_SIGNALS = ("fundamentals", "health", "researchGrade", "peerGrade", "catalyst")   # a fund has none of these
ETP_MARKET_SIGNALS = ("crossAsset", "optionsMarket", "credit", "warning")   # market-wide: an ETP carries them in its own direction
# z194: credit ratings (TRAPP2-CREDIT data/ratings.json, model-implied on the S&P scale). Every trade considers them:
# a 'credit' signal in the blend (30 % rating level + 70 % credit trend: CDS, option-implied vol, migration; a fund's is
# its holdings' trend, signed by its leverage), no new long at CCC+ or worse, sizes cut for weaker credits, a long closed
# when its issuer falls into distress or is downgraded CREDIT_EXIT_NOTCHES+ (confirmed a session later).
CREDIT_GATE_NOTCH = int(_envf("RUNNER_CREDIT_GATE_NOTCH", 17))      # CCC+ (17) or worse: no new longs
CREDIT_EXIT_NOTCHES = int(_envf("RUNNER_CREDIT_EXIT_NOTCHES", 2))   # notches of downgrade since entry that close a long
CREDIT_MAX_AGE_DAYS = _envf("RUNNER_CREDIT_MAX_AGE_DAYS", 4)        # older ratings: context only (no signal / gate / size / exit)
CREDIT_IG = 10                                                      # BBB- : investment grade at or better
# z195: a new listing (an IPO - Anthropic's ANT on 11-19 - or a spin-off) waits for this many sessions of price history
# before the bot opens a position in it. 0 turns the wait off.
MIN_HISTORY_SESSIONS = int(_envf("RUNNER_MIN_HISTORY_SESSIONS", 20))
ETP_SECTOR_SIGNALS = ("regimeGrade", "fed")                # sector / beta tilts: not defined for an ETP
SIGNAL_BASIS = "z192"                                      # components recorded fund-clean and exposure-signed
POSITION_PCT = _envf("RUNNER_POSITION_PCT", _MODE["pos_pct"]) / 100.0
CASH_RESERVE_PCT = _envf("RUNNER_CASH_RESERVE_PCT", _MODE["reserve"]) / 100.0
MIN_SCORE = _envf("RUNNER_MIN_SCORE", _MODE["min_score"])
DRY_RUN = os.environ.get("RUNNER_DRY_RUN", "") in ("1", "true", "yes")
# z186: SHORTS ON by default - owner direction §1d (10-21): "longs and shorts". Before z186 this defaulted off and
# no workflow set it, so all 57 runner trades were long and nothing in the book could gain when its longs fell.
# A strongly negative score (<= SHORT_SCORE, with a fired trigger) opens a short sale in a liquid, borrowable stock
# (risk_book.shortable) or a put. Repo Variable RUNNER_ALLOW_SHORTS = 0 turns them off again.
ALLOW_SHORTS = os.environ.get("RUNNER_ALLOW_SHORTS", "1").strip().lower() not in ("0", "false", "no", "off")
SHORT_SCORE = -MIN_SCORE - 0.10   # a bit more conviction to short
# z186: the blend leans long - in the 11-14 scan the 16 names at or below -0.35 were 11 penny stocks and 5 futures,
# none of them a stock you can short; no liquid, borrowable name gets that low. So the short bar is ALSO relative:
# a shortable stock in the bottom SHORT_PCTL of the session's scores, net bearish (<= SHORT_FLOOR) and with a bearish
# trigger fired, qualifies - the way a long/short book shorts its weakest names, not only the broken ones.
SHORT_PCTL = _envf("RUNNER_SHORT_PCTL", 10) / 100
SHORT_FLOOR = _envf("RUNNER_SHORT_FLOOR", -0.15)
SHORT_SLOTS = int(_envf("RUNNER_SHORT_SLOTS", 2))   # session slots kept for the best shorts, when there are any
# z190: a short sale the rules never allowed (a fund / ETP - rb.fund_reason) is unwound at the next pass, minimum hold or
# not: 11-16 the runner shorted GLD, SOXL, TAN and WGMI while a throttled profile fetch had wiped TRAPP2-1's fund flags.
# Repo Variable RUNNER_UNWIND_UNSHORTABLE = 0 keeps such positions instead.
UNWIND_UNSHORTABLE = os.environ.get("RUNNER_UNWIND_UNSHORTABLE", "1").strip().lower() not in ("0", "false", "no", "off")
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


FX_ASOF = {"at": None}


def load_fx():
    """USD-per-unit map from TRAPP2-1/data/fx/rates.json (the file the app uses)."""
    d = fetch_json(FX_URL)
    rates = (d.get("rates") if isinstance(d, dict) else {}) or {}
    FX_ASOF["at"] = d.get("updatedAt") if isinstance(d, dict) else None
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
            # z166: a previous close on the other side of a split (PARA 10-28: closeyest 0.425, opened 4.46 after a
            # 1-for-10) is not a price - use today's open instead, or leave the ticker unpriced
            if fld == "closeyest" and price:
                op = _f(r.get("priceopen"))
                if op and op > 0 and ca.split_like(op / price):
                    price, fld = op, "priceopen"
            if price is None:
                continue
            ccy = (r.get("currency") or "USD")
            f = usd_factor(ccy, fx)
            if f is None:
                no_rate[t] = ccy
                continue
            # z187: market caps are in the quote currency's MAIN unit (Glencore: price in pence, cap in pounds) and the
            # statements in the reporting currency (TSM: TWD) - both to USD here, so size tiers and P/S compare like
            # with like across listings
            mc_f = fx.get(_SUBUNIT.get(ccy.strip(), (ccy.strip().upper(), 1.0))[0])
            fin = str(r.get("financialCurrency") or "").strip()
            fin_f = fx.get(_SUBUNIT.get(fin, (fin.upper(), 1.0))[0]) if fin else mc_f
            mc_raw = _f(r.get("marketcap"))
            rows[t] = {
                "ticker": t, "price": round(price * f, 6),
                "priceLocal": price, "currency": ccy, "fxRate": f, "scan": scan,
                "asOf": r.get("fetched_at") if fld == "price" else None,   # z121: a close is never a live quote
                "assetClass": r.get("asset_class"),
                "isetf": str(r.get("isetf") or "").lower(),          # z186: funds aren't single-name shorts
                "isfund": str(r.get("isfund") or "").lower(),        # z190: ... nor are mutual / closed-end funds
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
                # z187: currency-consistent valuation from the books' pipeline (priced at the row's price) + the raw
                # pieces to rebuild it for a row written before z187
                "priceToSales": _f(r.get("priceToSales")),
                "evToEbitda": _f(r.get("evToEbitda")),
                "valuationNote": str(r.get("valuationNote") or ""),
                "valuationZ187": "valuationNote" in r,
                "financialCurrency": fin or ccy,
                "marketcapUSD": (mc_raw * mc_f) if (mc_raw and mc_f) else None,
                "finUsd": fin_f,
                "industry": r.get("industry") or "",
                "nameWas": r.get("nameWas"),
                "nameChangedAt": r.get("nameChangedAt"),
            }
    n_scan = sum(1 for v in rows.values() if v["scan"])
    n_conv = sum(1 for v in rows.values() if v["fxRate"] != 1.0)
    log(f"universe: {len(rows)} priced tickers in USD ({n_scan} scannable, {len(rows) - n_scan} price-only; "
        f"{n_conv} converted from local units)")
    if no_rate:
        log(f"  {len(no_rate)} skipped - no FX rate: "
            + ", ".join(f"{t}({c})" for t, c in sorted(no_rate.items())[:10]))
    n_fx = add_fx_instruments(rows, fx)                      # z164: currencies as USD-priced instruments
    if n_fx:
        log(f"  {n_fx} currencies tradable as FX:<CCY> (USD per unit, rates of {FX_ASOF.get('at')})")
    try:                                                     # z202: market cap by the formula (before the medians use it)
        apply_formula_mcap(rows)
    except Exception as e:
        log(f"  formula market caps unavailable ({type(e).__name__}: {e}) - Yahoo's kept")
    try:                                                     # z187: sector valuation medians for the relative step
        VAL_STATS.clear()
        VAL_STATS.update(valuation_stats(rows))
        al = VAL_STATS.get("_all") or {}
        z = sum(1 for v in rows.values() if v.get("valuationZ187"))
        log(f"  valuation: {max(0, len(VAL_STATS) - 1)} sectors with medians · universe P/E {al.get('pe', 0):.1f} · P/S {al.get('ps', 0):.2f} · "
            f"P/B {al.get('pb', 0):.2f} · EV/EBITDA {al.get('evEbitda', 0):.1f} · {z} rows priced by the z187 pipeline "
            f"(the rest rebuilt here from their raw fields, in USD)")
    except Exception as e:
        log(f"  valuation medians unavailable ({type(e).__name__}: {e}) - absolute valuation rules")
    return rows


# ------------------------------------------------------------------------- z202: market cap by the formula ---
# TRAPP2-ANALYTICS data/sec/shares.json (pipeline/shares.py): per ticker the share count to trust - the SEC's own cover
# count where Yahoo agrees (or the diluted count corroborates it), Yahoo's where they disagree. The bot's market cap
# (size tiers, P/S and the valuation medians, the $2B short floor) = the USD price x that count. Checked at use: if
# the count Yahoo's market cap implies NOW is > 10 % from the file's (a split or an issue since the file was built),
# Yahoo's figure stays for that pass. The split check in the position reviews keeps reading YAHOO's market cap
# (marketcapYahoo): it relies on Yahoo's count moving with a split.
SHARES_URL = os.environ.get("RUNNER_SHARES_URL") or f"{RAW}/TRAPP2-ANALYTICS/main/data/sec/shares.json"
SHARES_GUARD = 0.10


def apply_formula_mcap(rows, doc=None):
    """rows (load_universe) in place -> {"sec": n, "yahoo": n, "moved": n}."""
    d = doc if doc is not None else fetch_json(SHARES_URL, timeout=30)
    tk = d.get("tickers") if isinstance(d, dict) else None
    out = {"sec": 0, "yahoo": 0, "moved": 0}
    if not isinstance(tk, dict):
        return out
    for t, r in rows.items():
        e = tk.get(t)
        if not isinstance(e, dict) or str(r.get("currency") or "USD").strip().upper() not in ("USD", ""):
            continue
        sh, px, ym = _f(e.get("s")), _f(r.get("price")), _f(r.get("marketcap"))
        if not sh or not px or px <= 0:
            continue
        if ym and abs(sh / (ym / px) - 1) > SHARES_GUARD:
            out["moved"] += 1                                   # Yahoo's count moved since the file: keep Yahoo this pass
            continue
        r["marketcapYahoo"] = ym
        r["marketcap"] = r["marketcapUSD"] = px * sh
        r["mcapSource"] = f"{e.get('p')}:{e.get('st')}"
        out["sec" if e.get("p") == "sec" else "yahoo"] += 1
    if any(out.values()):
        log(f"  market cap = price x shares: {out['sec']} on the SEC's own count, {out['yahoo']} on Yahoo's"
            + (f", {out['moved']} kept Yahoo's (its count moved > {SHARES_GUARD:.0%} since the file - a split / issue?)" if out["moved"] else "")
            + f" (TRAPP2-ANALYTICS data/sec/shares.json {d.get('date') or ''})")
    return out


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


# z164: currencies as instruments. FX:<CCY> is priced in USD per one unit of the currency (from the FX loop's
# rates, refreshed every 15 min), so a position's size is a number of currency units and its P&L is in dollars
# like any share position. Long FX:BRL = the real rises against the dollar; short = it falls.
FX_HISTORY = {}     # FX:CCY -> (pair, invert) - which TRAPP2-1 history file, and whether to invert its rate


def fx_history_source(ccy):
    for pair, inv in ((f"{ccy}USD=X", False), (f"{ccy}=X", True), (f"USD{ccy}=X", True)):
        yield pair, inv


def add_fx_instruments(rows, fx):
    n = 0
    for ccy in alt_research.FX_TRADABLE:
        up = fx.get(ccy)
        t = f"FX:{ccy}"
        if not up or t in rows:
            continue
        rows[t] = {"ticker": t, "price": round(up, 8), "priceLocal": up, "currency": "USD", "fxRate": 1.0, "scan": True,
                   "asOf": FX_ASOF.get("at"), "assetClass": "fx", "sector": "Currencies", "name": f"{ccy} (USD per {ccy})",
                   "repo": "TRAPP2-1", "changepct": None, "beta": None}
        n += 1
    return n


_HIST = {}     # z186: (TICKER, repo) -> [(date, close)] - one fetch per ticker per pass (scan, management and risk)


def load_history(ticker, repo):
    """Daily closes for a ticker, newest last. None if absent."""
    d = load_history_dated(ticker, repo)
    return [c for _, c in d] if d else None


def load_history_dated(ticker, repo):
    """z186: [(date, close)] newest last (None if absent), cached for the pass."""
    key = (str(ticker).upper(), repo)
    if key not in _HIST:
        _HIST[key] = _load_history_dated(ticker, repo)
    return _HIST[key]


def _load_history_dated(ticker, repo):
    if str(ticker).upper().startswith("FX:"):
        ccy = str(ticker).upper()[3:]
        base = HISTORY_BASE.get("TRAPP2-1")
        for pair, inv in fx_history_source(ccy):
            data = fetch_json(f"{base}/{pair}.json", timeout=30)
            if not isinstance(data, list) or len(data) < 30:
                continue
            closes = []
            for bar in data:
                try:
                    c = float(bar.get("close") if bar.get("close") is not None else bar.get("price"))
                except (TypeError, ValueError):
                    continue
                if c and c > 0:
                    closes.append((str(bar.get("date") or "")[:10], 1.0 / c if inv else c))
            if len(closes) >= 30:
                FX_HISTORY[ticker] = (pair, inv)
                return closes
        return None
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
            closes.append((str(bar.get("date") or "")[:10], c))
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
    """Read TRAPP2-1/data/macro/quad.json. z177: its `current` is the market NOWCAST while that is fresh (else the
    official GDP/CPI quad) - one answer all day; the official quad and whether the two agree ride along in the note
    (the log used to call the nowcast "FRED quad"). Returns (quad:int, note:str) or (None, reason)."""
    data = fetch_json(MACRO_QUAD_URL, timeout=20)
    if isinstance(data, dict):
        cur = data.get("current") or {}
        q = cur.get("quad")
        if isinstance(q, int) and 1 <= q <= 4:
            kind = "market nowcast" if cur.get("method") == "market-nowcast" else "official GDP/CPI"
            off = data.get("official") or data.get("officialLatest") or {}
            agree = (data.get("agree") or {}).get("note")
            extra = f" · official Q{off.get('quad')} as of {off.get('asOf')}" if off.get("quad") and kind != "official GDP/CPI" else ""
            return q, f"quad {q} ({kind}, {cur.get('label','')}, as of {cur.get('asOf','?')}){extra}" + (f" · {agree}" if agree else "")
    return None, "no quad file"


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

_OPT_SUMMARY = {"doc": None}


def _opt_summary_row(ticker):
    """z167: TRAPP2-OPTIONS summary row (atmIv30, ivRank ...) for the IV regime."""
    if _OPT_SUMMARY["doc"] is None:
        d = fetch_json(f"{OPTIONS_BASE}/summary.json", timeout=30)
        _OPT_SUMMARY["doc"] = (d.get("tickers") if isinstance(d, dict) else None) or {}
    return _OPT_SUMMARY["doc"].get(str(ticker).upper())


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

_OPT_SIGNALS = {"doc": None}
OPT_SIGNAL_MAX_AGE_DAYS = 5


def _opt_signal_row(ticker):
    """z196: TRAPP2-OPTIONS signals.json - every name's extractOptionsSignal, computed by the pipeline from the same
    chain (one file a pass instead of one chain per candidate; ~2,300 names since z196). None if absent or stale."""
    if _OPT_SIGNALS["doc"] is None:
        d = fetch_json(OPTIONS_SIGNALS_URL, timeout=30)
        _OPT_SIGNALS["doc"] = (d.get("tickers") if isinstance(d, dict) else None) or {}
    row = _OPT_SIGNALS["doc"].get(str(ticker).upper())
    if not isinstance(row, list) or len(row) < 6 or row[0] is None:
        return None
    try:
        age = ((NOW_UTC or datetime.now(timezone.utc)).date() - datetime.strptime(str(row[5])[:10], "%Y-%m-%d").date()).days
    except (TypeError, ValueError):
        return None
    if age > OPT_SIGNAL_MAX_AGE_DAYS:
        return None
    return {"iv": row[0], "skew": row[1], "termSlope": row[2], "ivSignal": row[3], "source": "signals.json"}


def extract_options_signal(ticker):
    """Per-name forward signal from the option chain, faithful to the app's
    extractOptionsSignal: ATM IV level + put/call skew + term structure → a small
    signed signal (rich IV / heavy downside skew / backwardation lean negative).
    Returns dict or None. z196: the pipeline's signals.json first (same arithmetic, same chain), else the chain."""
    pre = _opt_signal_row(ticker)
    if pre:
        return pre
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

OPT_MARKET_MAX_AGE_DAYS = 3
OPT_MARKET_SWEEP_CAP = 200      # z196: the legacy per-chain sweep (no market.json) reads at most this many chains


def compute_options_market(universe_tickers):
    """Aggregate option chains across the manifest into a market-wide read:
    index-led IV → vol regime + a signed market-trend tilt. Faithful to the app's
    computeOptionsMarketSignal. Returns dict or None.
    z196: TRAPP2-OPTIONS now precomputes this read (market.json, the same arithmetic over the index ETFs and the
    full-ladder names) - the manifest lists ~2,300 chains, and sweeping them one HTTP request at a time would take
    the whole pass. The sweep remains the fallback, capped (index ETFs first)."""
    if _OPT_MARKET["done"]:
        return _OPT_MARKET["data"]
    mk = fetch_json(OPTIONS_MARKET_URL, timeout=30)
    app = (mk or {}).get("appSignal") if isinstance(mk, dict) else None
    if app and app.get("indexIV") is not None:
        try:
            age = ((NOW_UTC or datetime.now(timezone.utc)) - datetime.strptime(str(mk.get("asOf"))[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)).days
        except (TypeError, ValueError):
            age = 99
        if age <= OPT_MARKET_MAX_AGE_DAYS:
            data = {k: app.get(k) for k in ("count", "indexIV", "breadthIV", "avgSkew", "volRegime", "marketTrend")}
            data["source"] = "market.json"
            data["breadth"] = (mk.get("breadth") or None)
            _OPT_MARKET.update(done=True, data=data)
            return data
    man = fetch_json(OPTIONS_MANIFEST_URL, timeout=20)
    tickers = (man.get("tickers") if isinstance(man, dict) else None) or []
    tickers = [t for t in MKT_OPT_INDEXES if t in tickers] + [t for t in tickers if t not in MKT_OPT_INDEXES]
    tickers = [t for t in tickers if t != "run_report"][:OPT_MARKET_SWEEP_CAP]
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
    # z194: credit - the issuer's rating level and credit trend (a fund's: its holdings'), weighted like peer grade
    "credit": 0.12,
    # z196: early warning - market-wide risk (published loss / lending / funding / labour data vs their history) and
    # the name's sector exposure to the current warnings; the learning layer decides how much it is worth
    "warning": 0.08,
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


HEADCOUNT_BASE = f"{RAW}/TRAPP2-ANALYTICS/main/data/headcount"
_HC = {}                 # z151: {TICKER: {employees, yoyPct, fromPeakPct, date, cadence}}


_SEC = {}               # z158: {TICKER: {red30, caution30, insider, lastEvent}}
_SHORT = {"volume": {}, "interest": {}}
NO_LONG_ITEMS = {"1.03": "bankruptcy", "4.02": "non-reliance on past financials", "3.01": "delisting notice"}


_RATINGS = {"loaded": False, "by": {}, "asOf": None, "stale": True, "summary": None}


def load_ratings():
    """z194: TRAPP2-CREDIT's ratings, read once a pass. Stale (older than CREDIT_MAX_AGE_DAYS) or missing: recorded on
    trades as context, but no signal, gate, size or exit rests on them."""
    if _RATINGS["loaded"]:
        return _RATINGS
    d = fetch_json(RATINGS_URL, timeout=45)
    by = (d.get("ratings") or {}) if isinstance(d, dict) else {}
    as_of = d.get("date") if isinstance(d, dict) else None
    age = None
    try:
        age = ((NOW_UTC or datetime.now(timezone.utc)).date() - datetime.strptime(str(as_of)[:10], "%Y-%m-%d").date()).days
    except (TypeError, ValueError):
        pass
    _RATINGS.update(loaded=True, by=by, asOf=as_of, stale=(age is None or age > CREDIT_MAX_AGE_DAYS),
                    summary=(d.get("summary") if isinstance(d, dict) else None))
    spy = by.get("SPY") or {}
    log(f"credit ratings: {len(by)} instruments as of {as_of or '-'}" + (f" ({age} day(s) old)" if age is not None else "")
        + (" - STALE: context only" if _RATINGS["stale"] else "")
        + (f" · S&P 500 look-through {spy.get('r')} (WARF {((spy.get('lt') or {}).get('warf'))})" if spy.get("r") else ""))
    return _RATINGS


def credit_of(ticker):
    """The rating record of an instrument, or None."""
    return _RATINGS["by"].get(str(ticker or "").upper())


def credit_card(cr):
    """A compact rating snapshot for a trade record: letter, notch, outlook, signal, trend, source, as-of."""
    if not cr:
        return None
    return {"r": cr.get("r"), "n": cr.get("n"), "o": cr.get("o") or "stable", "sig": cr.get("sig"), "tr": cr.get("tr"),
            "src": cr.get("src"), "k": cr.get("k"), "asOf": _RATINGS.get("asOf"), "stale": bool(_RATINGS.get("stale"))}


def credit_size_mult(cr, short=False, etp=None):
    """z194: the size multiplier for a new position's credit. A long in a weaker credit is smaller - low-rated and
    distressed stocks earn less and fall harder (Campbell, Hilscher & Szilagyi 2008; Avramov, Chordia, Jostova &
    Philipov 2009); a short in the strongest credits is smaller (quality rarely collapses). Long an inverse product =
    short its underlying. NR / stale / missing: 1.0."""
    if not cr or _RATINGS["stale"] or not cr.get("n"):
        return 1.0
    n = int(cr["n"])
    eff_short = bool(short) != bool(etp and (etp.get("lev") or 0) < 0)
    if eff_short:
        return 0.85 if n <= 4 else 1.0
    return 1.0 if n <= CREDIT_IG else 0.85 if n <= 13 else 0.65 if n <= 16 else 0.5


def credit_gate(ticker, direction, etp=None):
    """z194: a hard guard on credit, like the 8-K one - no new long in a distressed credit (CCC+ or worse, or D). A long
    in an inverse product is a short of its underlying and is not barred; shorts are not barred. -> reason or None."""
    cr = None if _RATINGS["stale"] else credit_of(ticker)
    if not cr or not cr.get("n") or int(cr["n"]) < CREDIT_GATE_NOTCH or direction == "short":
        return None
    if etp and (etp.get("lev") or 0) < 0:
        return None
    return f"rated {cr.get('r')} (CCC+ or worse) - no new longs in a distressed credit"


def seasoning_gate(ticker, repo):
    """z195: no new position until a listing has MIN_HISTORY_SESSIONS daily closes. With a handful of bars there is no
    trend, momentum or RSI to read (they need 15-64), the stock's volatility is unknown and its first weeks are price
    discovery - a position would rest on fundamentals and peers alone, at an IPO's widest swings. Exits are never
    blocked. -> reason or None."""
    if MIN_HISTORY_SESSIONS <= 0 or not repo:
        return None
    closes = load_history(ticker, repo)
    n = len(closes) if closes else 0
    if n >= MIN_HISTORY_SESSIONS:
        return None
    return (f"{n} session{'' if n == 1 else 's'} of price history (< {MIN_HISTORY_SESSIONS}) - a new listing is still in "
            f"price discovery (RUNNER_MIN_HISTORY_SESSIONS)")


# ------------------------------------------------------------------------------------------- z196: early warning ---
# TRAPP2-ANALYTICS data/warning/signals.json ("the digging", owner 11-19): published loss / lending / funding / labour
# / collateral / leverage data scored against their own history, a composite RISK percentile, the regime (stress
# building, market blindness, organic expansion ...), the industries each warning hits, and a backtest that says
# whether acting on it has paid. In the bot:
#   * a 'warning' signal in the blend - market risk and the name's industry exposure (+ organic growth only when the
#     backtest earned it); a positive (risk-on) lean only when the risk read itself earned it
#   * longs sized down as risk rises: x0.9 / 0.8 / 0.7 from the 60th / 75th / 90th percentile (never below x0.9 unless
#     the backtest earned the throttle), x(1 - 0.35 e) in an exposed industry, a levered long x0.85 more at 75+
#   * a short in an exposed industry up to x1.15 - only when earned
#   * no new leveraged long fund at "extreme" risk in acute stress / stress spreading
#   * warningAtEntry on every trade (context: the journal shows whether the warnings paid)
# Stale (> WARNING_MAX_AGE_DAYS) or provisional (no FRED history yet): context only - no signal, size or gate.
WARNING_MAX_AGE_DAYS = _envf("RUNNER_WARNING_MAX_AGE_DAYS", 4)
_WARN = {"loaded": False, "doc": None, "stale": True, "age": None}


def load_warning():
    if _WARN["loaded"]:
        return _WARN
    d = fetch_json(WARNING_URL, timeout=45)
    doc = d if isinstance(d, dict) and d.get("risk") is not None else None
    age = None
    try:
        age = ((NOW_UTC or datetime.now(timezone.utc)).date() - datetime.strptime(str((doc or {}).get("date"))[:10], "%Y-%m-%d").date()).days
    except (TypeError, ValueError):
        pass
    stale = doc is None or age is None or age > WARNING_MAX_AGE_DAYS or bool(doc.get("provisional"))
    _WARN.update(loaded=True, doc=doc, stale=stale, age=age)
    if doc:
        log(f"early warning: {doc.get('headline')}" + (f" ({age} day(s) old)" if age is not None else "")
            + (" - STALE / provisional: context only" if stale else ""))
    else:
        log("early warning: no read yet (TRAPP2-ANALYTICS data/warning/signals.json) - trades run without it")
    return _WARN


def _warn_live():
    return None if _WARN["stale"] else _WARN["doc"]


def _warn_earned(key):
    return bool((((_warn_live() or {}).get("backtest") or {}).get("earned") or {}).get(key))


def warning_exposure(sector, industry):
    """How hard the current warnings hit this industry / sector: (e 0..1, why) - the stronger of the two matches."""
    ex = (_warn_live() or {}).get("exposures") or {}
    best = (0.0, None)
    for kind, name in (("industry", industry), ("sector", sector)):
        r = (ex.get(kind) or {}).get(name) if name else None
        if r and float(r.get("e") or 0) > best[0]:
            best = (float(r["e"]), f"{name}: {r.get('why')}")
    return best


def warning_signal(sector, industry, etp=None):
    """The 'warning' component: -0.5 R - 0.6 e (+ 0.3 G in an earned organic expansion), R = (risk - 50) / 40 and
    G = (growth percentile - 50) / 40 clipped to [-1, 1], e = the industry exposure (none for a fund - its market lean
    is signed by its leverage in _etp_sign_market). Leans positive on low risk only when the risk read earned it."""
    doc = _warn_live()
    if not doc or doc.get("risk") is None:
        return None
    r = max(-1.0, min(1.0, (float(doc["risk"]) - 50) / 40))
    e = 0.0 if etp else warning_exposure(sector, industry)[0]
    sig = -0.5 * r - 0.6 * e
    if doc.get("regime") == "organic expansion" and _warn_earned("organicUpside"):
        gp = ((doc.get("gauges") or {}).get("growth") or {}).get("pct")
        if gp is not None:
            sig += 0.3 * max(0.0, min(1.0, (float(gp) - 50) / 40))
    if not _warn_earned("throttle"):
        sig = min(sig, 0.0)
    return round(max(-1.0, min(1.0, sig)), 3)


# z198: the hypothesis lab's verdicts reach the bot. A hypothesis the lab SUPPORTS (12+ signal months, 2+ episodes
# caught, lift >= 1.5, permutation p <= 0.10 over 1960-2026) whose signal is ON today and points down (a drawdown or a
# recession ahead, or below-median returns) trims new longs once more - evidence earned, not assumed. Rejected and
# inconclusive hypotheses move nothing; shorts are never enlarged by it.
LAB_SIZE = _envf("RUNNER_LAB_SIZE", 0.9)


def warning_lab_on():
    """-> ids of supported, bearish hypotheses that are ON in the live read."""
    out = []
    for h in (((_warn_live() or {}).get("lab") or {}).get("nowOn") or []):
        if (h or {}).get("status") != "supported":
            continue
        oc = str(h.get("outcome") or "")
        if oc.startswith(("DD", "REC")) or (oc == "ABOVE_MED12" and h.get("expect") == "lower"):
            out.append(str(h.get("id")))
    return out


def warning_size_mult(sector, industry, short=False, etp=None):
    """-> (multiplier, why). Risk-reducing by design: longs smaller as risk rises and in exposed industries; the only
    increases (a short in an exposed industry, a long in an earned organic expansion) need the backtest's say-so."""
    doc = _warn_live()
    if not doc or doc.get("risk") is None:
        return 1.0, None
    risk = float(doc["risk"])
    lev = float((etp or {}).get("lev") or 0)
    eff_short = bool(short) != (lev < 0)
    e, why = (0.0, None) if etp else warning_exposure(sector, industry)
    if eff_short:
        m = 1.0 + (0.15 * e if _warn_earned("throttle") else 0.0)
        return round(min(1.15, m), 3), (why if m != 1.0 else None)
    base = 1.0 if risk < 60 else 0.9 if risk < 75 else 0.8 if risk < 90 else 0.7
    if not _warn_earned("throttle"):
        base = max(base, 0.9)
    if lev > 1 and risk >= 75:
        base *= 0.85
    m = base * (1 - 0.35 * e)
    lab = warning_lab_on()
    if lab:
        m *= LAB_SIZE                                 # z198: a supported, bearish hypothesis is on
    if doc.get("regime") == "organic expansion" and risk < 40 and _warn_earned("organicUpside") and not lev and not lab:
        m = min(1.05, m * 1.05)
    m = round(max(0.5, min(1.05, m)), 3)
    if m == 1.0:
        return 1.0, None
    return m, " · ".join([f"risk {risk:.0f} ({doc.get('level')})"] + ([why] if e > 0 else []) + ([f"lab: {', '.join(lab)} on"] if lab else []))


def warning_gate(direction, etp=None):
    """No new leveraged LONG fund (2x+) at extreme risk while markets are in acute stress / stress spreading - the
    daily reset compounds against you in exactly those weeks. Inverse funds and shorts are not barred. -> reason or None."""
    doc = _warn_live()
    if not doc or direction == "short" or not etp or float(etp.get("lev") or 0) < 2:
        return None
    if doc.get("level") == "extreme" and doc.get("regime") in ("acute stress", "stress spreading"):
        return f"early warning: {doc.get('level')} risk in {doc.get('regime')} - no new leveraged long fund ({float(etp['lev']):+g}x)"
    return None


def warning_card(sector=None, industry=None):
    """The early-warning read on the trade date, for the journal."""
    doc = _WARN.get("doc")
    if not doc:
        return None
    g = doc.get("gauges") or {}
    e, why = (0.0, None) if _WARN["stale"] else warning_exposure(sector, industry)
    return {"risk": doc.get("risk"), "level": doc.get("level"), "regime": doc.get("regime"),
            "fund": (g.get("fundamental") or {}).get("pct"), "mkt": (g.get("market") or {}).get("pct"),
            "frag": (g.get("fragility") or {}).get("pct"), "growth": (g.get("growth") or {}).get("pct"),
            "blind": any((f or {}).get("kind") == "blindness" for f in doc.get("flags") or []),
            "exposure": round(e, 3), "why": why, "asOf": doc.get("date"), "stale": bool(_WARN["stale"]),
            "labOn": [] if _WARN["stale"] else warning_lab_on()}           # z198: supported, bearish hypotheses on that day


# -------------------------------------------------------------------------------------------- z197: IPO calendar ---
# TRAPP2-ANALYTICS data/web/ipo.json (the Business Web build from config/ipo_watch.json): lock-up releases and the
# underwriters' research quiet period for recent IPOs (Anthropic 11-19-2026, SpaceX 06-12-2026). Owner 11-19: the banks
# behind an IPO have already made their fees, and their analysts "pump their bags" with high targets - "it's something
# that needs to be considered". In the bot:
#   * no NEW long from IPO_LOCKUP_BEFORE sessions before a major lock-up release to IPO_LOCKUP_AFTER sessions after it -
#     the supply overhang when insiders and early holders may sell (unlock days have carried negative abnormal returns
#     in the academic studies, larger for venture-backed issuers). Shorts and exits are never blocked.
#   * a long opened in the first IPO_QUIET_SESSIONS sessions after the research quiet period ends - when the syndicate's
#     own analysts initiate (conflicted coverage) - is sized x IPO_QUIET_SIZE
#   * ipoAtEntry on the trade (next event, sessions to it, lead underwriters, fees, conflicted window): the journal
#     grades whether these rules paid - evidence, not assumption
# Dated events only: a release defined by a rule ("after the Q3 results") or a price trigger is context, never a gate.
IPO_URL = os.environ.get("RUNNER_IPO_URL") or f"{RAW}/TRAPP2-ANALYTICS/main/data/web/ipo.json"
IPO_LOCKUP_BEFORE = int(_envf("RUNNER_IPO_LOCKUP_BEFORE", 3))
IPO_LOCKUP_AFTER = int(_envf("RUNNER_IPO_LOCKUP_AFTER", 1))
IPO_QUIET_SESSIONS = int(_envf("RUNNER_IPO_QUIET_SESSIONS", 5))
IPO_QUIET_SIZE = _envf("RUNNER_IPO_QUIET_SIZE", 0.85)
_IPO = {"loaded": False, "events": [], "ipos": {}, "asOf": None}


def load_ipo():
    if _IPO["loaded"]:
        return _IPO
    d = fetch_json(IPO_URL, timeout=30)
    d = d if isinstance(d, dict) else {}
    ev = [e for e in (d.get("events") or []) if isinstance(e, dict) and e.get("ticker")]
    _IPO.update(loaded=True, events=ev, ipos=d.get("ipos") or {}, asOf=d.get("asOf"))
    dated = [e for e in ev if e.get("date")]
    if dated:
        log("IPO calendar: " + " · ".join(f"{e['ticker']} {e['kind']} {e['date']}" + (" (major)" if e.get("major") else "")
                                          for e in dated[:6]))
    elif not d:
        log("IPO calendar: none yet (TRAPP2-ANALYTICS data/web/ipo.json) - no lock-up guard")
    return _IPO


def _ipo_session_of(d, strictly_after=False):
    """The session an event lands on: the date itself when it's a session, else the next one (a Sunday quiet-period
    end -> Monday). strictly_after: the first session AFTER the date (research may publish the day after the quiet
    period's last day)."""
    if strictly_after:
        d += timedelta(days=1)
    for _ in range(10):
        try:
            if mh.day_status(d)[0]:
                return d
        except Exception:
            if d.weekday() < 5:
                return d
        d += timedelta(days=1)
    return d


def _ipo_sessions(event_date, today, strictly_after=False):
    """Sessions from the event's session to today: + after it, - before it (counting the event's session), 0 on it.
    None when a date can't be read."""
    a, b = tb._d(event_date), tb._d(today)
    if not a or not b:
        return None
    a = _ipo_session_of(a, strictly_after)
    if a == b:
        return 0
    if b > a:
        return tb.sessions_between(a, b, mh.day_status)
    return -tb.sessions_between(b, a, mh.day_status)


def _ipo_events(ticker):
    t = str(ticker or "").upper()
    return [e for e in _IPO["events"] if str(e.get("ticker") or "").upper() == t and e.get("date")]


def ipo_gate(ticker, direction, today):
    """No new long into a major lock-up release (IPO_LOCKUP_BEFORE sessions before to IPO_LOCKUP_AFTER after).
    -> reason or None."""
    if direction == "short":
        return None
    for e in _ipo_events(ticker):
        if e.get("kind") != "lock-up" or not e.get("major"):
            continue
        s = _ipo_sessions(e["date"], today)
        if s is not None and -IPO_LOCKUP_BEFORE <= s <= IPO_LOCKUP_AFTER:
            when = "today" if s == 0 else (f"in {-s} session{'s' if s < -1 else ''}" if s < 0 else f"{s} session{'s' if s > 1 else ''} ago")
            sh = e.get("shares")
            return (f"IPO lock-up release {e['date']} ({when}){f' - {sh / 1e6:,.0f}M shares' if sh else ''}: no new long into "
                    f"the supply overhang")
    return None


def ipo_size_mult(ticker, short, today):
    """A long in the sessions right after the underwriters' research quiet period ends (conflicted initiations) is
    smaller. -> (multiplier, why)."""
    if short:
        return 1.0, None
    for e in _ipo_events(ticker):
        if e.get("kind") != "quiet-period":
            continue
        s = _ipo_sessions(e["date"], today, strictly_after=True)          # initiations: the session after it ends
        if s is not None and 0 <= s < IPO_QUIET_SESSIONS:
            return IPO_QUIET_SIZE, f"syndicate initiations window (quiet period ended {e['date']})"
    return 1.0, None


def ipo_card(ticker, today):
    """The IPO context on the trade date, for the journal (None for anything that isn't a tracked IPO)."""
    t = str(ticker or "").upper()
    info = _IPO["ipos"].get(t)
    if not info:
        return None
    nxt = None
    for e in sorted(_ipo_events(t), key=lambda e: e["date"]):
        s = _ipo_sessions(e["date"], today, strictly_after=e.get("kind") == "quiet-period")
        if s is not None and s <= 0:
            nxt = {"date": e["date"], "kind": e.get("kind"), "major": bool(e.get("major")), "sessionsTo": -s}
            break
    q = ipo_size_mult(t, False, today)
    uw = info.get("underwriters") or {}
    fees = info.get("fees") or {}
    return {"firstTrade": info.get("firstTrade"), "ipoPrice": info.get("price"), "next": nxt,
            "leadUnderwriters": uw.get("lead"), "syndicate": uw.get("count"), "fees": fees.get("owner") or fees.get("estimates"),
            "conflictedWindow": q[0] != 1.0, "lockupGate": bool(ipo_gate(t, "long", today))}


# ------------------------------------------------------------------------------------------ z201: today's trends ---
# TRAPP2-ANALYTICS data/trends/latest.json (pipeline/trends.py, every 30 min on weekdays 0915-1845 ET): the S&P 500's
# close-to-close run and how rare it is (today's provisional close counted in the session), its drawdown from the
# 52-week high, the 50-day average, VIX, distribution days, breadth (members up, equal- vs cap-weighted) and credit
# (HYG) scored into one market read - calm / under pressure / sell-off / heavy sell-off, with correction (-10%) and
# bear market (-20%) overlays. Owner 11-20: "this can be used for the bot as an indicator, it doesn't necessarily need
# to be displayed individually to a stock but just considered." In the bot:
#   * new equity LONGS (stocks, long equity funds) sized x TREND_SIZE (0.9) while the read is a sell-off / heavy
#     sell-off or the S&P 500 is in a correction / bear market - one trim, not one per criterion. A prior, not an
#     earned edge: the journal decides (below). Shorts, inverse funds, rates / commodity / volatility funds, crypto
#     and futures are untouched - never enlarged by it.
#   * trendsAtEntry on every trade (the read, the S&P run incl. today's provisional close, gap / from-open,
#     drawdown, VIX, breadth, EW vs CW) - the record that will say whether buying into sell-offs paid.
# Stale (the doc's session > TREND_MAX_AGE_DAYS calendar days old) or missing: context only - no size change.
#
# z202 - CONSIDERED, NOT JUST RECORDED (owner 11-21: "make sure this new trends strategy is relevant and being
# considered by the BOT and QUANT ... improve these trading schemas without faster data for now"):
#   * THE RECORD DECIDES THE TRIM. trend_evidence() labels every closed equity long by the market read as known at its
#     entry day's open (pipeline/market_tape.py - six of the seven criteria, from daily closes, so the whole journal
#     is labelled, not only trades since z201) and compares longs opened in a sell-off with the rest: per entry-day
#     means (trades on one day share one market - as the quant lab clusters by session), Welch t, >= TREND_MIN_DAYS
#     days and TREND_MIN_TRADES trades a side. "worse" (t <= -2) deepens the trim to TREND_SIZE_EARNED (0.8);
#     "better" (t >= 2) lifts it (1.0); until then the 0.9 is a stated prior. Never an increase.
#   * NET LONG BETA in a sell-off: the risk book's net beta cap for the long side x TREND_BETA (0.75: +-45 % -> +34 %
#     long) while the read is a sell-off / heavy sell-off or the S&P 500 is in a correction / bear market - new longs
#     can't push the book's market exposure up into a falling tape; shorts and hedges (which shrink it) pass as before.
#   * trendScores (learning rollups): how trades did by the market they were opened in (calm / pressure / selloff x
#     long / short) - the read's own track record, beside creditScores.
TRENDS_URL = os.environ.get("RUNNER_TRENDS_URL") or f"{RAW}/TRAPP2-ANALYTICS/main/data/trends/latest.json"
TREND_SIZE = _envf("RUNNER_TREND_SIZE", 0.9)
TREND_SIZE_EARNED = _envf("RUNNER_TREND_SIZE_EARNED", 0.8)       # z202: when the record says sell-off longs did worse
TREND_BETA = _envf("RUNNER_TREND_BETA", 0.75)                   # z202: the long side's net beta cap in a sell-off
TREND_MIN_DAYS = int(_envf("RUNNER_TREND_MIN_DAYS", 8))         # z202: entry days a side before the record decides
TREND_MIN_TRADES = int(_envf("RUNNER_TREND_MIN_TRADES", 20))    # z202: ... and trades a side
TREND_T = _envf("RUNNER_TREND_T", 2.0)                          # z202: |Welch t| the difference must clear
TREND_MAX_AGE_DAYS = _envf("RUNNER_TREND_MAX_AGE_DAYS", 4)
TREND_STATES = ("sell-off", "heavy sell-off")
TREND_OVERLAYS = ("correction", "bear market")
_TREND = {"loaded": False, "doc": None, "stale": True, "age": None, "evidence": None}
_TAPE = {"loaded": False, "H": None, "memo": {}}


def load_trends():
    if _TREND["loaded"]:
        return _TREND
    d = fetch_json(TRENDS_URL, timeout=30)
    doc = d if isinstance(d, dict) and isinstance(d.get("read"), dict) and isinstance(d.get("indices"), dict) else None
    age = None
    try:
        age = ((NOW_UTC or datetime.now(timezone.utc)).date() - datetime.strptime(str((doc or {}).get("date"))[:10], "%Y-%m-%d").date()).days
    except (TypeError, ValueError):
        pass
    stale = doc is None or age is None or age < 0 or age > TREND_MAX_AGE_DAYS
    _TREND.update(loaded=True, doc=doc, stale=stale, age=age)
    if doc:
        c = trend_card() or {}
        m, why = trend_size_mult()
        log(f"today's trends: {c.get('label')} ({c.get('on')} of {c.get('of')}) · S&P 500 {c.get('spxChg')}% "
            f"({'final' if c.get('spxFinal') else 'provisional ' + str(c.get('etTime') or '')}) · run {c.get('run')} through "
            f"{c.get('runThrough')}" + (f" ({age} day(s) old)" if age else "")
            + (" - STALE: context only" if stale else f" · new equity longs x{m:.2f}" if m != 1.0 else " · no size change"))
    else:
        log("today's trends: no read yet (TRAPP2-ANALYTICS data/trends/latest.json) - trades run without it")
    return _TREND


def trend_size_mult(short=False, etp=None, alt=None):
    """-> (multiplier, why). One trim on a new equity long while the market reads sell-off / heavy sell-off or the
    S&P 500 is in a correction / bear market. Never enlarges anything; a short or an inverse fund is untouched."""
    doc = None if _TREND["stale"] else _TREND["doc"]
    if not doc or TREND_SIZE >= 1.0:
        return 1.0, None
    lev = float((etp or {}).get("lev") or 0)
    if alt or (etp and etp.get("underlying") != "equity") or (bool(short) != (lev < 0)):
        return 1.0, None
    r = doc.get("read") or {}
    if r.get("state") in TREND_STATES or r.get("overlay") in TREND_OVERLAYS:
        ev = (_TREND.get("evidence") or {}).get("verdict")       # z202: the bot's own record decides
        m = TREND_SIZE_EARNED if ev == "worse" else 1.0 if ev == "better" else TREND_SIZE
        if m >= 1.0:
            return 1.0, None
        tail = "the record: sell-off longs did worse" if ev == "worse" else f"a prior - the record: {ev or 'not yet'}"
        return round(max(0.5, m), 3), f"S&P 500 {r.get('label')}, {r.get('on')} of {r.get('of')} · {tail}"
    return 1.0, None


def trend_beta_scale():
    """z202: -> (scale on the long side's net beta cap, why). x TREND_BETA while the market reads sell-off / heavy
    sell-off or the S&P 500 is in a correction / bear market; 1.0 otherwise, stale or missing."""
    doc = None if _TREND["stale"] else _TREND["doc"]
    if not doc or TREND_BETA >= 1.0:
        return 1.0, None
    r = doc.get("read") or {}
    if r.get("state") in TREND_STATES or r.get("overlay") in TREND_OVERLAYS:
        return round(max(0.25, TREND_BETA), 3), f"S&P 500 {r.get('label')}"
    return 1.0, None


def load_tape():
    """z202: TRAPP2-1's daily ^GSPC / ^VIX / HYG / SPY (pipeline/market_tape.py), once per pass."""
    if not _TAPE["loaded"]:
        _TAPE["loaded"] = True
        try:
            _TAPE["H"] = mt.load_histories(lambda url: fetch_json(url, timeout=30))
        except Exception as e:
            log(f"market tape unavailable ({type(e).__name__}: {e})")
            _TAPE["H"] = None
    return _TAPE["H"]


def read_at_open(day):
    """z202: the market read as known at the open of `day` (calm / pressure / selloff + the numbers) or None."""
    if not day:
        return None
    d = str(day)[:10]
    if d not in _TAPE["memo"]:
        H = load_tape()
        _TAPE["memo"][d] = mt.state_on(d, H) if H and (H.get("spx") or ([],))[0] else None
    return _TAPE["memo"][d]


def _equity_long(t):
    """z202: a closed SHARES trade that is in effect long equities (an inverse fund is a short; crypto, futures, FX,
    indices and rates / commodity / volatility funds are not equity reads)."""
    if t.get("instrument") not in (None, "shares"):
        return False
    tk = str(t.get("ticker") or "").upper()
    if tk.endswith("-USD") or tk.endswith("-USDT") or tk.endswith("=F") or tk.endswith("=X") or tk.startswith("^") or "." in tk:
        return False                                     # crypto, futures, FX, indices, foreign listings (their own market)
    etp = t.get("etp") or None
    if etp and etp.get("underlying") not in (None, "equity"):
        return False
    lev = float((etp or {}).get("lev") or 0)
    return (t.get("direction") != "short") != (lev < 0)


TREND_RET_CAP = 50.0                    # z202: a trade's return counted at most +-50 % (the June app-era journal has FX-basis
                                        # outliers, e.g. a +403 % 000660.KS long) - one bad record can't decide the verdict


def _ret_pct(t, cap=None):
    n = float(t.get("notional") or t.get("dollars") or 0)
    if n <= 0:
        return None
    r = _trade_pl(t) / n * 100
    return max(-cap, min(cap, r)) if cap else r


def trend_evidence(state):
    """z202: did equity longs opened in a sell-off (the read at the entry day's open) do better or worse than the rest?
    Per entry-day mean returns (trades on one day share one market), Welch t between the two groups of days.
    -> {verdict: not yet | worse | better | no difference, why, selloff {...}, other {...}, t, rule}"""
    days = {"selloff": {}, "other": {}}
    trades = {"selloff": 0, "other": 0}
    for t in trades_list(state):
        if t.get("status") != "closed" or not _equity_long(t):
            continue
        r = _ret_pct(t, TREND_RET_CAP)
        st = (read_at_open(t.get("entryDate") or str(t.get("entryAt") or "")[:10]) or {}).get("state")
        if r is None or not st:
            continue
        g = "selloff" if st == "selloff" else "other"
        days[g].setdefault(str(t.get("entryDate") or t.get("entryAt"))[:10], []).append(r)
        trades[g] += 1

    def side(g):
        xs = [sum(v) / len(v) for v in days[g].values()]
        n = len(xs)
        m = sum(xs) / n if n else None
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else None
        return {"days": n, "trades": trades[g], "meanPct": round(m, 3) if m is not None else None,
                "sdPct": round(sd, 3) if sd is not None else None}, m, sd, n
    a, ma, sa, na = side("selloff")
    b, mb, sb, nb = side("other")
    t_val = None
    if sa is not None and sb is not None:
        se = math.sqrt(sa * sa / na + sb * sb / nb)
        if se > 0:
            t_val = round((ma - mb) / se, 2)
    out = {"selloff": a, "other": b, "t": t_val,
           "rule": (f"US equity longs (shares; inverse funds count as shorts; crypto, futures, foreign listings and non-equity funds out) "
                    f"by the market read at the entry day's open (pipeline/market_tape.py: selloff vs calm + pressure); returns capped at "
                    f"+-{TREND_RET_CAP:g} %; per entry-day means, Welch t; needs >= {TREND_MIN_DAYS} days and >= {TREND_MIN_TRADES} trades a side and |t| >= {TREND_T:g}")}
    enough = na >= TREND_MIN_DAYS and nb >= TREND_MIN_DAYS and trades["selloff"] >= TREND_MIN_TRADES and trades["other"] >= TREND_MIN_TRADES
    if not enough:
        out["verdict"] = "not yet"
        out["why"] = (f"sell-off longs on {na} of {TREND_MIN_DAYS} days needed ({trades['selloff']} of {TREND_MIN_TRADES} trades)"
                      f" · the rest on {nb} days ({trades['other']} trades)")
    else:
        diff = f"longs opened in sell-offs averaged {ma:+.2f}% a trade-day vs {mb:+.2f}% (t {t_val})"
        out["verdict"] = "worse" if (t_val is not None and t_val <= -TREND_T) else "better" if (t_val is not None and t_val >= TREND_T) else "no difference"
        out["why"] = diff
    return out


def trend_card():
    """Today's market read on the trade date, for the journal (None before the first trends run)."""
    doc = _TREND.get("doc")
    if not doc:
        return None
    r = doc.get("read") or {}
    spx = (doc.get("indices") or {}).get("^GSPC") or {}
    s = spx.get("streak") or {}
    pv = (spx.get("provisional") or {}).get("now")
    ins = doc.get("internals") or {}
    run = (-1 if s.get("dir") == "down" else 1) * int(s.get("n") or 0) if s.get("dir") else 0
    run_now = run                                     # the run if today closed where it is now
    if pv in ("red", "green"):
        same = (pv == "red") == (run < 0) and run != 0
        run_now = (run + (-1 if run < 0 else 1)) if same else (-1 if pv == "red" else 1)
    return {"state": r.get("state"), "overlay": r.get("overlay"), "label": r.get("label"), "on": r.get("on"), "of": r.get("of"),
            "spxChg": spx.get("chg"), "spxFinal": bool(spx.get("final")), "gap": spx.get("gap"), "fromOpen": spx.get("fromOpen"),
            "run": run, "runThrough": s.get("through"), "runNow": run_now,
            "dd52": (spx.get("high52") or {}).get("dd"), "vix": ((doc.get("vol") or {}).get("^VIX") or {}).get("close"),
            "pctUp": ins.get("pctUp"), "ew": ins.get("ew"), "cw": ins.get("cw"),
            "date": doc.get("date"), "etTime": doc.get("etTime"), "session": doc.get("state"), "stale": bool(_TREND["stale"]),
            "atOpen": (lambda m: {k: m.get(k) for k in ("state", "on", "of", "asOf", "run", "dd")} if m else None)(
                read_at_open(datetime.now(timezone.utc).strftime("%Y-%m-%d") if NOW_UTC is None else NOW_UTC.strftime("%Y-%m-%d")))}


def credit_exit(pos, today, review_only=False):
    """z194: a long whose issuer falls into distress (CCC+ or worse) closes at once; a long downgraded
    CREDIT_EXIT_NOTCHES+ since entry, or from investment grade to high yield, closes when the next session confirms it
    (a one-day model swing never closes a position); a short is covered on the mirror-image upgrade. Companies held as
    shares only (a fund's rating is its index's - its trend is in the signal). Positions opened before z194 are measured
    from the first rating the bot saw (creditBaseline)."""
    if pos.get("etp") or (pos.get("instrument") or "shares") != "shares":
        return None
    cr = credit_of(pos.get("ticker"))
    if not cr or _RATINGS["stale"] or cr.get("k") != "co" or not cr.get("n"):
        return None
    n = int(cr["n"])
    long_ = (pos.get("direction") or "long") != "short"
    if long_ and n >= CREDIT_GATE_NOTCH:
        return "credit-distress"
    ent = pos.get("creditAtEntry") or pos.get("creditBaseline")
    if not ent or not ent.get("n"):
        if not review_only:
            pos["creditBaseline"] = dict(credit_card(cr), date=today)
        return None
    n0 = int(ent["n"])
    if long_:
        broke = (n - n0 >= CREDIT_EXIT_NOTCHES) or (n0 <= CREDIT_IG < n)
        why = f"downgraded {ent.get('r')} -> {cr.get('r')}"
    else:
        broke = (n0 - n >= CREDIT_EXIT_NOTCHES) or (n <= CREDIT_IG < n0)
        why = f"upgraded {ent.get('r')} -> {cr.get('r')}"
    w = pos.get("creditWatch")
    if not broke:
        if w and not review_only:
            pos.pop("creditWatch", None)
            log(f"  {pos.get('ticker')}: credit watch cleared ({cr.get('r')})")
        return None
    if w and str(w.get("date") or "") < today:
        return "credit-downgrade" if long_ else "credit-upgrade"
    if not w and not review_only:
        pos["creditWatch"] = {"date": today, "r": cr.get("r"), "n": n, "why": why}
        log(f"  {pos.get('ticker')}: credit watch - {why}; closes if the next session confirms it")
    return None


def load_filings_and_short():
    """z158: SEC events / insiders (TRAPP2-ANALYTICS data/sec) and FINRA short data (data/short)."""
    global _SEC, _SHORT
    j = fetch_json(f"{RAW}/TRAPP2-ANALYTICS/main/data/sec/index.json", timeout=30)
    _SEC = (j or {}).get("tickers") or {} if isinstance(j, dict) else {}
    s = fetch_json(f"{RAW}/TRAPP2-ANALYTICS/main/data/short/index.json", timeout=30)
    _SHORT = {"volume": (s or {}).get("volume") or {}, "interest": (s or {}).get("interest") or {}} if isinstance(s, dict) else {"volume": {}, "interest": {}}
    log(f"filings / short: SEC on file for {len(_SEC)} companies ({sum(1 for v in _SEC.values() if v.get('red30'))} with a red-flag 8-K in 30 days) · "
        f"short volume {len(_SHORT['volume'])} · short interest {len(_SHORT['interest'])}")


def filings_of(ticker):
    """z158 context for a trade record: recent 8-K flags, insider buying / selling (90 days), short selling."""
    t = str(ticker or "").upper()
    s, v, i = _SEC.get(t) or {}, _SHORT["volume"].get(t), _SHORT["interest"].get(t)
    if not (s or v or i):
        return None
    ins = s.get("insider") or {}
    return {"red30": s.get("red30") or [], "caution30": s.get("caution30") or [], "events30": s.get("events30"),
            "insiderNet90": ins.get("net"), "insiderBuyers90": ins.get("buyers"), "insiderSellers90": ins.get("sellers"), "clusterBuy": ins.get("cluster"),
            "shortVol5d": (v or {}).get("ratio5d"), "shortVolZ": (v or {}).get("z"),
            "shortInterestChgPct": (i or {}).get("changePct"), "daysToCover": (i or {}).get("daysToCover")}


def volatility_context():
    """z158: CBOE term structure and SKEW (TRAPP2-1 data/extras/vol.json) - context, not scored."""
    v = fetch_json(f"{RAW}/TRAPP2-1/main/data/extras/vol.json", timeout=30)
    if not isinstance(v, dict) or not v.get("latest"):
        return None
    L = v["latest"]
    return {"vix": L.get("VIX"), "vix3m": L.get("VIX3M"), "vixTo3m": v.get("vixTo3m"), "structure": v.get("structure"),
            "vixPctile1y": v.get("vixPctile1y"), "skew": L.get("SKEW"), "skewPctile1y": v.get("skewPctile1y"), "date": v.get("date")}


def labor_context():
    """z151 - the market's hiring breadth (TRAPP2-ANALYTICS headcount): share of companies cutting vs
    growing staff year over year and the aggregate change. Recorded on every trade as context, NOT scored
    (evidence before scoring): one company trimming staff raises an eyebrow; many at once is how a
    slowdown shows up in the labor data."""
    global _HC
    idx = fetch_json(f"{HEADCOUNT_BASE}/index.json", timeout=30)
    _HC = (idx or {}).get("tickers") or {} if isinstance(idx, dict) else {}
    s = fetch_json(f"{HEADCOUNT_BASE}/summary.json", timeout=30)
    n = (s or {}).get("now") if isinstance(s, dict) else None
    if not n:
        return None
    pc = n.get("pctCutting")
    read_ = "broad" if (pc or 0) >= 40 else "spreading" if (pc or 0) >= 25 else "contained"
    return {"pctCutting": pc, "pctGrowing": n.get("pctGrowing"), "aggregateYoY": n.get("aggregateYoY"),
            "medianYoY": n.get("medianYoY"), "companies": n.get("companies"), "read": read_, "asOf": (s or {}).get("asOf")}


def headcount_of(ticker):
    """z151: a company's headcount for the trade record (context): employees, YoY, from peak."""
    h = _HC.get(str(ticker or "").upper())
    if not h:
        return None
    e = h.get("employees") or 0
    return {"employees": e, "yoyPct": h.get("yoyPct"), "fromPeakPct": h.get("fromPeakPct"), "date": h.get("date"),
            "scale": "large employer (10k+)" if e >= 10000 else "mid (1k-10k)" if e >= 1000 else "small (<1k)"}


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


def _pattern_edge(rec):
    """z184 (patterns-v4): the edge a template's record has EARNED - its date-matched 20-bar edge when the whole
    month-clustered 95% interval clears zero, 0 (neutral trust) when it doesn't (trustEdgePct). Older stats files
    without it fall back to the raw edge vs the 20-year baseline."""
    if not isinstance(rec, dict):
        return None
    v = rec.get("trustEdgePct")
    return v if isinstance(v, (int, float)) else rec.get("edgeVsBaselinePct")


def _pattern_trust(edge, rec):
    """z184: STATUS 2 - "a template only becomes a buy/sell signal once its record shows an edge". A record carrying a
    verdict (patterns-v4) earns trust only when EARNED (0.5 + edge/4, 0.5..1.0); UNPROVEN or HARMFUL = 0, no weight -
    its calls keep being recorded, so it can earn its way in. Older records: the old formula (0.15..1.0, 0.5 neutral)."""
    v = (rec or {}).get("edgeVerdict") if isinstance(rec, dict) else None
    if v is not None:
        return max(0.5, min(1.0, 0.5 + edge / 4.0)) if (v == "earned" and isinstance(edge, (int, float))) else 0.0
    return max(0.15, min(1.0, 0.5 + (edge / 4.0))) if isinstance(edge, (int, float)) else 0.5


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
        edge = _pattern_edge((t or {}).get("backtest") or {})
        vrec = (t or {}).get("backtest") or {}               # z184: whose verdict governs the trust
        src = "all"
        # Regime-aware: use the template's record IN THE CURRENT MARKET REGIME
        # (SPY trend: up / mixed / down) once it has enough judged calls - a 52-week
        # breakout earns trust in up-trends and loses it in down-trends.
        rg = (((t or {}).get("byRegime") or {}).get("trend") or {}).get(now or "")
        if rg and rg.get("usable") and isinstance(_pattern_edge(rg), (int, float)):
            edge, src, vrec = _pattern_edge(rg), now, rg
        # z104: backtest -> LIVE transition. The live record (calls made since the
        # live ledger started, judged only on later bars) is the honest test. Blend
        # it in by sample size against a fixed prior weight for the backtest, so
        # live evidence takes over as it accumulates (at LIVE_PRIOR_N live calls
        # it carries half the weight) and a backtest edge that doesn't hold live
        # loses trust.
        live = (t or {}).get("live") or {}
        ln, le = live.get("judged") or 0, _pattern_edge(live)
        w = 0.0
        if ln >= LIVE_MIN_N and isinstance(le, (int, float)) and isinstance(edge, (int, float)):
            w = ln / (ln + LIVE_PRIOR_N)
            edge = (1 - w) * edge + w * le
            src = f"{src}+live{ln}"
        # earned trust: +4pp 20-day edge -> 1.0, no edge -> 0.5, negative -> floor 0.15
        # z184: with a patterns-v4 verdict, only an EARNED edge carries weight (unproven / harmful -> 0)
        rel[k] = _pattern_trust(edge, vrec)
        # z103: breakout-volume split (patterns-v3) - a volume-confirmed call
        # earns the confirmed record's trust, a light-volume call the other.
        for vk, vs in (((t or {}).get("byVolume") or {}).items()):
            ve = _pattern_edge(vs or {})
            if (vs or {}).get("usable") and isinstance(ve, (int, float)):
                if w:                                    # same live blend as the template trust
                    ve = (1 - w) * ve + w * le
                rel_vol.setdefault(k, {})[vk] = _pattern_trust(ve, vs)
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


VAL_STATS = {}           # z187: sector -> {pe, ps, pb, evEbitda: median}; "_all" = the whole equity universe
VAL_METRICS = (("pe", "pe"), ("ps", "priceToSales"), ("pb", "priceToBook"), ("evEbitda", "evToEbitda"))
VAL_MIN_PEERS = 8
# how much P/B says about value, by sector: book is the business for lenders / insurers / REITs and matters for asset-heavy
# sectors; for software, brands and pharma it mostly measures past buybacks
PB_SECTOR_W = {"Financial Services": 1.0, "Real Estate": 0.9, "Energy": 0.55, "Basic Materials": 0.55, "Utilities": 0.55,
               "Industrials": 0.4}
VAL_W = {"pe": 1.0, "ps": 0.7, "evEbitda": 0.8}


def _norm_fund(u):
    """Map master.json fields to the engine's names + derive missing ratios.
    Faithful to the app's normalizeEngineFields - z187: in ONE currency (USD) and at the row's price:
      * market cap in USD (size tiers compared Tokyo yen caps with dollar caps)
      * P/S from the pipeline (z187 rows) or market cap / revenue both converted to USD (TSM was 0.4 "cheap", truly ~13)
      * P/B from the pipeline; for a row written before z187, rebuilt as market cap / total equity in USD - never
        Yahoo's (it mixes currencies for ADRs - TSM 76, ASML 1,284 - and share classes - BRK-B 0.001); never for
        negative book
      * Debt/Equity: Yahoo's is a PERCENT (84 = 0.84x) - the app divides by 100 on load, the bot read it raw, so almost
        every company scored as "debt > 2x equity" (-20 on balance-sheet strength)"""
    def n(v):
        try:
            x = float(v)
            return x if (x == x and x not in (float("inf"), float("-inf"))) else None
        except (TypeError, ValueError):
            return None
    note = str(u.get("valuationNote") or "")
    z187 = bool(u.get("valuationZ187"))
    cur = str(u.get("currency") or "USD").strip()
    q_main = _SUBUNIT.get(cur, (cur.upper(), 1.0))[0]
    fcur = str(u.get("financialCurrency") or cur).strip()
    same_ccy = _SUBUNIT.get(fcur, (fcur.upper(), 1.0))[0] == q_main
    mc_raw, mc_usd, fin_usd = n(u.get("marketcap")), n(u.get("marketcapUSD")), n(u.get("finUsd"))
    if mc_raw is None:
        mc_raw = n(u.get("marketCap"))
    eq = n(u.get("totalEquity"))
    f = {
        "marketCap": mc_usd if mc_usd is not None else (mc_raw if q_main == "USD" else None),
        "eps": n(u.get("eps")),
        "netMargin": n(u.get("profitMargin")),
        "freeCashFlow": n(u.get("freeCashFlow")),
        "roe": n(u.get("returnOnEquity")),
        "debtToEquity": (n(u.get("debtToEquity")) / 100.0) if n(u.get("debtToEquity")) is not None else None,
        "currentRatio": n(u.get("currentRatio")),
        "cash": n(u.get("cash")),
        "totalDebt": n(u.get("totalDebt")),
        "totalEquity": eq,
        "pe": n(u.get("pe")),
        "priceToBook": None,
        "evToEbitda": None,
        "revenueGrowth": n(u.get("revenueGrowth")),
        "payoutRatio": n(u.get("payoutRatio")),
        "mcapSuspect": "mcap?" in note,
        "negBook": ("neg-book" in note) or (eq is not None and eq < 0),
    }
    if f["debtToEquity"] is None:
        d = n(u.get("totalDebt"))
        if d is not None and eq and eq > 0:
            f["debtToEquity"] = d / eq
    ps = n(u.get("priceToSales"))
    rev = n(u.get("revenue"))
    if ps is None and not z187 and rev and rev > 0:          # a row written before z187: rebuild, in one currency
        if mc_usd and fin_usd:
            ps = mc_usd / (rev * fin_usd)
        elif same_ccy and mc_raw:
            ps = mc_raw / rev
    f["priceToSales"] = ps if (ps is not None and ps > 0) else None
    pb, eve = n(u.get("priceToBook")), n(u.get("evToEbitda"))
    if z187:
        f["priceToBook"] = pb if (pb is not None and pb > 0 and not f["negBook"]) else None
        f["evToEbitda"] = eve if (eve is not None and eve > 0) else None
    else:                                                   # a row from before z187 - the same rules as research
        if eq and eq > 0:
            if mc_usd and fin_usd:
                f["priceToBook"] = mc_usd / (eq * fin_usd)
            elif same_ccy and mc_raw:
                f["priceToBook"] = mc_raw / eq
        if same_ccy:                                        # Yahoo's cross-currency EV ratios are wrong - dropped
            f["evToEbitda"] = eve if (eve is not None and eve > 0) else None
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


def _val_inputs(f):
    """The multiples the valuation step may use: P/E always (per share, currency-correct); P/S, P/B and EV/EBITDA only
    when the market cap they divide is trusted (a 'mcap?' row - an ADR-ratio or one-share-class cap - keeps P/E only)."""
    out = {}
    if f.get("pe") and f["pe"] > 0:
        out["pe"] = f["pe"]
    if not f.get("mcapSuspect"):
        for m, k in VAL_METRICS[1:]:
            v = f.get(k)
            if v and v > 0:
                out[m] = v
    return out


def valuation_stats(rows):
    """z187: median P/E, P/S, P/B and EV/EBITDA per sector (>= 8 companies with the figure) and for the whole equity
    universe - what 'cheap' means is relative: a grocer at 0.3x sales isn't cheap, software at 6x may be."""
    by = {}
    for u in rows.values():
        if (u.get("assetClass") or "Equity") != "Equity" or str(u.get("isetf") or "") == "true":
            continue
        vals = _val_inputs(_norm_fund(u))
        for key in (u.get("sector") or "Unknown", "_all"):
            d = by.setdefault(key, {})
            for m, v in vals.items():
                d.setdefault(m, []).append(v)
    out = {}
    for key, d in by.items():
        med = {}
        for m, vs in d.items():
            if len(vs) >= VAL_MIN_PEERS:
                vs = sorted(vs)
                k = len(vs) // 2
                med[m] = vs[k] if len(vs) % 2 else (vs[k - 1] + vs[k]) / 2
        if med:
            out[key] = med
    return out


def valuation_points(f, sector, quality):
    """z187: the valuation step, SECTOR-RELATIVE. Each multiple vs its sector median (the universe's when the sector
    has < 8 peers): -tanh(ln(multiple / median) / 0.5) - 30 % cheaper scores about +0.6, twice as dear about -0.9.
    Weights: P/E 1.0 (a quality company's premium counts 40 % less against it), P/S 0.7 (1.0 when there are no
    earnings), EV/EBITDA 0.8, P/B by sector (1.0 banks / insurers ... 0.3 software). 50 = fairly valued; 14..86.
    None when no multiple has a median to compare with (the absolute rules then apply)."""
    st, al = VAL_STATS.get(sector or "Unknown") or {}, VAL_STATS.get("_all") or {}
    use = _val_inputs(f)
    num = den = 0.0
    for m, v in use.items():
        med = st.get(m) or al.get(m)
        if not med or med <= 0:
            continue
        s = -math.tanh(math.log(v / med) / 0.5)
        if m == "pe" and quality and s < 0:
            s *= 0.6
        w = PB_SECTOR_W.get(sector, 0.3) if m == "pb" else VAL_W[m]
        if m == "ps" and "pe" not in use:
            w = 1.0
        num += w * s
        den += w
    return (50.0 + 36.0 * num / den) if den else None


def valuation_card(u):
    """z187: the multiples a trade was judged on, next to its sector's medians - on the trade record (Bets tab)."""
    try:
        f = _norm_fund(u)
        sec = u.get("sector") or "Unknown"
        st, al = VAL_STATS.get(sec) or {}, VAL_STATS.get("_all") or {}
        out = {"sector": sec, "note": u.get("valuationNote") or None}
        for m, k in VAL_METRICS:
            v = f.get(k)
            if v and v > 0:
                out[m] = round(v, 2)
                med = st.get(m) or al.get(m)
                if med:
                    out[m + "Median"] = round(med, 2)
        quality = (f["roe"] is not None and f["roe"] > 0.15) or (f["revenueGrowth"] is not None and f["revenueGrowth"] > 0.15)
        pts = valuation_points(f, sec, quality) if VAL_STATS else None
        out["points"] = round(pts, 1) if pts is not None else None
        return out
    except Exception:
        return None


def recently_renamed(u, today, days=30):
    """z187: the pipeline saw the company name behind this symbol change (renamed, or the symbol now belongs to someone
    else - PARA went from Paramount to a micro-cap): its history and fundamentals don't describe one company, so it is
    not opened for 30 days. Held positions are still managed."""
    at = u.get("nameChangedAt")
    if not at:
        return False
    try:
        return (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(str(at)[:10], "%Y-%m-%d")).days <= days
    except ValueError:
        return False


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
        if eq is not None and eq < 0:
            # z187: negative book from buybacks at a cash-generating company (MCD, ABBV, SBUX) is a capital-return
            # choice, not distress - a smaller deduction; with no profits or free cash flow it stays a red flag
            cash_gen = (fcf is not None and fcf > 0) and (nm is None or nm > 0)
            pts -= 8 if cash_gen else 25
        steps[3] = max(0.0, min(100.0, pts))
    else:
        steps[3] = None

    pe, pb, ps, rg = f["pe"], f["priceToBook"], f["priceToSales"], f["revenueGrowth"]
    quality = (roe is not None and roe > 0.15) or (rg is not None and rg > 0.15)
    rel = valuation_points(f, u.get("sector"), quality) if VAL_STATS else None
    if rel is not None:
        steps[4] = max(0.0, min(100.0, rel))                 # z187: sector-relative
    elif pe is not None or pb is not None or ps is not None:
        pts = 50.0
        if pe is not None and pe > 0:
            fair = 35 if quality else 20
            if pe < fair * 0.6: pts += 18
            elif pe < fair: pts += 6
            elif pe < fair * 1.7: pts -= 6
            else: pts -= 18
        if ps is not None:
            if ps < 2: pts += 8
            elif ps > 12: pts -= 10
        if pb is not None and 0 < pb < 1: pts += 10           # z187: never for negative book (was "below book - value")
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
    # z192: a fund / ETP has no company fundamentals - an ETF's distribution yield alone made the reasoning chain score
    # SQQQ +0.50 "fundamentals" and +0.50 "health" (a bear fund's T-bill income), and the horizon read it as a slow,
    # long-term thesis ("core", 8-month plan). Funds are judged on their own chart and the market signals.
    fund = rb.fund_reason(u)
    etp = rb.etp_exposure(u)
    if etp:
        fund = fund or "a leveraged / inverse product"
    if not fund:
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
    cs = None if fund else tb.catalyst_signal(tb.catalyst_info(_CAL, tk, ctx.get("today") or datetime.now(timezone.utc).strftime("%Y-%m-%d")))
    if cs is not None and abs(cs) > 0.03:
        comps["catalyst"] = cs
    # z194: credit - the issuer's rating level and trend; a fund's (unlevered) holdings' trend, signed below for an ETP
    cr = None if _RATINGS["stale"] else credit_of(tk)
    if cr and cr.get("sig") is not None and abs(cr["sig"]) > 0.02:
        comps["credit"] = float(cr["sig"])
    # z196: early warning - market risk + this industry's exposure to the current warnings (a fund: market lean only)
    wsig = warning_signal(u.get("sector"), u.get("industry"), etp)
    if wsig is not None and abs(wsig) > 0.03:
        comps["warning"] = wsig
    if etp:
        _etp_sign_market(comps, etp, ctx.get("xa"))
    return {"comps": comps, "signed": blend_score(comps, ctx["weights"], ctx["weight_mods"]), "etp": etp}


def _market_cross_asset(xa):
    """The broad-market cross-asset read (global indices less the safe-haven pull) - cross_asset_for_sector's default."""
    if not xa:
        return None
    idx, sh = xa.get("indices"), xa.get("safeHaven")
    if idx is None and sh is None:
        return None
    return max(-1.0, min(1.0, (idx or 0) - 0.5 * (sh or 0)))


def _etp_sign_market(comps, etp, xa):
    """z192: a leveraged / inverse ETP's market-wide signals in ITS direction. A risk-off tape is a buy signal for a
    bear fund: before z192 SPXS (-3x S&P 500) counted cross-asset -1.00 and options-market -0.44 against its own long.
    Equity products take the broad-market read times the sign of their leverage; sector / beta tilts (regime, Fed) are
    not defined for them; a rates / commodity / volatility product takes no equity-market signal."""
    for k in ETP_SECTOR_SIGNALS + ("crossAsset",):
        comps.pop(k, None)
    sgn = 1.0 if etp["lev"] > 0 else -1.0
    if comps.get("credit") is not None:
        # z194: the rating file gives the UNLEVERED underlying's credit trend (a volatility product's is already "credit
        # stress lifts it") - a bear fund gains when its index's credit deteriorates
        comps["credit"] = round(sgn * comps["credit"], 3)
    if etp.get("underlying") != "equity":
        comps.pop("optionsMarket", None)
        comps.pop("warning", None)          # z196: an equity-market read says nothing about a rates / commodity / vol fund
        return comps
    if comps.get("warning") is not None:
        comps["warning"] = round(sgn * comps["warning"], 3)   # z196: a bear fund gains when the warning is right
    mkt = _market_cross_asset(xa)
    if mkt is not None:
        comps["crossAsset"] = round(sgn * mkt, 3)
    if comps.get("optionsMarket") is not None:
        comps["optionsMarket"] = round(sgn * comps["optionsMarket"], 3)
    return comps


# ------------------------------------------------------------------------------------------- z192: learning ----
# The bot learns from EVERY closed trade (owner, 11-17). Before z192 the server bot never did: learnedWeights stayed {}
# (every signal at 1.0) and signalScores / styleScores were an app export frozen at 8 trades - the trades were recorded,
# nothing was learned from them. retrain() REPLAYS the whole journal in exit order on each pass - so the weights are a
# pure function of the trades (reproducible, and a restated P/L re-teaches) - with the app's botRetrain rule: a signal
# that leaned the trade's way on a winner is trusted more, on a loser less (x(1 +/- 0.05 x |signal|), kept in
# 0.25..4 - a signal is never removed nor dominant). A flat trade is neither a win nor a loss (the win-rate rule); a
# regime signal learns only when the call was dialed in and right (z93); an exit the strategy didn't choose (a rule
# unwind) teaches at half weight; a trade without recorded signals (app era) teaches the style / exit statistics.
LEARN_RATE = _envf("RUNNER_LEARN_RATE", 0.05)
LEARN_MIN, LEARN_MAX = 0.25, 4.0
LEARN_FORCED = _envf("RUNNER_LEARN_FORCED", 0.5)
FORCED_EXITS = ("not-shortable",)
REGIME_SIGNALS = ("regimeGrade", "regime", "quad")


def _trade_pl(t):
    """What the trade put into the bankroll - exit P/L and its trims (the ledger's and the win rate's number)."""
    return round(ts.exit_realized(t) + ts.trim_total(t), 2)


def _lesson_components(t, universe=None):
    """The signals a closed trade teaches, as they should have read at entry. A z192+ trade recorded them clean; an
    older one is cleaned here: a fund's company signals dropped, an ETP's market signals put in its direction."""
    comps = {}
    for k, v in (t.get("components") or {}).items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v != 0:
            comps[k] = v
    if not comps or t.get("signalBasis") == SIGNAL_BASIS:
        return comps, []
    tk = str(t.get("ticker") or "").upper()
    u = dict((universe or {}).get(tk) or {})
    u.setdefault("ticker", tk)
    if not u.get("name"):
        u["name"] = t.get("name")
    etp = rb.etp_exposure(u)
    notes = []
    if rb.fund_reason(u) or etp:
        gone = [k for k in COMPANY_SIGNALS if k in comps]
        for k in gone:
            comps.pop(k)
        if gone:
            notes.append("fund: no company signals (" + ", ".join(gone) + ")")
    if etp:
        sgn = 1.0 if etp["lev"] > 0 else -1.0
        gone = [k for k in ETP_SECTOR_SIGNALS if k in comps]
        for k in gone:
            comps.pop(k)
        for k in ETP_MARKET_SIGNALS:
            if k in comps:
                if etp.get("underlying") != "equity":
                    comps.pop(k)
                    gone.append(k)
                elif sgn < 0:
                    comps[k] = -comps[k]
        notes.append(f"{etp['lev']:+g}x {etp.get('underlying')} ETP: market signals in its direction"
                     + (f", {', '.join(gone)} dropped" if gone else ""))
    return comps, notes


def retrain(state, universe=None, now_iso=None):
    """Replay every closed trade (exit order) into learnedWeights; stamp each trade's lesson; summarise in
    state["learning"]. Idempotent - the same journal gives the same weights."""
    def _key(t):
        return (str(t.get("exitAt") or t.get("exitDate") or ""), str(t.get("entryAt") or t.get("entryDate") or ""), str(t.get("id") or ""))
    closed = sorted((t for t in trades_list(state) if t.get("status") == "closed"), key=_key)
    w = {}
    tally = {"closed": len(closed), "taught": 0, "flat": 0, "noSignals": 0, "forced": 0, "nudges": 0}
    for t in closed:
        pl = _trade_pl(t)
        verdict = "win" if pl > 0 else ("loss" if pl < 0 else "flat")
        comps, notes = _lesson_components(t, universe)
        forced = str(t.get("exitReason") or "").split(" ")[0] in FORCED_EXITS
        kw = LEARN_FORCED if forced else 1.0
        rc = t.get("regimeContext") if isinstance(t.get("regimeContext"), dict) else {}
        regime_ok = bool(rc) and rc.get("dialedIn") is True and t.get("regimeCallCorrect") is not False
        dir_sign = -1 if t.get("direction") == "short" else 1
        nudges = {}
        if verdict != "flat":
            for k, v in sorted(comps.items()):
                if k in REGIME_SIGNALS and not regime_ok:
                    continue
                agreed = (v > 0) == (dir_sign > 0)
                d = (1 if (verdict == "win") == agreed else -1) * LEARN_RATE * kw * min(1.0, abs(v))
                w[k] = max(LEARN_MIN, min(LEARN_MAX, w.get(k, 1.0) * (1 + d)))
                nudges[k] = round(d, 4)
        if not comps:
            tally["noSignals"] += 1
            notes = notes + ["no signals recorded at entry - teaches the style / exit statistics only"]
        elif verdict == "flat":
            tally["flat"] += 1
        else:
            tally["taught"] += 1
        tally["forced"] += 1 if forced else 0
        tally["nudges"] += len(nudges)
        t["lesson"] = {"verdict": verdict, "pl": pl, "weight": kw, "signals": {k: round(v, 3) for k, v in sorted(comps.items())},
                       "nudges": nudges, "notes": notes}
    lw = state.get("learnedWeights") if isinstance(state.get("learnedWeights"), dict) else {}
    before = dict(lw)
    lw.clear()
    lw.update({k: round(v, 4) for k, v in sorted(w.items())})
    state["learnedWeights"] = lw
    ranked = sorted(lw.items(), key=lambda kv: kv[1])
    state["learning"] = dict(tally, **{
        "rule": "replay every closed trade: x(1 +/- %g x |signal|) toward signals that leaned the winning way; flat = no lesson; "
                "regime signals only when dialed in and right; forced exits x%g; weights kept in %g..%g" % (LEARN_RATE, LEARN_FORCED, LEARN_MIN, LEARN_MAX),
        "rate": LEARN_RATE, "bounds": [LEARN_MIN, LEARN_MAX], "forcedWeight": LEARN_FORCED,
        "at": now_iso or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "weakest": [[k, v] for k, v in ranked[:3]], "strongest": [[k, v] for k, v in ranked[::-1][:3]],
        "changed": sorted(k for k in set(before) | set(lw) if abs(before.get(k, 1.0) - lw.get(k, 1.0)) > 1e-9)})
    return state["learning"]


def learning_rollups(state, universe=None):
    """signalScores / styleScores from the journal (the app's rollup: a signal's win rate when it leaned the trade's
    way, decisive trades only; a style's win rate and P/L)."""
    w = state.get("learnedWeights") if isinstance(state.get("learnedWeights"), dict) else {}
    sig, sty = {}, {}
    for t in trades_list(state):
        if t.get("status") != "closed":
            continue
        pl = _trade_pl(t)
        s = sty.setdefault(str(t.get("style") or "other"), {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0})
        s["trades"] += 1
        s["wins"] += 1 if pl > 0 else 0
        s["losses"] += 1 if pl < 0 else 0
        s["pnl"] += pl
        if pl == 0:
            continue
        dir_sign = -1 if t.get("direction") == "short" else 1
        comps, _ = _lesson_components(t, universe)
        for k, v in comps.items():
            p = sig.setdefault(k, {"trades": 0, "wins": 0, "lean": 0, "leanWon": 0})
            p["trades"] += 1
            p["wins"] += 1 if pl > 0 else 0
            if (v > 0) == (dir_sign > 0):
                p["lean"] += 1
                p["leanWon"] += 1 if pl > 0 else 0
    state["signalScores"] = sorted(({"signal": k, "trades": p["trades"], "leanN": p["lean"],
                                     "winRateWhenLeaning": round(p["leanWon"] / p["lean"], 3) if p["lean"] else None,
                                     "currentWeight": round(float(w.get(k, 1.0)), 3)} for k, p in sig.items()),
                                   key=lambda x: -(x["winRateWhenLeaning"] or 0))
    # z194: how trades did by the credit they were opened on (bucket at entry) - the rating's own track record
    cb = {}
    for t in trades_list(state):
        if t.get("status") != "closed" or not (t.get("creditAtEntry") or {}).get("n"):
            continue
        n = int(t["creditAtEntry"]["n"])
        b = "AAA-AA" if n <= 4 else "A" if n <= 7 else "BBB" if n <= 10 else "BB" if n <= 13 else "B" if n <= 16 else "CCC-D"
        k = f"{b}|{'short' if t.get('direction') == 'short' else 'long'}"
        pl = _trade_pl(t)
        e = cb.setdefault(k, {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0})
        e["trades"] += 1
        e["wins"] += 1 if pl > 0 else 0
        e["losses"] += 1 if pl < 0 else 0
        e["pnl"] += pl
    # z202: how trades did by the market they were opened in (the read at the entry day's open) - the tape's own record
    tsc = {}
    for t in trades_list(state):
        if t.get("status") != "closed":
            continue
        st = (read_at_open(t.get("entryDate") or str(t.get("entryAt") or "")[:10]) or {}).get("state")
        if not st:
            continue
        k = f"{st}|{'short' if t.get('direction') == 'short' else 'long'}"
        pl, rp = _trade_pl(t), _ret_pct(t, TREND_RET_CAP)
        e = tsc.setdefault(k, {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "ret": []})
        e["trades"] += 1
        e["wins"] += 1 if pl > 0 else 0
        e["losses"] += 1 if pl < 0 else 0
        e["pnl"] += pl
        if rp is not None:
            e["ret"].append(rp)
    order = {"calm": 0, "pressure": 1, "selloff": 2}
    state["trendScores"] = [{"read": k.split("|")[0], "direction": k.split("|")[1], "trades": e["trades"],
                             "winRate": round(e["wins"] / (e["wins"] + e["losses"]), 3) if (e["wins"] + e["losses"]) else None,
                             "totalPnl": round(e["pnl"], 2), "avgReturnPct": round(sum(e["ret"]) / len(e["ret"]), 3) if e["ret"] else None}
                            for k, e in sorted(tsc.items(), key=lambda kv: (order.get(kv[0].split("|")[0], 9), kv[0]))]
    state["trendEvidence"] = trend_evidence(state)
    state["creditScores"] = [{"bucket": k.split("|")[0], "direction": k.split("|")[1], "trades": e["trades"],
                              "winRate": round(e["wins"] / (e["wins"] + e["losses"]), 3) if (e["wins"] + e["losses"]) else None,
                              "totalPnl": round(e["pnl"], 2)} for k, e in sorted(cb.items())]
    state["styleScores"] = sorted(({"style": k, "trades": s["trades"],
                                    "winRate": round(s["wins"] / (s["wins"] + s["losses"]), 3) if (s["wins"] + s["losses"]) else None,
                                    "totalPnl": round(s["pnl"], 2)} for k, s in sty.items()), key=lambda x: -x["totalPnl"])


def log_learning(state):
    lr = state.get("learning") or {}
    if not lr:
        return
    fmt = lambda xs: ", ".join(f"{k} x{v:.2f}" for k, v in xs) or "-"
    log(f"learning: {lr.get('closed')} closed trades replayed - {lr.get('taught')} taught, {lr.get('flat')} flat, "
        f"{lr.get('noSignals')} without signals, {lr.get('forced')} forced (x{lr.get('forcedWeight')}) · "
        f"most trusted {fmt(lr.get('strongest') or [])} · least {fmt(lr.get('weakest') or [])}"
        + (f" · moved this pass: {', '.join(lr['changed'][:6])}" if lr.get("changed") else ""))


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
    try:
        load_ratings()                                  # z194
    except Exception as e:
        log(f"credit ratings unavailable ({type(e).__name__}: {e})")
    try:
        load_warning()                                  # z196
    except Exception as e:
        log(f"early warning unavailable ({type(e).__name__}: {e})")
    spy_data = fetch_json(SPY_HISTORY_URL, timeout=30)
    spy_closes = [c for c in ((_f(b.get("close")) or _f(b.get("price"))) for b in spy_data) if c and c > 0] if isinstance(spy_data, list) else None
    xa = compute_cross_asset()
    opt_market = compute_options_market(list(universe.keys()))
    regime = compute_regime(spy_closes, opt_market)
    quad = resolve_quad(xa, regime.get("mode"))[0]          # z177: quad.json (macro_quad.json never existed - a 404 every pass)
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
        # z166: a one-pass jump by a split-like ratio (x10, x0.5 ...) with the market cap unchanged is a split the
        # book hasn't applied, not a move - confirm it with Yahoo's split record now; if it can't be confirmed,
        # freeze the position (no stop / target / trim on a price that can't be trusted) and say so every pass.
        prev = _f(pos.get("lastPrice"))
        if prev and px and (pos.get("instrument") or "shares") == "shares" and ca.split_like(px / prev):
            mc_now, mc_prev = _f(u.get("marketcapYahoo") or u.get("marketcap")), _f(pos.get("lastMarketCap"))   # z202: Yahoo's (moves with a split)
            real_move = bool(mc_now and mc_prev and not (0.6 <= mc_now / mc_prev <= 1.6))
            if not real_move:
                confirmed = []
                if CA_STATE.get("store") is not None:
                    CA_STATE["store"].get(pos.get("ticker"), force=True)
                    confirmed = ca.process({"trades": [pos]}, CA_STATE["store"], today, restate_fn=ts.arith_exit_pl, log=log)
                    CA_STATE["store"].save()
                prev = _f(pos.get("lastPrice"))
                if confirmed and prev and not ca.split_like(px / prev):
                    log(f"  {pos.get('ticker')}: split confirmed and applied - managed on the adjusted basis")
                else:
                    pos["priceAnomaly"] = {"since": (pos.get("priceAnomaly") or {}).get("since") or today, "date": today,
                                           "prev": prev, "now": px, "ratio": round(px / prev, 4) if prev else None,
                                           "marketCapNow": mc_now, "marketCapPrev": mc_prev}
                    msg = (f"{pos.get('ticker')} {pos.get('direction')} → FROZEN: ${prev:,.4f} → ${px:,.4f} (x{px / prev:.2f}) in one pass"
                           + (" with the market cap unchanged" if (mc_now and mc_prev) else " with no market cap to tell")
                           + " - a split-like jump with no split on record; no stop / target / trim until it's confirmed")
                    log("  ! " + msg)
                    review_lines.append(msg)
                    continue
        pos.pop("priceAnomaly", None)
        pos["lastMarketCap"] = _f(u.get("marketcapYahoo") or u.get("marketcap"))
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
        # z192: a daily-reset leveraged / inverse product is a session-by-session signal trade - never a core / long
        # hold. Opened before z192 (SQQQ, SPXS: "core", a 15-session minimum hold, an 8-month plan) it is re-planned
        # once on the short horizon; the old plan is kept in pos["etp"]["replannedFrom"].
        etp = rb.etp_exposure(u) if (pos.get("instrument") or "shares") == "shares" else None
        etp_note = ""
        if etp:
            if _replan_etp(pos, etp, today):
                rf = pos["etp"]["replannedFrom"]
                was = (rf.get("horizonType") or rf.get("horizonClass") or "unplanned") + \
                    (f", a {rf['minHoldDays']}-session minimum hold" if rf.get("minHoldDays") else "")
                log(f"  {pos.get('ticker')}: daily-reset {etp['lev']:+g}x product re-planned (z192) - short horizon, minimum hold "
                    f"{pos.get('minHoldDays')} sessions, stop ${pos.get('stopPrice')}, target ${pos.get('targetPrice')}, "
                    f"held while its signal is >= {ETP_HOLD_SCORE:.2f} (was {was})")
            htype = pos.get("horizonType") or "swing"
            vol_a = tb.realized_vol(load_history(pos.get("ticker"), u.get("repo")) or [])
            drag = rb.etp_drag(etp["lev"], vol_a / math.sqrt(252)) if vol_a else None
            pos["etp"]["dragPerSession"] = round(drag, 6) if drag is not None else None
            etp_note = f"daily-reset {etp['lev']:+g}x" + (f", drag ~{drag * 100:.2f} %/session" if drag is not None else "")
        # z138: the position's plan - sessions held, the minimum hold, the catalyst ahead
        days_held = tb.sessions_between(pos.get("entryDate"), today, mh.day_status)
        min_hold = int(pos.get("minHoldDays") or 0)
        in_hold = days_held < min_hold
        hc = tb.horizon_class(pos)
        cat = tb.catalyst_info(_CAL, pos.get("ticker"), today)
        if pos.get("instrument") == "option" and pos.get("optionLegs"):
            # z167: a vertical spread - valued leg by leg; exits by its own rules (credit: keep 50 %, stop at 2x the credit)
            chain = _load_option_chain(pos["ticker"])
            ed = tb._d(pos.get("optionExpiry"))
            dte = (ed - tb._d(today)).days if ed else None
            value, exit_v, src = tb.spread_value(pos, chain, px, today)
            if value is None:
                review_lines.append(tb.review(pos, today, price=px, ret=0.0, days_held=days_held, min_hold=min_hold, decision="hold",
                                              reason="no mark for a leg this pass", score_now=signed_now, cat=cat, dte=dte))
                continue
            pos["optionMark"], pos["optionMarkAt"], pos["optionMarkSource"] = value, today, src
            pos["pnl"] = tb.spread_pnl(pos, value)
            ret_s = pos["pnl"] / (pos.get("notional") or 1)
            pos["returnPct"] = round(ret_s * 100, 2)
            exit_reason = tb.spread_exit(pos, value, dte, hc)
            if exit_reason and in_hold and exit_reason != "expiry" and not exit_reason.startswith("credit-stop"):
                exit_reason = None                              # inside the minimum hold: only expiry or a credit stop
            if not exit_reason and not in_hold and aligned_now is not None and aligned_now <= -CORE_EXIT_SCORE:
                exit_reason = "thesis-break"
            if exit_reason and review_only:
                review_lines.append(tb.review(pos, today, price=px, ret=ret_s, days_held=days_held, min_hold=min_hold, decision="flag",
                                              reason=f"{exit_reason} - acts at the next session's first pass", score_now=signed_now, cat=cat,
                                              mark=value, mark_src=src, dte=dte))
                continue
            if exit_reason:
                settle = value if exit_reason == "expiry" else exit_v
                r = tb.spread_pnl(pos, settle, fee_sides=2)
                pos.update({"status": "closed", "exitDate": today, "exitPrice": px, "optionExit": round(settle, 4),
                            "exitReason": exit_reason, "realizedPL": r, "pnl": r, "won": r > 0,
                            "returnPct": round(r / (pos.get("notional") or 1) * 100, 2)})
                state["bankroll"] = round(state.get("bankroll", STARTING_BANKROLL) + r, 2)
                closed_now.append((pos["ticker"], exit_reason, r))
                review_lines.append(tb.review(pos, today, price=px, ret=ret_s, days_held=days_held, min_hold=min_hold, decision="close",
                                              reason=f"{exit_reason} - closed at {settle:+.2f}/sh ({src})", score_now=signed_now, cat=cat,
                                              mark=value, mark_src=src, dte=dte))
            else:
                review_lines.append(tb.review(pos, today, price=px, ret=ret_s, days_held=days_held, min_hold=min_hold, decision="hold",
                                              reason=("credit spread: keep 50 % of the credit, stop at 2x it, close a week out" if pos.get("optionCredit")
                                                      else "debit spread: take 60 % of the maximum gain, stop at half the debit"),
                                              score_now=signed_now, cat=cat, mark=value, mark_src=src, dte=dte))
            continue
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

        stop0 = pos.get("stopPrice")
        if htype == "core":
            if ret <= -CORE_HARD_STOP:
                exit_reason = "hard-stop"
            elif stop0 and ret < 0 and ((direction == 1 and px <= stop0) or (direction == -1 and px >= stop0)):
                # z186: a stop set at entry is a FLOOR. Re-scoring a losing swing trade as "core" used to drop it to
                # the 25 % hard stop (MPC, VLO: -8 % stops, closed at -17 %) - converting a loser into an investment.
                exit_reason = "stop-loss"
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
            elif etp and aligned_now is not None and aligned_now < ETP_HOLD_SCORE:
                # z192: a daily-reset product decays every session it's held - it stays only while its signal does
                exit_reason = "signal-faded"
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

        if not exit_reason and rb.pos_loss_capped(pos, state.get("bankroll", STARTING_BANKROLL)):
            # z186: no single name may cost the book more than RUNNER_RISK_POS_LOSS_CAP (0.75 %) - a backstop for
            # gaps and for positions sized before the risk book
            exit_reason = "risk-cap"
        if not exit_reason and UNWIND_UNSHORTABLE and direction == -1 and (pos.get("instrument") or "shares") == "shares":
            _why = rb.fund_reason(u)
            if _why:
                # z190: the rule (§1d 3) says a short sale needs a single stock - this one never qualified
                exit_reason = "not-shortable"
                log(f"  {pos.get('ticker')}: short sale in {_why} - unwound (RUNNER_UNWIND_UNSHORTABLE=0 keeps it)")
        if not exit_reason and not is_alt:
            exit_reason = credit_exit(pos, today, review_only)     # z194: distress / confirmed downgrade (short: upgrade)
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
            pos["creditAtExit"] = credit_card(credit_of(pos.get("ticker")))    # z194
            pos.pop("creditWatch", None)
            state["bankroll"] = round(state.get("bankroll", STARTING_BANKROLL) + r, 2)
            closed_now.append((pos["ticker"], exit_reason, total_r))
        # z138: the review - what it is, how it's doing, what's ahead, what the bot decided and why
        if exit_reason:
            decision, why = "close", f"{exit_reason} at ${px:,.2f}"
        elif trimmed_now and trimmed_now[-1][0] == pos.get("ticker") and pos.get("trims") and pos["trims"][-1].get("date") == today:
            decision, why = "trim", f"tier {pos.get('trimTier')} at {ret * 100:+.1f}% - riding the rest"
        elif in_hold:
            decision, why = "hold", f"minimum hold ({days_held}/{min_hold} sessions) - only a −{hold_disaster:.0%} move closes it"
        elif etp:
            decision, why = "hold", (f"{etp_note}: held while its signal is >= {ETP_HOLD_SCORE:.2f} in its direction"
                                     + (f" (now {aligned_now:+.2f})" if aligned_now is not None else " (not re-scored this pass)"))
        elif aligned_now is not None and aligned_now < MIN_SCORE:
            decision, why = "hold", f"signal faded to {aligned_now:+.2f} - exits at the stop / target or the {pos.get('horizonDaysPlanned') or pos.get('horizonDays') or '—'}-session horizon"
        else:
            decision, why = "hold", ("thesis intact" + (f" (aligned {aligned_now:+.2f} in its direction)" if aligned_now is not None else ""))
        crn = credit_of(pos.get("ticker"))
        if crn and crn.get("r") and crn.get("r") != "NR":
            ent = pos.get("creditAtEntry") or pos.get("creditBaseline") or {}
            why += f" · credit {crn['r']}" + (f" (was {ent.get('r')})" if ent.get("r") and ent.get("r") != crn["r"] else "") \
                + (f", {crn['o']}" if crn.get("o") and crn.get("o") != "stable" else "") \
                + (" - on watch" if pos.get("creditWatch") else "")
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


def _replan_etp(pos, etp, today):
    """z192: record the product's exposure on the position and put it on the short horizon (once). True if re-planned."""
    rec = pos.get("etp") if isinstance(pos.get("etp"), dict) else {}
    rec.update({"lev": etp["lev"], "underlying": etp.get("underlying"), "basis": etp.get("basis")})
    pos["etp"] = rec
    if tb.horizon_class(pos) == "short" and pos.get("horizonType") == "swing":
        return False
    hz = tb.HZ["short"]
    entry = _f(pos.get("entryPrice"))
    is_short = pos.get("direction") == "short"
    rec["replannedFrom"] = {k: pos.get(k) for k in ("horizonClass", "horizonType", "minHoldDays", "horizonDaysPlanned",
                                                    "horizonDays", "stopPrice", "targetPrice", "plannedExitBy")}
    rec["replannedAt"] = today
    pos["horizonClass"], pos["horizonType"] = "short", "swing"
    pos["minHoldDays"] = min(int(pos.get("minHoldDays") or hz["min_hold"]), hz["min_hold"])
    pos["horizonDaysPlanned"] = pos["horizonDays"] = hz["days"]
    if entry:
        pos["stopPrice"] = round(entry * ((1 + hz["stop"]) if is_short else (1 - hz["stop"])), 4)
        pos["targetPrice"] = round(entry * ((1 - hz["target"]) if is_short else (1 + hz["target"])), 4)
    ed = tb._d(pos.get("entryDate"))
    if ed:
        pos["plannedExitBy"] = (ed + timedelta(days=round(hz["days"] * 7 / 5))).strftime("%Y-%m-%d")
    pos["horizonReason"] = (f"daily-reset {etp['lev']:+g}x product (re-planned z192): short horizon, held only while its "
                            f"signal is >= {ETP_HOLD_SCORE:.2f} in its direction")
    return True


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
                    research=None, patterns=None, regime_ctx=None, risk=None):
    bankroll = state.get("bankroll", STARTING_BANKROLL)
    book = (risk or {}).get("book")                     # z186: the risk book (None = pre-z186 sizing, tests only)
    rscale = (risk or {}).get("scale", 1.0)
    spy_rets = (risk or {}).get("spyRets") or {}
    if book is not None and book.halted:
        log(f"  risk: daily halt - the book is {((risk or {}).get('dayChange') or 0):+.2%} today (halt at -{rb.DAY_HALT:.0%}); "
            f"no new entries this session (positions are still managed)")
        return []
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
    alt_seen = {"crypto": 0, "future": 0, "fx": 0, "bond": 0, "country": 0}
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
    hedge_room = opt_room                              # z186: a short against a net-long book may use half the reserve too
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
    all_signed = []                                    # z186: the score distribution, logged - are there shorts to find?
    bear = []                                          # z186: (signed, ticker, what happened) for every name at the short bar
    bear_pool = []                                     # z186: tradable bearish ideas (bar applied after the scan)
    renamed = []                                       # z187: the company behind the symbol changed - not traded for 30 days
    for tk, u in universe.items():
        if tk in held:
            continue
        # Index levels (^VIX, ^MOVE, ^GSPC ...) are reference series, not
        # instruments - they can't be bought, so they never become positions.
        # They still feed regime / cross-asset signals elsewhere.
        if tk.startswith("^"):
            continue
        if not u.get("scan", True):
            continue   # price-only book (TRAPP2-3 / TRAPP2-4): managed when held, never scanned
        if recently_renamed(u, today):
            renamed.append(f"{tk} (was '{u.get('nameWas')}')")
            continue
        sc = score_symbol(tk, u, ctx)
        scanned += 1
        if sc is None:
            continue
        comps, signed = sc["comps"], sc["signed"]
        all_signed.append(signed)
        conf = compute_confidence(comps, signed)
        trigs = compute_triggers(comps, signed, TRIGGER_LEVEL)
        # Crypto & futures: no filings, earnings or research grade - price alone
        # isn't a thesis. They must clear the research gate (drivers, supply
        # chain, news, regime, seasonality) at a stricter bar.
        cls = alt_research.asset_class(tk, u)
        if signed <= SHORT_SCORE and cls:
            bear.append((signed, tk, f"{cls} - judged by the research gate"))
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
                               "entryReason": entry_reason, "etp": sc.get("etp")})
        elif signed <= SHORT_FLOOR and len(trigs) < MIN_TRIGGERS:
            if signed <= SHORT_SCORE:
                bear.append((signed, tk, "no bearish trigger fired"))
        elif signed <= SHORT_FLOOR and len(trigs) >= MIN_TRIGGERS:
            # z138: a bearish idea trades as a short sale (RUNNER_ALLOW_SHORTS) or as a put - defined risk
            # z186: shorts on by default; a short sale needs a liquid, borrowable stock (else a put, if a chain exists).
            # Collected here; the session's short bar (absolute OR bottom decile) is applied after the scan.
            can_short, short_why = rb.shortable(u) if ALLOW_SHORTS else (False, "shorts off")
            if not can_short and not tb.chain_fresh(_load_option_chain(tk), today):
                if signed <= SHORT_SCORE:
                    bear.append((signed, tk, f"not shortable ({short_why}) and no option chain"))
                continue
            bear_pool.append({"ticker": tk, "score": abs(signed), "direction": "short", "signed": signed,
                              "components": comps, "confidence": conf, "triggers": trigs,
                              "price": u["price"], "sector": u["sector"], "name": u["name"],
                              "priceLocal": u.get("priceLocal"), "currency": u.get("currency"), "fxRate": u.get("fxRate"),
                              "putOnly": not can_short, "putWhy": short_why, "etp": sc.get("etp")})
    # z186: the session's short bar - the absolute bar, or the bottom SHORT_PCTL of today's scores (never above
    # SHORT_FLOOR): the weakest names a long/short book would short, not only the broken ones
    short_bar = SHORT_SCORE
    if all_signed:
        qq = sorted(all_signed)
        short_bar = min(SHORT_FLOOR, max(SHORT_SCORE, qq[min(len(qq) - 1, int(SHORT_PCTL * len(qq)))]))
    for bc in bear_pool:
        if bc["signed"] <= short_bar:
            candidates.append(bc)
            bear.append((bc["signed"], bc["ticker"], "short candidate" if not bc["putOnly"] else f"put candidate ({bc['putWhy']})"))
        else:
            bear.append((bc["signed"], bc["ticker"], f"above today's short bar {short_bar:+.2f}"))
    candidates.sort(key=lambda c: c["score"], reverse=True)
    n_long = sum(1 for c in candidates if c["direction"] == "long")
    n_short = sum(1 for c in candidates if c["direction"] == "short")
    log(f"  mode={RUNNER_MODE} regime={mode} quad={quad} · scanned {scanned} · {n_long} long / {n_short} short pass "
        f"(min {min_score:.2f} long, {short_bar:.2f} short today{'' if ALLOW_SHORTS else ' - puts only, shorts off'})")
    if all_signed:
        q = sorted(all_signed)
        pq = lambda f: q[min(len(q) - 1, int(f * len(q)))]
        log(f"  scores: min {q[0]:+.2f} · 5% {pq(0.05):+.2f} · median {pq(0.5):+.2f} · 95% {pq(0.95):+.2f} · max {q[-1]:+.2f} · "
            f"{sum(1 for x in q if x <= short_bar)} at or below today's short bar, {sum(1 for x in q if x >= min_score)} at or above the long bar")
    if renamed:
        log(f"  {len(renamed)} not traded - the company behind the symbol changed in the last 30 days (renamed or reassigned): "
            + ", ".join(renamed[:10]))
    if bear:
        why_n = {}
        for _, _, w in bear:
            k = w.split(" (")[0].split(" - ")[0]
            why_n[k] = why_n.get(k, 0) + 1
        log("  short side: " + ", ".join(f"{k} {v}" for k, v in sorted(why_n.items(), key=lambda x: -x[1]))
            + " · weakest: " + ", ".join(f"{tk} {sg:+.2f} ({w})" for sg, tk, w in sorted(bear)[:5]))
    if patterns:
        log(f"  pattern factor: {pat_seen['aligned']} names with a bullish pattern read · "
            f"{pat_seen['nudged']} nudged over the bar · {pat_seen['overruled']} strong patterns overruled by the other factors")
    n_alt_pass = sum(1 for c in candidates if c.get("alt"))
    if sum(alt_seen.values()):
        log(f"  alt research: {alt_seen['crypto']} crypto + {alt_seen['future']} futures + {alt_seen['fx']} currencies + "
            f"{alt_seen['bond']} bond ETFs + {alt_seen['country']} country ETFs scanned · "
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
    # z186: a mix of directions (owner direction §1d) - when there are short ideas, keep up to SHORT_SLOTS of the
    # session's slots for the best of them; ranked by score alone, longs (>= +0.25) would always crowd them out
    shorts_ranked = [c for c in candidates if c["direction"] == "short"]
    if shorts_ranked and slots >= 3 and SHORT_SLOTS > 0:
        want = shorts_ranked[:min(SHORT_SLOTS, len(shorts_ranked))]
        if not all(c in candidates[:slots] for c in want):
            for c in want:
                candidates.remove(c)
            pos = max(0, slots - len(want))
            for k, c in enumerate(want):
                candidates.insert(pos + k, c)
            log(f"  reserved {len(want)} slot(s) for short(s): " + ", ".join(f"{c['ticker']} {-c['score']:+.2f}" for c in want))
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
        # z158: a hard guard, not a score - no NEW long in a company that filed a bankruptcy, non-reliance or
        # delisting 8-K in the last 30 days (bearish ideas are still allowed)
        red = [x for x in ((_SEC.get(str(c["ticker"]).upper()) or {}).get("red30") or []) if x in NO_LONG_ITEMS]
        if red and c.get("direction") != "short":
            log(f"  skip {c['ticker']}: 8-K {', '.join(f'{x} ({NO_LONG_ITEMS[x]})' for x in red)} in the last 30 days - no new longs")
            continue
        blocked = (credit_gate(c["ticker"], c.get("direction"), c.get("etp"))     # z194
                   or seasoning_gate(c["ticker"], (universe.get(c["ticker"]) or {}).get("repo"))   # z195
                   or warning_gate(c.get("direction"), c.get("etp"))                               # z196
                   or ipo_gate(c["ticker"], c.get("direction"), today))                            # z197
        if blocked:
            log(f"  skip {c['ticker']}: {blocked}")
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
        elif c.get("etp"):
            # z192: its exposure resets every session and decays with volatility - a session-by-session signal trade
            htype, hclass, steered = "swing", "short", False
            h_why = (f"daily-reset {c['etp']['lev']:+g}x product: short horizon, held only while its signal is "
                     f">= {ETP_HOLD_SCORE:.2f} in its direction")
        cell = f"{c.get('direction')}|{hclass}"
        option = None
        use_hedge = False
        chain = None if alt else _load_option_chain(c["ticker"])
        cat_in_window = bool(cat and cat.get("days") is not None and 1 <= cat["days"] <= tb.HZ[hclass]["days"] + 5
                             and abs(c["components"].get("catalyst") or 0) >= 0.10)
        if not alt and (conf >= tb.OPT_CONF or cat_in_window or c.get("putOnly")) and tb.chain_fresh(chain, today):
            prem_open = sum((p.get("notional") or 0) for p in trades_list(state) if p.get("status") == "open" and p.get("instrument") == "option")
            # z167: the IV regime decides what to trade - sell premium when IV is rich, verticals in the middle,
            # buy premium when it's cheap (defined risk everywhere: credit spreads, never naked shorts)
            u_c = universe.get(c["ticker"].upper()) or {}
            reg = tb.iv_regime(_opt_summary_row(c["ticker"]), load_history(c["ticker"], u_c.get("repo")) if u_c.get("repo") else None)
            structure = tb.choose_structure(c.get("direction"), reg)
            if structure in ("long_call", "long_put"):
                option, o_why = tb.pick_contract(chain, c.get("direction"), hclass, today, cat if cat_in_window else None)
            else:
                option, o_why = tb.pick_spread(chain, structure, hclass, today, cat if cat_in_window else None)
                if not option:
                    log(f"  {c['ticker']}: {structure.replace('_', ' ')} not available ({o_why}) - a long {('call' if c.get('direction') == 'long' else 'put')} instead")
                    structure = "long_call" if c.get("direction") == "long" else "long_put"
                    option, o_why = tb.pick_contract(chain, c.get("direction"), hclass, today, cat if cat_in_window else None)
            if option:
                option["structure"], option["ivRegime"] = structure, reg
                log(f"  {c['ticker']}: IV {reg.get('regime')} ({reg.get('basis')}) → {structure.replace('_', ' ')}")
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
        # z186: the risk book's view of this name - its volatility, beta and recent returns (clusters)
        rets = sig = beta_v = None
        if book is not None and not alt:
            uu = universe.get(str(c["ticker"]).upper()) or {}
            dated = load_history_dated(c["ticker"], uu.get("repo")) if uu.get("repo") else None
            rets = rb.returns_by_date(dated) if dated else None
            sig = rb.sigma(rets) if rets else None
            beta_v = rb.beta(rets, spy_rets) if (rets and spy_rets) else None
        if option:
            # z186: premium 0.4 % → 1.0 % of the bankroll with confidence (deep conviction 1.5 %), times the drawdown
            # throttle; the premium (or a spread's max loss) is the most it can lose. Was 1 % → 2.5 % (4 % deep).
            budget = tb.option_budget(bankroll, conf, explore) * rscale
            cap_now = min(budget, max(deployable, opt_room) if constrained else deployable)
            risk_per = (option["maxLoss"] if option.get("legs") else option["ask"]) * 100      # z167: the most one contract can lose
            contracts = int(cap_now // risk_per)
            if contracts < 1:
                log(f"  {c['ticker']}: one contract (${risk_per:,.0f} at risk) is more than the ${cap_now:,.0f} available "
                    f"(budget ${budget:,.0f}{f', option room left ${opt_room:,.0f}' if constrained else ''})")
                if c.get("putOnly"):
                    continue
                option = None
        if option and book is not None:
            o_risk = contracts * (option["maxLoss"] if option.get("legs") else option["ask"]) * 100
            ok_r, why_r = book.check(c["ticker"], c["sector"], c.get("direction"), o_risk, -o_risk if is_short else o_risk,
                                     beta_v, rets, o_risk, instrument="option")
            if not ok_r:
                log(f"  risk: skip {c['ticker']} option - {why_r}")
                continue
        if option:
            shares = 0
            notional = round(contracts * (option["maxLoss"] if option.get("legs") else option["ask"]) * 100, 2)   # capital at risk
            fees = round(tb.OPT_FEE * contracts * (len(option.get("legs") or []) or 1), 2)
            opt_room -= notional
        else:
            # z186: the stop this trade will carry (set below by its horizon) decides its size - the loss at the stop
            # is the budget, not the notional
            if alt:
                stop_pct = alt["stopPct"]
            elif hclass in ("short", "medium"):
                stop_pct = tb.HZ[hclass]["stop"]
            elif hclass == "long":
                stop_pct = CORE_HARD_STOP
            else:
                stop_pct = SWING_STOP
            ceiling = bankroll * POSITION_PCT * 1.6                          # the pre-z186 maximum stays the maximum
            if book is not None:
                size, risk_amt, eff_stop, size_why = rb.size_shares(bankroll, conf, stop_pct, sig, rscale, ceiling)
                if alt:
                    size *= alt["sizeMult"]
                if explore and not alt:
                    size *= 0.7
                cmult = credit_size_mult(credit_of(c["ticker"]), is_short, c.get("etp"))
                if cmult != 1.0:
                    size *= cmult                  # z194: weaker credit, smaller long
                    size_why = f"{size_why} · credit {credit_of(c['ticker']).get('r')} x{cmult:.2f}"
                _u = universe.get(c["ticker"]) or {}
                wmult, wwhy = warning_size_mult(_u.get("sector") or c.get("sector"), _u.get("industry"), is_short, c.get("etp"))
                if wmult != 1.0:
                    size *= wmult                  # z196: early warning - risk percentile and industry exposure
                    size_why = f"{size_why} · warning x{wmult:.2f} ({wwhy})"
                imult, iwhy = ipo_size_mult(c["ticker"], is_short, today)
                if imult != 1.0:
                    size *= imult                  # z197: conflicted-coverage window after an IPO's quiet period
                    size_why = f"{size_why} · IPO x{imult:.2f} ({iwhy})"
                tmult, twhy = trend_size_mult(is_short, c.get("etp"), alt)
                if tmult != 1.0:
                    size *= tmult                  # z201: today's trends - a sell-off / correction trims new equity longs
                    size_why = f"{size_why} · trends x{tmult:.2f} ({twhy})"
                risk_amt = size * eff_stop
                ok_r, why_r = book.check(c["ticker"], c["sector"], c.get("direction"), size, -size if is_short else size,
                                         beta_v, rets, risk_amt)
                if not ok_r:
                    log(f"  risk: skip {c['ticker']} {c.get('direction')} - {why_r}")
                    continue
            # z186: a short sale against a net-long book lowers its risk - like an option, it may use half the reserve
            use_hedge = is_short and book is not None and book.net_beta > 0 and deployable < bankroll * 0.02
            if constrained and deployable < bankroll * 0.02 and not use_hedge:
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
            if book is None:
                # pre-z186: confidence-scaled sizing 0.6x (low) → 1.4x (high), hard-capped at 1.6x base
                size = min(bankroll * POSITION_PCT * (0.6 + 0.8 * conf), deployable, ceiling)
                if alt:
                    size *= alt["sizeMult"]    # smaller + volatility-scaled for crypto / futures
                if explore and not alt:
                    size *= 0.7                # z138: a first trade in an empty cell is a smaller, exploratory one
                size_why = "notional"
                cmult = credit_size_mult(credit_of(c["ticker"]), is_short, c.get("etp"))
                if cmult != 1.0:
                    size *= cmult                  # z194: weaker credit, smaller long
                    size_why = f"notional · credit {credit_of(c['ticker']).get('r')} x{cmult:.2f}"
                _u = universe.get(c["ticker"]) or {}
                wmult, wwhy = warning_size_mult(_u.get("sector") or c.get("sector"), _u.get("industry"), is_short, c.get("etp"))
                if wmult != 1.0:
                    size *= wmult                  # z196: early warning
                    size_why = f"{size_why} · warning x{wmult:.2f} ({wwhy})"
                imult, iwhy = ipo_size_mult(c["ticker"], is_short, today)
                if imult != 1.0:
                    size *= imult                  # z197: conflicted-coverage window after an IPO's quiet period
                    size_why = f"{size_why} · IPO x{imult:.2f} ({iwhy})"
                tmult, twhy = trend_size_mult(is_short, c.get("etp"), alt)
                if tmult != 1.0:
                    size *= tmult                  # z201: today's trends
                    size_why = f"{size_why} · trends x{tmult:.2f} ({twhy})"
            room = hedge_room if use_hedge else deployable
            size = min(size, room)
            if size < bankroll * MIN_POSITION_PCT:
                continue                       # z186: too small (risk budget or cash) - the next idea may still fit (a short via the hedge room)
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
        if book is not None:
            g_new, s_new = (notional, -notional if is_short else notional)
            r_new = notional if option else notional * (eff_stop if not option else 1.0)
            book.add(c["ticker"], c["sector"], c.get("direction"), g_new, s_new, beta_v, rets, new_risk=r_new, new=True,
                     instrument="option" if option else "shares")
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
            "signalBasis": SIGNAL_BASIS,                        # z192: fund-clean, exposure-signed - taught as recorded
            "etp": c.get("etp"),                                # z192: a leveraged / inverse product's exposure
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
            # z186: what the risk book decided - the loss budget, the stop it sized against, the throttle
            "risk": ({"riskPct": round((notional if option else notional * eff_stop) / bankroll * 100, 3),
                      "sizedBy": "premium budget" if option else size_why,
                      "effStopPct": None if option else round(eff_stop * 100, 2),
                      "dailyVolPct": round(sig * 100, 2) if sig else None, "beta": round(beta_v, 2) if beta_v is not None else None,
                      "throttle": rscale} if book is not None else None),
            "headcount": headcount_of(c["ticker"]),             # z151: context, not scored
            "filings": filings_of(c["ticker"]),                 # z158: 8-K flags, insiders, short selling - context
            "valuation": valuation_card(universe.get(c["ticker"]) or {}),   # z187: the multiples it was judged on
            "creditAtEntry": credit_card(credit_of(c["ticker"])),   # z194: the rating it was opened on (exits measure from it)
            "warningAtEntry": warning_card((universe.get(c["ticker"]) or {}).get("sector") or c.get("sector"),
                                           (universe.get(c["ticker"]) or {}).get("industry")),   # z196: the early-warning read
            "ipoAtEntry": ipo_card(c["ticker"], today),         # z197: lock-up / quiet-period context (None if not an IPO)
            "trendsAtEntry": trend_card(),                      # z201: the market read (sell-off, S&P run, breadth) that day
        }
        if option and option.get("legs"):
            # z167: a vertical spread - two legs, one expiry; optionNet is the signed entry value per share
            # (+ a debit paid, - a credit received); notional / maxLoss = the most it can lose
            short_leg = next(l for l in option["legs"] if l["side"] == "short")
            long_leg = next(l for l in option["legs"] if l["side"] == "long")
            trade.update({"instrument": "option", "optionStructure": option["structure"], "optionLegs": option["legs"],
                          "optionType": option["type"], "optionStrike": short_leg["strike"] if option["credit"] else long_leg["strike"],
                          "optionExpiry": option["expiry"], "optionContracts": contracts, "optionNet": option["net"],
                          "optionPremium": round(abs(option["net"]), 4), "optionCredit": bool(option["credit"]),
                          "optionMark": round(sum((l.get("mid") or l["price"]) * (1 if l["side"] == "long" else -1) for l in option["legs"]), 4),
                          "optionMarkAt": today, "optionDelta": option.get("delta"), "optionIV": option.get("iv"), "optionWidth": option["width"],
                          "maxLossPerShare": option["maxLoss"], "maxGainPerShare": option["maxGain"], "breakeven": option["breakeven"],
                          "maxGain": round(option["maxGain"] * 100 * contracts, 2), "optionChainAsOf": option.get("chainAsOf"),
                          "ivRegime": option.get("ivRegime"), "multiplier": 100, "maxLoss": notional, "leverage": 1})
            k_long, k_short = long_leg["strike"], short_leg["strike"]
            trade["rationale"].append(
                f"{option['structure'].replace('_', ' ').upper()} {option['type'].upper()}S {k_long:g}/{k_short:g} exp {option['expiry']} ({option['dte']}d) ×{contracts} · "
                + (f"credit ${-option['net']:.2f}/sh (${-option['net'] * 100 * contracts:,.0f} received)" if option["credit"] else f"debit ${option['net']:.2f}/sh")
                + f" · max loss ${notional:,.0f} · max gain ${option['maxGain'] * 100 * contracts:,.0f} · breakeven ${option['breakeven']:.2f}"
                + f" · IV {option['ivRegime'].get('regime')}: {option['ivRegime'].get('basis')}")
        elif option:
            trade.update({"instrument": "option", "optionType": option["type"], "optionStrike": option["strike"],
                          "optionExpiry": option["expiry"], "optionContracts": contracts, "optionPremium": option["ask"],
                          "optionMark": option["mid"], "optionMarkAt": today, "optionDelta": option.get("delta"),
                          "optionIV": option.get("iv"), "optionChainAsOf": option.get("chainAsOf"), "multiplier": 100,
                          "optionStructure": option.get("structure") or ("long_call" if option["type"] == "call" else "long_put"),
                          "ivRegime": option.get("ivRegime"), "maxLoss": notional, "leverage": 1})
            trade["rationale"].append(
                f"OPTION {option['type'].upper()} {option['strike']} exp {option['expiry']} ({option['dte']}d) ×{contracts} @ ${option['ask']:.2f} ask"
                f" · Δ {option.get('delta')} · premium ${notional:,.0f} = the most it can lose" +
                (" · catalyst inside the horizon" if cat_in_window else f" · conviction {conf:.2f}") +
                (f" · IV {option['ivRegime'].get('regime')}: {option['ivRegime'].get('basis')}" if option.get("ivRegime") else ""))
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
        if not option and is_short and book is not None and use_hedge:
            hedge_room -= notional
        committed += notional
        cash -= notional
        deployable -= notional
        primary_count[primary] = primary_count.get(primary, 0) + 1
        cell_count[cell] = cell_count.get(cell, 0) + 1
        state.setdefault("trades", []).append(trade)
        opened.append((c["ticker"], notional, c["score"]))
        if not state.get("startedAt"):
            state["startedAt"] = today
    if book is not None and book.rejects:
        log("  risk: skipped " + ", ".join(f"{k} {v}" for k, v in sorted(book.rejects.items(), key=lambda x: -x[1]))
            + f" · new risk today {book.new_risk / max(1.0, bankroll):.2%} of the book (cap {rb.NEW_RISK_PER_DAY:.1%})")
    for tk, notional, sc in opened:
        t = next((x for x in reversed(trades_list(state)) if x.get("ticker") == tk and x.get("status") == "open"), {})
        if t.get("optionLegs"):
            what = (f"{t['optionStructure'].replace('_', ' ').upper()} {'/'.join(format(l['strike'], 'g') for l in t['optionLegs'])} "
                    f"{t.get('optionExpiry')} ×{t.get('optionContracts')} ({'credit' if t.get('optionCredit') else 'debit'} {abs(t.get('optionNet') or 0):.2f}/sh)")
        else:
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

    # z186: takeaways from the journal itself (the old ones were an app export's, frozen at 8 trades)
    tk = []
    if win_rate is not None:
        tk.append(f"Win rate {win_rate * 100:.1f}% over {len(wins) + len(losses)} settled trades; net P&L ${total_pnl:,.2f}"
                  + (f"; profit factor {pf}" if pf is not None else "") + ".")
    by_exit = {}
    for b in closed:
        k = str(b.get("exitReason") or "?").split(" ")[0]
        by_exit[k] = by_exit.get(k, 0.0) + _pl(b)
    if by_exit:
        worst = min(by_exit.items(), key=lambda x: x[1])
        best = max(by_exit.items(), key=lambda x: x[1])
        tk.append(f"Costliest exit type: {worst[0]} (${worst[1]:,.0f}); best: {best[0]} (${best[1]:,.0f}).")
    by_style = {}
    for b in closed:
        k = str(b.get("style") or "other")
        by_style.setdefault(k, []).append(_pl(b))
    if by_style:
        ws = min(by_style.items(), key=lambda x: sum(x[1]))
        tk.append(f"Weakest style: {ws[0]} (${sum(ws[1]):,.0f} over {len(ws[1])} trades).")
    state["takeaways"] = tk
    try:                                       # z192: signal / style scores from the journal (were frozen at 8 trades)
        learning_rollups(state, universe)
    except Exception as e:
        log(f"learning rollups unavailable ({type(e).__name__}: {e})")
    state["schema"] = "valuatio-bot-training/v1"
    state["generatedAt"] = datetime.now(timezone.utc).isoformat()
    state["startingBankroll"] = STARTING_BANKROLL
    return state


# ----------------------------- risk book (z186) ---------------------------------
def spy_returns(spy_data):
    """SPY's daily returns by date from its history file (the market for every beta)."""
    if not isinstance(spy_data, list):
        return {}
    dated = []
    for b in spy_data:
        c = _f(b.get("close")) or _f(b.get("price"))
        if c and c > 0 and b.get("date"):
            dated.append((str(b["date"])[:10], c))
    return rb.returns_by_date(sorted(dated))


def build_risk(state, universe, today, spy_rets, dialed):
    """z186: the open book as the risk checks see it - drawdown from the high-water mark (-> the throttle on new
    risk), today's change (-> the daily halt), every open position's exposure by sector, beta and recent returns."""
    bankroll = float(state.get("bankroll", STARTING_BANKROLL))
    open_pos = [p for p in trades_list(state) if p.get("status") == "open"]
    value_now = bankroll + sum((p.get("pnl") or 0) for p in open_pos)
    dd = rb.drawdown(state.get("equityCurve"), value_now)
    chg = rb.day_change(state.get("equityCurve"), value_now, today)
    halted = chg is not None and chg <= -rb.DAY_HALT
    book = rb.Book(bankroll, dialed, dd["scale"], halted, today)
    book.long_beta_scale, book.long_beta_why = trend_beta_scale()   # z202: a sell-off tightens the long side's net beta cap
    for p in open_pos:
        tk = str(p.get("ticker") or "").upper()
        u = universe.get(tk) or {}
        g, sgn = rb.position_exposure(p, p.get("lastPrice") or u.get("price"))
        dated = load_history_dated(tk, u.get("repo")) if u.get("repo") else None
        rets = rb.returns_by_date(dated) if dated else None
        b = rb.beta(rets, spy_rets) if (rets and spy_rets) else None
        book.add(tk, p.get("sector") or u.get("sector"), p.get("direction"), g, sgn, b, rets,
                 instrument="option" if p.get("instrument") == "option" else "shares")
    return {"book": book, "scale": dd["scale"], "dd": dd, "dayChange": chg, "spyRets": spy_rets}


def risk_report(risk, when):
    """state['risk'] - what the app shows: the throttle, exposures, the caps and what they turned down."""
    if not risk:
        return None
    out = {"at": when, "drawdown": risk["dd"], "dayChange": None if risk["dayChange"] is None else round(risk["dayChange"], 4),
           "halted": risk["book"].halted, "shortsOn": ALLOW_SHORTS, "dialedIn": risk["book"].dialed,
           "limits": dict(rb.knobs(), optionBookCap=tb.OPT_BOOK_PCT, optionTradePct=list(tb.OPT_TRADE_PCT), optionDeepPct=tb.DEEP_PCT)}
    out.update(risk["book"].summary())
    return out


def log_risk(risk):
    b, dd = risk["book"], risk["dd"]
    sm = b.summary()
    top = ", ".join(f"{k} {v:.0%}" for k, v in list(sm["bySector"].items())[:4])
    log(f"risk: book ${dd['value']:,.0f} · high-water ${dd['hwm']:,.0f} ({dd['hwmDate']}) · drawdown {dd['dd']:.1%} → new risk x{dd['scale']:.2f}"
        f" · today {('%+.2f%%' % (risk['dayChange'] * 100)) if risk['dayChange'] is not None else 'n/a'}{' · HALT' if b.halted else ''}")
    log(f"risk: shares {sm['gross']:.0%} of the book (cap {rb.GROSS_CAP:.0%}) · options at risk {sm['optionRisk']:.1%} (cap {tb.OPT_BOOK_PCT:.0%}) · "
        f"net {sm['net']:+.0%} · net beta {sm['netBeta']:+.0%} (cap ±{rb.NET_BETA_CAP[b.dialed]:.0%}"
        + (f", long side +{rb.NET_BETA_CAP[b.dialed] * b.long_beta_scale:.0%} - {b.long_beta_why}" if getattr(b, "long_beta_scale", 1.0) < 1.0 else "")
        + f") · top sectors {top or 'none'}"
        + (f" · beyond the ±{rb.SECTOR_CAP:.0%} sector cap: {', '.join(sm['overSectorCap'])} (only trades that shrink them)" if sm["overSectorCap"] else "")
        + f" · shorts {'on' if ALLOW_SHORTS else 'OFF'}")


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
    if t.endswith("-USD") or t.endswith("=F") or t.startswith("FX:"):
        return True                     # z164: FX:<CCY> - currencies trade around the clock on weekdays
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
    ca_done = []
    try:                                       # z166: splits (adjust / restate) and dividends, every pass
        CA_STATE["store"] = ca.Store(STATE_FILE.parent / "corporate_actions.json")
        CA_STATE["fx"] = load_fx()
        for t in trades_list(state):           # a held name that jumped by a split-like ratio: ask Yahoo now
            uu, lp = universe.get(str(t.get("ticker") or "").upper()), _f(t.get("lastPrice"))
            if t.get("status") == "open" and uu and lp and ca.split_like(uu["price"] / lp):
                CA_STATE["store"].get(t.get("ticker"), force=True)
        ca_done = ca.process(state, CA_STATE["store"], today, restate_fn=ts.arith_exit_pl,
                             fx_of=lambda c: usd_factor(c, CA_STATE["fx"]), log=log)
        CA_STATE["store"].save()
    except Exception as e:
        log(f"corporate actions unavailable this pass ({type(e).__name__}: {e})")
    before = state.get("bankroll")
    reconcile_ledger(state, before, today)       # z117: pnl repair + dated opening adjustment
    state["bankroll"] = ts.ledger_bankroll(state, STARTING_BANKROLL)
    if before is not None and abs((before or 0) - state["bankroll"]) >= 0.01:
        log(f"bankroll re-derived from the journal: {before:,.2f} → {state['bankroll']:,.2f}")
    sig_before = {t.get("id"): _sig(t) for t in trades_list(state)}
    if app_done:
        log(f"applied {len(app_done)} app action(s)")
    try:                                       # z192: learn from every closed trade (replayed - idempotent)
        retrain(state, universe)
        log_learning(state)
    except Exception as e:
        log(f"learning unavailable this pass ({type(e).__name__}: {e})")

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
            try:                                       # z186: end-of-day exposures for the app
                rc_e = load_regime_certainty()
                risk_e = build_risk(state, universe, sess["date"], spy_returns(fetch_json(SPY_HISTORY_URL, timeout=30)),
                                    bool(rc_e and rc_e.get("dialedIn")))
                log_risk(risk_e)
                state["risk"] = risk_report(risk_e, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
            except Exception as e:
                log(f"risk report unavailable ({type(e).__name__}: {e})")
            finalize_state(state, sig_before)
            recompute(state, universe, today)
            _write_state(state, "after-close review of open positions (no trades)")
            return
        if not app_done and not ca_done:
            log("market closed - no trades; journal unchanged (nothing written)")
            return
        finalize_state(state, sig_before)
        recompute(state, universe, today)
        _write_state(state, f"closed-market pass: {len(app_done)} bookkeeping action(s), {len(ca_done)} corporate action(s), no trades")
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
    # z177: TRAPP2-1/data/macro/quad.json - the market nowcast while fresh, else the official GDP/CPI quad, with the
    # official quad and the agreement in the note. Falls back to the market-implied proxy only if the file is absent.
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
    labor = None
    try:
        labor = labor_context()
    except Exception as e:
        log(f"labor context failed: {e}")
    vol_ctx = None
    try:
        load_ratings()                                  # z194: credit ratings - signal, entry gate, size, exits
    except Exception as e:
        log(f"credit ratings unavailable ({type(e).__name__}: {e}) - trades run without them")
    try:
        load_warning()                                  # z196: early warning - signal, size, levered-long guard, context
    except Exception as e:
        log(f"early warning unavailable ({type(e).__name__}: {e}) - trades run without it")
    try:
        load_ipo()                                      # z197: IPO lock-up / quiet-period calendar - gate, size, context
    except Exception as e:
        log(f"IPO calendar unavailable ({type(e).__name__}: {e}) - trades run without it")
    try:
        load_trends()                                   # z201: today's trends - the market read: size, context
    except Exception as e:
        log(f"today's trends unavailable ({type(e).__name__}: {e}) - trades run without them")
    try:                                                # z202: the record decides the trim (and is logged every pass)
        _TREND["evidence"] = trend_evidence(state)
        ev = _TREND["evidence"]
        m_open = read_at_open(today)
        log(f"today's trends - the record: {ev['verdict']} ({ev['why']})"
            + (f" · the read at today's open: {m_open['state']} ({m_open['on']} of {m_open['of']}, through {m_open['asOf']})" if m_open else ""))
    except Exception as e:
        log(f"trend record unavailable ({type(e).__name__}: {e})")
    try:
        load_filings_and_short()
        vol_ctx = volatility_context()
        if vol_ctx:
            log(f"volatility (context, not scored): VIX {vol_ctx['vix']} / VIX3M {vol_ctx['vix3m']} = {vol_ctx['vixTo3m']} {vol_ctx['structure']} · SKEW {vol_ctx['skew']}")
    except Exception as e:
        log(f"filings / short / volatility context failed: {e}")
    # z157: the data-health monitor (TRAPP2-ANALYTICS, every 30 min) - which inputs are behind schedule
    health = None
    try:
        hdoc = fetch_json(f"{RAW}/TRAPP2-ANALYTICS/main/data/health.json", timeout=20)
        if isinstance(hdoc, dict):
            health = {"asOf": hdoc.get("asOf"), "botInputsDown": hdoc.get("botInputsDown") or [], "summary": hdoc.get("summary")}
            if health["botInputsDown"]:
                log(f"DATA HEALTH: bot inputs behind schedule - {', '.join(health['botInputsDown'])} (recorded on trades)")
    except Exception as e:
        log(f"data health unavailable: {e}")
    if labor:
        log(f"labor (context, not scored): {labor.get('pctCutting')} % of {labor.get('companies')} companies cutting headcount YoY, "
            f"{labor.get('pctGrowing')} % growing, aggregate {labor.get('aggregateYoY')} % - {labor.get('read')} · headcount on file for {len(_HC)} companies")
    regime_ctx = {"regime": (rc or {}).get("regime"), "quad": (rc or {}).get("quadNowcast"), "quadOfficial": (rc or {}).get("quadOfficial"),
                  "certainty": (rc or {}).get("certainty"), "dialedIn": not agnostic, "botMode": regime.get("mode"),
                  "bondTrend": bond, "fed": fed_ctx, "curve": curve, "credit": credit, "labor": labor, "dataHealth": health,
                  "volatility": vol_ctx}
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
    if closed:
        try:                                   # z192: this pass's closes teach before the entry scan
            retrain(state, universe)
            log_learning(state)
        except Exception as e:
            log(f"learning unavailable after the reviews ({type(e).__name__}: {e})")
    risk = None
    try:                                               # z186: the risk book - sizes, caps, throttle, halt
        risk = build_risk(state, universe, today, spy_returns(spy_data), not agnostic)
        log_risk(risk)
    except Exception as e:
        log(f"risk book unavailable ({type(e).__name__}: {e}) - entries use the pre-z186 sizing")
    ok_entry, why = entry_window(sess, state)
    eligible = {t: u for t, u in fresh.items() if session_tradable(t)} if ok_entry else {}
    if ok_entry and eligible:
        log(f"entries: {len(eligible)} tickers with a fresh quote whose home market is open now")
        opened = open_new_trades(state, eligible, today, regime, grades, peer_ranks, xa, quad, opt_market, fed,
                                 research=ctx["research"], patterns=ctx["patterns"], regime_ctx=regime_ctx, risk=risk)
        state["lastEntryScan"] = {"date": sess["date"], "et": sess["et"], "opened": len(opened)}
    elif ok_entry:
        opened = []
        log("entries: no fresh, session-tradable quotes yet - the next pass scans (today's scan not used up)")
    else:
        opened = []
        log(f"entries: {why}")
    if risk:
        state["risk"] = risk_report(risk, now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"))
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
