#!/usr/bin/env python3
"""
alt_research.py - research layer for NON-EQUITY instruments (crypto, futures).

A stock has filings, earnings, a research grade and peers. Crypto and futures have
none of that, so price momentum alone is not a thesis. Before the bot may enter one,
this module builds an asset-specific research case from what actually drives it:

  drivers       macro / cross-asset forces for that asset (USD, real yields,
                liquidity, rates, risk appetite, the Nasdaq for crypto, ...)
  supplyChain   the physical / economic chain around it: freight (dry-bulk and
                tanker rates), producers, input costs, processors, downstream
                demand, spot-ETF / miner flows for crypto
  news          headline sentiment for the asset: an asset-specific event
                dictionary (supply shocks, policy, hacks, ETF flows ...) plus the
                XTRAPP human-trained lexicon, recency-weighted
  regime        the macro quad (growth / inflation) mapped per asset group
  seasonality   20 years of the contract's own history: the average 20-day
                forward return from this point in the calendar (crop cycles,
                driving / heating seasons). Futures only.

The entry gate is STRICTER than for equities:
  * technical score >= MIN_SCORE + 0.10, in the trade direction
  * research composite >= 0.25 in the trade direction
    (>= 0.35 when there is no news coverage - less evidence, higher bar)
  * coverage: drivers + at least two other research components present
  * no research component strongly against the trade (<= -0.50)
  * news not against the trade (<= -0.30) when there are >= 2 matched headlines
Positions are smaller (crypto 0.5x, futures 0.6x the equity size, further scaled
down by volatility), stops / targets are volatility-based, and concurrency is
capped (2 crypto, 3 futures, 1 per futures complex). The full research case is
stored on the trade, so every entry can be audited.

Every input is optional: a missing file or series just drops that component
(and the coverage rule then decides whether there is still enough evidence).
"""
import math
import re
from datetime import datetime, timezone

# ----------------------------------------------------------------- config ----
MIN_SCORE_BUMP = 0.10          # technical bar above the equity MIN_SCORE
RESEARCH_MIN = 0.25            # research composite needed (with news)
RESEARCH_MIN_NO_NEWS = 0.35    # ... without any news coverage
VETO = -0.50                   # any single component this far against = no trade
NEWS_VETO = -0.30              # news this far against (>= 2 headlines) = no trade
EXIT_FLIP = -0.25              # held position: research this far against = exit
WEIGHTS = {"drivers": 0.30, "supplyChain": 0.25, "news": 0.20, "regime": 0.15, "seasonality": 0.10}
MAX_OPEN = {"crypto": 2, "future": 3}
MAX_PER_GROUP = 1              # futures complexes (energy, grains, ...)
SIZE_MULT = {"crypto": 0.5, "future": 0.6}
TARGET_DAILY_VOL = {"crypto": 0.025, "future": 0.015}
STOP_RULE = {"crypto": (2.5, 0.10, 0.22), "future": (2.0, 0.05, 0.12)}   # k*sigma*sqrt(10), clamp
NEWS_WINDOW_DAYS = 14
NEWS_HALF_LIFE_DAYS = 3.0

# --------------------------------------------------------------- profiles ----
# sign: +1 = when this rises, the asset tends to rise; -1 = the opposite.
# ("macro", id) entries are FRED transforms computed below.
CRYPTO = {
    "group": "crypto", "seasonality": False,
    "drivers": [("QQQ", +1, "Nasdaq risk appetite"), ("UUP", -1, "US dollar"),
                ("^VIX", -1, "equity volatility"), ("TLT", +1, "long bonds / easier money"),
                (("macro", "m2_yoy"), +1, "M2 liquidity growth"),
                (("macro", "real_yield_chg"), -1, "real yields")],
    "chain": [("IBIT", +1, "spot BTC ETF (flows)"), ("ETHA", +1, "spot ETH ETF (flows)"),
              ("COIN", +1, "exchange volumes (Coinbase)"), ("MSTR", +1, "treasury buyers (Strategy)"),
              ("WGMI", +1, "miners"), ("BLOK", +1, "blockchain equities")],
    "keywords": ["crypto", "bitcoin", "btc", "ether", "ethereum", "solana", "stablecoin", "blockchain", "token"],
    "events": {
        "etf inflow": +1, "inflows": +1, "approval": +1, "approved": +1, "adoption": +1,
        "treasury purchase": +1, "buys bitcoin": +1, "halving": +1, "rate cut": +1,
        "institutional": +1, "all-time high": +1, "record high": +1,
        "hack": -1, "exploit": -1, "stolen": -1, "outflow": -1, "outflows": -1,
        "sec sues": -1, "lawsuit": -1, "crackdown": -1, "ban": -1, "bankruptcy": -1,
        "insolvent": -1, "liquidation": -1, "liquidations": -1, "depeg": -1, "delisted": -1,
        "rate hike": -1, "sell-off": -1, "selloff": -1,
    },
    "regime": {1: +0.6, 2: +0.4, 3: -0.6, 4: -0.4},
}
_ENERGY = {
    "group": "energy", "seasonality": True,
    "drivers": [("UUP", -1, "US dollar"), ("XLE", +1, "energy equities"), ("SPY", +1, "growth / demand"),
                (("macro", "real_yield_chg"), -1, "real yields")],
    "chain": [("BWET", +1, "tanker rates"), ("XOM", +1, "producers"), ("USO", +1, "crude ETF flows"),
              ("UNG", +1, "gas ETF flows"), ("BDRY", +1, "dry-bulk demand")],
    "keywords": ["oil", "crude", "brent", "wti", "opec", "gasoline", "diesel", "refinery", "natural gas",
                 "lng", "pipeline", "energy"],
    "events": {
        "opec cut": +1, "output cut": +1, "production cut": +1, "supply cut": +1, "sanctions": +1,
        "outage": +1, "disruption": +1, "hurricane": +1, "inventory draw": +1, "draw": +1,
        "refinery fire": +1, "attack": +1, "embargo": +1, "cold snap": +1, "heat wave": +1,
        "opec hike": -1, "output increase": -1, "production increase": -1, "build": -1,
        "inventory build": -1, "glut": -1, "oversupply": -1, "demand slump": -1, "ceasefire": -1,
        "recession": -1, "mild weather": -1,
    },
    "regime": {1: +0.2, 2: +0.7, 3: +0.4, 4: -0.7},
}
_GRAINS = {
    "group": "grains", "seasonality": True,
    "drivers": [("UUP", -1, "US dollar (export competitiveness)"), ("BRL=X", -1, "weaker real = more Brazilian selling"),
                ("NG=F", +1, "natural gas -> fertilizer cost"),
                ("CL=F", +1, "crude -> ethanol / biodiesel demand"),
                (("macro", "real_yield_chg"), -1, "real yields")],
    "chain": [("BDRY", +1, "dry-bulk freight (grain shipping)"), ("ADM", +1, "processors (ADM)"),
              ("BG", +1, "processors (Bunge)"), ("MOS", +1, "fertilizer (Mosaic)"),
              ("CF", +1, "nitrogen (CF)"), ("NTR", +1, "fertilizer (Nutrien)"), ("DE", +1, "farm equipment demand")],
    "keywords": ["corn", "wheat", "soybean", "soybeans", "soy", "grain", "grains", "crop", "harvest",
                 "planting", "usda", "wasde", "rice", "oats", "farm"],
    "events": {
        "drought": +1, "dry weather": +1, "heat wave": +1, "frost": +1, "flood": +1, "export ban": +1,
        "crop failure": +1, "lower yield": +1, "yield cut": +1, "shortage": +1, "war": +1,
        "black sea": +1, "strong demand": +1, "china buys": +1, "china purchase": +1,
        "record harvest": -1, "bumper": -1, "bumper crop": -1, "rain": -1, "favorable weather": -1,
        "higher yield": -1, "surplus": -1, "glut": -1, "export sales fall": -1, "cancelled": -1,
        "tariff": -1,
    },
    "regime": {1: -0.2, 2: +0.6, 3: +0.5, 4: -0.6},
}
_SOFTS = {
    "group": "softs", "seasonality": True,
    "drivers": [("UUP", -1, "US dollar"), ("BRL=X", -1, "weaker real = more Brazilian selling (coffee, sugar)"),
                ("CL=F", +1, "crude -> sugar ethanol / transport"),
                (("macro", "real_yield_chg"), -1, "real yields")],
    "chain": [("BDRY", +1, "dry-bulk freight"), ("SBLK", +1, "bulk shippers")],
    "keywords": ["coffee", "sugar", "cocoa", "cotton", "orange juice", "arabica", "robusta", "brazil",
                 "ivory coast", "ghana"],
    "events": {
        "drought": +1, "frost": +1, "dry weather": +1, "el nino": +1, "la nina": +1, "disease": +1,
        "shortage": +1, "deficit": +1, "export ban": +1, "smaller crop": +1, "lower output": +1,
        "rain": -1, "bumper": -1, "record crop": -1, "surplus": -1, "higher output": -1, "glut": -1,
    },
    "regime": {1: 0.0, 2: +0.5, 3: +0.4, 4: -0.5},
}
_PRECIOUS = {
    "group": "precious", "seasonality": True,
    "drivers": [("UUP", -1, "US dollar"), (("macro", "real_yield_chg"), -1, "real yields"),
                ("^VIX", +1, "risk aversion"), ("TLT", +1, "falling yields"),
                (("macro", "m2_yoy"), +1, "M2 liquidity")],
    "chain": [("NEM", +1, "gold miners (Newmont)"), ("GLD", +1, "gold ETF flows"), ("SLV", +1, "silver ETF flows")],
    "keywords": ["gold", "silver", "platinum", "palladium", "bullion", "precious metal"],
    "events": {
        "central bank buying": +1, "central bank purchases": +1, "safe haven": +1, "rate cut": +1,
        "geopolitical": +1, "war": +1, "inflows": +1, "record high": +1, "mine strike": +1,
        "rate hike": -1, "strong dollar": -1, "outflows": -1, "risk-on": -1, "selling": -1,
    },
    "regime": {1: -0.3, 2: +0.3, 3: +0.7, 4: +0.4},
}
_BASE = {
    "group": "base", "seasonality": True,
    "drivers": [("UUP", -1, "US dollar"), ("SPY", +1, "global growth"),
                ("USDCNY=X", -1, "yuan strength = China demand")],
    "chain": [("FCX", +1, "copper miners (Freeport)"), ("SCCO", +1, "copper miners (Southern Copper)"),
              ("CPER", +1, "copper ETF flows"), ("LIT", +1, "battery / EV demand"), ("TAN", +1, "solar build-out"),
              ("BDRY", +1, "dry-bulk freight (ore)")],
    "keywords": ["copper", "metals", "mining", "smelter", "china", "infrastructure", "grid"],
    "events": {
        "mine closure": +1, "strike": +1, "disruption": +1, "stimulus": +1, "shortage": +1,
        "deficit": +1, "export curbs": +1, "infrastructure": +1,
        "surplus": -1, "slowdown": -1, "property crisis": -1, "weak demand": -1, "tariff": -1,
    },
    "regime": {1: +0.4, 2: +0.7, 3: -0.2, 4: -0.7},
}
_LIVESTOCK = {
    "group": "livestock", "seasonality": True,
    "drivers": [("ZC=F", -1, "feed cost (corn)"), ("ZM=F", -1, "feed cost (soymeal)"), ("SPY", +1, "consumer demand")],
    "chain": [("UUP", -1, "export competitiveness")],
    "keywords": ["cattle", "hogs", "pork", "beef", "livestock", "meat", "screwworm", "swine"],
    "events": {
        "herd": +1, "tight supply": +1, "shortage": +1, "export demand": +1, "screwworm": +1,
        "disease": -1, "swine fever": -1, "recall": -1, "oversupply": -1, "weak demand": -1,
    },
    "regime": {1: +0.2, 2: +0.4, 3: 0.0, 4: -0.4},
}
_RATES = {
    "group": "rates", "seasonality": False,
    "drivers": [(("macro", "y10_chg"), -1, "10y yield change"), (("macro", "y2_chg"), -1, "2y yield (Fed path)"),
                ("^VIX", +1, "flight to quality"), ("SPY", -1, "risk appetite")],
    "chain": [("TLT", +1, "long-bond ETF flows"), ("TIP", +1, "inflation-linked demand")],
    "keywords": ["treasury", "treasuries", "yield", "yields", "bond", "bonds", "fed", "fomc", "rate", "rates", "auction"],
    "events": {
        "rate cut": +1, "dovish": +1, "cooling inflation": +1, "slowdown": +1, "flight to safety": +1,
        "weak jobs": +1, "recession": +1, "strong auction": +1,
        "rate hike": -1, "hawkish": -1, "hot inflation": -1, "sticky inflation": -1, "deficit": -1,
        "weak auction": -1, "strong jobs": -1,
    },
    "regime": {1: +0.2, 2: -0.6, 3: -0.3, 4: +0.7},
}
_INDEX = {
    "group": "index", "seasonality": False,
    "drivers": [("^VIX", -1, "volatility"), ("TLT", +1, "easier money"), ("UUP", -1, "US dollar"),
                (("macro", "real_yield_chg"), -1, "real yields")],
    "chain": [("QQQ", +1, "tech leadership"), ("SPY", +1, "breadth"), ("RSP", +1, "equal-weight breadth")],
    "keywords": ["stocks", "s&p", "nasdaq", "dow", "equities", "wall street", "earnings"],
    "events": {"rate cut": +1, "beat": +1, "rally": +1, "record high": +1, "stimulus": +1,
               "rate hike": -1, "miss": -1, "sell-off": -1, "selloff": -1, "recession": -1, "downgrade": -1},
    "regime": {1: +0.7, 2: +0.4, 3: -0.6, 4: -0.4},
}
FUTURE_ROOTS = {
    **{r: _ENERGY for r in ("CL", "BZ", "NG", "RB", "HO", "QM", "QG")},
    **{r: _GRAINS for r in ("ZC", "ZW", "KE", "ZS", "ZL", "ZM", "ZO", "ZR", "XC", "XW", "XK")},
    **{r: _SOFTS for r in ("KC", "SB", "CC", "CT", "OJ", "LBS")},
    **{r: _PRECIOUS for r in ("GC", "SI", "PL", "PA", "MGC", "SIL")},
    **{r: _BASE for r in ("HG", "ALI")},
    **{r: _LIVESTOCK for r in ("LE", "HE", "GF")},
    **{r: _RATES for r in ("ZT", "ZF", "ZN", "ZB", "UB", "TN")},
    **{r: _INDEX for r in ("ES", "NQ", "YM", "RTY", "MES", "MNQ", "MYM", "M2K")},
}
# Per-contract keyword refinements (news must be about THIS contract, not its sector).
CONTRACT_KEYWORDS = {
    "CL": ["crude", "wti", "oil", "opec"], "BZ": ["brent", "crude", "oil", "opec"],
    "NG": ["natural gas", "lng", "gas prices"], "RB": ["gasoline", "refinery"], "HO": ["heating oil", "diesel", "distillate"],
    "ZC": ["corn"], "ZW": ["wheat"], "KE": ["wheat"], "ZS": ["soybean", "soybeans", "soy"],
    "ZL": ["soybean oil", "soyoil", "biodiesel"], "ZM": ["soymeal", "soybean meal"], "ZO": ["oats"], "ZR": ["rice"],
    "KC": ["coffee", "arabica"], "SB": ["sugar"], "CC": ["cocoa"], "CT": ["cotton"], "OJ": ["orange juice", "orange"],
    "GC": ["gold", "bullion"], "SI": ["silver"], "PL": ["platinum"], "PA": ["palladium"], "HG": ["copper"],
    "LE": ["cattle", "beef"], "GF": ["feeder cattle", "cattle"], "HE": ["hogs", "pork", "swine"],
}
CRYPTO_KEYWORDS = {
    "BTC": ["bitcoin", "btc"], "ETH": ["ethereum", "ether", "eth"], "SOL": ["solana", "sol"],
    "XRP": ["xrp", "ripple"], "DOGE": ["dogecoin", "doge"], "ADA": ["cardano"], "AVAX": ["avalanche"],
}


def asset_class(ticker, row=None):
    """'crypto' | 'future' | None (equity / FX / index - not handled here)."""
    t = (ticker or "").upper()
    ac = str((row or {}).get("assetClass") or (row or {}).get("asset_class") or "").lower()
    if t.endswith("=F") or ac == "future":
        return "future"
    if ac == "crypto" or re.fullmatch(r"[A-Z0-9]{2,10}-USD", t):
        return "crypto"
    return None


def profile_for(ticker, row=None):
    cls = asset_class(ticker, row)
    if cls == "crypto":
        base = (ticker or "").upper().split("-")[0]
        kw = CRYPTO_KEYWORDS.get(base, [base.lower()])
        return cls, dict(CRYPTO, keywords=kw + ["crypto"])
    if cls == "future":
        root = (ticker or "").upper().replace("=F", "")
        prof = FUTURE_ROOTS.get(root)
        if not prof:
            return cls, None
        kw = CONTRACT_KEYWORDS.get(root) or prof["keywords"]
        return cls, dict(prof, keywords=kw)
    return None, None


# ---------------------------------------------------------------- helpers ----
def _clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def _closes_with_dates(series):
    out = []
    for bar in series or []:
        if not isinstance(bar, dict):
            continue
        c = bar.get("close", bar.get("price"))
        try:
            c = float(c)
        except (TypeError, ValueError):
            continue
        if c > 0 and math.isfinite(c) and bar.get("date"):
            out.append((str(bar["date"])[:10], c))
    return out


def momentum_z(closes, n=20):
    """n-day return scaled by its own volatility -> [-1, 1] (|z| of 3 = full)."""
    if not closes or len(closes) < n + 21:
        return None
    rets = [closes[i] / closes[i - 1] - 1 for i in range(len(closes) - 60, len(closes)) if i > 0]
    if len(rets) < 20:
        return None
    m = sum(rets) / len(rets)
    sd = math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1)) or 1e-9
    r_n = closes[-1] / closes[-1 - n] - 1
    return _clip(r_n / (sd * math.sqrt(n)) / 3.0)


def daily_vol(closes, n=60):
    if not closes or len(closes) < n + 1:
        return None
    rets = [closes[i] / closes[i - 1] - 1 for i in range(len(closes) - n, len(closes))]
    m = sum(rets) / len(rets)
    return math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1))


def seasonality(dated, today, horizon=20, min_years=8, lookback_years=15):
    """Average forward `horizon`-day return starting at today's calendar date in
    each of the last `lookback_years` years -> score in [-1, 1] (t-stat / 3),
    plus the stats. None if fewer than `min_years` usable years."""
    if not dated or len(dated) < 260 * min_years:
        return None
    try:
        td = datetime.strptime(today, "%Y-%m-%d")
    except ValueError:
        return None
    dates = [d for d, _ in dated]
    rets = []
    for y in range(td.year - lookback_years, td.year):
        key = f"{y}-{td.month:02d}-{td.day:02d}"
        i = next((k for k, d in enumerate(dates) if d >= key), None)
        if i is None or i + horizon >= len(dated) or dates[i][:4] != str(y):
            continue
        rets.append(dated[i + horizon][1] / dated[i][1] - 1)
    if len(rets) < min_years:
        return None
    m = sum(rets) / len(rets)
    sd = math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1)) or 1e-9
    t = m / (sd / math.sqrt(len(rets)))
    hit = sum(1 for r in rets if r > 0) / len(rets)
    return {"score": _clip(t / 3.0), "meanPct": round(m * 100, 2), "hitRate": round(hit, 2),
            "years": len(rets), "horizonDays": horizon}


# ------------------------------------------------------------ data access ----
class ResearchData:
    """Lazily loads and caches everything the research layer reads."""

    def __init__(self, fetch_json, raw, universe, today):
        self.fetch_json = fetch_json
        self.raw = raw
        self.universe = universe or {}
        self.today = today
        self._hist = {}
        self._macro = {}
        self._news = None
        self._lexicon = None

    def history(self, ticker):
        t = (ticker or "").upper()
        if t in self._hist:
            return self._hist[t]
        row = self.universe.get(t) or {}
        repos = [row.get("repo")] if row.get("repo") else []
        repos += [r for r in ("TRAPP2", "TRAPP2-1", "TRAPP2-2", "TRAPP2-3") if r not in repos]
        dated = None
        for repo in repos:
            data = self.fetch_json(f"{self.raw}/{repo}/main/data/history/{t}.json", timeout=30)
            if isinstance(data, list) and len(data) >= 60:
                dated = _closes_with_dates(data)
                break
        self._hist[t] = dated
        return dated

    def closes(self, ticker):
        d = self.history(ticker)
        return [c for _, c in d] if d else None

    def fred(self, sid):
        if sid in self._macro:
            return self._macro[sid]
        d = self.fetch_json(f"{self.raw}/TRAPP2-1/main/data/macro/{sid}.json", timeout=30)
        obs = (d.get("observations") if isinstance(d, dict) else None) or []
        vals = []
        for o in obs:
            try:
                v = float(o.get("value"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                vals.append((o.get("date"), v))
        self._macro[sid] = vals or None
        return self._macro[sid]

    def macro_signal(self, name):
        """FRED transforms, signed so that 'rising' = positive."""
        if name == "m2_yoy":
            s = self.fred("M2SL")
            if not s or len(s) < 16:
                return None
            yoy_now = s[-1][1] / s[-13][1] - 1
            yoy_prev = s[-4][1] / s[-16][1] - 1
            return _clip((yoy_now - yoy_prev) / 0.01)          # 1pp accel in M2 growth = full
        if name in ("real_yield_chg", "y10_chg", "y2_chg"):
            sid = "DGS2" if name == "y2_chg" else "DGS10"
            s = self.fred(sid)
            if not s or len(s) < 70:
                return None
            chg = s[-1][1] - s[-64][1]                          # ~3 months of trading days
            if name == "real_yield_chg":
                cpi = self.fred("CPIAUCSL")
                if cpi and len(cpi) >= 16:
                    infl_now = cpi[-1][1] / cpi[-13][1] - 1
                    infl_3m = cpi[-4][1] / cpi[-16][1] - 1
                    chg -= (infl_now - infl_3m) * 100           # nominal minus inflation change
            return _clip(chg / 0.75)                            # 75bp move = full
        return None

    def news_items(self):
        if self._news is not None:
            return self._news
        items = []
        d = self.fetch_json(f"{self.raw}/TRAPP2-ANALYTICS/main/data/news/latest.json", timeout=30)
        for a in (d.get("items") if isinstance(d, dict) else None) or []:
            items.append(a)
        x = self.fetch_json(f"{self.raw}/XTRAPP/main/data/xtrapp_data.json", timeout=30)
        if isinstance(x, dict):
            items += [a for a in (x.get("articles") or []) if isinstance(a, dict)]
            lex = x.get("lexicon") if isinstance(x.get("lexicon"), dict) else {}
            self._lexicon = {"bull": lex.get("bull") or {}, "bear": lex.get("bear") or {}}
        seen, out = set(), []
        for a in items:
            u = a.get("url") or a.get("headline")
            if u and u not in seen:
                seen.add(u)
                out.append(a)
        self._news = out
        return out

    def lexicon(self):
        self.news_items()
        return self._lexicon or {"bull": {}, "bear": {}}


# ------------------------------------------------------------- components ----
def _basket(entries, data):
    vals, parts = [], []
    for ent, sign, label in entries:
        if not sign:
            continue
        if isinstance(ent, tuple) and ent[0] == "macro":
            v = data.macro_signal(ent[1])
            name = ent[1]
        else:
            v = momentum_z(data.closes(ent))
            name = ent
        if v is None:
            continue
        vals.append(sign * v)
        parts.append({"input": name, "label": label, "value": round(sign * v, 3)})
    if len(vals) < 2:
        return None, parts
    return _clip(sum(vals) / len(vals)), parts


def _age_days(dt_str, now):
    if not dt_str:
        return None
    s = str(dt_str)
    try:
        if s.isdigit():
            ts = int(s)
            ts = ts / 1000 if ts > 1e12 else ts
            d = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            d = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
        return (now - d).total_seconds() / 86400
    except Exception:
        return None


def news_score(ticker, prof, data, now=None):
    now = now or datetime.now(timezone.utc)
    t = (ticker or "").upper()
    kws = [k.lower() for k in prof.get("keywords") or []]
    events = prof.get("events") or {}
    lex = data.lexicon()
    num = den = 0.0
    hits = []
    for a in data.news_items():
        age = _age_days(a.get("datetime") or a.get("publishedAt") or a.get("date"), now)
        if age is None or age < -1 or age > NEWS_WINDOW_DAYS:
            continue
        text = f" {a.get('headline') or ''} {a.get('summary') or ''} ".lower()
        tagged = t in [str(x).upper() for x in ([a.get("ticker")] + list(a.get("tickers") or [])) if x]
        if not tagged and not any(re.search(rf"\b{re.escape(k)}\b", text) for k in kws):
            continue
        s = 0.0
        for phrase, d in events.items():
            if phrase in text:
                s += d
        for phrase, w in (lex.get("bull") or {}).items():
            if re.search(rf"\b{re.escape(str(phrase).lower())}\b", text):
                s += 0.25 * float(w or 1)
        for phrase, w in (lex.get("bear") or {}).items():
            if re.search(rf"\b{re.escape(str(phrase).lower())}\b", text):
                s -= 0.25 * float(w or 1)
        sent = a.get("sentiment")
        if isinstance(sent, (int, float)):
            s += float(sent)
        elif isinstance(sent, str):
            s += {"positive": 0.5, "bullish": 0.5, "negative": -0.5, "bearish": -0.5}.get(sent.lower(), 0.0)
        w = 0.5 ** (max(age, 0) / NEWS_HALF_LIFE_DAYS)
        num += w * _clip(s / 2.0)
        den += w
        hits.append({"headline": (a.get("headline") or "")[:120], "ageDays": round(age, 1),
                     "score": round(_clip(s / 2.0), 2)})
    if not hits:
        return None, 0, []
    hits.sort(key=lambda h: h["ageDays"])
    return _clip(num / den if den else 0.0), len(hits), hits[:5]


# ------------------------------------------------------------- evaluation ----
def evaluate(ticker, row, tech_signed, quad, data, today, min_score, allow_shorts=False):
    """Research case for one crypto / futures instrument.
    Returns None for anything this module doesn't handle (equities etc.)."""
    cls, prof = profile_for(ticker, row)
    if not cls:
        return None
    out = {"assetClass": cls, "group": (prof or {}).get("group"), "eligible": False,
           "reasons": [], "components": {}, "detail": {}}
    if not prof:
        out["reasons"].append("no research profile for this contract")
        return out

    comps = {}
    v, parts = _basket(prof["drivers"], data)
    if v is not None:
        comps["drivers"] = v
    out["detail"]["drivers"] = parts
    v, parts = _basket(prof["chain"], data)
    if v is not None:
        comps["supplyChain"] = v
    out["detail"]["supplyChain"] = parts
    nv, ncount, nhits = news_score(ticker, prof, data)
    if nv is not None:
        comps["news"] = nv
    out["detail"]["news"] = {"matched": ncount, "top": nhits}
    if isinstance(quad, int) and quad in prof["regime"]:
        comps["regime"] = prof["regime"][quad]
        out["detail"]["regime"] = {"quad": quad}
    if prof.get("seasonality"):
        s = seasonality(data.history(ticker), today)
        if s:
            comps["seasonality"] = s["score"]
            out["detail"]["seasonality"] = s

    out["components"] = {k: round(v, 3) for k, v in comps.items()}
    w = {k: WEIGHTS[k] for k in comps}
    research = (sum(comps[k] * w[k] for k in comps) / sum(w.values())) if w else None
    out["research"] = round(research, 3) if research is not None else None
    out["newsCoverage"] = ncount

    # ---- the gate ----
    if "drivers" not in comps:
        out["reasons"].append("no driver data")
    if len(comps) < 3:
        out["reasons"].append(f"only {len(comps)} research component(s) - need drivers + 2 more")
    tech = tech_signed if tech_signed is not None else 0.0
    direction = None
    bar = RESEARCH_MIN if ncount else RESEARCH_MIN_NO_NEWS
    if research is not None:
        if tech >= min_score + MIN_SCORE_BUMP and research >= bar:
            direction = "long"
        elif allow_shorts and tech <= -(min_score + MIN_SCORE_BUMP) and research <= -bar:
            direction = "short"
        else:
            out["reasons"].append(
                f"tech {tech:+.2f} / research {research:+.2f} below the alt bar "
                f"(tech >= {min_score + MIN_SCORE_BUMP:.2f}, research >= {bar:.2f}"
                f"{'' if ncount else ', no news coverage'})")
    d = 1 if direction != "short" else -1
    against = [k for k, x in comps.items() if d * x <= VETO]
    if direction and against:
        out["reasons"].append("vetoed - strongly against: " + ", ".join(against))
    if direction and ncount >= 2 and d * comps.get("news", 0) <= NEWS_VETO:
        out["reasons"].append("vetoed - news flow against the trade")
    if out["reasons"]:
        return out

    closes = data.closes(ticker)
    sig = daily_vol(closes) if closes else None
    k, lo, hi = STOP_RULE[cls]
    stop_pct = _clip((k * sig * math.sqrt(10)) if sig else hi, lo, hi)
    size_mult = SIZE_MULT[cls] * (min(1.0, TARGET_DAILY_VOL[cls] / sig) if sig else 0.5)
    out.update({
        "eligible": True, "direction": direction,
        "rankScore": round(0.5 * abs(tech) + 0.5 * abs(research), 3),
        "stopPct": round(stop_pct, 4), "targetPct": round(2 * stop_pct, 4),
        "sizeMult": round(size_mult, 3), "dailyVol": round(sig, 4) if sig else None,
    })
    return out


def research_flip(pos, row, quad, data, today):
    """For a HELD alt position: research composite now (signed for the position's
    direction) - the runner exits when it has turned decisively against."""
    ev = evaluate(pos.get("ticker"), row, 0.0, quad, data, today, min_score=9.0)   # gate irrelevant here
    if not ev or ev.get("research") is None:
        return None, ev
    d = -1 if pos.get("direction") == "short" else 1
    return d * ev["research"], ev
