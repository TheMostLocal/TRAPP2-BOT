# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   pipeline/market_tape.py
#  z202 - THE MARKET TAPE, as known at a session's open - the same rules as TRAPP2-QUANT's quant/tape.py (keep the two
#  in step), so the bot's record and the quant lab speak one definition. Six of Today's Trends' seven criteria
#  (TRAPP2-ANALYTICS pipeline/trends.py), computed from completed sessions only (TRAPP2-1's daily ^GSPC with volume,
#  ^VIX, HYG) - so ANY trade, past or future, can be labelled by the market it was opened in (no look-ahead):
#     run 4+ lower closes · 5 %+ below the 52-week high · below the 50-day · VIX 25+ or +30 % in 5 sessions ·
#     5+ distribution days in 25 · HYG -1 % in 5 sessions          0-1 on: calm · 2: pressure · 3+: selloff
#  The bot uses it for the RECORD (bot_runner.trend_evidence / trendScores: did longs opened in sell-offs do better or
#  worse than the rest?) - the live 7-criterion read (trend_card, TRAPP2-ANALYTICS data/trends/latest.json) sizes
#  today's trades. Owner 11-21: the trends "relevant and being considered by the BOT and QUANT".
# ============================================================
import json
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta

RAW = "https://raw.githubusercontent.com/TheMostLocal"
HIST = RAW + "/TRAPP2-1/main/data/history/{}.json"
TRENDS_URL = RAW + "/TRAPP2-ANALYTICS/main/data/trends/latest.json"
SPX, VIX, HYG, SPY = "^GSPC", "^VIX", "HYG", "SPY"
STATES = ((0, "calm"), (2, "pressure"), (3, "selloff"))
DAY_BAND = 0.0075                   # +-0.75 % on the day = "flat"
MAX_GAP_DAYS = 6                    # the history must reach within 6 calendar days of the session, or no state


def _get(url, timeout=30):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "valuatio-quant"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def _num(v):
    try:
        x = float(v)
        return x if x == x and x not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


def series(rows):
    """History rows [{date, close, volume}] -> (dates, closes, volumes): sorted, one per date, valid closes only."""
    by = {}
    for r in (rows if isinstance(rows, list) else []):            # anything but a list of rows (an error page) = nothing
        if not isinstance(r, dict):
            continue
        d, c = str(r.get("date") or "")[:10], _num(r.get("close"))
        if len(d) == 10 and c and c > 0:
            by[d] = (c, _num(r.get("volume")))
    ds = sorted(by)
    return ds, [by[d][0] for d in ds], [by[d][1] for d in ds]


def load_histories(fetch=None):
    """{'spx', 'vix', 'hyg', 'spy'} -> (dates, closes, volumes) from TRAPP2-1 (one request each, ~300 KB)."""
    fetch = fetch or _get
    out = {}
    for k, sym in (("spx", SPX), ("vix", VIX), ("hyg", HYG), ("spy", SPY)):
        out[k] = series(fetch(HIST.format(urllib.parse.quote(sym, safe=""))))
    return out


def _upto(ser, day):
    """The part of a series strictly before `day` (completed sessions as known at that session's open)."""
    ds, cs, vs = ser or ([], [], [])
    k = 0
    for i, d in enumerate(ds):
        if d < day:
            k = i + 1
        else:
            break
    return ds[:k], cs[:k], vs[:k]


def state_on(day, H):
    """The market read as known at the open of `day` (ISO date or date). -> dict or None (too little history /
    the history doesn't reach the session)."""
    day = day.isoformat() if isinstance(day, date) else str(day)[:10]
    ds, cs, vs = _upto((H or {}).get("spx"), day)
    if len(cs) < 60:
        return None
    try:
        if (date.fromisoformat(day) - date.fromisoformat(ds[-1])).days > MAX_GAP_DAYS:
            return None
    except ValueError:
        return None
    crit = {}
    run = 0
    for i in range(len(cs) - 1, 0, -1):                     # straight lower closes through the prior close
        if cs[i] < cs[i - 1]:
            run += 1
        else:
            break
    up = 0
    if not run:
        for i in range(len(cs) - 1, 0, -1):
            if cs[i] > cs[i - 1]:
                up += 1
            else:
                break
    crit["run"] = run >= 4
    hi = max(cs[-252:])
    dd = cs[-1] / hi - 1
    crit["drawdown"] = dd <= -0.05
    m50 = sum(cs[-50:]) / 50
    crit["trend"] = cs[-1] < m50
    vd, vc, _ = _upto(H.get("vix"), day)
    vix = vix5 = None
    if vc and vd[-1] == ds[-1]:
        vix = vc[-1]
        vix5 = (vc[-1] / vc[-6] - 1) if len(vc) > 6 and vc[-6] else None
        crit["fear"] = vix >= 25 or (vix5 is not None and vix5 >= 0.30)
    dist = None
    if len(vs) >= 26 and any(vs[-25:]):                     # as trends.py: sessions without a volume are skipped
        dist = sum(1 for k in range(len(cs) - 25, len(cs)) if vs[k] and vs[k - 1] and cs[k] / cs[k - 1] - 1 <= -0.002 and vs[k] > vs[k - 1])
        crit["distribution"] = dist >= 5
    hd, hc, _ = _upto(H.get("hyg"), day)
    hyg5 = None
    if len(hc) > 6 and hd[-1] == ds[-1]:
        hyg5 = hc[-1] / hc[-6] - 1
        crit["credit"] = hyg5 <= -0.01
    on = sum(1 for v in crit.values() if v)
    state = [name for lo, name in STATES if on >= lo][-1]
    return {"state": state, "on": on, "of": len(crit), "asOf": ds[-1], "run": -run if run else up,
            "dd": round(dd * 100, 2), "vs50": round((cs[-1] / m50 - 1) * 100, 2), "vix": round(vix, 2) if vix else None,
            "vix5": round(vix5 * 100, 1) if vix5 is not None else None, "dist": dist,
            "hyg5": round(hyg5 * 100, 2) if hyg5 is not None else None,
            "criteria": {k: bool(v) for k, v in crit.items()}}


def prior_close(ser, day):
    """SPY's close on the last session before `day` (None when the history doesn't reach it)."""
    day = day.isoformat() if isinstance(day, date) else str(day)[:10]
    ds, cs, _ = _upto(ser, day)
    if not cs:
        return None
    try:
        if (date.fromisoformat(day) - date.fromisoformat(ds[-1])).days > MAX_GAP_DAYS:
            return None
    except ValueError:
        return None
    return cs[-1]


def day_bucket(move):
    """SPY's move from its prior close (a fraction) -> 'dn' / 'flat' / 'up' (None when unknown)."""
    if move is None:
        return None
    return "dn" if move <= -DAY_BAND else "up" if move >= DAY_BAND else "flat"
