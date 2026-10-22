# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   pipeline/trade_book.py
#
#  z138 - HOW THE BOT HOLDS THINGS. Pure helpers used by bot_runner.py:
#   • Three horizons for longs AND shorts - SHORT (~2 weeks), MEDIUM (~2 months), LONG
#     (~6+ months) - each with its own minimum hold, stop, target and option expiry window.
#     Nothing the bot opens is closed the next day: inside the minimum hold only a disaster
#     move (shares) or expiry settlement (options) can close it.
#   • A mix: the open book is steered toward ~35 % short / 40 % medium / 25 % long, so the
#     record covers every kind of trade (the data trains on variety, not one habit).
#   • Options: calls for longs, puts for shorts (defined risk - the premium is the most it
#     can lose), used for high conviction and for catalysts inside the horizon; the expiry
#     is chosen for the horizon (time management) and marked from the TRAPP2-OPTIONS chain.
#   • Catalysts: the next earnings date + the name's beat record (TRAPP2-ANALYTICS
#     earnings calendar) - timing for the short horizon and a signal in the blend.
#   • A review of every open position on every pass: what it is, how it's doing, what's
#     coming (catalyst / stop / target / expiry), and what the bot decided and why.
# ============================================================
import math
from datetime import date, datetime, timedelta

HZ = {
    #          planned days, min hold (sessions), stop / target (shares, frac), disaster stop inside the hold,
    #          option expiry window (calendar days), option take-profit (x premium), roll-off DTE
    "short":  {"days": 10,  "min_hold": 2,  "stop": 0.07, "target": 0.12, "disaster": 0.12,
               "dte": (14, 45),   "opt_tp": 1.00, "exit_dte": 5,  "delta": 0.45},
    "medium": {"days": 40,  "min_hold": 5,  "stop": 0.12, "target": 0.25, "disaster": 0.18,
               "dte": (45, 120),  "opt_tp": 1.50, "exit_dte": 14, "delta": 0.50},
    "long":   {"days": 180, "min_hold": 15, "stop": None, "target": None, "disaster": 0.25,
               "dte": (180, 560), "opt_tp": 2.00, "exit_dte": 30, "delta": 0.65},
}
MIX_TARGET = {"short": 0.35, "medium": 0.40, "long": 0.25}
OPT_CONF = 0.68                 # conviction that earns an options trade (or a catalyst inside the horizon)
OPT_TRADE_PCT = (0.010, 0.025)  # premium per option trade: 1.0 % → 2.5 % of the bankroll with confidence
DEEP_CONV, DEEP_PCT = 0.85, 0.04  # deep conviction (≥ 0.85): up to 4 % of the bankroll in premium - the big-money trades


def option_budget(bankroll, conf, explore=False):
    """Premium for one option trade. Exploratory trades are smaller, but never below the normal floor."""
    if conf >= DEEP_CONV:
        return bankroll * DEEP_PCT                     # deep conviction is sized as such, exploratory or not
    pct = OPT_TRADE_PCT[0] + (OPT_TRADE_PCT[1] - OPT_TRADE_PCT[0]) * max(0.0, min(1.0, (conf - 0.5) / (DEEP_CONV - 0.5)))
    if explore:
        pct = max(OPT_TRADE_PCT[0], pct * 0.7)
    return bankroll * pct
OPT_BOOK_PCT = 0.12             # all premium at risk ≤ 12 % of the bankroll
OPT_STOP = 0.50                 # −50 % of the premium (after the minimum hold)
OPT_FEE = 0.65                  # $ per contract per side
MIN_OI, MIN_VOL = 50, 5         # liquidity
MAX_SPREAD = 0.15               # (ask − bid) ≤ 15 % of mid (or ≤ $0.10)
CHAIN_MAX_AGE_D = 3             # a chain older than this isn't traded on


def _d(s):
    try:
        return date.fromisoformat(str(s)[:10])
    except (TypeError, ValueError):
        return None


def sessions_between(d0, d1, day_status=None):
    """NYSE sessions after d0 up to and including d1 (weekends and, with market_hours, holidays out)."""
    a, b = _d(d0), _d(d1)
    if not a or not b or b <= a:
        return 0
    n, d = 0, a
    while d < b:
        d += timedelta(days=1)
        if d.weekday() >= 5:
            continue
        if day_status:
            try:
                if not day_status(d)[0]:
                    continue
            except Exception:
                pass
        n += 1
    return n


# ---------------------------------------------------------------- catalysts
def catalyst_info(cal, ticker, today):
    """{type, date, days, time, est, beatRate, avgSurprise, n} for the next earnings, or None."""
    c = (cal or {}).get(str(ticker or "").upper())
    if not c:
        return None
    nx = c.get("next") or {}
    days = None
    if nx.get("date"):
        dd = _d(nx["date"])
        if dd:
            days = (dd - _d(today)).days
    return {"type": "earnings", "date": nx.get("date"), "days": days, "time": nx.get("time") or "",
            "est": nx.get("est"), "beatRate": c.get("beatRate"), "avgSurprise": c.get("avgSurprise"), "n": c.get("n") or 0}


def catalyst_signal(info):
    """Signed −0.5…+0.5: a name that beats (and by how much) leans long into its report, one that
    misses leans short; strongest with the report 1–15 days out, fading beyond 45 days."""
    if not info or info.get("beatRate") is None:
        return None
    br, av, n = info["beatRate"], info.get("avgSurprise") or 0.0, info.get("n") or 0
    v = (br - 0.5) * 1.0 + max(-0.25, min(0.25, av / 20.0))
    d = info.get("days")
    t = 1.0 if (d is not None and 1 <= d <= 15) else 0.6 if (d is not None and 16 <= d <= 45) else 0.3
    v *= t * (1.0 if n >= 4 else 0.5)
    return round(max(-0.5, min(0.5, v)), 3)


# ---------------------------------------------------------------- horizon + mix
def open_mix(trades):
    m = {"short": 0, "medium": 0, "long": 0}
    for t in trades:
        if t.get("status") == "open":
            m[horizon_class(t)] += 1
    return m


def horizon_class(t):
    h = t.get("horizonClass")
    if h in HZ:
        return h
    return "long" if t.get("horizonType") == "core" else "medium"     # pre-z138 trades: core → long, swing → medium


def choose_horizon(conf, lean, cat, direction, mix, core_conf=0.62):
    """The horizon the signals ask for, nudged toward the mix the book is short of."""
    aligned_cat = cat and cat.get("days") is not None and 2 <= cat["days"] <= 12 and \
        cat.get("beatRate") is not None and ((cat["beatRate"] >= 0.6) if direction == "long" else (cat["beatRate"] <= 0.45))
    if aligned_cat:
        want, why = "short", f"earnings in {cat['days']}d - a catalyst trade"
    elif conf >= core_conf and lean >= 0.55:
        want, why = "long", f"conf {conf:.2f} on slow signals (lean {lean:.2f})"
    elif lean < 0.35:
        want, why = "short", f"momentum-led (lean {lean:.2f})"
    else:
        want, why = "medium", f"mixed signals (lean {lean:.2f})"
    total = sum(mix.values()) or 1
    share = {k: mix[k] / total for k in mix}
    over = share[want] - MIX_TARGET[want]
    steered = False
    if over > 0.10 and not aligned_cat:
        # a borderline name may take the neighbouring horizon the book lacks
        nbrs = {"short": ["medium"], "medium": ["short", "long"], "long": ["medium"]}[want]
        need = sorted(nbrs, key=lambda k: share[k] - MIX_TARGET[k])
        alt = need[0]
        if share[alt] - MIX_TARGET[alt] < -0.05 and (alt != "long" or conf >= core_conf - 0.07):
            why += f"; steered {want}→{alt} (book {share[want]:.0%} {want} vs {MIX_TARGET[want]:.0%} target)"
            want, steered = alt, True
    return want, why, steered


# ---------------------------------------------------------------- options
def chain_fresh(chain, today):
    if not isinstance(chain, dict) or not chain.get("expiries"):
        return False
    a = _d(chain.get("asOf"))
    return bool(a and (_d(today) - a).days <= CHAIN_MAX_AGE_D)


def pick_contract(chain, direction, hclass, today, cat=None):
    """Call for a long, put for a short: expiry inside the horizon's window (and after the catalyst
    + 5 days), the strike nearest the horizon's delta, liquid and with a sane spread. Fill at the ask."""
    if not chain_fresh(chain, today):
        return None, "no fresh chain"
    lo, hi = HZ[hclass]["dte"]
    t0 = _d(today)
    after = (_d(cat["date"]) + timedelta(days=5)) if (cat and cat.get("date") and _d(cat["date"])) else None
    def window(a, b):
        out = []
        for e in chain.get("expiries") or []:
            ed = _d(e.get("expiry"))
            if not ed:
                continue
            dte = (ed - t0).days
            if a <= dte <= b and (after is None or ed >= after):
                out.append((dte, e))
        return out
    exps = window(lo, hi)
    if not exps:
        # the chains sample expiries (weeklies, then monthlies, then LEAPS): with nothing inside the
        # window, take the nearest one past it (up to 1.8x) - more time than planned, never less
        exps = sorted(window(lo, int(hi * 1.8)), key=lambda x: x[0])[:1]
    if not exps:
        return None, f"no expiry {lo}-{int(hi * 1.8)} days out"
    mid_target = (lo + hi) / 2 if hclass != "long" else lo + 60
    exps.sort(key=lambda x: abs(x[0] - mid_target))
    kind = "call" if direction == "long" else "put"
    want = HZ[hclass]["delta"]
    for dte, e in exps[:3]:
        best = None
        for o in e.get("calls" if kind == "call" else "puts") or []:
            bid, ask, mid = o.get("bid") or 0, o.get("ask") or 0, o.get("mid") or 0
            if not (ask > 0 and mid > 0):
                continue
            if (o.get("openInterest") or 0) < MIN_OI and (o.get("volume") or 0) < MIN_VOL:
                continue
            if (ask - bid) > max(0.10, MAX_SPREAD * mid):
                continue
            dl = o.get("delta")
            score = abs(abs(dl) - want) if dl is not None else abs((o.get("strike") or 0) - (chain.get("spot") or 0)) / max(1e-9, chain.get("spot") or 1)
            if best is None or score < best[0]:
                best = (score, o)
        if best:
            o = best[1]
            return {"type": kind, "strike": o["strike"], "expiry": e["expiry"], "dte": dte, "ask": o["ask"], "bid": o.get("bid"),
                    "mid": o.get("mid"), "delta": o.get("delta"), "iv": o.get("iv"), "oi": o.get("openInterest"),
                    "chainAsOf": chain.get("asOf"), "spot": chain.get("spot")}, "ok"
    return None, "no liquid contract near the target delta"


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(kind, S, K, T, r, iv):
    if not (S and K and T and iv) or S <= 0 or K <= 0 or T <= 0 or iv <= 0:
        return max(0.0, (S - K) if kind == "call" else (K - S)) if (S and K) else None
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if kind == "call":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def option_mark(pos, chain, spot, today):
    """(mark per share, bid per share, source). The same contract in the latest chain (mid; bid for
    an exit) - else Black-Scholes at the entry IV - and intrinsic from the expiry date on."""
    kind, K, exp = pos.get("optionType"), pos.get("optionStrike"), pos.get("optionExpiry")
    ed, t0 = _d(exp), _d(today)
    intrinsic = max(0.0, (spot - K) if kind == "call" else (K - spot)) if (spot and K) else None
    if ed and t0 and t0 >= ed:
        return intrinsic, intrinsic, "expiry (intrinsic)"
    if isinstance(chain, dict):
        for e in chain.get("expiries") or []:
            if str(e.get("expiry"))[:10] != str(exp)[:10]:
                continue
            for o in e.get("calls" if kind == "call" else "puts") or []:
                if abs((o.get("strike") or 0) - K) < 1e-6 and (o.get("mid") or 0) > 0:
                    return float(o["mid"]), float(o.get("bid") or o["mid"]), f"chain {str(chain.get('asOf'))[:16]}"
    if ed and t0 and spot:
        T = max(1, (ed - t0).days) / 365.0
        p = bs_price(kind, spot, K, T, 0.04, pos.get("optionIV") or 0.4)
        if p is not None:
            return p, p * 0.97, "model (Black-Scholes, entry IV)"
    return None, None, "no mark"


def option_pnl(pos, px_per_share, fee_sides=1):
    c = pos.get("optionContracts") or 0
    return round((px_per_share - (pos.get("optionPremium") or 0)) * 100 * c - OPT_FEE * c * fee_sides, 2)


# ---------------------------------------------------------------- the review
def review(pos, today, *, price, ret, days_held, min_hold, decision, reason, score_now=None, cat=None,
           mark=None, mark_src=None, dte=None):
    """One open position, this pass: a line for the log and a record on the trade."""
    hc = horizon_class(pos)
    planned = pos.get("horizonDaysPlanned") or HZ[hc]["days"]
    parts = [f"{pos.get('ticker')} {pos.get('direction')}", f"{hc} (day {days_held} of ~{planned}"
             + (f", min hold {min_hold}" if days_held < min_hold else "") + ")"]
    if pos.get("instrument") == "option":
        parts.append(f"{pos.get('optionType', '').upper()} {pos.get('optionStrike')} {str(pos.get('optionExpiry'))[:10]} ×{pos.get('optionContracts')}"
                     + (f" · premium ${pos.get('optionPremium'):.2f}→${mark:.2f}" if mark is not None else "") + (f" · {dte} DTE" if dte is not None else ""))
    parts.append(f"{ret * 100:+.1f}%")
    if score_now is not None:
        parts.append(f"score {score_now:+.2f} (entry {pos.get('conviction', 0):+.2f})")
    if cat and cat.get("date"):
        parts.append(f"earnings {cat['date']}" + (f" ({cat['days']}d)" if cat.get("days") is not None else ""))
    if pos.get("instrument") != "option" and (pos.get("stopPrice") or pos.get("targetPrice")):
        parts.append(f"stop {pos.get('stopPrice')} / target {pos.get('targetPrice')}")
    line = " · ".join(str(p) for p in parts) + f" → {decision.upper()}: {reason}"
    rec = {"at": today, "decision": decision, "reason": reason, "price": price, "returnPct": round(ret * 100, 2),
           "daysHeld": days_held, "horizonClass": hc, "scoreNow": score_now, "markSource": mark_src}
    if mark is not None:
        rec["optionMark"] = round(mark, 4)
    if cat:
        rec["catalyst"] = {k: cat.get(k) for k in ("date", "days", "time")}
    pos["lastReview"] = rec
    log = pos.setdefault("reviews", [])
    if log and log[-1].get("at") == today and log[-1].get("decision") == decision:
        log[-1] = rec                                          # one entry per day per decision
    else:
        log.append(rec)
    del log[:-12]
    return line
