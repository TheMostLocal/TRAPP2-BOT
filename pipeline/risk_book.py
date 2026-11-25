# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   pipeline/risk_book.py
#  z186 - THE RISK BOOK: how much one trade may lose, how much of the book one theme may hold, and how the bot
#  takes less risk while it is losing. Signals pick the trades; this decides their size and whether the book can
#  take them. Every knob is a repo Variable (RUNNER_RISK_*); the defaults are below.
#
#  Why (the 10-30 -> 11-09 drawdown, -9.4 % while SPY was -0.4 %): it was the book's construction, not the market.
#    * one theme, one day: eight energy longs opened 10-16 / 10-17 at ~3.8 % each (~30 % of the book), all
#      correlated, all stopped or timed out at -6 % .. -18 % (energy -$5,773 over 15 trades);
#    * options sized like stock: 1-2.5 % of the bankroll of premium on low-conviction calls, five medium-term calls
#      all stopped at -57 .. -67 %, one put credit spread lost $2,926 (2.6 % of the book in one trade);
#    * risk not budgeted: every position ~3.8 % of the book whatever its stop (7 % / 12 % / 25 %) or volatility -
#      a core position risked 3.5x what a short-term one did;
#    * no shorts: RUNNER_ALLOW_SHORTS was off, so all 57 runner trades were long (owner direction §1d: longs AND
#      shorts) - nothing in the book could gain when its longs fell;
#    * a stop that moved: a swing trade re-scored as "core" lost its stop (MPC, VLO: -8 % stops, closed at -17 %).
#
#  What it does: risk-budgeted sizes (the loss at the stop, not the notional, is what's fixed), caps per sector and
#  per correlated cluster (a sector's NET is capped, so a short against a long-heavy sector is a hedge, not a breach),
#  a net beta cap that depends on whether the regime call is dialed in (§1b), a cap on the
#  new risk taken per session, a per-position loss cap, a drawdown throttle (smaller new trades while the book is
#  below its high-water mark - it keeps trading, explore mode keeps collecting data) and a daily halt.
#  z190: short sales need a single stock by every sign (fund_reason) and a KNOWN $2B+ market cap; a short the rule
#  never allowed is unwound by the runner (RUNNER_UNWIND_UNSHORTABLE).
#  z192: etp_exposure() - a leveraged / inverse product's signed daily leverage and what it tracks, and etp_drag() -
#  what holding it past one session costs (the runner signs its market signals and manages it session by session).
#  z205: liquidity() - a position is a small share of what trades in a day: at most LIQ_PCT of the 20-session average
#  dollar volume (TRAPP2-ANALYTICS data/derived/market.json, the runner's u["dv20USD"]), none in a name trading under
#  MIN_DV a day; a short sale also needs MIN_SHORT_DV a day (borrow, a clean cover). Equities / ETFs only.
#  z206: shortable(u, covering) - no new short sale into a CROWDED short: MAX_SHORT_SI (20 %) or more of the float sold
#  short already (borrow cost, recalls, squeeze risk), and SQUEEZE_SHORT_SI (10 %) while the tape shows shorts being
#  covered (TRAPP2-ANALYTICS' flows: a short-covering rally today, or the session after one). The name gets a put
#  instead, when it has an option chain.
#  z207: crowded_short(u, covering) - FINRA's official short interest first (the runner's u["shortPctSO"]: shares short
#  at the latest settlement / the share count the runner trusts, and u["daysToCover"]): MAX_SHORT_SO (15 %) of the
#  shares outstanding, or MAX_SHORT_DTC (10) days to cover with DTC_MIN_SO (5 %)+ short; SQUEEZE_SHORT_SO (8 %) /
#  SQUEEZE_SHORT_DTC (7) while shorts are being covered. Yahoo's short % of float (20 % / 10 %) only where FINRA has no
#  reading - it disagreed with FINRA's own numbers by 3-300x on 11-25 (WHR 0.1 % vs 32 %, CHTR 45 % vs 9 %).
# ============================================================
import math
import os
import re


def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


RISK_PER_TRADE = _envf("RUNNER_RISK_PER_TRADE", 0.40) / 100       # loss at the effective stop, confidence 0.5
RISK_PER_TRADE_MAX = _envf("RUNNER_RISK_PER_TRADE_MAX", 0.60) / 100   # ... at confidence 1.0
VOL_STOP_MULT = _envf("RUNNER_RISK_VOL_STOP_MULT", 2.5)            # effective stop >= 2.5 daily sigmas (gaps, noise)
MAX_POS_DAILY_VOL = _envf("RUNNER_RISK_MAX_POS_VOL", 0.12) / 100   # one position's 1-sigma day <= 0.12 % of the book
SECTOR_CAP = _envf("RUNNER_RISK_SECTOR_CAP", 15) / 100             # one sector's NET <= 15 % of the bankroll
SECTOR_GROSS_CAP = _envf("RUNNER_RISK_SECTOR_GROSS", 30) / 100      # ... and its gross (longs + shorts) <= 30 %
SECTOR_NEW_MAX = int(_envf("RUNNER_RISK_SECTOR_NEW", 2))           # new positions per sector per session
CLUSTER_CORR = _envf("RUNNER_RISK_CLUSTER_CORR", 0.70)             # same-direction names this correlated = one bet
CLUSTER_MAX = int(_envf("RUNNER_RISK_CLUSTER_MAX", 2))             # positions per correlated cluster
NET_BETA_CAP = {False: _envf("RUNNER_RISK_NET_BETA", 45) / 100,     # |net beta-weighted exposure| / bankroll
                True: _envf("RUNNER_RISK_NET_BETA_DIALED", 80) / 100}  # ... once the regime call is dialed in
GROSS_CAP = _envf("RUNNER_RISK_GROSS_CAP", 110) / 100              # gross shares notional / bankroll
NEW_RISK_PER_DAY = _envf("RUNNER_RISK_NEW_PER_DAY", 2.5) / 100     # sum of new trades' risk at the stop, per session
POS_LOSS_CAP = _envf("RUNNER_RISK_POS_LOSS_CAP", 0.75) / 100       # an open position losing more is closed (risk-cap)
DAY_HALT = _envf("RUNNER_RISK_DAY_HALT", 2.0) / 100                # book down this much on the day -> no new entries
DD_WINDOW = int(_envf("RUNNER_RISK_DD_WINDOW", 90))                # high-water mark over the last 90 curve points
DD_STEPS = ((0.04, 1.00), (0.08, 0.70), (0.12, 0.50), (9.99, 0.35))   # drawdown -> scale on new risk
MIN_SHORT_PRICE = _envf("RUNNER_RISK_MIN_SHORT_PX", 10)            # no short sales under $10 ...
MIN_SHORT_CAP = _envf("RUNNER_RISK_MIN_SHORT_CAP", 2e9)            # ... or under a $2B market cap (borrow, squeezes)
LIQ_PCT = _envf("RUNNER_RISK_LIQ_PCT", 2.0) / 100                  # z205: one position <= 2 % of a day's average dollar volume
MIN_DV = _envf("RUNNER_RISK_MIN_DV", 250e3)                        # ... no new position under $250K traded a day
MIN_SHORT_DV = _envf("RUNNER_RISK_MIN_SHORT_DV", 10e6)             # ... no short sale under $10M a day
MAX_SHORT_SI = _envf("RUNNER_RISK_MAX_SHORT_SI", 20.0) / 100       # z206: no short sale where 20 %+ of the float is short
SQUEEZE_SHORT_SI = _envf("RUNNER_RISK_SQUEEZE_SHORT_SI", 10.0) / 100   # ... 10 %+ while shorts are being covered
MAX_SHORT_SO = _envf("RUNNER_RISK_MAX_SHORT_SO", 15.0) / 100       # z207: FINRA - 15 %+ of the shares outstanding short
MAX_SHORT_DTC = _envf("RUNNER_RISK_MAX_SHORT_DTC", 10.0)           # ... or 10+ days to cover (with DTC_MIN_SO+ short)
SQUEEZE_SHORT_SO = _envf("RUNNER_RISK_SQUEEZE_SHORT_SO", 8.0) / 100    # ... 8 %+ while shorts are being covered
SQUEEZE_SHORT_DTC = _envf("RUNNER_RISK_SQUEEZE_SHORT_DTC", 7.0)        # ... or 7+ days to cover
DTC_MIN_SO = _envf("RUNNER_RISK_DTC_MIN_SO", 5.0) / 100            # days to cover counts only with 5 %+ of the shares short
# z190: a fund is recognised by ANY sign of one. TRAPP2-1 (the macro book) lists only funds / ETPs among its US symbols
# (plus foreign listings, FX, crypto, futures, indices); on 11-16 a throttled Yahoo answer blanked their profiles (no
# name, isetf "false", no market cap) and the runner shorted GLD, SOXL (3x), TAN and WGMI as if they were stocks.
FUND_REPOS = {r.strip() for r in os.environ.get("RUNNER_RISK_FUND_REPOS", "TRAPP2-1").split(",") if r.strip()}
_LEVERAGED = re.compile(r"(?i)^(?=.*\b(?:etf|etn|shares|fund)\b).*\b(?:[1-4](?:\.\d+)?x|ultra(?:pro)?|inverse|leveraged|bull|bear)\b")
VOL_WIN, BETA_WIN, CORR_WIN = 60, 500, 60


# ------------------------------------------------------------------------------------------- statistics ----
def returns_by_date(dated):
    """[(date, close)] ascending -> {date: log return} (a return is dated by the day it ends)."""
    out = {}
    for (d0, c0), (d1, c1) in zip(dated, dated[1:]):
        if c0 and c1 and c0 > 0 and c1 > 0:
            r = math.log(c1 / c0)
            if abs(r) < math.log(2.5):                 # z185: a jump beyond x2.5 is a data question, not a move
                out[d1] = r
    return out


def _tail(rets, n):
    ks = sorted(rets)[-n:]
    return {k: rets[k] for k in ks}


def sigma(rets, n=VOL_WIN):
    """Daily volatility of the last n returns (None with fewer than 20)."""
    x = list(_tail(rets, n).values())
    if len(x) < 20:
        return None
    m = sum(x) / len(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def _paired(a, b, n):
    ks = sorted(set(a) & set(b))[-n:]
    return [a[k] for k in ks], [b[k] for k in ks]


def beta(rets, mkt, n=BETA_WIN):
    """The beta the risk book uses: two years against SPY, shrunk a third of the way to 1 (Blume) - a one-year beta
    swings too much to cap on (this year COP's was -0.67 over 250 sessions, +0.44 over 500). A structurally inverse
    product (raw beta below -0.5: SQQQ, SPXS) keeps its raw beta - held long, it is a short in this book."""
    b = beta_raw(rets, mkt, n)
    if b is None or b < -0.5:
        return b
    return (2.0 / 3.0) * b + 1.0 / 3.0


def beta_raw(rets, mkt, n=BETA_WIN):
    """OLS beta to the market over the last n common dates (None with fewer than 30)."""
    x, y = _paired(mkt, rets, n)
    if len(x) < 30:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    vx = sum((a - mx) ** 2 for a in x)
    if vx <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / vx


def corr(a, b, n=CORR_WIN):
    x, y = _paired(a, b, n)
    if len(x) < 30:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    if sx <= 0 or sy <= 0:
        return None
    return sum((p - mx) * (q - my) for p, q in zip(x, y)) / (sx * sy)


# ---------------------------------------------------------------------------------- the book's condition ----
def drawdown(curve, value_now, window=DD_WINDOW):
    """High-water mark over the last `window` daily book values (and now) -> {hwm, hwmDate, value, dd, scale}.
    A rolling mark: an old restatement or deposit falls out of it instead of setting the bar forever."""
    pts = [p for p in (curve or []) if isinstance(p, dict) and isinstance(p.get("value"), (int, float))][-window:]
    hwm, hwm_d = value_now, "now"
    for p in pts:
        if p["value"] > hwm:
            hwm, hwm_d = p["value"], p.get("date")
    dd = max(0.0, 1 - value_now / hwm) if hwm > 0 else 0.0
    scale = next(s for lim, s in DD_STEPS if dd < lim)
    return {"hwm": round(hwm, 2), "hwmDate": hwm_d, "value": round(value_now, 2), "dd": round(dd, 4), "scale": scale}


def day_change(curve, value_now, today):
    """Today's book change against the last close before today (None without one)."""
    prev = [p for p in (curve or []) if isinstance(p, dict) and str(p.get("date") or "") < today and isinstance(p.get("value"), (int, float))]
    if not prev or prev[-1]["value"] <= 0:
        return None
    return value_now / prev[-1]["value"] - 1


def position_exposure(pos, price):
    """(gross $, signed $) a position puts on: shares at their market value; options and spreads at the most they
    can lose (premium / max loss) - their dollar risk, which is what a sector cap should count."""
    if pos.get("instrument") == "option":
        g = float(pos.get("maxLoss") or pos.get("notional") or 0)
        # "direction" is the view (a put bought on a bearish idea is "short"; a put credit spread sold on a bullish
        # one is "long"), whatever the contract
        return g, (-g if pos.get("direction") == "short" else g)
    sh = float(pos.get("shares") or 0)
    px = price or pos.get("lastPrice") or pos.get("entryPrice") or 0
    g = abs(sh * float(px)) if sh else float(pos.get("notional") or 0)
    return g, (-g if pos.get("direction") == "short" else g)


class Book:
    """The open book as the risk checks see it: exposures by sector, net beta, the held names' returns (for
    clusters), the risk already added this session. Built once per pass; updated as trades are added."""

    def __init__(self, bankroll, regime_dialed=False, scale=1.0, halted=False, today=None):
        self.bankroll = float(bankroll)
        self.dialed = bool(regime_dialed)
        self.scale, self.halted, self.today = scale, halted, today
        self.gross = self.net = self.net_beta = self.option_risk = 0.0
        self.sector = {}                # sector -> {"gross", "n", "new"}
        self.held = []                  # (ticker, direction(+1/-1), returns by date)
        self.new_risk = 0.0
        self.new_count = 0
        self.rejects = {}
        self.long_beta_scale = 1.0      # z202: the long side's net beta cap x this (bot_runner.trend_beta_scale: a sell-off)
        self.long_beta_why = None

    def add(self, ticker, sector, direction, gross, signed, beta_v=None, rets=None, new_risk=0.0, new=False, instrument="shares"):
        s = self.sector.setdefault(sector or "Unknown", {"gross": 0.0, "net": 0.0, "n": 0, "new": 0})
        s["gross"] += gross
        s["net"] += signed
        s["n"] += 1
        if new:
            s["new"] += 1
            self.new_risk += new_risk
            self.new_count += 1
        if instrument == "option":
            self.option_risk += gross
        else:
            self.gross += gross
        self.net += signed
        self.net_beta += signed * (beta_v if beta_v is not None else 1.0)
        if rets:
            self.held.append((ticker, 1 if signed >= 0 else -1, rets))

    def cluster_of(self, rets, direction):
        """Held names in the candidate's direction whose returns move with it (corr >= CLUSTER_CORR)."""
        if not rets:
            return []
        out = []
        for t, d, r in self.held:
            if d != direction:
                continue
            c = corr(rets, r)
            if c is not None and c >= CLUSTER_CORR:
                out.append((t, round(c, 2)))
        return out

    def check(self, ticker, sector, direction, gross, signed, beta_v, rets, risk, instrument="shares"):
        """-> (ok, reason). Order: halt, sector caps and count, cluster, gross, net beta, new risk today. A trade that
        shrinks its sector's net exposure (a short against a long-heavy sector) passes the net cap - it is a hedge."""
        B = self.bankroll
        why = None
        sec = self.sector.get(sector or "Unknown", {"gross": 0.0, "net": 0.0, "n": 0, "new": 0})
        known = (sector or "Unknown") not in ("Unknown", "")
        net_after = sec["net"] + signed
        if self.halted:
            why = "daily halt"
        elif known and abs(net_after) > SECTOR_CAP * B and abs(net_after) > abs(sec["net"]):
            why = f"sector cap: {sector} net {sec['net'] / B:+.0%} {'+' if signed >= 0 else '-'} {gross / B:.1%} beyond ±{SECTOR_CAP:.0%}"
        elif known and sec["gross"] + gross > SECTOR_GROSS_CAP * B:
            why = f"sector gross cap: {sector} {sec['gross'] / B:.0%} + {gross / B:.1%} > {SECTOR_GROSS_CAP:.0%}"
        elif known and sec["new"] >= SECTOR_NEW_MAX:
            why = f"sector new-trade limit: {SECTOR_NEW_MAX} new {sector} today"
        else:
            cl = self.cluster_of(rets, 1 if signed >= 0 else -1)
            if len(cl) >= CLUSTER_MAX:
                why = "cluster: moves with " + ", ".join(f"{t} ({c:+.2f})" for t, c in cl[:3])
            elif instrument != "option" and self.gross + gross > GROSS_CAP * B:
                why = f"gross cap: {(self.gross + gross) / B:.0%} > {GROSS_CAP:.0%}"
            else:
                nb = self.net_beta + signed * (beta_v if beta_v is not None else 1.0)
                cap = NET_BETA_CAP[self.dialed] * B * (self.long_beta_scale if nb > 0 else 1.0)    # z202: tighter long side in a sell-off
                if abs(nb) > cap and abs(nb) > abs(self.net_beta):
                    why = (f"net beta cap: {nb / B:+.0%} (cap ±{NET_BETA_CAP[self.dialed]:.0%}, regime {'dialed in' if self.dialed else 'not dialed in'}"
                           + (f"; long side +{NET_BETA_CAP[self.dialed] * self.long_beta_scale:.0%} - {self.long_beta_why}" if nb > 0 and self.long_beta_scale < 1.0 else "")
                           + ")")
                elif self.new_risk + risk > NEW_RISK_PER_DAY * B:
                    why = f"new-risk cap: {(self.new_risk + risk) / B:.2%} of the book today > {NEW_RISK_PER_DAY:.1%}"
        if why:
            key = why.split(":")[0]
            self.rejects[key] = self.rejects.get(key, 0) + 1
            return False, why
        return True, ""

    def summary(self):
        B = self.bankroll or 1
        sect = sorted(((k, v["net"], v["gross"]) for k, v in self.sector.items()), key=lambda x: -abs(x[1]))
        return {"gross": round(self.gross / B, 4), "net": round(self.net / B, 4), "netBeta": round(self.net_beta / B, 4),
                "optionRisk": round(self.option_risk / B, 4),
                "bySector": {k: round(n / B, 4) for k, n, g in sect},                      # net, signed
                "bySectorGross": {k: round(g / B, 4) for k, n, g in sect},
                "overSectorCap": [k for k, n, g in sect if abs(n) > SECTOR_CAP * B and k not in ("Unknown", "")],
                "newRisk": round(self.new_risk / B, 4), "newTrades": self.new_count, "rejects": dict(self.rejects),
                "longBetaScale": self.long_beta_scale, "longBetaWhy": self.long_beta_why}


# ------------------------------------------------------------------------------------------------ sizing ----
def per_trade_risk(conf):
    c = max(0.0, min(1.0, ((conf or 0.5) - 0.5) / 0.5))
    return RISK_PER_TRADE + (RISK_PER_TRADE_MAX - RISK_PER_TRADE) * c


def size_shares(bankroll, conf, stop_pct, sig, scale=1.0, cap_notional=None):
    """-> (notional, risk $, effective stop, why). The loss at the effective stop - the planned stop, or 2.5 daily
    sigmas if the name moves more than that - is the budget; a volatile name gets a smaller position, a 25 % core
    stop gets a smaller one than a 7 % short-term stop. Also: one position's 1-sigma day <= MAX_POS_DAILY_VOL."""
    eff = max(stop_pct or 0.0, VOL_STOP_MULT * sig if sig else 0.0)
    if eff <= 0:
        eff = 0.10                                    # no stop and no history: assume 10 %
    budget = bankroll * per_trade_risk(conf) * scale
    n1 = budget / eff
    n2 = (bankroll * MAX_POS_DAILY_VOL / sig) if sig else n1
    notional = min(n1, n2, cap_notional if cap_notional is not None else n1)
    why = "risk budget" if notional == n1 else ("volatility cap" if notional == n2 else "size cap")
    return notional, notional * eff, eff, why


def fund_reason(u):
    """Why a universe row is a fund / ETP and not a single stock, or "" (z190: any one sign is enough - the ETF flag,
    the fund flag, a US symbol from the macro book, a leveraged / inverse product's name)."""
    t = str(u.get("ticker") or "").upper()
    if str(u.get("isetf") or "").lower() == "true":
        return "a fund (ETF)"
    if str(u.get("isfund") or "").lower() == "true":
        return "a fund"
    if t and "." not in t and str(u.get("repo") or "") in FUND_REPOS:
        return f"an ETF / ETP from the macro book ({u.get('repo')})"
    if _LEVERAGED.search(str(u.get("name") or "")):
        return "a leveraged / inverse product"
    return ""


# ------------------------------------------------------------------------------------------- z192: ETPs ----
# A leveraged / inverse exchange-traded product resets its exposure every session: SQQQ is -3x the Nasdaq-100 for ONE
# day. Held longer it lags L x the move by the volatility drag, about (L^2 - L) / 2 x sigma^2 a session (sigma = the
# underlying's daily vol: SQQQ ~0.10 % a session at a 1.3 % Nasdaq, ~2 % a month) - a hold needs a live signal every
# session. Its market-wide signals carry its sign (a risk-off tape is a buy signal for a bear fund), and it has no
# company fundamentals (before z192 a bear fund's T-bill income scored SQQQ +0.5 "fundamentals" and +0.5 "health").
_ETP_KNOWN = {   # (signed daily leverage, what it tracks) - names can be blank on a throttled night
    "SQQQ": (-3, "equity"), "TQQQ": (3, "equity"), "QID": (-2, "equity"), "QLD": (2, "equity"), "PSQ": (-1, "equity"),
    "SPXS": (-3, "equity"), "SPXL": (3, "equity"), "SPXU": (-3, "equity"), "UPRO": (3, "equity"),
    "SDS": (-2, "equity"), "SSO": (2, "equity"), "SH": (-1, "equity"),
    "SDOW": (-3, "equity"), "UDOW": (3, "equity"), "DXD": (-2, "equity"), "DDM": (2, "equity"), "DOG": (-1, "equity"),
    "TZA": (-3, "equity"), "TNA": (3, "equity"), "TWM": (-2, "equity"), "UWM": (2, "equity"), "RWM": (-1, "equity"),
    "SOXS": (-3, "equity"), "SOXL": (3, "equity"), "TECS": (-3, "equity"), "TECL": (3, "equity"),
    "FAZ": (-3, "equity"), "FAS": (3, "equity"), "LABD": (-3, "equity"), "LABU": (3, "equity"),
    "YANG": (-3, "equity"), "YINN": (3, "equity"), "NUGT": (2, "equity"), "DUST": (-2, "equity"),
    "JNUG": (2, "equity"), "JDST": (-2, "equity"),
    "TMF": (3, "rates"), "TMV": (-3, "rates"), "TBT": (-2, "rates"), "UBT": (2, "rates"),
    "UCO": (2, "commodity"), "SCO": (-2, "commodity"), "BOIL": (2, "commodity"), "KOLD": (-2, "commodity"),
    "UGL": (2, "commodity"), "GLL": (-2, "commodity"), "AGQ": (2, "commodity"), "ZSL": (-2, "commodity"),
    "UVXY": (1.5, "volatility"), "SVXY": (-0.5, "volatility"),
}
_FUND_WORD = re.compile(r"(?i)\b(?:etf|etn|fund|trust)\b|shares\b")          # "ProShares", "Direxion ... Shares", "... ETF"
_MULT = re.compile(r"(?i)(?<![\d.])([1-4](?:\.\d+)?)\s*x\b")                 # 2X, 3x, 1.5x
_INVERSE = re.compile(r"(?i)\bultra\s*(?:pro\s*)?short|\bbear\b|\binverse\b|(?<!ultra-)(?<!ultra )\bshort\b"
                      r"(?![\s-]*(?:term|duration|maturity|dated|income|bond|treasury|government|corporate|muni))")
_UNDERLYING = (("volatility", re.compile(r"(?i)\bvix\b|volatility")),
               ("rates", re.compile(r"(?i)treasur|\bbonds?\b|\byear\b|\btips\b|high yield")),
               ("commodity", re.compile(r"(?i)crude|\bwti\b|\bbrent\b|natural gas|\bgold\b(?!\s*miners)|\bsilver\b|copper|commodit")),
               ("currency", re.compile(r"(?i)\bdollar\b|\beuro\b|\byen\b|currenc")))


def etp_exposure(u):
    """A leveraged / inverse ETP's daily exposure: {"lev": signed leverage, "underlying": equity | rates | commodity |
    volatility | currency, "basis": table | name}, or None for a stock or a plain (1x long) fund. Known symbols first,
    then the name - only a fund's name (ProShares' Ultra = 2x, UltraPro = 3x; else an explicit multiple; a short /
    bear / inverse fund is -1x unless it says more; "Short-Term", "Ultra-Short Income" are durations, not shorts)."""
    t = str(u.get("ticker") or "").upper()
    if t in _ETP_KNOWN:
        lev, und = _ETP_KNOWN[t]
        return {"lev": float(lev), "underlying": und, "basis": "table"}
    name = str(u.get("name") or "")
    if not name or not (fund_reason(u) or _FUND_WORD.search(name)):
        return None
    m = _MULT.search(name)
    pro = "proshares" in name.lower()
    mag = float(m.group(1)) if m else (3.0 if pro and re.search(r"(?i)\bultra\s*pro", name) else
                                       2.0 if pro and re.search(r"(?i)\bultra", name) else None)
    inverse = bool(_INVERSE.search(name))
    if mag is None:
        if not inverse:
            return None
        mag = 1.0
    if not inverse and mag == 1.0:
        return None                                    # a plain 1x long fund
    und = next((k for k, rx in _UNDERLYING if rx.search(name)), "equity")
    return {"lev": -mag if inverse else mag, "underlying": und, "basis": "name"}


def etp_drag(lev, etp_daily_vol):
    """The volatility drag of holding a daily-reset product one more session, as a fraction: (L^2 - L) / 2 x the
    underlying's variance, the underlying's vol read off the product's own (sigma_etp / |L|)."""
    try:
        lev, v = float(lev), float(etp_daily_vol)
    except (TypeError, ValueError):
        return None
    if not lev or not math.isfinite(v) or v <= 0:
        return None
    return (lev * lev - lev) / 2.0 * (v / abs(lev)) ** 2


def liquidity(u):
    """z205 -> (notional cap, skip reason). The 20-session average dollar volume in USD (u["dv20USD"]): under MIN_DV a
    day -> (0, why) - no new position; else (LIQ_PCT x it, None). Unknown (no file, a future / currency / index, whose
    volume x price is no notional) -> (None, None): no liquidity rule."""
    dv = u.get("dv20USD")
    if dv is None or not equity_like(u):
        return None, None
    if dv < MIN_DV:
        return 0.0, f"thin: ${dv / 1e3:,.0f}K traded a day on average (< ${MIN_DV / 1e3:,.0f}K)"
    return LIQ_PCT * dv, None


_NOT_EQUITY = re.compile(r"^\^|=F$|=X$|-USD$|^[A-Z]{1,6}\d{6}[CP]\d{8}$")


def equity_like(u):
    """A stock or an ETF (what trades on an exchange with a share volume): not a future, currency, index, crypto pair or
    option contract - their volume x price is not what the position would be a share of."""
    t = str(u.get("ticker") or "").upper()
    cls = str(u.get("assetClass") or "Equity").strip().lower()
    return cls in ("equity", "etf", "") and not _NOT_EQUITY.search(t)


def shortable(u, covering=False):
    """A short sale needs a liquid, borrowable single stock (§1d 3): not a fund or an ETP, >= $10, and a market cap we
    KNOW is >= $2B (in USD - z190: an unknown cap no longer passes). Paper trading has no borrow desk; this keeps the
    shorts realistic. A name that fails gets a put instead, when it has an option chain. z205: and, when its average
    dollar volume is known, >= MIN_SHORT_DV a day. z206: and not a crowded short - short interest (Yahoo's short % of
    float) under MAX_SHORT_SI, under SQUEEZE_SHORT_SI while the tape shows shorts being covered (covering=True).
    z207: FINRA's official reading first - crowded_short()."""
    why = fund_reason(u)
    if why:
        return False, f"{why} - no single-name short"
    px = u.get("price") or 0
    mc = u.get("marketcapUSD") if u.get("marketcapUSD") is not None else u.get("marketcap")
    if px < MIN_SHORT_PRICE:
        return False, f"price ${px:.2f} < ${MIN_SHORT_PRICE:.0f}"
    if mc is None:
        return False, "market cap unknown - can't confirm a $2B+ company"
    if mc < MIN_SHORT_CAP:
        return False, f"market cap ${mc / 1e9:.1f}B < ${MIN_SHORT_CAP / 1e9:.0f}B"
    dv = u.get("dv20USD")
    if dv is not None and dv < MIN_SHORT_DV:
        return False, f"${dv / 1e6:.1f}M traded a day < ${MIN_SHORT_DV / 1e6:.0f}M (borrow, a clean cover)"
    crowded, why = crowded_short(u, covering)
    if crowded:
        return False, why
    return True, ""


def crowded_short(u, covering=False):
    """(crowded?, why). z207: FINRA's reading first - shares short / shares outstanding (u["shortPctSO"]) and days to
    cover (u["daysToCover"]); Yahoo's short % of float only where FINRA has no share of shares to give."""
    when = " while shorts are being covered" if covering else ""
    so, dtc = u.get("shortPctSO"), u.get("daysToCover")
    dbar = SQUEEZE_SHORT_DTC if covering else MAX_SHORT_DTC
    if so is not None and 0 <= so <= 1.0:
        bar = SQUEEZE_SHORT_SO if covering else MAX_SHORT_SO
        st = f"FINRA, settled {u['shortSettle']}" if u.get("shortSettle") else "FINRA"
        if so >= bar:
            return True, f"{so:.0%} of the shares outstanding sold short ({st}; >= {bar:.0%}{when}) - a crowded short"
        if dtc is not None and dtc >= dbar and so >= DTC_MIN_SO:
            return True, (f"{dtc:.1f} days to cover with {so:.0%} of the shares short ({st}; >= {dbar:.0f} days{when}) "
                          "- a crowded short")
        return False, ""
    si = u.get("shortPctFloat")
    bar = SQUEEZE_SHORT_SI if covering else MAX_SHORT_SI
    if si is not None and 0 < si <= 1.0:
        if si >= bar:
            return True, (f"{si:.0%} of the float already sold short (>= {bar:.0%}{when}) - a crowded short "
                          "(Yahoo's figure; no FINRA share of shares)")
        if dtc is not None and dtc >= dbar and si >= DTC_MIN_SO:
            return True, (f"{dtc:.1f} days to cover (FINRA) with {si:.0%} of the float short (Yahoo; >= {dbar:.0f} days{when}) "
                          "- a crowded short")
    return False, ""


def pos_loss_capped(pos, bankroll):
    """True when an open position's loss exceeds POS_LOSS_CAP of the bankroll (shares only - an option's premium is
    already its most it can lose)."""
    if pos.get("instrument") == "option":
        return False
    return (pos.get("pnl") or 0) < -POS_LOSS_CAP * bankroll


def knobs():
    return {"riskPerTrade": [RISK_PER_TRADE, RISK_PER_TRADE_MAX], "volStopMult": VOL_STOP_MULT, "maxPosDailyVol": MAX_POS_DAILY_VOL,
            "sectorCap": SECTOR_CAP, "sectorGrossCap": SECTOR_GROSS_CAP, "sectorNewMax": SECTOR_NEW_MAX, "clusterCorr": CLUSTER_CORR, "clusterMax": CLUSTER_MAX,
            "netBetaCap": {"notDialedIn": NET_BETA_CAP[False], "dialedIn": NET_BETA_CAP[True]}, "grossCap": GROSS_CAP,
            "newRiskPerDay": NEW_RISK_PER_DAY, "posLossCap": POS_LOSS_CAP, "dayHalt": DAY_HALT,
            "drawdownSteps": [list(x) for x in DD_STEPS], "ddWindow": DD_WINDOW,
            "liquidity": {"pctOfDay": LIQ_PCT, "minDollarVolume": MIN_DV, "minShortDollarVolume": MIN_SHORT_DV},
            "shortCrowding": {"maxShortPctFloat": MAX_SHORT_SI, "whileCovering": SQUEEZE_SHORT_SI,
                              "finra": {"maxShortPctShares": MAX_SHORT_SO, "maxDaysToCover": MAX_SHORT_DTC,
                                        "whileCovering": {"pctShares": SQUEEZE_SHORT_SO, "daysToCover": SQUEEZE_SHORT_DTC},
                                        "daysToCoverNeedsPct": DTC_MIN_SO}}}
