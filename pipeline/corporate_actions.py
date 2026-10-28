#!/usr/bin/env python3
# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   pipeline/corporate_actions.py
#  z166 - SPLITS AND DIVIDENDS for the bot's book.
#  2026-10-27: PARA did a 1-for-10 reverse split at the open. The book still held 5,982.84 pre-split shares
#  short at $0.54, read the post-split $3.63 as a 572 % loss, hit its $0.68 stop and booked -$18,486.97. On the
#  adjusted basis (598.28 shares at $5.40) that cover was a +$1,058.96 gain.
#  Source: Yahoo's chart events (splits with numerator / denominator and ex-date, dividends with amount and
#  ex-date), per ticker, cached in data/corporate_actions.json and refreshed daily - or at once when a held
#  name jumps by a split-like ratio. Applied every pass, before any position is marked:
#    SPLIT     open position: shares x r; entry, stop, target, peak and last prices / r (value unchanged);
#              a trade CLOSED on or after the ex-date on the old basis is RESTATED (exit P/L on the adjusted
#              basis) and listed in state["restatements"]. r = new shares per old share (1-for-10 -> 0.1).
#    DIVIDEND  held through the ex-date (entered before it, still held at its open): long +amount x shares,
#              short -amount x shares, booked as income on the trade (dividends[]); the ledger counts it in
#              the bankroll (trade_sync.ledger_bankroll), and the app does the same.
#  Options (OCC adjusts the contract) and leveraged-ETF lines are flagged, not adjusted.
# ============================================================
import json
import math
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{t}?range={rng}&interval=1d&events=div%2Csplits"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
COMMON_RATIOS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 75, 80, 100, 150, 200, 250)
REFRESH_HOURS = 20
RESTATE_WINDOW_DAYS = None          # every trade the book ever made (it started in 2026; Yahoo's events go back a year)
PRICE_KEYS = ("entryPrice", "entryPriceLocal", "stopPrice", "targetPrice", "peakPrice", "troughPrice", "trailStop")


def _et_date(epoch):
    try:
        from zoneinfo import ZoneInfo
        return datetime.fromtimestamp(int(epoch), timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()
    except Exception:
        return datetime.fromtimestamp(int(epoch) - 4 * 3600, timezone.utc).date().isoformat()


def split_like(ratio, tol=0.15):
    """The common split factor a price ratio is close to (as new shares per old share), or None.
    A price that goes x10 is a 1-for-10 reverse split -> 0.1."""
    if not ratio or ratio <= 0 or not math.isfinite(ratio):
        return None
    if 0.55 < ratio < 1.8:
        return None
    up = ratio >= 1
    big = ratio if up else 1 / ratio
    for f in COMMON_RATIOS:
        if abs(big - f) / f <= tol:
            return 1 / f if up else float(f)
    return None


def parse_chart(j):
    """Yahoo chart JSON -> {"splits": [{exDate, ratio, label}], "dividends": [{exDate, amount}]}."""
    res = (((j or {}).get("chart") or {}).get("result") or [None])[0] or {}
    ev = res.get("events") or {}
    splits, divs = [], []
    for v in (ev.get("splits") or {}).values():
        try:
            num, den = float(v.get("numerator")), float(v.get("denominator"))
        except (TypeError, ValueError):
            continue
        if num > 0 and den > 0 and abs(num - den) > 1e-9:
            splits.append({"exDate": _et_date(v.get("date")), "ratio": round(num / den, 10), "label": f"{num:g}-for-{den:g}"})
    for v in (ev.get("dividends") or {}).values():
        try:
            amt = float(v.get("amount"))
        except (TypeError, ValueError):
            continue
        if amt > 0:
            divs.append({"exDate": _et_date(v.get("date")), "amount": round(amt, 6)})
    cur = (res.get("meta") or {}).get("currency")
    return {"splits": sorted(splits, key=lambda x: x["exDate"]), "dividends": sorted(divs, key=lambda x: x["exDate"]), "currency": cur}


def fetch_yahoo(ticker, rng="1y", timeout=20):
    t = str(ticker).upper()
    if t.startswith(("FX:", "^")) or t.endswith(("=X", "=F", "-USD")):
        return {"splits": [], "dividends": [], "currency": None}
    for host in ("query1", "query2"):
        try:
            req = urllib.request.Request(CHART.format(t=t, rng=rng).replace("query1", host), headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return parse_chart(json.loads(r.read().decode("utf-8")))
        except Exception:
            time.sleep(0.6)
    return None


class Store:
    """data/corporate_actions.json: {asOf, tickers: {T: {fetchedAt, splits, dividends, currency}}}."""

    def __init__(self, path, fetch=None, now=None):
        self.path, self.fetch = Path(path), (fetch or (lambda t: fetch_yahoo(t)))
        self.now = now or (lambda: datetime.now(timezone.utc))
        try:
            self.doc = json.loads(self.path.read_text())
        except Exception:
            self.doc = {"tickers": {}}
        self.doc.setdefault("tickers", {})
        self.changed = False

    def get(self, ticker, force=False):
        t = str(ticker).upper()
        cur = self.doc["tickers"].get(t)
        stale = True
        if cur and cur.get("fetchedAt"):
            try:
                age = (self.now() - datetime.fromisoformat(cur["fetchedAt"])).total_seconds() / 3600
                stale = age >= REFRESH_HOURS
            except Exception:
                pass
        if force or stale or not cur:
            got = self.fetch(t)
            if got is not None:
                cur = dict(got, fetchedAt=self.now().isoformat(timespec="seconds"))
                self.doc["tickers"][t] = cur
                self.changed = True
        return cur or {"splits": [], "dividends": []}

    def save(self):
        if self.changed:
            self.doc["asOf"] = self.now().isoformat(timespec="seconds")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.doc, indent=1))
        return self.changed


def _applied(t, kind, ex):
    return any(a.get("type") == kind and a.get("exDate") == ex for a in (t.get("corporateActions") or []))


def _shares(t):
    try:
        return float(t.get("shares") or 0)
    except (TypeError, ValueError):
        return 0.0


def apply_split(t, s, today, restate_fn=None):
    """Adjust one trade for split s (r = new shares per old). Returns a log line or None."""
    r = float(s["ratio"])
    ex = s["exDate"]
    inst = t.get("instrument") or "shares"
    if inst != "shares":
        if not _applied(t, "split-flag", ex):
            t.setdefault("corporateActions", []).append({"type": "split-flag", "exDate": ex, "ratio": r, "label": s.get("label"),
                                                         "note": "option / leveraged line - the contract is adjusted by OCC or the fund; not changed here",
                                                         "at": today})
            return f"{t.get('ticker')}: {s.get('label')} split on {ex} - {inst} line flagged for review (not adjusted)"
        return None
    before = {k: t.get(k) for k in ("shares", "sharesAtExit") + PRICE_KEYS + ("lastPrice", "pnl", "realizedPL")}
    closed = t.get("status") == "closed"
    for k in ("shares", "sharesAtExit"):
        if t.get(k) not in (None, ""):
            t[k] = round(float(t[k]) * r, 6)
    for k in PRICE_KEYS:
        if t.get(k) not in (None, ""):
            t[k] = round(float(t[k]) / r, 6)
    if not closed and t.get("lastPrice") not in (None, ""):
        t["lastPrice"] = round(float(t["lastPrice"]) / r, 6)            # the last mark was on the old basis
    rec = {"type": "split", "exDate": ex, "ratio": r, "label": s.get("label"), "at": today, "before": before}
    line = f"{t.get('ticker')}: {s.get('label')} split on {ex} - {before.get('shares')} -> {t.get('shares')} shares, entry {before.get('entryPrice')} -> {t.get('entryPrice')}"
    if closed and restate_fn:
        new_pnl = restate_fn(t)
        if new_pnl is not None:
            new_pnl = round(float(new_pnl), 2)
            old = t.get("pnl")
            t["pnl"] = new_pnl
            if t.get("realizedPL") is not None:
                t["realizedPL"] = new_pnl
            try:
                notional = float(t.get("notional") or 0)
                if notional:
                    t["returnPct"] = round(new_pnl / notional * 100, 2)
            except (TypeError, ValueError):
                pass
            t["won"] = new_pnl > 0
            rec.update({"restated": True, "pnlBefore": old, "pnlAfter": new_pnl, "delta": round(new_pnl - float(old or 0), 2)})
            t["exitNote"] = (f"closed on the old share basis the day of a {s.get('label')} split ({ex}); restated on the adjusted basis "
                             f"(z166): P/L {old} -> {new_pnl}")
            line += f" · RESTATED P/L {old} -> {new_pnl}"
    t.setdefault("corporateActions", []).append(rec)
    return line


def shares_at(t, ex):
    """Shares held at the ex-date's open: the current count plus anything trimmed on or after the ex-date
    (sold after the record point still earned it). Trims without a share count can't be placed; ignored."""
    sh = _shares(t)
    if t.get("status") == "closed":
        try:
            sh = float(t.get("sharesAtExit") or t.get("shares") or 0)
        except (TypeError, ValueError):
            sh = 0.0
    for x in (t.get("trims") or []):
        if isinstance(x, dict) and str(x.get("date") or "")[:10] >= ex:
            try:
                sh += float(x.get("shares") or 0)
            except (TypeError, ValueError):
                pass
    return sh


def apply_dividend(t, d, today, fx=1.0):
    ex, amt = d["exDate"], float(d["amount"])
    if _applied(t, "dividend", ex):
        return None
    sh = shares_at(t, ex)
    if sh <= 0:
        return None
    sign = -1 if t.get("direction") == "short" else 1
    cash = round(sign * amt * sh * (fx or 1.0), 2)
    t.setdefault("dividends", []).append({"exDate": ex, "perShare": amt, "shares": round(sh, 6), "amount": cash, "fx": fx, "bookedOn": today,
                                         "note": "paid by the short" if sign < 0 else "income to the long"})
    t["dividendIncome"] = round(sum(float(x.get("amount") or 0) for x in t["dividends"]), 2)
    t.setdefault("corporateActions", []).append({"type": "dividend", "exDate": ex, "perShare": amt, "amount": cash, "at": today})
    return f"{t.get('ticker')}: dividend {amt:g}/sh ex {ex} x {sh:,.2f} sh {'(short pays)' if sign < 0 else ''} = {cash:+,.2f}"


def process(state, store, today, restate_fn=None, fx_of=None, log=print, window_days=RESTATE_WINDOW_DAYS):
    """Every trade still open, and every trade closed in the last `window_days`, against its ticker's splits and
    dividends. Splits first (a dividend after a split is paid on the new share count). Returns the log lines."""
    from datetime import date, timedelta
    cutoff = (date.fromisoformat(today) - timedelta(days=window_days)).isoformat() if window_days else ""
    lines = []
    for t in state.get("trades") or []:
        tk = str(t.get("ticker") or "").upper()
        if not tk or tk.startswith(("FX:", "^")):
            continue
        open_ = t.get("status") == "open"
        if not open_ and str(t.get("exitDate") or "")[:10] < cutoff:
            continue
        ev = store.get(tk)
        entry = str(t.get("entryDate") or "")[:10]
        last = today if open_ else str(t.get("exitDate") or today)[:10]
        for s in ev.get("splits") or []:
            if entry < s["exDate"] <= last and not _applied(t, "split", s["exDate"]) and not _applied(t, "split-flag", s["exDate"]):
                ln = apply_split(t, s, today, restate_fn)
                if ln:
                    lines.append(ln)
                    if (t.get("corporateActions") or [{}])[-1].get("restated"):
                        state.setdefault("restatements", []).append({"id": t.get("id"), "ticker": tk, "exDate": s["exDate"], "split": s.get("label"),
                                                                     "pnlBefore": t["corporateActions"][-1]["pnlBefore"], "pnlAfter": t["pnl"],
                                                                     "delta": t["corporateActions"][-1]["delta"], "at": today})
        if (t.get("instrument") or "shares") == "shares":
            fx = 1.0
            if fx_of and ev.get("currency") and ev.get("currency") != "USD":
                fx = fx_of(ev.get("currency")) or 1.0
            for d in ev.get("dividends") or []:
                if entry < d["exDate"] <= last:
                    ln = apply_dividend(t, d, today, fx)
                    if ln:
                        lines.append(ln)
    for ln in lines:
        log("  corporate action: " + ln)
    return lines
