"""
trade_sync.py — one merge rule for bot trades across the runner, Supabase and
the app (z112). Shared by bot_runner.py (pull) and sync_bot_to_supabase.py (push).

The same trade can be changed in two places (the runner's journal in this repo,
and the app - which writes bot_trades in Supabase directly with its anon key).
Which version wins:
  1. a CLOSED version beats an OPEN one (a close is never undone by a stale copy)
  2. more trims beats fewer (trims are history; never drop one)
  3. otherwise the later `updatedAt` wins (stamped by whoever changed the trade)
  4. no stamps on either side -> keep the local (repo) version

z117 - the ledger's honesty checks (mirrored in app.js):
  * repair_pnl: a closed share trade whose pnl contradicts its own entry / exit /
    shares while realizedPL matches them to the cent takes realizedPL. The
    original stays in pnlRecorded and ledgerFix says why. (21 trades closed
    2026-10-04 carried pnl from a stale mark: booked -$845.48, real +$4,852.46.)
  * ledger_asof: the ledger as of a date - capital, trims and exits dated on or
    before it - so a stored bankroll is reconciled against what had been
    realized BY ITS OWN DATE, never against later events.
"""
from datetime import datetime, timezone

STATUS_FIELDS = ("status", "exitDate", "exitPrice", "exitReason", "pnl", "realizedPL", "won", "returnPct",
                 "trims", "realizedTrim", "partialSells", "shares", "notional", "sharesAtExit", "closedBy",
                 "holdDays", "updatedAt", "pnlRecorded", "ledgerFix")
# Money in / out of the book that is not trading P/L: deposits, withdrawals and the
# dated opening adjustment. Same set in app.js (botLedgerBankroll, botNetCapital).
CAPITAL_TYPES = ("deposit", "withdrawal", "adjustment")
PNL_TOL = 0.05      # dollars; realizedPL and the price arithmetic are both rounded to cents


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ntrims(t):
    return len(t.get("trims") or []) + len(t.get("partialSells") or [])


def winner(local, remote):
    """-> 'local' | 'remote' for two versions of the same trade id."""
    # z116: 'void' (a trade from a runner run whose state was never saved) is final
    if remote.get("status") == "void" and local.get("status") != "void":
        return "remote"
    if local.get("status") == "void":
        return "local"
    lc, rc = local.get("status") == "closed", remote.get("status") == "closed"
    if rc and not lc:
        return "remote"
    if lc and not rc:
        return "local"
    if _ntrims(remote) > _ntrims(local):
        return "remote"
    if _ntrims(local) > _ntrims(remote):
        return "local"
    lu, ru = str(local.get("updatedAt") or ""), str(remote.get("updatedAt") or "")
    if ru and ru > lu:
        return "remote"
    return "local"


def exit_realized(t):
    """Realized at the exit itself, trims excluded (they're counted on their own).
    Runner semantics: trims[] + realizedTrim folded into pnl at close.
    App legacy semantics: partialSells[] kept OUT of pnl."""
    pnl = float(t.get("pnl") or 0)
    if t.get("trims") and t.get("realizedTrim") is not None:
        return pnl - float(t.get("realizedTrim") or 0)
    return pnl


def trim_total(t):
    tr = t.get("trims") or t.get("partialSells") or []
    return sum(float((x or {}).get("realized") or 0) for x in tr)


def ledger_bankroll(state, start):
    """start + capital in/out + every trim + every closed trade's exit P/L.
    Same rule as the app (botLedgerBankroll) - so both always agree."""
    trades = state.get("trades") or []
    capital = sum(float(x.get("amount") or 0) for x in (state.get("transactions") or [])
                  if x.get("type") in CAPITAL_TYPES)
    trims = sum(trim_total(t) for t in trades)
    closed = sum(exit_realized(t) for t in trades if t.get("status") == "closed")
    return round(start + capital + trims + closed, 2)


def ledger_asof(state, start, asof, exclude_ids=()):
    """The ledger AS OF a date (YYYY-MM-DD): start + capital entries dated on/before
    it + trims dated on/before it + exit P/L of trades closed on/before it. Undated
    trims can't be placed in time and are left out here (they still count in
    ledger_bankroll)."""
    d = str(asof or "")[:10]
    cap = sum(float(x.get("amount") or 0) for x in (state.get("transactions") or [])
              if x.get("type") in CAPITAL_TYPES and x.get("id") not in exclude_ids
              and str(x.get("ts") or "")[:10] <= d)
    real = 0.0
    for t in state.get("trades") or []:
        for x in (t.get("trims") or t.get("partialSells") or []):
            if (x or {}).get("date") and str(x["date"])[:10] <= d:
                real += float(x.get("realized") or 0)
        if t.get("status") == "closed" and t.get("exitDate") and str(t["exitDate"])[:10] <= d:
            real += exit_realized(t)
    return round(start + cap + real, 2)


def arith_exit_pl(t):
    """sign x shares x (exit - entry) x leverage - fees (+ realizedTrim under runner
    trims semantics) for a CLOSED plain share trade; None when it can't be derived
    from the record alone (options, leveraged ETFs, missing prices or shares)."""
    if t.get("status") != "closed" or (t.get("instrument") or "shares") != "shares":
        return None
    try:
        e, x = float(t.get("entryPrice") or 0), float(t.get("exitPrice") or 0)
        sh = float(t.get("sharesAtExit") or 0) or float(t.get("shares") or 0)
        lev, fees = float(t.get("leverage") or 1), float(t.get("fees") or 0)
    except (TypeError, ValueError):
        return None
    if e <= 0 or x <= 0 or sh <= 0:
        return None
    v = (-1.0 if t.get("direction") == "short" else 1.0) * (x - e) * sh * lev - fees
    if t.get("trims") and t.get("realizedTrim") is not None:
        v += float(t.get("realizedTrim") or 0)
    return v


def pnl_contradicts_prices(t):
    """True only when TWO witnesses agree against pnl: the record's own prices and
    its realizedPL match each other (to the cent), and pnl doesn't."""
    a, rp = arith_exit_pl(t), t.get("realizedPL")
    if a is None or not isinstance(rp, (int, float)) or (t.get("direction") == "short" and t.get("dividendCarry")):
        return False
    return abs(rp - a) <= PNL_TOL and abs(float(t.get("pnl") or 0) - rp) > PNL_TOL


def repair_pnl(trades, now=None):
    """-> [(id, pnl_was, pnl_now)]. Idempotent; stamps updatedAt so the repaired
    version wins in Supabase and on every device (winner rule 3)."""
    now, out = now or now_iso(), []
    for t in trades:
        if not pnl_contradicts_prices(t):
            continue
        was, new = t.get("pnl"), round(float(t["realizedPL"]), 2)
        if t.get("pnlRecorded") is None:
            t["pnlRecorded"] = was
        t["pnl"], t["won"] = new, new > 0
        sh = float(t.get("sharesAtExit") or 0) or float(t.get("shares") or 0)
        cap = sh * float(t.get("entryPrice") or 0)
        if cap > 0:
            t["returnPct"] = round(new / cap * 100, 2)
        t["ledgerFix"] = {"at": now, "pnlWas": was, "pnlNow": new,
                          "rule": "pnl contradicted entry/exit/shares; realizedPL matched them (z117)"}
        t["updatedAt"] = now
        out.append((t.get("id"), float(was or 0), new))
    return out


def is_runner_id(tid):
    """Runner-made trade ids look like TICKER-YYYY-MM-DD-runner-<epoch>."""
    return "-runner-" in str(tid or "")
