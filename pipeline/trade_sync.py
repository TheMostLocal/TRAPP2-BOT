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
"""
from datetime import datetime, timezone

STATUS_FIELDS = ("status", "exitDate", "exitPrice", "exitReason", "pnl", "realizedPL", "won", "returnPct",
                 "trims", "realizedTrim", "partialSells", "shares", "notional", "sharesAtExit", "closedBy",
                 "holdDays", "updatedAt")


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
                  if x.get("type") in ("deposit", "withdrawal", "adjustment"))
    trims = sum(trim_total(t) for t in trades)
    closed = sum(exit_realized(t) for t in trades if t.get("status") == "closed")
    return round(start + capital + trims + closed, 2)


def is_runner_id(tid):
    """Runner-made trade ids look like TICKER-YYYY-MM-DD-runner-<epoch>."""
    return "-runner-" in str(tid or "")
