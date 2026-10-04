#!/usr/bin/env python3
"""
sync_port_to_supabase.py — push data/portfolio_data.json -> Supabase
`portfolio_positions` (the shared ANALYTICS/PORT project). Server-side
(GitHub Actions), SERVICE-ROLE key.

WHY THE OLD VERSION PUSHED 0 ROWS:
  It was a copy of the BOT sync — it read d.get("trades") (portfolio_data.json
  has no "trades"; positions live under "portfolio", history under
  "transactions") AND wrote a "trade" key when the table column is "position".
  So it built an empty list AND the wrong shape.

WHAT IT PUSHES (all -> portfolio_positions, jsonb column `position`, key `id`):
  - each portfolio entry            _kind='entry'        (role + ticker + notes -> thesis,
                                                           marketValue/pnl for Long/Short lots)
  - each transaction (immutable)    _kind='transaction'  (stable txn: id, no duplication)
  - cash / goodGlobe index+curve /  _kind='cash'|'index'|'indexCurve'|'snapshot'  (singletons)
    a restore snapshot

FRESH, NO DUPLICATES:
  Every row upserts on `id` (resolution=merge-duplicates) — re-running never
  duplicates. A RECONCILE step then deletes rows that no longer exist locally
  (so a removed position/watch disappears), while KEEPING transaction history
  (txn: ids) and the snapshot singleton. Mirrors the app's own sync exactly.

TIMEZONE:
  updated_at is Eastern wall-clock tagged +00:00 so Supabase's UTC display shows
  local Eastern time (see now_iso).

Env (repo secrets):
  SUPABASE_URL           the shared ANALYTICS/PORT project URL
  SUPABASE_SERVICE_ROLE  the service_role key   (SUPABASE_SERVICE_KEY also accepted)
"""
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

# Repo owner — resolved at runtime so the pipeline follows the repos to any
# GitHub account. Actions sets GITHUB_REPOSITORY_OWNER automatically;
# VALUATIO_OWNER (repo variable/env) overrides; legacy owner is the fallback.
_GH_OWNER = (__import__("os").environ.get("VALUATIO_OWNER")
             or __import__("os").environ.get("GITHUB_REPOSITORY_OWNER")
             or "GoodGlobeLLC").strip()

URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
KEY = (os.environ.get("SUPABASE_SERVICE_ROLE")
       or os.environ.get("SUPABASE_SERVICE_KEY")
       or os.environ.get("SUPABASE_KEY")
       or os.environ.get("SUPABASE_ANON_KEY") or "")
TABLE = "portfolio_positions"
DATA = "data/portfolio_data.json"

RAW = f"https://raw.githubusercontent.com/{_GH_OWNER}"
MASTER_SOURCES = [f"{RAW}/TRAPP2/main/data/master.json",
                  f"{RAW}/TRAPP2-2/main/data/master.json",
                  f"{RAW}/TRAPP2-1/main/data/master.json",
                  f"{RAW}/TRAPP2-3/main/data/master.json"]  # gap-fill only (first-wins)
ACTIVE_ROLES = {"long", "short", "buy to open", "sell to open"}
SHORT_ROLES = {"short", "sell to open"}


# --------------------------------------------------------------- timezone ----
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:
    _ET = None


def _eastern_now():
    if _ET is not None:
        return datetime.now(_ET)
    u = datetime.now(timezone.utc); y = u.year

    def nth_sunday(month, n):
        d = datetime(y, month, 1, tzinfo=timezone.utc)
        return 1 + ((6 - d.weekday()) % 7) + (n - 1) * 7
    start = datetime(y, 3, nth_sunday(3, 2), 7, tzinfo=timezone.utc)
    end = datetime(y, 11, nth_sunday(11, 1), 6, tzinfo=timezone.utc)
    return u + timedelta(hours=(-4 if start <= u < end else -5))


def now_iso():
    """Eastern wall-clock tagged +00:00 so Supabase's UTC display shows local."""
    return _eastern_now().replace(tzinfo=timezone.utc).isoformat()


# ------------------------------------------------------------------ http -----
def _req(method, path, body=None, headers=None):
    h = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    if headers:
        h.update(headers)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def fetch_json(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "valuatio-port-sync"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def _num(v):
    try:
        f = float(v)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------- id helpers (app parity) -
def _js_num_str(v):
    n = _num(v)
    n = 0.0 if n is None else n
    return str(int(n)) if n == int(n) else repr(n)


def _b36(n):
    if n == 0:
        return "0"
    digs = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while n:
        n, r = divmod(n, 36)
        out = digs[r] + out
    return out


def _stable_txn_id(t):
    tid = t.get("id")
    if tid and str(tid).startswith("txn:"):
        return str(tid)
    if tid and str(tid).strip():
        return f"txn:{tid}"
    key = "|".join([
        str(t.get("type", "") or "").lower(),
        str(t.get("ticker", "") or "").upper(),
        _js_num_str(t.get("qty")),
        _js_num_str(t.get("price")),
        str(t.get("ts") or t.get("date") or ""),
    ])
    h = 0
    for ch in key:
        h = (h << 5) - h + ord(ch)
        h &= 0xFFFFFFFF
        if h & 0x80000000:
            h -= 0x100000000
    return f"txn:c{_b36(h & 0xFFFFFFFF)}"


def _ensure_port_id(rec):
    if rec.get("id") and str(rec["id"]).strip():
        return str(rec["id"])
    tk = str(rec.get("ticker") or rec.get("symbol") or "POS").upper()
    d = rec.get("openDate") or rec.get("date") or rec.get("addedAt") or ""
    rec["id"] = f"{tk}-{d}" if d else f"{tk}-{_b36(abs(hash(json.dumps(rec, sort_keys=True))) & 0xFFFFFFFF)}"
    return rec["id"]


# --------------------------------------------------------------- enrichment --
def _load_price_map():
    px = {}
    for url in MASTER_SOURCES:
        d = fetch_json(url)
        rows = d.values() if isinstance(d, dict) else (d if isinstance(d, list) else [])
        for r in rows:
            if not isinstance(r, dict):
                continue
            tk = (r.get("ticker") or r.get("symbol") or "").upper()
            p = _num(r.get("price")) or _num(r.get("fmpPrice")) or _num(r.get("close")) or _num(r.get("last"))
            if tk and p is not None and tk not in px:
                px[tk] = p
    return px


def _enrich(rec, price_map, total_mv):
    """Field bridges (schema column paths) + marketValue/pnl for active lots."""
    k = rec.get("_kind")
    if k in ("snapshot", "index", "indexCurve", "cash"):
        return rec
    if k == "transaction":
        to = dict(rec)
        if rec.get("type") is not None and to.get("action") is None:
            to["action"] = rec["type"]
        if rec.get("qty") is not None and to.get("shares") is None:
            to["shares"] = _num(rec["qty"])
        if (rec.get("ts") or rec.get("date")) and to.get("date") is None:
            to["date"] = rec.get("ts") or rec.get("date")
        return to

    out = dict(rec)
    tk = str(rec.get("ticker") or "").upper()
    if not tk:
        return out
    # schema reads shares/avgCost/openDate; app stores qty/costBasis/addedAt.
    if rec.get("qty") is not None and out.get("shares") is None:
        out["shares"] = _num(rec["qty"])
    if rec.get("costBasis") is not None and out.get("avgCost") is None:
        out["avgCost"] = _num(rec["costBasis"])
    if rec.get("addedAt") and out.get("openDate") is None:
        out["openDate"] = rec["addedAt"]
        out.setdefault("open_date", rec["addedAt"])
    # exit date from the most recent sell note; thesis from the first note.
    notes = rec.get("notes") if isinstance(rec.get("notes"), list) else []
    if out.get("exitDate") is None:
        sells = [n for n in notes if isinstance(n, dict) and n.get("soldAction")]
        if sells and sells[-1].get("ts"):
            out["exitDate"] = sells[-1]["ts"]; out["exit_date"] = sells[-1]["ts"]
    if out.get("thesis") is None and notes:
        first = notes[0]
        out["thesis"] = str(first.get("text") if isinstance(first, dict) else first)

    # marketValue / pnl / weight for ACTIVE lots (Long/Short/…) with qty.
    role = str(rec.get("position") or "").lower()
    qty, avg = _num(rec.get("qty")), _num(rec.get("costBasis"))
    if role in ACTIVE_ROLES and qty:
        price = price_map.get(tk) or avg
        if price is not None:
            direction = -1.0 if role in SHORT_ROLES else 1.0
            mv = abs(qty * price)
            out["marketValue"] = round(mv, 2)
            if avg is not None:
                pnl = (price - avg) * qty * direction
                out["pnl"] = round(pnl, 2)
                basis = abs(avg * qty)
                if basis:
                    out["pnlPct"] = round(pnl / basis * 100, 4)
                    out["returnPct"] = out["pnlPct"]
            if total_mv:
                out["weight"] = round(mv / total_mv * 100, 4)
    return out


# ------------------------------------------------------------- build rows ----
def build_records(d):
    recs = []
    for e in (d.get("portfolio") or []):
        if isinstance(e, dict):
            recs.append(dict(e, _kind="entry"))
    for t in (d.get("transactions") or []):
        if isinstance(t, dict):
            r = dict(t, _kind="transaction")
            r["id"] = _stable_txn_id(t)
            recs.append(r)
    if d.get("cashPosition") is not None:
        recs.append({"id": "__cash__", "_kind": "cash", "value": d["cashPosition"]})
    if d.get("goodGlobeIndex") is not None:
        recs.append({"id": "__goodglobe_index__", "_kind": "index", "value": d["goodGlobeIndex"]})
    if d.get("goodGlobeCurve") is not None:
        recs.append({"id": "__goodglobe_curve__", "_kind": "indexCurve", "value": d["goodGlobeCurve"]})
    recs.append({"id": "__portfolio_snapshot__", "_kind": "snapshot",
                 "counts": d.get("counts"), "generatedAt": d.get("generatedAt"), "schema": d.get("schema")})
    return recs


def _existing_ids():
    st, body = _req("GET", f"/rest/v1/{TABLE}?select=id")
    if st == 200:
        try:
            return {r["id"] for r in json.loads(body) if isinstance(r, dict) and "id" in r}
        except Exception:
            return set()
    print(f"  (could not list existing ids: HTTP {st})")
    return set()


def main():
    if not URL or not KEY:
        print("X Missing creds. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE (or SUPABASE_SERVICE_KEY).")
        return 1
    if not os.path.exists(DATA):
        print(f"No {DATA} — nothing to sync.")
        return 0
    d = json.load(open(DATA))
    if not isinstance(d, dict):
        print("portfolio_data.json is not a wrapped snapshot object — skipping.")
        return 0

    records = build_records(d)
    price_map = _load_price_map()
    print(f"Portfolio -> Supabase ({URL}) · {len(records)} record(s), {len(price_map)} prices")

    # total active market value (for weight_pct).
    total_mv = 0.0
    for r in records:
        if r.get("_kind") == "entry" and str(r.get("position") or "").lower() in ACTIVE_ROLES:
            qty = _num(r.get("qty"))
            if qty:
                px = price_map.get(str(r.get("ticker") or "").upper()) or _num(r.get("costBasis"))
                if px:
                    total_mv += abs(qty * px)

    now = now_iso()
    rows, local_ids = [], set()
    for rec in records:
        if rec.get("_kind") in ("entry", "transaction"):
            _ensure_port_id(rec)
            pos = _enrich(rec, price_map, total_mv)
            rows.append({"id": rec["id"], "position": pos, "updated_at": now})
        else:
            rows.append({"id": rec["id"], "position": _enrich(rec, price_map, total_mv), "updated_at": now})
        local_ids.add(rec["id"])

    # upsert (fresh, no duplicates)
    sent = 0
    for i in range(0, len(rows), 100):
        chunk = rows[i:i + 100]
        st, body = _req("POST", f"/rest/v1/{TABLE}?on_conflict=id", chunk,
                        {"Prefer": "resolution=merge-duplicates,return=minimal"})
        if st in (200, 201, 204):
            sent += len(chunk)
        else:
            print(f"  upsert chunk {i} -> HTTP {st}: {body[:240]}")
    print(f"  upserted {sent}/{len(rows)}")

    # RECONCILE: drop rows gone locally — but keep txn history + the snapshot.
    # Guard: only reconcile when we actually have local records, so an empty or
    # unreadable file can never wipe the table.
    deleted = 0
    if local_ids:
        stale = [i for i in _existing_ids()
                 if i not in local_ids
                 and not (isinstance(i, str) and i.startswith("txn:"))
                 and i != "__portfolio_snapshot__"]
        for i in range(0, len(stale), 50):
            chunk = stale[i:i + 50]
            in_list = ",".join('"%s"' % str(x).replace('"', "") for x in chunk)
            st, body = _req("DELETE", f"/rest/v1/{TABLE}?id=in.({urllib.parse.quote(in_list)})", None,
                            {"Prefer": "return=minimal"})
            if st in (200, 204):
                deleted += len(chunk)
            else:
                print(f"  reconcile delete -> HTTP {st}: {body[:200]}")
    print(f"OK portfolio sync complete — {sent} upserted, {deleted} stale row(s) removed")
    return 0


import urllib.parse  # noqa: E402  (used in main's reconcile)

if __name__ == "__main__":
    sys.exit(main())
