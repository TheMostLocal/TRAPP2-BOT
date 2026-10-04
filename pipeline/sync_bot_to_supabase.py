#!/usr/bin/env python3
"""
sync_bot_to_supabase.py — push bot_training_data.json → Supabase bot_trades.

Server-side (GitHub Actions). Uses the SERVICE ROLE key. The bot file's `trades`
array IS the transaction history; each trade row is upserted by id so the
`bot_trades` table (and the bot_trades_closed / bot_signal_performance views)
stay current without opening the app.

Env (repo secrets):
  SUPABASE_URL          the BOT project URL
  SUPABASE_SERVICE_KEY  the service_role key
"""
import json, os, sys, urllib.request, urllib.error
from datetime import datetime, timezone

# ---- Supabase credentials (same block in every Valuatio sync script) --------
# Picks whichever configured key is actually a SERVICE key (legacy JWT with
# role=service_role, or a new sb_secret_ key), so a wrong value in ONE of the
# two secret names can't silently downgrade writes to anon. Never prints keys.
import base64 as _sb_b64, json as _sb_json, os as _sb_os, re as _sb_re
def _sb_claims(k):
    try:
        seg = k.split(".")[1]; seg += "=" * (-len(seg) % 4)
        return _sb_json.loads(_sb_b64.urlsafe_b64decode(seg))
    except Exception:
        return {}
def _sb_kind(k):
    k = (k or "").strip()
    if not k: return "missing"
    if k.startswith("sb_secret_"): return "secret"
    if k.startswith("sb_publishable_"): return "publishable"
    if k.count(".") == 2: return _sb_claims(k).get("role") or "jwt(no role)"
    return "unrecognized"
def _sb_url():
    u = (_sb_os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    return _sb_re.sub(r"/rest/v1$", "", u)
def _sb_pick_key():
    names = ("SUPABASE_SERVICE_ROLE", "SUPABASE_SERVICE_KEY", "SUPABASE_KEY", "SUPABASE_ANON_KEY")
    vals = [(n, (_sb_os.environ.get(n) or "").strip()) for n in names]
    have = [(n, v) for n, v in vals if v]
    good = [(n, v) for n, v in have if _sb_kind(v) in ("service_role", "secret")]
    name, key = (good or have or [(None, "")])[0]
    print("[supabase] " + (", ".join(f"{n}={_sb_kind(v)}" for n, v in have) or "no keys set")
          + f" -> using {name or 'none'}")
    if key and _sb_kind(key) not in ("service_role", "secret"):
        print(f"::warning::{name} is a '{_sb_kind(key)}' key, not service_role/secret - "
              "service-only tables (ticker_snapshot, regime_timeline, bot_equity) will reject writes")
    distinct = {v for n, v in have if n in names[:2]}
    if len(distinct) > 1:
        print("::warning::SUPABASE_SERVICE_ROLE and SUPABASE_SERVICE_KEY differ - set both to the same service key")
    ref = _sb_claims(key).get("ref") if key.count(".") == 2 else None
    m = _sb_re.match(r"https://([a-z0-9]+)\.supabase\.co$", _sb_url())
    if ref and m and ref != m.group(1):
        print(f"::error::key belongs to Supabase project '{ref}' but SUPABASE_URL points at '{m.group(1)}' (keys from the other project?)")
    return key
def _sb_finite(obj):
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _sb_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sb_finite(v) for v in obj]
    return obj
# -----------------------------------------------------------------------------
URL = _sb_url()
KEY = _sb_pick_key()
TABLE = "bot_trades"
DATA = "data/bot_training_data.json"

def _req(method, path, body=None, headers=None):
    h = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    if headers: h.update(headers)
    data = json.dumps(_sb_finite(body), allow_nan=False).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

# ---- Scheduled-run freshness guard -------------------------------------------
# bot_runner rewrites generatedAt on every real run and mirrors to Supabase in
# the same job. The hourly SCHEDULED run is only a backstop; replaying an old
# repo copy would revert trades the app has since updated in Supabase. So a
# scheduled run only syncs a file generated within BOT_SYNC_MAX_AGE_H (default
# 96 h = covers the Fri -> Mon gap between weekday bot runs).
MAX_AGE_H = float(os.environ.get("BOT_SYNC_MAX_AGE_H") or 96)


def _stale_for_schedule(d):
    if (os.environ.get("SYNC_TRIGGER") or "").strip() != "schedule":
        return False
    ts = d.get("generatedAt") if isinstance(d, dict) else None
    try:
        gen = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if gen.tzinfo is None:
            gen = gen.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - gen).total_seconds() / 3600
    except Exception:
        age_h = None
    if age_h is not None and age_h <= MAX_AGE_H:
        return False
    shown = "unknown age" if age_h is None else f"{age_h:.0f}h old"
    print(f"::notice::bot: scheduled sync SKIPPED - repo file is {shown} (generatedAt={ts}); "
          "the bot-runner job mirrors fresh trades itself, and a push/manual run always syncs.")
    return True
# -----------------------------------------------------------------------------


def main():
    if not URL or not KEY:
        print("✗ Missing creds. Set SUPABASE_URL and one of SUPABASE_SERVICE_KEY / SUPABASE_SERVICE_ROLE as repo secrets."); return 1
    if not os.path.exists(DATA):
        print(f"No {DATA} — nothing to sync."); return 0
    d = json.load(open(DATA))
    if _stale_for_schedule(d):
        return 0
    # Accept both the standalone training export and an XTRAPP-style dump.
    trades = d.get("trades")
    if trades is None and isinstance(d.get("bot"), dict):
        trades = d["bot"].get("bets")
    trades = trades or []
    now = datetime.now(timezone.utc).isoformat()

    rows = []
    for i, t in enumerate(trades):
        tid = t.get("id") or f"{t.get('ticker','TX')}-{t.get('entryDate','')}-{i}"
        rows.append({"id": tid, "trade": t, "updated_at": now})

    sent, failed = 0, 0
    for i in range(0, len(rows), 100):
        chunk = rows[i:i+100]
        st, body = _req("POST", f"/rest/v1/{TABLE}?on_conflict=id", chunk,
                        {"Prefer": "resolution=merge-duplicates,return=minimal"})
        if st in (200, 201, 204): sent += len(chunk)
        else:
            print(f"::error::bot_trades upsert chunk {i} -> HTTP {st}: {body[:200]}")
            failed += 1
    if failed:
        print(f"X bot -> Supabase: {sent}/{len(rows)} trade(s) upserted, {failed} chunk(s) rejected")
        return 1
    print(f"✓ bot → Supabase: upserted {sent} trade(s)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
