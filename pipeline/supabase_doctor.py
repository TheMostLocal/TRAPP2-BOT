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
"""
supabase_doctor.py — one-shot health check for a Valuatio Supabase project.
Run from Actions (workflow "Supabase doctor"). Reports, without printing keys:
  * which secret names are set and what KIND of key each holds
  * whether the key's project ref matches SUPABASE_URL
  * per table: exists? row count? can the chosen key WRITE to it?
Write probes use an obvious sentinel row and delete it right after.
Exit 1 if anything is wrong, so the run goes red.
"""
import sys, json, urllib.request, urllib.error
from datetime import datetime, timezone

TABLES = {
    # table: (conflict column, probe row)
    "bot_trades":          ("id", {"id": "__doctor__", "trade": {"doctor": True}}),
    "bot_equity":          ("t", {"t": "1970-01-01T00:00:00Z", "value": 0, "source": "doctor"}),
    "research_grades":     ("ticker", {"ticker": "__DOCTOR__", "grades": {"doctor": True}}),
    "analytics_kv":        ("key", {"key": "__doctor__", "value": {"doctor": True}}),
    "portfolio_positions": ("id", {"id": "__doctor__", "position": {"doctor": True}}),
    "regime_snapshots":    ("id", {"id": "__doctor__", "snapshot": {"regime": "doctor"}}),
    "regime_timeline":     ("id", {"id": "__doctor__", "entry": {"regime": "doctor"}}),
    "ticker_snapshot":     ("ticker", {"ticker": "__DOCTOR__", "name": "doctor"}),
}
PROJECT_TABLES = {
    "bot": ["bot_trades", "bot_equity"],
    "analytics": ["research_grades", "analytics_kv", "portfolio_positions",
                  "regime_snapshots", "regime_timeline", "ticker_snapshot"],
}

def req(method, path, body=None, extra=None):
    h = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    h.update(extra or {})
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(URL + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, resp.read().decode(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), dict(e.headers or {})
    except Exception as e:
        return 0, str(e), {}

def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    tables = PROJECT_TABLES.get(which, [t for v in PROJECT_TABLES.values() for t in v])
    bad = 0
    print(f"Supabase doctor — project '{which or 'all'}' — {datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z")
    if not URL:
        print("::error::SUPABASE_URL is not set"); return 1
    if not _sb_re.match(r"https://[a-z0-9]+\.supabase\.co$", URL):
        print(f"::warning::SUPABASE_URL looks unusual: expected https://<ref>.supabase.co (got a {len(URL)}-char value)")
    if not KEY:
        print("::error::no Supabase key set"); return 1
    if _sb_kind(KEY) not in ("service_role", "secret"):
        bad += 1
    for t in tables:
        conflict, probe = TABLES[t]
        st, body, hdr = req("GET", f"/rest/v1/{t}?select=*&limit=1", extra={"Prefer": "count=exact"})
        if st == 404 or "PGRST205" in body or "does not exist" in body:
            print(f"::error::{t:20s} MISSING — run the project SQL in _SUPABASE/"); bad += 1; continue
        if st not in (200, 206):
            print(f"::error::{t:20s} read HTTP {st}: {body[:160]}"); bad += 1; continue
        rng = hdr.get("Content-Range") or hdr.get("content-range") or "?"
        rows = rng.split("/")[-1]
        st2, body2, _ = req("POST", f"/rest/v1/{t}?on_conflict={conflict}", [probe],
                            {"Prefer": "resolution=merge-duplicates,return=minimal"})
        ok = st2 in (200, 201, 204)
        val = str(probe[conflict])
        if ok:
            req("DELETE", f"/rest/v1/{t}?{conflict}=eq.{urllib.request.quote(val)}")
        print(f"  {t:20s} rows={rows:>7}  write={'OK' if ok else f'FAIL HTTP {st2}'}"
              + ("" if ok else f"  {body2[:140]}"))
        if not ok:
            bad += 1
            if st2 in (401, 403) or "row-level security" in body2:
                print(f"::error::{t}: write rejected by RLS — the key in use is not a service/secret key for THIS project")
    print("RESULT: " + ("all good" if not bad else f"{bad} problem(s) — see annotations above"))
    return 1 if bad else 0

if __name__ == "__main__":
    URL = _sb_url()
    KEY = _sb_pick_key()
    sys.exit(main())
