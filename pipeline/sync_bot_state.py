#!/usr/bin/env python3
# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   pipeline/sync_bot_state.py
#
#  z159 - PHASE 2: the journal, live. After each runner pass (bot_loop.sh) the journal the runner just
#  committed to GitHub is upserted as ONE row ('journal') of the Bot project's bot_state table, so the app
#  sees a pass the moment it ends. Skipped when the journal hasn't changed since the last push (same
#  generatedAt). Uses the same service-key picking as sync_bot_to_supabase.py. Best-effort.
# ============================================================
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = Path(__file__).resolve().parent.parent
JOURNAL = ROOT / "data" / "bot_training_data.json"
STATE = Path(os.environ.get("RUNNER_TEMP") or "/tmp") / "bot_state_pushed.txt"


def main():
    try:
        import sync_bot_to_supabase as sb                      # the shared credential block (service key only)
        url, key = sb.URL, sb.KEY
    except Exception as e:
        print(f"[bot_state] credentials unavailable ({e}) - skipped")
        return 0
    if not url or not key:
        return 0
    try:
        doc = json.loads(JOURNAL.read_text())
    except Exception as e:
        print(f"[bot_state] journal unreadable ({e})")
        return 0
    gen = str(doc.get("generatedAt") or "")
    try:
        if gen and STATE.read_text().strip() == gen:
            print(f"[bot_state] journal unchanged ({gen}) - nothing to push")
            return 0
    except Exception:
        pass
    clean = sb._sb_finite(doc) if hasattr(sb, "_sb_finite") else doc     # NaN / inf never reach JSON
    body = json.dumps([{"id": "journal", "doc": clean, "generated_at": gen or None}], allow_nan=False).encode()
    req = urllib.request.Request(f"{url}/rest/v1/bot_state?on_conflict=id", method="POST", data=body,
                                 headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "Prefer": "resolution=merge-duplicates,return=minimal"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            print(f"[bot_state] journal {gen} pushed ({len(body) // 1024} KB, HTTP {r.status})")
        try:
            STATE.write_text(gen)
        except Exception:
            pass
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")[:200]
        hint = " - run supabase/bot_state.sql in the Bot project" if e.code in (404,) or "bot_state" in msg else ""
        print(f"::warning::bot_state push failed: HTTP {e.code} {msg}{hint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
