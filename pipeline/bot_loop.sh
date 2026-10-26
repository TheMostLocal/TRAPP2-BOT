#!/usr/bin/env bash
# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   pipeline/bot_loop.sh
#
#  z144 - THE RUNNER, LOOPED. GitHub drops scheduled runs at busy times (10-22: the runner wrote at
#  13:57 ET and not again; the equity tape got 2 of ~18 points). Instead of one pass per cron
#  trigger, one job now runs a pass every 30 minutes on the Eastern :05 / :35 grid for up to
#  LOOP_BUDGET_MIN, and an hourly trigger queues the next job behind it (concurrency group, one
#  pending) - so the book is managed continuously even when some triggers never fire.
#  Each pass = the same steps the one-shot workflow ran: start from the latest main → runner
#  (session-gated inside: trades only while the NYSE is open; after the close one review) →
#  verify → commit the journal (+ app_actions) → Supabase mirror (if configured) → equity point.
# ============================================================
set -u
BUDGET=$(( ${LOOP_BUDGET_MIN:-170} * 60 ))
START=$(date +%s)
BRANCH="${GITHUB_REF_NAME:-main}"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

latest() {   # z117: always commit from the newest main, never from the tree the job started on
  git fetch -q origin "$BRANCH" && { git merge -q --ff-only FETCH_HEAD 2>/dev/null || git reset -q --hard FETCH_HEAD; }
}

push_retry() {
  for i in 1 2 3 4 5; do
    if git push -q origin "HEAD:$BRANCH"; then echo "  pushed (attempt $i)"; return 0; fi
    git fetch -q origin "$BRANCH"
    git merge -X ours "origin/$BRANCH" --no-edit -q 2>/dev/null || true
    if [ -f .git/MERGE_HEAD ]; then
      git checkout --ours -- data/bot_training_data.json 2>/dev/null || true
      git add -A data/bot_training_data.json 2>/dev/null || true
      git commit -q --no-edit 2>/dev/null || true
    fi
    sleep $(( i * 3 ))
  done
  echo "::warning::push failed after 5 attempts"
  return 1
}

one_pass() {
  latest
  python pipeline/bot_runner.py || { echo "::warning::runner pass failed"; return 0; }
  python3 -c 'import json,sys; d=json.load(open("data/bot_training_data.json")); sys.exit(0 if isinstance(d.get("bankroll"),(int,float)) else 1)' \
    || { echo "::error::bankroll missing or not numeric - not committing"; git checkout -q -- data 2>/dev/null; return 0; }
  git add -A data/bot_training_data.json                       # z116: one path per git add
  git add -A data/app_actions/ 2>/dev/null || true
  if ! git diff --staged --quiet; then
    git commit -q -m "Bot run $(date -u +%Y-%m-%dT%H:%M:%SZ)"   # no [skip ci]: supabase-sync.yml keys off this push
    push_retry
  else
    echo "  journal unchanged"
  fi
  if [ -n "${SUPABASE_URL:-}" ] && [ -f pipeline/sync_bot_to_supabase.py ]; then
    python pipeline/sync_bot_to_supabase.py || echo "::warning::Supabase mirror failed (book unaffected)"
    python pipeline/sync_bot_state.py || true                 # z159: the whole journal, live, for the app
  fi
  if [ -f pipeline/bot_equity_snapshot.py ]; then
    python3 pipeline/bot_equity_snapshot.py || true
    git add data/bot_equity_history.json 2>/dev/null || true
    if ! git diff --staged --quiet; then
      git commit -q -m "equity tape: $(date -u +%FT%TZ) [skip ci]"
      push_retry
    fi
  fi
}

# z161: when this job stops for its TIME BUDGET while its window is still open, it queues the next run of its own
# workflow - GitHub's scheduler went hours without starting anything (2026-10-26: no intraday run by 0900 ET).
# Needs `actions: write` and GH_TOKEN (the job's own token) in the workflow; the concurrency group queues the new
# run behind this one. GitHub's schedule and the Supabase heartbeat (supabase/heartbeat.sql) stay as backstops.
chain_next() {
  [ -n "${GH_TOKEN:-}" ] || { echo "::notice::GH_TOKEN not set - not chaining"; return 0; }
  local wf="${GITHUB_WORKFLOW_REF:-}"
  wf="${wf%@*}"; wf="${wf##*/}"
  { [ -n "$wf" ] && [ -n "${GITHUB_REPOSITORY:-}" ]; } || return 0
  if gh workflow run "$wf" --ref "${GITHUB_REF_NAME:-main}" -R "$GITHUB_REPOSITORY" >/dev/null 2>&1; then
    echo "chained: the next run of $wf is queued"
  else
    echo "::warning::could not queue the next run of $wf (does the workflow grant actions: write?)"
  fi
}

next_grid_wait() {   # seconds until the next :05 or :35 (minute ≡ 5 mod 30), at least 60 s away
  local now m s w
  now=$(date +%s); m=$(( (now / 60) % 60 )); s=$(( now % 60 ))
  w=$(( ( (5 - m % 30 + 30) % 30 ) * 60 - s ))
  [ "$w" -lt 60 ] && w=$(( w + 1800 ))
  echo "$w"
}

i=0
while :; do
  i=$((i + 1))
  echo "── pass $i · $(date -u +%H:%M:%S)Z · $(TZ=America/New_York date +'%a %H:%M') ET"
  one_pass
  wait=$(next_grid_wait)
  if [ $(( $(date +%s) - START + wait + 240 )) -gt "$BUDGET" ]; then
    echo "budget reached after $i pass(es)"
    if [ "$(TZ=America/New_York date +%u)" -le 5 ] && [ "$((10#$(TZ=America/New_York date +%H%M)))" -lt 1630 ]; then chain_next; fi
    break
  fi
  echo "  next pass in $(( wait / 60 )) min"
  sleep "$wait"
done
