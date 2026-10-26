-- ============================================================
--  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
--  >>> FILE PATH:   supabase/bot_state.sql
--  z159 - PHASE 2: the runner's journal, live. Run ONCE in the Bot Supabase project → SQL Editor. Safe to
--  re-run. One row ('journal') holding the journal exactly as the runner wrote it to GitHub; the app reads
--  it the moment a pass finishes (GitHub's raw cache is 5 minutes). The journal is public in TRAPP2-BOT.
-- ============================================================

create table if not exists public.bot_state (
  id            text primary key,
  doc           jsonb not null,
  generated_at  timestamptz,
  updated_at    timestamptz not null default now()
);

alter table public.bot_state enable row level security;
drop policy if exists "read bot_state" on public.bot_state;
create policy "read bot_state" on public.bot_state for select to anon, authenticated using (true);
grant select on public.bot_state to anon, authenticated;
grant all    on public.bot_state to service_role;
