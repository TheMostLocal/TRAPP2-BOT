#!/usr/bin/env python3
"""
market_hours.py - WHEN THE BOT MAY TRADE (z121): the NYSE regular session, in US
Eastern time. Same calendar rules as the books' tape writer
(TRAPP2*/pipeline/append_intraday_snapshot.py) and the app (botmirror.js), so the
three can never disagree:

  * weekdays only;
  * full-day holidays computed by rule (they never expire): New Year's Day (a
    Saturday New Year is NOT observed on Dec 31), Martin Luther King Jr. Day
    (3rd Mon Jan), Washington's Birthday (3rd Mon Feb), Good Friday, Memorial Day
    (last Mon May), Juneteenth (Jun 19, from 2022), Independence Day, Labor Day
    (1st Mon Sep), Thanksgiving (4th Thu Nov), Christmas - a Saturday holiday is
    observed on Friday, a Sunday one on Monday;
  * 1:00 pm ET early closes: the day after Thanksgiving, Jul 3 when Jul 4 falls
    Tue-Fri, Christmas Eve on a weekday (unless it is the observed holiday);
  * ad-hoc closures (national days of mourning, emergencies): EXTRA_CLOSED below,
    or the repo Variable RUNNER_CLOSED_DATES="YYYY-MM-DD,YYYY-MM-DD".

Regular session 09:30-16:00 ET (13:00 on an early-close day). DST comes from
zoneinfo (America/New_York); without tzdata the US rule (2nd Sunday of March ->
1st Sunday of November, in force since 2007) is used.
"""
import os
from datetime import datetime, timezone, timedelta, date as _date

OPEN_MIN = 9 * 60 + 30
CLOSE_MIN = 16 * 60
EARLY_CLOSE_MIN = 13 * 60
EXTRA_CLOSED = {
    "2025-01-09": "National Day of Mourning (President Carter)",
}


def et_now(utc_dt=None):
    utc_dt = utc_dt or datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        return utc_dt.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        pass
    y = utc_dt.year

    def _nth_sunday(month, nth):
        d = _date(y, month, 1)
        d += timedelta(days=(6 - d.weekday()) % 7)
        return d + timedelta(days=7 * (nth - 1))
    dst_start = datetime.combine(_nth_sunday(3, 2), datetime.min.time(), timezone.utc).replace(hour=7)
    dst_end = datetime.combine(_nth_sunday(11, 1), datetime.min.time(), timezone.utc).replace(hour=6)
    return utc_dt + timedelta(hours=-4 if dst_start <= utc_dt < dst_end else -5)


def _easter(year):
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return _date(year, month, day + 1)


def _observed(d, allow_friday=True):
    if d.weekday() == 5:
        return d - timedelta(days=1) if allow_friday else None
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth_weekday(year, month, weekday, nth):
    d = _date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(days=7 * (nth - 1))


def _last_weekday(year, month, weekday):
    d = _date(year, 12, 31) if month == 12 else _date(year, month + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


_CACHE = {}


def holidays(year):
    """{date: name} - full-day closures of the year."""
    if year in _CACHE:
        return _CACHE[year]
    h = {}
    ny = _observed(_date(year, 1, 1), allow_friday=False)
    if ny:
        h[ny] = "New Year's Day"
    h[_nth_weekday(year, 1, 0, 3)] = "Martin Luther King Jr. Day"
    h[_nth_weekday(year, 2, 0, 3)] = "Washington's Birthday"
    h[_easter(year) - timedelta(days=2)] = "Good Friday"
    h[_last_weekday(year, 5, 0)] = "Memorial Day"
    if year >= 2022:
        h[_observed(_date(year, 6, 19))] = "Juneteenth"
    h[_observed(_date(year, 7, 4))] = "Independence Day"
    h[_nth_weekday(year, 9, 0, 1)] = "Labor Day"
    h[_nth_weekday(year, 11, 3, 4)] = "Thanksgiving Day"
    h[_observed(_date(year, 12, 25))] = "Christmas Day"
    _CACHE[year] = h
    return h


def early_closes(year):
    out = {_nth_weekday(year, 11, 3, 4) + timedelta(days=1)}
    jul4 = _date(year, 7, 4)
    if jul4.weekday() in (1, 2, 3, 4):
        out.add(jul4 - timedelta(days=1))
    xmas_eve = _date(year, 12, 24)
    if xmas_eve.weekday() < 5:
        out.add(xmas_eve)
    hol = holidays(year)
    return {d for d in out if d.weekday() < 5 and d not in hol}


def _extra():
    out = dict(EXTRA_CLOSED)
    for s in (os.environ.get("RUNNER_CLOSED_DATES") or "").replace(";", ",").split(","):
        s = s.strip()
        if len(s) == 10:
            out[s] = "closed (RUNNER_CLOSED_DATES)"
    return out


def day_status(d):
    """(is_trading_day, close_minute_ET, label) for a calendar date."""
    if d.weekday() >= 5:
        return False, None, "weekend"
    ex = _extra().get(d.isoformat())
    if ex:
        return False, None, ex
    name = holidays(d.year).get(d)
    if name:
        return False, None, name
    if d in early_closes(d.year):
        return True, EARLY_CLOSE_MIN, "early close (1:00 pm ET)"
    return True, CLOSE_MIN, "regular"


def next_open(et_dt):
    """ET datetime of the next regular-session open after et_dt (today's, if before 09:30)."""
    d, m = et_dt.date(), et_dt.hour * 60 + et_dt.minute
    for k in range(0, 20):
        day = d + timedelta(days=k)
        ok, _close, _ = day_status(day)
        if not ok or (k == 0 and m >= OPEN_MIN):
            continue
        return datetime(day.year, day.month, day.day, 9, 30, tzinfo=et_dt.tzinfo)
    return None


def session_now(utc_dt=None):
    utc_dt = utc_dt or datetime.now(timezone.utc)
    et = et_now(utc_dt)
    d, m = et.date(), et.hour * 60 + et.minute
    ok, close, label = day_status(d)
    is_open = bool(ok and OPEN_MIN <= m < close)
    state = "open" if is_open else (label if not ok else ("pre-market" if m < OPEN_MIN else "after the close"))
    nxt = None if is_open else next_open(et)
    if is_open:
        text = (f"NYSE open · {et.strftime('%a %Y-%m-%d %H:%M')} ET · closes {close // 60:02d}:{close % 60:02d} ET"
                + (" (early close)" if close == EARLY_CLOSE_MIN else ""))
    else:
        text = (f"NYSE closed ({state}) · {et.strftime('%a %Y-%m-%d %H:%M')} ET"
                + (f" · next open {nxt.strftime('%a %Y-%m-%d %H:%M')} ET" if nxt else ""))
    return {"open": is_open, "state": state, "label": label, "date": d.isoformat(), "minute": m,
            "openMin": OPEN_MIN, "closeMin": close, "minutesToClose": (close - m) if is_open else None,
            "et": et.isoformat(timespec="minutes"), "nextOpenET": nxt.isoformat(timespec="minutes") if nxt else None,
            "text": text}


if __name__ == "__main__":
    s = session_now()
    print(s["text"])
