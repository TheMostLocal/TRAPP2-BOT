# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   tests/test_short_interest.py
#  z207 - FINRA's official short interest in the crowded-short rule, and open positions priced when no book has them:
#    * apply_short_interest: shares short / the trusted share count (apply_formula_mcap's u["sharesOut"]), days to cover,
#      the change (none across a split), a settlement over 35 days old not used
#    * risk_book.crowded_short: 15 %+ of the shares, or 10+ days to cover with 5 %+ short (8 % / 7 days while shorts
#      are being covered); FINRA's reading over Yahoo's short % of float; Yahoo only where FINRA has no share
#    * price_orphans: a held name no book priced (the macro book out: PBRG, ZSL, SH) from Yahoo's chart endpoint - USD
#      only, a quote <= 4 days old, Yahoo's trade time as asOf; unpricedSince set and cleared
#    * a price-only row is managed on price (stop / target) and never re-scored or closed on a horizon it can't read
#  Run: python -m unittest discover -s tests
# ============================================================
import os
import sys
import unittest
from datetime import datetime, timezone, date
from pathlib import Path

os.environ.setdefault("RUNNER_DRY_RUN", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import bot_runner as br  # noqa: E402
import risk_book as rb  # noqa: E402

NOW = datetime(2026, 11, 25, 19, 0, tzinfo=timezone.utc)                  # 1400 ET, in session


def stock(**kw):
    u = {"ticker": "XYZ", "price": 50.0, "marketcap": 5e9, "marketcapUSD": 5e9, "dv20USD": 50e6, "assetClass": "Equity",
         "isetf": "false", "isfund": "", "name": "XYZ Corp", "scan": True}
    u.update(kw)
    return u


class Crowded(unittest.TestCase):
    def test_finra_standing_rule(self):
        self.assertFalse(rb.crowded_short(stock(shortPctSO=0.12, daysToCover=4.0))[0])
        ok, why = rb.crowded_short(stock(shortPctSO=0.16, daysToCover=4.0, shortSettle="2026-11-13"))
        self.assertTrue(ok)
        self.assertIn("16% of the shares outstanding sold short (FINRA, settled 2026-11-13; >= 15%) - a crowded short", why)
        ok, why = rb.crowded_short(stock(shortPctSO=0.06, daysToCover=11.2))
        self.assertTrue(ok)
        self.assertIn("11.2 days to cover with 6% of the shares short", why)
        self.assertFalse(rb.crowded_short(stock(shortPctSO=0.04, daysToCover=14.0))[0])         # days to cover needs 5 %+ short

    def test_finra_covering_window(self):
        self.assertTrue(rb.crowded_short(stock(shortPctSO=0.09, daysToCover=2.0), covering=True)[0])
        ok, why = rb.crowded_short(stock(shortPctSO=0.06, daysToCover=7.5), covering=True)
        self.assertTrue(ok)
        self.assertIn(">= 7 days while shorts are being covered", why)
        self.assertFalse(rb.crowded_short(stock(shortPctSO=0.07, daysToCover=3.0), covering=True)[0])

    def test_finra_over_yahoo_and_yahoo_where_finra_has_none(self):
        # WHR 11-25: Yahoo 0.1 % of float, FINRA 32 % of the shares - FINRA decides
        self.assertTrue(rb.crowded_short(stock(shortPctFloat=0.001, shortPctSO=0.32))[0])
        # CHTR: Yahoo 45 %, FINRA 8.8 % - not crowded on FINRA's reading
        self.assertFalse(rb.crowded_short(stock(shortPctFloat=0.45, shortPctSO=0.088, daysToCover=3.0))[0])
        ok, why = rb.crowded_short(stock(shortPctFloat=0.25))                                     # no FINRA share: Yahoo's
        self.assertTrue(ok)
        self.assertIn("(Yahoo's figure; no FINRA share of shares)", why)
        self.assertFalse(rb.shortable(stock(shortPctSO=0.2))[0])
        self.assertTrue(rb.shortable(stock(shortPctSO=0.02))[0])


class ApplyShortInterest(unittest.TestCase):
    def test_share_of_shares_days_to_cover_split_and_stale(self):
        rows = {"AAA": stock(ticker="AAA", sharesOut=10e6), "BBB": stock(ticker="BBB"), "CCC": stock(ticker="CCC", sharesOut=5e6),
                "DDD": stock(ticker="DDD", sharesOut=1e6), "EEE": stock(ticker="EEE", sharesOut=1e6, scan=False)}
        doc = {"interest": {"AAA": {"settlementDate": "2026-11-13", "shares": 1.8e6, "changePct": 12.5, "daysToCover": 6.0},
                            "BBB": {"settlementDate": "2026-11-13", "shares": 2e6, "daysToCover": 12.0},
                            "CCC": {"settlementDate": "2026-11-13", "shares": 4e5, "changePct": 300.0, "split": True},
                            "DDD": {"settlementDate": "2026-09-30", "shares": 5e5},
                            "EEE": {"settlementDate": "2026-11-13", "shares": 5e5}},
               "volume": {"AAA": {"z": 2.4}}}
        out = br.apply_short_interest(rows, doc=doc, today=date(2026, 11, 25))
        self.assertEqual((out["names"], out["pctShares"], out["stale"], out["settle"]), (4, 3, 1, "2026-11-13"))
        self.assertAlmostEqual(rows["AAA"]["shortPctSO"], 0.18)
        self.assertEqual((rows["AAA"]["daysToCover"], rows["AAA"]["shortChgPct"], rows["AAA"]["shortVolZ"]), (6.0, 12.5, 2.4))
        self.assertNotIn("shortPctSO", rows["BBB"])                       # no trusted share count: days to cover only
        self.assertEqual(rows["BBB"]["daysToCover"], 12.0)
        self.assertIsNone(rows["CCC"]["shortChgPct"])                     # a split between settlements
        self.assertNotIn("shortSettle", rows["DDD"])                      # 56 days old: not used
        self.assertEqual(out["crowded"], 1)                               # AAA (18 %); EEE (50 %) isn't scannable
        self.assertEqual(br.apply_short_interest(rows, doc={}), {"names": 0, "pctShares": 0, "crowded": 0, "settle": None, "stale": 0})

    def test_formula_mcap_keeps_the_share_count(self):
        rows = {"AAA": stock(ticker="AAA", price=10.0, marketcap=1.0e9, currency="USD")}
        br.apply_formula_mcap(rows, doc={"date": "2026-11-25", "tickers": {"AAA": {"s": 1.01e8, "p": "sec", "st": "cover"}}})
        self.assertEqual(rows["AAA"]["sharesOut"], 1.01e8)

    def test_wired_after_the_share_counts(self):
        src = Path(br.__file__).read_text()
        self.assertLess(src.index("        apply_formula_mcap(rows)\n"), src.index("        apply_short_interest(rows)\n"))
        self.assertIn("price_orphans(universe, state, now_utc)", src)


def chart(price, ts, ccy="USD", it="ETF", name="ProShares Short S&P500", closes=(34.6, 34.4)):
    """Yahoo's v8 chart result: daily bars stamped at the 0930 ET open (1430 UTC), today's bar the live price."""
    days = [ts.replace(hour=14, minute=30) - __import__("datetime").timedelta(days=k) for k in range(len(closes), -1, -1)]
    return {"meta": {"regularMarketPrice": price, "regularMarketTime": int(ts.timestamp()), "currency": ccy,
                     "instrumentType": it, "longName": name, "chartPreviousClose": 35.0},
            "timestamp": [int(d.timestamp()) for d in days],
            "indicators": {"quote": [{"close": list(closes) + [price]}]}}


def pos(tk, **kw):
    p = {"id": f"t-{tk}", "ticker": tk, "direction": "long", "instrument": "shares", "status": "open", "entryPrice": 34.315,
         "entryDate": "2026-11-10", "shares": 100.0, "horizonClass": "short", "horizonDaysPlanned": 3, "horizonType": "swing",
         "stopPrice": 31.9129, "targetPrice": 38.4328, "lastPrice": 34.21}
    p.update(kw)
    return p


class Orphans(unittest.TestCase):
    def test_priced_from_yahoo_usd_only_fresh_only(self):
        state = {"trades": [pos("SH"), pos("ZSL"), pos("PBRG"), pos("VOD.L"), pos("AAPL"), pos("OLD", status="closed")]}
        universe = {"AAPL": stock(ticker="AAPL")}
        feed = {"SH": chart(34.12, NOW), "ZSL": chart(50.01, NOW, name="ProShares UltraShort Silver"),
                "VOD.L": chart(71.2, NOW, ccy="GBp", it="EQUITY", name="Vodafone")}

        def fetch(t):
            if t == "PBRG":
                raise TimeoutError("no answer")
            return feed.get(t)
        got = br.price_orphans(universe, state, NOW, fetch=fetch)
        self.assertEqual(got, ["SH", "ZSL"])
        sh = universe["SH"]
        self.assertEqual((sh["price"], sh["priceOnly"], sh["priceSource"], sh["scan"], sh["isetf"]), (34.12, True, "yahoo-direct", False, "true"))
        self.assertEqual(sh["asOf"], NOW.isoformat(timespec="seconds"))
        self.assertAlmostEqual(sh["changepct"], (34.12 / 34.4 - 1) * 100, places=3)
        self.assertEqual(rb.etp_exposure(universe["ZSL"])["lev"], -2.0)                         # the fund's name still reads
        self.assertIn("SH", br.fresh_universe(universe, NOW))                                    # a live quote: can act
        by = {p["ticker"]: p for p in state["trades"]}
        self.assertEqual((by["PBRG"].get("unpricedSince"), by["VOD.L"].get("unpricedSince")), ("2026-11-25", "2026-11-25"))
        self.assertNotIn("unpricedSince", by["SH"])
        self.assertNotIn("VOD.L", universe)                                                      # pence: not a USD price
        self.assertEqual(br.orphan_row("VOD.L", feed["VOD.L"], NOW), (None, "quoted in GBp"))
        # the next pass: PBRG priced (a 3-day-old last trade still counts, but isn't fresh enough to act)
        feed["PBRG"] = chart(66.0, NOW.replace(day=22), it="EQUITY", name="PBRG Inc")

        def fetch2(t):
            return feed.get(t)
        u2 = {"AAPL": stock(ticker="AAPL")}
        br.price_orphans(u2, state, NOW, fetch=fetch2)
        self.assertNotIn("unpricedSince", by["PBRG"])
        self.assertNotIn("PBRG", br.fresh_universe(u2, NOW))
        feed["PBRG"] = chart(66.0, NOW.replace(day=18), it="EQUITY")                             # a week old: refused
        u3 = {}
        br.price_orphans(u3, {"trades": [pos("PBRG")]}, NOW, fetch=fetch2)
        self.assertNotIn("PBRG", u3)

    def test_a_price_only_row_is_managed_on_price_and_never_re_scored(self):
        calls = []
        real = br.score_symbol
        br.score_symbol = lambda *a, **k: calls.append(a) or None
        try:
            # past its 3-session horizon, quote above the stop: held (no horizon exit on a signal it couldn't read)
            state = {"bankroll": 100_000.0, "trades": [pos("SH", entryDate="2026-11-10")]}
            universe = {}
            br.price_orphans(universe, state, NOW, fetch=lambda t: chart(34.12, NOW))
            br.manage_open_positions(state, universe, "2026-11-25", ctx={"weights": {}})
            p = state["trades"][0]
            self.assertEqual(p["status"], "open")
            self.assertEqual(calls, [])
            self.assertEqual(p["lastPrice"], 34.12)
            # through the stop: closed at the quote
            state2 = {"bankroll": 100_000.0, "trades": [pos("SH")]}
            u2 = {}
            br.price_orphans(u2, state2, NOW, fetch=lambda t: chart(31.5, NOW))
            br.manage_open_positions(state2, u2, "2026-11-25", ctx={"weights": {}})
            self.assertEqual((state2["trades"][0]["status"], state2["trades"][0]["exitPrice"]), ("closed", 31.5))
        finally:
            br.score_symbol = real


if __name__ == "__main__":
    unittest.main()
