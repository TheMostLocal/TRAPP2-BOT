# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   tests/test_liquidity.py
#  z205: liquidity by the formula - the 20-session average dollar volume from our own histories (TRAPP2-ANALYTICS
#  data/derived/market.json, z203) sizes each new share position: at most 2 % of a day's trading, none under $250K a
#  day, a short sale only at $10M+ a day. USD at the row's rate (pence included); a stale / unit-mismatched figure and
#  an old file are not used; futures, currencies, indices and crypto pairs carry no rule (volume x price is no notional).
#  Run: python -m unittest discover -s tests
# ============================================================
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("RUNNER_DRY_RUN", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import bot_runner as br  # noqa: E402
import risk_book as rb  # noqa: E402

NOW = datetime(2026, 11, 23, 15, 0, tzinfo=timezone.utc)


def row(t="X", **kw):
    base = {"ticker": t, "price": 50.0, "priceLocal": 50.0, "currency": "USD", "fxRate": 1.0, "scan": True,
            "assetClass": "Equity", "isetf": "false", "marketcap": 10e9, "marketcapUSD": 10e9, "sector": "Technology"}
    base.update(kw)
    return base


def entry(dv, px=50.0, q=None, a63=400000):
    e = {"d": "2026-11-20", "px": px, "dv20": dv, "a63": a63}
    if q:
        e["q"] = q
    return e


def doc(tickers, at="2026-11-22T10:39:02Z"):
    return {"build": "z203", "at": at, "date": "2026-11-20", "tickers": tickers}


class Rules(unittest.TestCase):
    def test_cap_floor_and_unknown(self):
        self.assertEqual(rb.liquidity(row()), (None, None))                    # no figure -> no rule
        cap, why = rb.liquidity(row(dv20USD=50e6))
        self.assertAlmostEqual(cap, rb.LIQ_PCT * 50e6)
        self.assertIsNone(why)
        cap, why = rb.liquidity(row(dv20USD=120e3))
        self.assertEqual(cap, 0.0)
        self.assertIn("thin", why)
        self.assertIn("$120K", why)

    def test_not_equity_has_no_rule(self):
        for t, cls in (("ES=F", "Future"), ("EURUSD=X", "Currency"), ("^GSPC", "Index"), ("BTC-USD", "Crypto"),
                       ("AAPL261218C00200000", "Equity"), ("GC=F", "Equity")):
            self.assertFalse(rb.equity_like(row(t, assetClass=cls)), t)
            self.assertEqual(rb.liquidity(row(t, assetClass=cls, dv20USD=10e3)), (None, None), t)
        for t, cls in (("AAPL", "Equity"), ("SPY", "ETF"), ("BRK-B", None), ("RY.TO", "Equity")):
            self.assertTrue(rb.equity_like(row(t, assetClass=cls)), t)

    def test_short_needs_dollar_volume(self):
        self.assertEqual(rb.shortable(row())[0], True)                         # unknown -> the z190 rules only
        self.assertEqual(rb.shortable(row(dv20USD=25e6))[0], True)
        ok, why = rb.shortable(row(dv20USD=4e6))
        self.assertFalse(ok)
        self.assertIn("$4.0M traded a day", why)

    def test_knobs(self):
        k = rb.knobs()["liquidity"]
        self.assertEqual(k, {"pctOfDay": rb.LIQ_PCT, "minDollarVolume": rb.MIN_DV, "minShortDollarVolume": rb.MIN_SHORT_DV})


class ApplyMarketStats(unittest.TestCase):
    def test_usd_foreign_pence_and_skips(self):
        rows = {"AAA": row("AAA"),
                "LLOY.L": row("LLOY.L", currency="GBp", priceLocal=5000.0, price=63.5, fxRate=0.0127),   # pence -> USD
                "SAP.DE": row("SAP.DE", currency="EUR", priceLocal=200.0, price=230.0, fxRate=1.15),
                "NOFX.X": row("NOFX.X", currency="ZZZ", fxRate=None),
                "OLD": row("OLD"), "UNIT": row("UNIT", priceLocal=5000.0), "PXOFF": row("PXOFF"), "NONE": row("NONE"),
                "ES=F": row("ES=F", assetClass="Future", priceLocal=6900.0), "^GSPC": row("^GSPC", assetClass="Index", priceLocal=6900.0)}
        d = doc({"AAA": entry(30e6), "LLOY.L": entry(2.0e11, px=5000.0), "SAP.DE": entry(400e6, px=200.0),
                 "NOFX.X": entry(9e6), "OLD": entry(9e6, q=["stale"]), "UNIT": entry(9e6, px=50.0),
                 "PXOFF": entry(9e6, q=["px:x0.415"]), "ZZZ": entry(1e6), "ES=F": entry(9e11, px=6900.0), "^GSPC": entry(9e12, px=6900.0)})
        out = br.apply_market_stats(rows, doc=d, now=NOW)
        self.assertEqual(out, {"rows": 3, "stale": 0, "units": 1})
        self.assertEqual(rows["AAA"]["dv20USD"], 30e6)
        self.assertEqual(rows["AAA"]["a63"], 400000)
        self.assertAlmostEqual(rows["LLOY.L"]["dv20USD"], 2.0e11 * 0.0127, delta=1)      # pence x volume -> $2.54B
        self.assertAlmostEqual(rows["SAP.DE"]["dv20USD"], 460e6, delta=1)
        self.assertEqual(rows["SAP.DE"]["dv20"], 400e6)                         # the quote currency's figure kept
        for t in ("NOFX.X", "OLD", "UNIT", "PXOFF", "NONE", "ES=F", "^GSPC"):                  # futures / indices: no notional
            self.assertNotIn("dv20USD", rows[t], t)

    def test_old_or_undated_file_not_used(self):
        for at in ("2026-11-10T10:00:00Z", "", None):
            rows = {"AAA": row("AAA")}
            out = br.apply_market_stats(rows, doc=doc({"AAA": entry(30e6)}, at=at), now=NOW)
            self.assertEqual(out["rows"], 0, at)
            self.assertEqual(out["stale"], 1, at)
            self.assertNotIn("dv20USD", rows["AAA"])

    def test_no_file(self):
        self.assertEqual(br.apply_market_stats({"AAA": row("AAA")}, doc={}, now=NOW)["rows"], 0)


class Card(unittest.TestCase):
    def test_trade_card(self):
        u = row(dv20USD=10e6, a63=250000)
        self.assertEqual(br.liquidity_card(u, 100e3), {"dv20": 10000000, "pctOfDay": 1.0, "a63": 250000})
        self.assertIsNone(br.liquidity_card(u, 100e3, option={"type": "call"}))   # an option: its own book
        self.assertIsNone(br.liquidity_card(row(), 100e3))


if __name__ == "__main__":
    unittest.main()
