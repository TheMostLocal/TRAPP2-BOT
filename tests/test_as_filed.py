# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   tests/test_as_filed.py
#  z204: the valuation multiples checked against the SEC filings as filed (TRAPP2-ANALYTICS
#  data/derived/fundamentals.json): a disputed denominator leaves its multiple out, the filing's figure prices the
#  multiple where Yahoo has none or the filing is newer, verified / Yahoo-newer rows unchanged; checked at use (Yahoo's
#  figure moved since the file -> not applied), USD only, an old file not used; the trade card says which.
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

NOW = datetime(2026, 11, 23, 15, 0, tzinfo=timezone.utc)


def row(**kw):
    base = {"ticker": "X", "price": 100.0, "currency": "USD", "financialCurrency": "USD", "marketcap": 10e9, "marketcapUSD": 10e9,
            "finUsd": 1.0, "sector": "Technology", "valuationNote": "", "valuationZ187": True, "assetClass": "Equity",
            "isetf": "false", "eps": 5.0, "pe": 20.0, "revenueUSD": 5e9, "priceToSales": 2.0, "bookUSD": 4e9, "priceToBook": 2.5,
            "ebitdaUSD": 1e9, "evToEbitda": 11.0, "totalDebt": 2e9, "cash": 1e9}
    base.update(kw)
    return base


def entry(use, y, t=None, b=None):
    return {"end": "2026-09-30", "filed": ["10-Q", "2026-11-03"], "use": use, "y": y,
            "t": t or {"eps": 5.0, "rev": 5e9, "ebitda": 1e9}, "b": b or {"eq": 4e9, "debt": 2e9, "cashT": 1e9}}


def doc(tickers, at="2026-11-23T03:10:00Z"):
    return {"build": "z204", "at": at, "tickers": tickers}


ALL_SEC = {"eps": "sec", "rev": "sec", "eq": "sec", "ebitda": "sec", "debt": "sec", "cash": "sec"}
AGREE = {"eps": [5.0, 0.0, "agree"], "rev": [5e9, 0.0, "agree"], "eq": [4e9, 0.0, "agree"], "ebitda": [1e9, 0.0, "agree"],
         "debt": [2e9, 0.0, "agree"], "cash": [1e9, 0.0, "agree"]}


class AsFiled(unittest.TestCase):
    def test_verified_rows_unchanged(self):
        rows = {"X": row()}
        out = br.apply_as_filed(rows, doc({"X": entry(ALL_SEC, AGREE)}), now=NOW)
        u = rows["X"]
        self.assertEqual((u["pe"], u["priceToSales"], u["priceToBook"], u["evToEbitda"]), (20.0, 2.0, 2.5, 11.0))
        self.assertEqual(out["verified"], 4)
        self.assertEqual(u["asFiled"]["multiples"], {"pe": "verified", "ps": "verified", "pb": "verified", "evEbitda": "verified"})

    def test_disputed_denominator_drops_its_multiple(self):
        rows = {"MAR": row(ticker="MAR", revenueUSD=7.581e9)}
        y = dict(AGREE, rev=[7.581e9, 2.6169, "check"])
        out = br.apply_as_filed(rows, doc({"MAR": entry(dict(ALL_SEC, rev=None), y)}), now=NOW)
        u = rows["MAR"]
        self.assertIsNone(u["priceToSales"])
        self.assertEqual(u["pe"], 20.0)
        self.assertEqual(out["dropped"], 1)
        self.assertEqual(u["asFiled"]["multiples"]["ps"], "disputed")
        self.assertNotIn("ps", br._val_inputs(br._norm_fund(u)))         # the valuation step never sees it

    def test_filing_prices_where_yahoo_has_none_or_is_older(self):
        rows = {"N": row(ticker="N", eps=None, pe=None, bookUSD=None, priceToBook=None, revenueUSD=4.6e9, priceToSales=2.17)}
        y = dict(AGREE, eps=[None, None, "only"], eq=[None, None, "only"], rev=[4.6e9, 0.087, "newer"])
        e = entry(ALL_SEC, y, t={"eps": 4.0, "rev": 5e9, "ebitda": 1e9}, b={"eq": 2e9, "debt": 2e9, "cashT": 1e9})
        br.apply_as_filed(rows, doc({"N": e}), now=NOW)
        u = rows["N"]
        self.assertAlmostEqual(u["pe"], 25.0)                 # 100 / 4.00 as filed
        self.assertAlmostEqual(u["priceToBook"], 5.0)         # 10 B / 2 B common equity
        self.assertAlmostEqual(u["priceToSales"], 2.0)        # 10 B / 5 B (the newer quarter)
        self.assertEqual(u["asFiled"]["multiples"], {"pe": "filing", "ps": "filing", "pb": "filing", "evEbitda": "verified"})

    def test_yahoo_newer_leaves_it(self):
        rows = {"O": row(ticker="O", eps=5.2, pe=19.23)}
        y = dict(AGREE, eps=[5.2, -0.04, "older"])
        out = br.apply_as_filed(rows, doc({"O": entry(dict(ALL_SEC, eps="yahoo"), y)}), now=NOW)
        self.assertEqual(rows["O"]["pe"], 19.23)
        self.assertNotIn("pe", rows["O"]["asFiled"]["multiples"])
        self.assertEqual(out["moved"], 0)

    def test_checked_at_use_yahoo_moved(self):
        rows = {"M": row(ticker="M", revenueUSD=9.9e9)}         # Yahoo corrected its revenue since the file was built
        y = dict(AGREE, rev=[7.581e9, 2.6, "check"])
        out = br.apply_as_filed(rows, doc({"M": entry(dict(ALL_SEC, rev=None), y)}), now=NOW)
        self.assertEqual(rows["M"]["priceToSales"], 2.0)
        self.assertEqual(out["moved"], 1)
        rows = {"F": row(ticker="F", eps=4.9, pe=20.4)}           # Yahoo has an EPS now where the file saw none
        br.apply_as_filed(rows, doc({"F": entry(ALL_SEC, dict(AGREE, eps=[None, None, "only"]), t={"eps": 4.0})}), now=NOW)
        self.assertEqual(rows["F"]["pe"], 20.4)

    def test_negative_or_missing_figures_never_price(self):
        rows = {"L": row(ticker="L", eps=None, pe=None)}
        br.apply_as_filed(rows, doc({"L": entry(ALL_SEC, dict(AGREE, eps=[None, None, "only"]), t={"eps": -1.2})}), now=NOW)
        self.assertIsNone(rows["L"]["pe"])                     # a loss: no P/E (the valuation step reads the rest)
        rows = {"Z": row(ticker="Z", eps=None, pe=None)}
        br.apply_as_filed(rows, doc({"Z": entry(ALL_SEC, dict(AGREE, eps=[None, None, "only"]), t={"eps": 23747.4})}), now=NOW)
        self.assertIsNone(rows["Z"]["pe"])                     # P/E 0.004 from the filing alone: implausible, not used

    def test_usd_only_and_old_file(self):
        rows = {"T": row(ticker="T", financialCurrency="TWD"), "U": row(ticker="U", currency="GBp")}
        out = br.apply_as_filed(rows, doc({"T": entry(dict(ALL_SEC, rev=None), AGREE), "U": entry(dict(ALL_SEC, rev=None), AGREE)}), now=NOW)
        self.assertEqual(out["checked"], 0)
        self.assertEqual(rows["T"]["priceToSales"], 2.0)
        rows = {"X": row()}
        out = br.apply_as_filed(rows, doc({"X": entry(dict(ALL_SEC, rev=None), AGREE)}, at="2026-11-01T03:00:00Z"), now=NOW)
        self.assertEqual(out["stale"], 1)
        self.assertEqual(rows["X"]["priceToSales"], 2.0)
        self.assertEqual(br.apply_as_filed({"X": row()}, {"nope": 1}, now=NOW)["checked"], 0)

    def test_ev_ebitda_needs_all_three(self):
        rows = {"E": row(ticker="E", evToEbitda=None, ebitdaUSD=None)}
        y = dict(AGREE, ebitda=[None, None, "only"], debt=[None, None, "partial"])
        br.apply_as_filed(rows, doc({"E": entry(dict(ALL_SEC, debt="yahoo"), y)}), now=NOW)
        self.assertIsNone(rows["E"]["evToEbitda"])            # debt incomplete before the z204 re-read: not priced from ours
        rows = {"E": row(ticker="E", evToEbitda=None, ebitdaUSD=None)}
        br.apply_as_filed(rows, doc({"E": entry(ALL_SEC, dict(AGREE, ebitda=[None, None, "only"]))}), now=NOW)
        self.assertAlmostEqual(rows["E"]["evToEbitda"], (10e9 + 2e9 - 1e9) / 1e9)

    def test_card_says_which(self):
        u = row(asFiled={"multiples": {"ps": "disputed"}}, priceToSales=None)
        br.VAL_STATS.clear()
        br.VAL_STATS.update({"_all": {"pe": 20.0, "ps": 2.0, "pb": 2.5, "evEbitda": 11.0}})
        try:
            c = br.valuation_card(u)
        finally:
            br.VAL_STATS.clear()
        self.assertEqual(c["asFiled"], {"ps": "disputed"})
        self.assertNotIn("ps", c)


if __name__ == "__main__":
    unittest.main()
