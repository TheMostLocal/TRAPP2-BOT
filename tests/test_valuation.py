# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   tests/test_valuation.py
#  z187: P/B, P/S and the valuation step - one currency, today's price, sector-relative.
#  Run: python -m unittest discover -s tests
# ============================================================
import math
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("RUNNER_DRY_RUN", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import bot_runner as br  # noqa: E402


def row(**kw):
    """A load_universe() row: USD market cap / FX already attached, like the runner builds it."""
    base = {"ticker": "X", "currency": "USD", "financialCurrency": "USD", "marketcap": 10e9, "marketcapUSD": 10e9, "finUsd": 1.0,
            "sector": "Technology", "valuationNote": "", "valuationZ187": True, "assetClass": "Equity", "isetf": "false"}
    base.update(kw)
    return base


class NormFund(unittest.TestCase):
    def test_debt_to_equity_is_a_percent(self):
        f = br._norm_fund(row(debtToEquity=84.0))
        self.assertAlmostEqual(f["debtToEquity"], 0.84)               # Yahoo's 84 = 0.84x, not 84x

    def test_pipeline_multiples_are_used_as_written(self):
        f = br._norm_fund(row(priceToBook=10.22, priceToSales=14.46, evToEbitda=18.9, pe=27.1))
        self.assertEqual((f["priceToBook"], f["priceToSales"], f["evToEbitda"]), (10.22, 14.46, 18.9))

    def test_negative_book_has_no_pb(self):
        f = br._norm_fund(row(priceToBook=-559.6, valuationNote="neg-book", totalEquity=-1.0e9))
        self.assertIsNone(f["priceToBook"])
        self.assertTrue(f["negBook"])
        g = br._norm_fund(row(valuationZ187=False, priceToBook=-559.6, totalEquity=-1.0e9))
        self.assertIsNone(g["priceToBook"])

    def test_legacy_adr_is_rebuilt_in_usd(self):
        # TSM before z187: Yahoo's P/B 76 (USD price / TWD book), P/S 0.4 if market cap / TWD revenue
        twd = 0.0304
        u = row(valuationZ187=False, financialCurrency="TWD", finUsd=twd, marketcap=2.07e12, marketcapUSD=2.07e12,
                priceToBook=76.09, totalEquity=6.998e12, revenue=4.9448e12, evToEbitda=0.9)
        f = br._norm_fund(u)
        self.assertAlmostEqual(f["priceToBook"], 2.07e12 / (6.998e12 * twd), places=6)   # ~9.7, not 76
        self.assertAlmostEqual(f["priceToSales"], 2.07e12 / (4.9448e12 * twd), places=6)  # ~13.8, not 0.4
        self.assertIsNone(f["evToEbitda"])                                 # Yahoo's cross-currency EV ratio dropped

    def test_legacy_share_class_pb(self):
        # BRK-B: Yahoo divides the class-B price by class-A book per share -> 0.0011
        f = br._norm_fund(row(valuationZ187=False, priceToBook=0.001105, marketcap=1.2597e12, marketcapUSD=1.2597e12,
                              totalEquity=7.174e11))
        self.assertAlmostEqual(f["priceToBook"], 1.2597e12 / 7.174e11, places=6)

    def test_market_cap_in_usd_for_size_tiers(self):
        f = br._norm_fund(row(currency="JPY", financialCurrency="JPY", marketcap=4.0e13, marketcapUSD=4.0e13 * 0.00625))
        self.assertAlmostEqual(f["marketCap"], 2.5e11)                    # 40 trillion yen is $250B, not $40T
        g = br._norm_fund(row(currency="JPY", financialCurrency="JPY", marketcap=4.0e13, marketcapUSD=None))
        self.assertIsNone(g["marketCap"])                                  # no rate -> unknown, never yen as dollars

    def test_suspect_cap_keeps_pe_only(self):
        f = br._norm_fund(row(pe=20.0, priceToBook=3.0, priceToSales=2.0, valuationNote="mcap?"))
        self.assertEqual(br._val_inputs(f), {"pe": 20.0})


class Valuation(unittest.TestCase):
    def setUp(self):
        br.VAL_STATS.clear()
        br.VAL_STATS.update({"Technology": {"pe": 25.0, "ps": 4.0, "pb": 5.0, "evEbitda": 16.0},
                             "Financial Services": {"pe": 12.0, "ps": 3.0, "pb": 1.4, "evEbitda": 11.0},
                             "_all": {"pe": 20.0, "ps": 2.0, "pb": 2.0, "evEbitda": 12.0}})

    def tearDown(self):
        br.VAL_STATS.clear()

    def test_at_the_median_is_fair(self):
        f = br._norm_fund(row(pe=25.0, priceToSales=4.0, priceToBook=5.0, evToEbitda=16.0))
        self.assertAlmostEqual(br.valuation_points(f, "Technology", False), 50.0)

    def test_cheaper_and_dearer(self):
        cheap = br.valuation_points(br._norm_fund(row(pe=17.5)), "Technology", False)       # 30 % below the median
        dear = br.valuation_points(br._norm_fund(row(pe=50.0)), "Technology", False)        # twice the median
        self.assertAlmostEqual(cheap, 50 + 36 * math.tanh(-math.log(0.7) / 0.5), places=6)
        self.assertGreater(cheap, 70)
        self.assertLess(dear, 20)

    def test_quality_premium_counts_less(self):
        f = br._norm_fund(row(pe=50.0))
        self.assertGreater(br.valuation_points(f, "Technology", True), br.valuation_points(f, "Technology", False))

    def test_relative_not_absolute(self):
        # the same 3x sales is dear for a bank-like sector median of 3 ... and cheap against software at 4x
        f = br._norm_fund(row(priceToSales=3.0))
        self.assertGreater(br.valuation_points(f, "Technology", False), 50)
        self.assertAlmostEqual(br.valuation_points(f, "Financial Services", False), 50.0)

    def test_pb_weight_by_sector(self):
        # P/B half the median: counts fully for a lender, a third as much for software
        bank = br._norm_fund(row(sector="Financial Services", pe=12.0, priceToBook=0.7))
        soft = br._norm_fund(row(pe=25.0, priceToBook=2.5))
        self.assertGreater(br.valuation_points(bank, "Financial Services", False) - 50,
                           br.valuation_points(soft, "Technology", False) - 50)

    def test_unknown_sector_falls_back_to_universe(self):
        f = br._norm_fund(row(sector="Nowhere", pe=20.0))
        self.assertAlmostEqual(br.valuation_points(f, "Nowhere", False), 50.0)

    def test_engine_uses_relative_step(self):
        a = br._fund_engine(row(pe=17.5, priceToSales=2.8, returnOnEquity=0.1, profitMargin=0.1))
        b = br._fund_engine(row(pe=50.0, priceToSales=8.0, returnOnEquity=0.1, profitMargin=0.1))
        self.assertGreater(a["conviction"], b["conviction"])

    def test_negative_book_buybacks_not_distress(self):
        common = dict(pe=22.0, debtToEquity=None, totalDebt=5e10, cash=1e9, currentRatio=1.0)
        buyback = br._fund_engine(row(totalEquity=-1e9, freeCashFlow=7e9, profitMargin=0.3, **common))
        distress = br._fund_engine(row(totalEquity=-1e9, freeCashFlow=-2e9, profitMargin=-0.2, **common))
        self.assertGreater(buyback["health"], distress["health"])

    def test_card(self):
        c = br.valuation_card(row(pe=17.5, priceToSales=4.0, priceToBook=5.0, sector="Technology"))
        self.assertEqual((c["pe"], c["peMedian"], c["ps"], c["psMedian"]), (17.5, 25.0, 4.0, 4.0))
        self.assertIsNotNone(c["points"])


class Medians(unittest.TestCase):
    def test_needs_eight_peers(self):
        rows = {f"T{i}": row(ticker=f"T{i}", pe=10.0 + i, sector="Utilities") for i in range(8)}
        rows.update({f"S{i}": row(ticker=f"S{i}", pe=30.0, sector="Tiny") for i in range(3)})
        st = br.valuation_stats(rows)
        self.assertAlmostEqual(st["Utilities"]["pe"], 13.5)
        self.assertNotIn("Tiny", st)                                       # 3 peers - the universe median is used
        self.assertIn("pe", st["_all"])

    def test_etfs_and_non_equities_left_out(self):
        rows = {f"E{i}": row(ticker=f"E{i}", pe=10.0, isetf="true") for i in range(10)}
        rows.update({f"F{i}": row(ticker=f"F{i}", pe=10.0, assetClass="Future") for i in range(10)})
        self.assertEqual(br.valuation_stats(rows), {})


class Renamed(unittest.TestCase):
    def test_thirty_days(self):
        self.assertTrue(br.recently_renamed({"nameChangedAt": "2026-11-01"}, "2026-11-15"))
        self.assertFalse(br.recently_renamed({"nameChangedAt": "2026-09-01"}, "2026-11-15"))
        self.assertFalse(br.recently_renamed({}, "2026-11-15"))


if __name__ == "__main__":
    unittest.main()
