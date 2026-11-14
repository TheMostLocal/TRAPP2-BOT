# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   tests/test_risk_book.py
#  z186: the risk book - sizes, caps, the drawdown throttle. Run: python -m unittest discover -s tests
# ============================================================
import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import risk_book as rb  # noqa: E402


def series(n=200, drift=0.0, vol=0.02, seed=1, start=100.0, shock=None):
    """[(date, close)] - business-day-ish dates; `shock` = a common return stream added (for correlation)."""
    rnd = random.Random(seed)
    out, px = [], start
    for k in range(n):
        r = drift + vol * rnd.gauss(0, 1) + (shock[k] if shock else 0.0)
        px *= math.exp(r)
        out.append((f"2026-{1 + k // 28:02d}-{1 + k % 28:02d}", px))
    return out


class Sizing(unittest.TestCase):
    def test_risk_budget_not_notional(self):
        B = 100_000
        # 7 % stop, quiet name: 0.40 % of the book at the stop -> 5.71 % notional, capped at the 4.8 % ceiling
        n, risk, eff, why = rb.size_shares(B, 0.5, 0.07, 0.01, 1.0, cap_notional=4_800)
        self.assertEqual((round(n), why), (4_800, "size cap"))
        # 25 % core stop: the same 0.40 % budget buys 1.6 % of the book (was ~3.8 % - 3.5x the risk of a 7 % stop)
        n, risk, eff, why = rb.size_shares(B, 0.5, 0.25, 0.01, 1.0, cap_notional=4_800)
        self.assertAlmostEqual(n, 1_600, places=6)
        self.assertAlmostEqual(risk, 400, places=6)
        self.assertEqual(why, "risk budget")

    def test_volatility_widens_the_effective_stop_and_caps_the_size(self):
        B = 100_000
        n, risk, eff, why = rb.size_shares(B, 0.5, 0.07, 0.04, 1.0, cap_notional=10_000)   # 4 % a day
        self.assertAlmostEqual(eff, 0.10)                     # 2.5 sigmas > the 7 % stop
        self.assertAlmostEqual(n, 3_000, places=6)            # the 0.12 %-a-day volatility cap binds before 4,000
        self.assertEqual(why, "volatility cap")

    def test_confidence_and_throttle(self):
        self.assertAlmostEqual(rb.per_trade_risk(1.0), rb.RISK_PER_TRADE_MAX)
        self.assertAlmostEqual(rb.per_trade_risk(0.3), rb.RISK_PER_TRADE)
        a = rb.size_shares(100_000, 0.5, 0.25, None, 1.0)[0]
        b = rb.size_shares(100_000, 0.5, 0.25, None, 0.5)[0]
        self.assertAlmostEqual(b, a / 2)


class Drawdown(unittest.TestCase):
    def test_steps_and_rolling_mark(self):
        curve = [{"date": f"d{k:03d}", "value": v} for k, v in enumerate([100_000] * 5 + [120_755] + [115_000] * 10)]
        d = rb.drawdown(curve, 111_394.44)
        self.assertEqual(d["hwm"], 120_755)
        self.assertAlmostEqual(d["dd"], 1 - 111_394.44 / 120_755, places=4)
        self.assertEqual(d["scale"], 0.70)                    # 7.75 % below the mark
        self.assertEqual(rb.drawdown(curve, 120_000)["scale"], 1.0)
        self.assertEqual(rb.drawdown(curve, 108_000)["scale"], 0.50)      # 10.6 %
        self.assertEqual(rb.drawdown(curve, 90_000)["scale"], 0.35)
        old = [{"date": "x", "value": 200_000}] + [{"date": f"d{k}", "value": 100_000} for k in range(rb.DD_WINDOW)]
        self.assertEqual(rb.drawdown(old, 100_000)["dd"], 0.0)   # an old peak falls out of the window

    def test_day_change(self):
        curve = [{"date": "2026-11-12", "value": 100_000}, {"date": "2026-11-13", "value": 101_000}]
        self.assertAlmostEqual(rb.day_change(curve, 98_980, "2026-11-16"), -0.02)
        self.assertAlmostEqual(rb.day_change(curve + [{"date": "2026-11-16", "value": 1}], 98_980, "2026-11-16"), -0.02)
        self.assertIsNone(rb.day_change([], 1, "2026-11-16"))


class Caps(unittest.TestCase):
    def setUp(self):
        self.B = 100_000.0

    def test_sector_cap_and_new_count(self):
        bk = rb.Book(self.B)
        bk.add("COP", "Energy", "long", 4_400, 4_400)
        bk.add("CVX", "Energy", "long", 4_400, 4_400)
        bk.add("MPC", "Energy", "long", 4_000, 4_000)
        ok, why = bk.check("XOM", "Energy", "long", 3_000, 3_000, 1.0, None, 100)
        self.assertFalse(ok)
        self.assertIn("sector cap", why)
        ok, _ = bk.check("XOM", "Energy", "long", 2_000, 2_000, 1.0, None, 100)
        self.assertTrue(ok)
        # a short against a long-heavy sector shrinks its net: a hedge, not a breach (until the 30 % gross cap)
        self.assertTrue(bk.check("OXY", "Energy", "short", 3_000, -3_000, 1.0, None, 100)[0])
        heavy = rb.Book(self.B)
        heavy.add("A", "Energy", "long", 14_000, 14_000)
        heavy.add("B", "Energy", "short", 14_000, -14_000)
        ok, why = heavy.check("C", "Energy", "short", 3_000, -3_000, 1.0, None, 100)
        self.assertFalse(ok)
        self.assertIn("gross cap", why)
        bk2 = rb.Book(self.B)
        for t in ("AAA", "BBB"):
            bk2.add(t, "Healthcare", "long", 1_000, 1_000, new=True, new_risk=10)
        ok, why = bk2.check("CCC", "Healthcare", "long", 1_000, 1_000, 1.0, None, 10)
        self.assertFalse(ok)
        self.assertIn("new-trade limit", why)

    def test_cluster(self):
        g = random.Random(7)
        common = [0.02 * g.gauss(0, 1) for _ in range(200)]
        a = rb.returns_by_date(series(shock=common, vol=0.004, seed=1))
        b = rb.returns_by_date(series(shock=common, vol=0.004, seed=2))
        c = rb.returns_by_date(series(shock=common, vol=0.004, seed=3))
        z = rb.returns_by_date(series(vol=0.02, seed=4))
        self.assertGreater(rb.corr(a, b), 0.9)
        bk = rb.Book(self.B)
        bk.add("MPC", "Energy", "long", 1_000, 1_000, rets=a)
        bk.add("VLO", "Energy", "long", 1_000, 1_000, rets=b)
        ok, why = bk.check("PSX", "Industrials", "long", 1_000, 1_000, 1.0, c, 10)
        self.assertFalse(ok)
        self.assertIn("cluster", why)
        self.assertTrue(bk.check("PSX", "Industrials", "short", 1_000, -1_000, 1.0, c, 10)[0], "a short isn't the same bet")
        self.assertTrue(bk.check("ZZZ", "Industrials", "long", 1_000, 1_000, 1.0, z, 10)[0])

    def test_net_beta_cap_lets_hedges_through(self):
        bk = rb.Book(self.B, regime_dialed=False)
        for k in range(10):
            bk.add(f"L{k}", f"S{k}", "long", 4_000, 4_000, beta_v=1.1)
        ok, why = bk.check("NEW", "Tech", "long", 4_000, 4_000, 1.2, None, 100)
        self.assertFalse(ok)
        self.assertIn("net beta cap", why)
        self.assertTrue(bk.check("SHRT", "Tech", "short", 4_000, -4_000, 1.2, None, 100)[0])
        self.assertTrue(bk.check("SQQQ", "Tech", "long", 3_000, 3_000, -3.3, None, 100)[0], "an inverse ETF bought long lowers beta")
        dialed = rb.Book(self.B, regime_dialed=True)
        for k in range(10):
            dialed.add(f"L{k}", f"S{k}", "long", 4_000, 4_000, beta_v=1.1)
        self.assertTrue(dialed.check("NEW", "Tech", "long", 4_000, 4_000, 1.2, None, 100)[0])

    def test_gross_new_risk_and_halt(self):
        bk = rb.Book(self.B)
        bk.add("X", "A", "long", 108_000, 0.0)
        self.assertIn("gross cap", bk.check("Y", "B", "long", 4_000, 0.0, 0.0, None, 10)[1])
        bk = rb.Book(self.B)
        bk.add("X", "A", "long", 1_000, 0.0, new=True, new_risk=2_400)
        self.assertIn("new-risk cap", bk.check("Y", "B", "long", 1_000, 0.0, 0.0, None, 200)[1])
        self.assertIn("daily halt", rb.Book(self.B, halted=True).check("Y", "B", "long", 1, 1, 1, None, 1)[1])

    def test_exposure_shortable_loss_cap(self):
        self.assertEqual(rb.position_exposure({"shares": 10, "direction": "short"}, 50.0), (500.0, -500.0))
        self.assertEqual(rb.position_exposure({"instrument": "option", "maxLoss": 900, "direction": "short", "optionType": "put"}, 1), (900.0, -900.0))
        self.assertEqual(rb.position_exposure({"instrument": "option", "maxLoss": 900, "direction": "long", "optionLegs": [1, 2]}, 1), (900.0, 900.0))
        self.assertFalse(rb.shortable({"price": 6.0, "marketcap": 5e9})[0])
        self.assertFalse(rb.shortable({"price": 60.0, "marketcap": 1e9})[0])
        self.assertFalse(rb.shortable({"price": 60.0, "marketcap": 5e10, "isetf": "true"})[0])
        self.assertTrue(rb.shortable({"price": 60.0, "marketcap": 5e10, "isetf": "false"})[0])
        self.assertTrue(rb.pos_loss_capped({"pnl": -800}, 100_000))
        self.assertFalse(rb.pos_loss_capped({"pnl": -700}, 100_000))
        self.assertFalse(rb.pos_loss_capped({"pnl": -5_000, "instrument": "option"}, 100_000))

    def test_returns_drop_data_jumps(self):
        r = rb.returns_by_date([("a", 100.0), ("b", 101.0), ("c", 10.0), ("d", 10.2)])
        self.assertEqual(sorted(r), ["b", "d"])               # the x0.1 day is a data question, not a return


if __name__ == "__main__":
    unittest.main()
