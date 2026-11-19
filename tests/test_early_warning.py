# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   tests/test_early_warning.py
#  z196: the early-warning read in every trade - the 'warning' signal (a fund's in its own direction), risk / industry
#  size multipliers (increases only when the backtest earned them), the leveraged-long guard, the journal card; and the
#  options market read from TRAPP2-OPTIONS market.json / signals.json instead of a sweep of ~2,300 chains.
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

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")


def doc(risk=50.0, level="moderate", regime="neutral", throttle=True, organic=False, growth=50.0, exposures=None, provisional=False, date=TODAY):
    return {"date": date, "risk": risk, "level": level, "regime": regime, "provisional": provisional,
            "headline": f"{level.upper()} risk", "gauges": {"growth": {"pct": growth}, "fundamental": {"pct": 60}, "market": {"pct": 40}},
            "flags": [{"kind": "blindness"}] if regime == "stress building" else [],
            "exposures": exposures or {"industry": {}, "sector": {}},
            "backtest": {"earned": {"throttle": throttle, "organicUpside": organic}}}


def set_warn(d, stale=False):
    br._WARN.update(loaded=True, doc=d, stale=stale)


def tearDownModule():
    br._WARN.update(loaded=False, doc=None, stale=True)
    br._OPT_MARKET.update(done=False, data=None)
    br._OPT_SIGNALS["doc"] = None


EXPO = {"industry": {"Credit Services": {"e": 0.8, "why": "consumer credit stress 91/100 (lead: Credit-card delinquencies)"}},
        "sector": {"Real Estate": {"e": 0.4, "why": "housing / mortgages stress 77/100"}}}


class Load(unittest.TestCase):
    def setUp(self):
        self.real = br.fetch_json
        br._WARN.update(loaded=False, doc=None, stale=True)

    def tearDown(self):
        br.fetch_json = self.real
        br._WARN.update(loaded=False, doc=None, stale=True)

    def test_fresh(self):
        br.fetch_json = lambda url, timeout=45: doc()
        self.assertFalse(br.load_warning()["stale"])

    def test_stale_old_or_provisional_or_missing(self):
        br.fetch_json = lambda url, timeout=45: doc(date="2026-01-02")
        self.assertTrue(br.load_warning()["stale"])
        br._WARN.update(loaded=False)
        br.fetch_json = lambda url, timeout=45: doc(provisional=True)
        self.assertTrue(br.load_warning()["stale"])
        br._WARN.update(loaded=False)
        br.fetch_json = lambda url, timeout=45: None
        self.assertTrue(br.load_warning()["stale"])
        self.assertEqual(br.warning_size_mult("Technology", "Software"), (1.0, None))
        self.assertIsNone(br.warning_signal("Technology", "Software"))


class Size(unittest.TestCase):
    def test_longs_shrink_as_risk_rises(self):
        ms = []
        for r in (30, 65, 80, 95):
            set_warn(doc(risk=r))
            ms.append(br.warning_size_mult("Technology", "Software")[0])
        self.assertEqual(ms, [1.0, 0.9, 0.8, 0.7])

    def test_unearned_throttle_is_mild(self):
        set_warn(doc(risk=95, throttle=False))
        self.assertEqual(br.warning_size_mult("Technology", "Software")[0], 0.9)

    def test_exposed_industry_smaller(self):
        set_warn(doc(risk=50, exposures=EXPO))
        m, why = br.warning_size_mult("Financial Services", "Credit Services")
        self.assertAlmostEqual(m, round(1 - 0.35 * 0.8, 3))
        self.assertIn("Credit Services", why)
        self.assertEqual(br.warning_size_mult("Real Estate", "REIT - Industrial")[0], round(1 - 0.35 * 0.4, 3))   # sector match

    def test_short_in_exposed_industry_bigger_only_when_earned(self):
        set_warn(doc(risk=80, exposures=EXPO))
        self.assertEqual(br.warning_size_mult("Financial Services", "Credit Services", short=True)[0], round(1 + 0.15 * 0.8, 3))
        set_warn(doc(risk=80, exposures=EXPO, throttle=False))
        self.assertEqual(br.warning_size_mult("Financial Services", "Credit Services", short=True)[0], 1.0)

    def test_levered_long_extra_cut_inverse_is_a_short(self):
        set_warn(doc(risk=80))
        self.assertEqual(br.warning_size_mult(None, None, etp={"lev": 3, "underlying": "equity"})[0], round(0.8 * 0.85, 3))
        self.assertEqual(br.warning_size_mult(None, None, etp={"lev": -3, "underlying": "equity"})[0], 1.0)

    def test_organic_upside_only_when_earned(self):
        set_warn(doc(risk=30, regime="organic expansion", organic=False))
        self.assertEqual(br.warning_size_mult("Industrials", "Railroads")[0], 1.0)
        set_warn(doc(risk=30, regime="organic expansion", organic=True))
        self.assertEqual(br.warning_size_mult("Industrials", "Railroads")[0], 1.05)

    def test_stale_is_context_only(self):
        set_warn(doc(risk=95, exposures=EXPO), stale=True)
        self.assertEqual(br.warning_size_mult("Financial Services", "Credit Services"), (1.0, None))
        card = br.warning_card("Financial Services", "Credit Services")
        self.assertTrue(card["stale"])
        self.assertEqual(card["exposure"], 0.0)


class Signal(unittest.TestCase):
    def test_risk_and_exposure_push_negative(self):
        set_warn(doc(risk=90, exposures=EXPO))
        self.assertEqual(br.warning_signal("Financial Services", "Credit Services"), round(-0.5 * 1.0 - 0.6 * 0.8, 3))
        self.assertEqual(br.warning_signal("Technology", "Software"), -0.5)

    def test_low_risk_leans_positive_only_when_earned(self):
        set_warn(doc(risk=10, throttle=True))
        self.assertEqual(br.warning_signal("Technology", "Software"), 0.5)
        set_warn(doc(risk=10, throttle=False))
        self.assertEqual(br.warning_signal("Technology", "Software"), 0.0)

    def test_earned_organic_growth_adds(self):
        set_warn(doc(risk=50, regime="organic expansion", organic=True, growth=90))
        self.assertEqual(br.warning_signal("Industrials", "Railroads"), 0.3)

    def test_fund_signed_by_its_leverage(self):
        set_warn(doc(risk=90))
        etp = {"lev": -3, "underlying": "equity"}
        comps = {"warning": br.warning_signal(None, None, etp)}
        br._etp_sign_market(comps, etp, None)
        self.assertEqual(comps["warning"], 0.5)           # a bear fund: high risk is FOR it
        comps = {"warning": br.warning_signal(None, None, {"lev": 2, "underlying": "rates"})}
        br._etp_sign_market(comps, {"lev": 2, "underlying": "rates"}, None)
        self.assertNotIn("warning", comps)                # an equity read says nothing about a rates fund

    def test_in_the_blend(self):
        self.assertIn("warning", br.BASE_WEIGHTS)
        self.assertIn("warning", br.ETP_MARKET_SIGNALS)


class Gate(unittest.TestCase):
    def test_no_levered_long_in_extreme_acute(self):
        set_warn(doc(risk=97, level="extreme", regime="acute stress"))
        self.assertIsNotNone(br.warning_gate("long", {"lev": 3, "underlying": "equity"}))
        self.assertIsNone(br.warning_gate("long", {"lev": -3, "underlying": "equity"}))     # inverse allowed
        self.assertIsNone(br.warning_gate("long", None))                                    # stocks: sizing, not a bar
        self.assertIsNone(br.warning_gate("short", {"lev": 3, "underlying": "equity"}))
        set_warn(doc(risk=97, level="extreme", regime="repair"))
        self.assertIsNone(br.warning_gate("long", {"lev": 3, "underlying": "equity"}))


class OptionsMarket(unittest.TestCase):
    def setUp(self):
        self.real = br.fetch_json
        br._OPT_MARKET.update(done=False, data=None)
        br._OPT_SIGNALS["doc"] = None
        br._OPT_CACHE.clear()

    def tearDown(self):
        br.fetch_json = self.real
        br._OPT_MARKET.update(done=False, data=None)
        br._OPT_SIGNALS["doc"] = None

    def test_market_json_no_chain_sweep(self):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        seen = []

        def fj(url, timeout=45):
            seen.append(url)
            if url.endswith("/market.json"):
                return {"asOf": now, "appSignal": {"count": 460, "indexIV": 0.21, "breadthIV": 0.33, "avgSkew": 0.01,
                                                   "volRegime": "normal", "marketTrend": 0.0}, "breadth": {"invertedShare": 0.2}}
            raise AssertionError(f"swept a chain: {url}")
        br.fetch_json = fj
        m = br.compute_options_market([])
        self.assertEqual((m["volRegime"], m["count"], m["source"]), ("normal", 460, "market.json"))
        self.assertEqual(len(seen), 1)

    def test_stale_market_json_falls_back_capped(self):
        urls = []

        def fj(url, timeout=45):
            urls.append(url)
            if url.endswith("/market.json"):
                return {"asOf": "2026-01-01T00:00:00Z", "appSignal": {"indexIV": 0.2}}
            if url.endswith("/manifest.json"):
                return {"tickers": ["run_report"] + [f"T{i}" for i in range(500)] + ["SPY"]}
            return None
        br.fetch_json = fj
        br.compute_options_market([])
        chains = [u for u in urls if u.endswith(".json") and "/options/T" in u or u.endswith("/SPY.json")]
        self.assertLessEqual(len([u for u in urls if "/options/" in u and not u.endswith(("market.json", "manifest.json", "signals.json"))]),
                             2 * br.OPT_MARKET_SWEEP_CAP)   # chain + legacy fallback per name at most
        self.assertTrue(any(u.endswith("/SPY.json") for u in urls))     # index ETFs first
        self.assertFalse(any(u.endswith("/run_report.json") for u in urls))

    def test_per_name_signal_from_signals_json(self):
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        def fj(url, timeout=45):
            if url.endswith("/signals.json"):
                return {"tickers": {"ABC": [0.5, 0.08, 0.14, -0.51, 30, day], "OLD": [0.3, 0.0, 0.0, 0.0, 30, "2026-01-01"]}}
            if url.endswith("/OLD.json"):
                return None
            raise AssertionError(f"loaded a chain: {url}")
        br.fetch_json = fj
        self.assertEqual(br.extract_options_signal("ABC")["ivSignal"], -0.51)
        self.assertIsNone(br.extract_options_signal("OLD"))       # stale row -> the chain (none here)


if __name__ == "__main__":
    unittest.main()
