# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   tests/test_credit.py
#  z194: credit ratings in every trade - the 'credit' signal (an ETP's in its own direction), the distressed-credit
#  entry guard, credit-scaled sizes, and the distress / confirmed-downgrade exits. Run: python -m unittest discover -s tests
# ============================================================
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("RUNNER_DRY_RUN", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import bot_runner as br  # noqa: E402

TODAY = "2026-11-18"


def set_ratings(by, stale=False):
    br._RATINGS.update(loaded=True, by=by, asOf="2026-11-18", stale=stale)


RATED = {
    "AAA1": {"k": "co", "r": "AA", "n": 3, "sig": 0.25},
    "IGCO": {"k": "co", "r": "BBB", "n": 9, "sig": 0.05},
    "BBCO": {"k": "co", "r": "BB", "n": 12, "sig": -0.10},
    "BCO": {"k": "co", "r": "B-", "n": 16, "sig": -0.20},
    "CCCO": {"k": "co", "r": "CCC", "n": 18, "sig": -0.35},
    "SQQQ": {"k": "etp", "r": "A", "n": 6, "sig": -0.30, "lev": -3},
    "GLD": {"k": "etf", "r": "NR", "n": None},
}


def tearDownModule():
    br._RATINGS.update(loaded=False, by={}, asOf=None, stale=True)


class Load(unittest.TestCase):
    def test_fresh_and_stale_files(self):
        real = br.fetch_json
        try:
            today = (br.NOW_UTC or br.datetime.now(br.timezone.utc)).strftime("%Y-%m-%d")
            br.fetch_json = lambda url, timeout=45: {"date": today, "ratings": {"SPY": {"r": "A-", "n": 7}}}
            br._RATINGS.update(loaded=False)
            r = br.load_ratings()
            self.assertFalse(r["stale"])
            self.assertEqual(br.credit_of("spy")["r"], "A-")
            br.fetch_json = lambda url, timeout=45: {"date": "2026-01-02", "ratings": {"SPY": {"r": "A-", "n": 7}}}
            br._RATINGS.update(loaded=False)
            self.assertTrue(br.load_ratings()["stale"])                    # weeks old: context only
            br.fetch_json = lambda url, timeout=45: None
            br._RATINGS.update(loaded=False)
            self.assertTrue(br.load_ratings()["stale"])                    # missing file
            self.assertIsNone(br.credit_of("SPY"))
        finally:
            br.fetch_json = real


class Gate(unittest.TestCase):
    def setUp(self):
        set_ratings(dict(RATED))

    def test_distressed_longs_barred(self):
        self.assertIn("CCC", br.credit_gate("CCCO", "long"))
        self.assertIsNone(br.credit_gate("CCCO", "short"))                  # a bearish idea is not barred
        self.assertIsNone(br.credit_gate("BCO", "long"))                    # B- is weak, not distressed
        self.assertIsNone(br.credit_gate("GLD", "long"))                    # no issuer, no gate
        self.assertIsNone(br.credit_gate("UNKNOWN", "long"))

    def test_inverse_product_is_a_short(self):
        set_ratings(dict(RATED, BEAR={"k": "etp", "r": "CCC", "n": 18}))
        self.assertIsNone(br.credit_gate("BEAR", "long", {"lev": -2}))

    def test_stale_ratings_are_context_only(self):
        set_ratings(dict(RATED), stale=True)
        self.assertIsNone(br.credit_gate("CCCO", "long"))
        self.assertEqual(br.credit_size_mult(RATED["BCO"]), 1.0)


class Size(unittest.TestCase):
    def setUp(self):
        set_ratings(dict(RATED))

    def test_long_multipliers(self):
        m = lambda t: br.credit_size_mult(RATED[t])
        self.assertEqual(m("AAA1"), 1.0)
        self.assertEqual(m("IGCO"), 1.0)
        self.assertEqual(m("BBCO"), 0.85)
        self.assertEqual(m("BCO"), 0.65)
        self.assertEqual(m("CCCO"), 0.5)
        self.assertEqual(br.credit_size_mult(RATED["GLD"]), 1.0)
        self.assertEqual(br.credit_size_mult(None), 1.0)

    def test_short_multipliers(self):
        self.assertEqual(br.credit_size_mult(RATED["AAA1"], short=True), 0.85)   # shorting quality: smaller
        self.assertEqual(br.credit_size_mult(RATED["BCO"], short=True), 1.0)
        # long an inverse fund = short its index: an A index is not "the strongest credit", no cut
        self.assertEqual(br.credit_size_mult(RATED["SQQQ"], short=False, etp={"lev": -3}), 1.0)


class Signal(unittest.TestCase):
    def test_etp_signal_in_its_direction(self):
        comps = {"credit": -0.30, "crossAsset": 0.2}
        br._etp_sign_market(comps, {"lev": -3, "underlying": "equity"}, None)
        self.assertAlmostEqual(comps["credit"], 0.30)                       # the index's credit worsening favors the bear fund
        comps = {"credit": 0.12}
        br._etp_sign_market(comps, {"lev": 1.5, "underlying": "volatility"}, None)
        self.assertAlmostEqual(comps["credit"], 0.12)                       # kept for a volatility product, its sign
        comps = {"credit": 0.12}
        br._etp_sign_market(comps, {"lev": -0.5, "underlying": "volatility"}, None)
        self.assertAlmostEqual(comps["credit"], -0.12)

    def test_weights_and_horizon(self):
        self.assertIn("credit", br.BASE_WEIGHTS)
        self.assertIn("credit", br.LONG_TERM_SIGNALS)
        self.assertIn("credit", br.ETP_MARKET_SIGNALS)
        self.assertNotIn("credit", br.COMPANY_SIGNALS)                      # a fund keeps its (holdings') credit signal

    def test_card(self):
        set_ratings(dict(RATED))
        c = br.credit_card(RATED["BBCO"])
        self.assertEqual((c["r"], c["n"], c["o"]), ("BB", 12, "stable"))
        self.assertIsNone(br.credit_card(None))


class Exit(unittest.TestCase):
    def pos(self, **k):
        p = {"ticker": "IGCO", "direction": "long", "instrument": "shares", "status": "open",
             "creditAtEntry": {"r": "BBB", "n": 9}}
        p.update(k)
        return p

    def test_distress_closes_at_once(self):
        set_ratings(dict(RATED, IGCO={"k": "co", "r": "CCC+", "n": 17}))
        self.assertEqual(br.credit_exit(self.pos(), TODAY), "credit-distress")

    def test_downgrade_needs_a_second_session(self):
        set_ratings(dict(RATED, IGCO={"k": "co", "r": "BB+", "n": 11}))
        p = self.pos()
        self.assertIsNone(br.credit_exit(p, TODAY))                         # day 1: on watch
        self.assertEqual(p["creditWatch"]["date"], TODAY)
        self.assertIsNone(br.credit_exit(p, TODAY))                         # same session: still waiting
        self.assertEqual(br.credit_exit(p, "2026-11-19"), "credit-downgrade")

    def test_recovery_clears_the_watch(self):
        p = self.pos(creditWatch={"date": "2026-11-17", "r": "BB+", "n": 11})
        set_ratings(dict(RATED, IGCO={"k": "co", "r": "BBB", "n": 9}))
        self.assertIsNone(br.credit_exit(p, TODAY))
        self.assertNotIn("creditWatch", p)

    def test_one_notch_inside_grade_holds(self):
        set_ratings(dict(RATED, IGCO={"k": "co", "r": "BBB-", "n": 10}))
        p = self.pos()
        self.assertIsNone(br.credit_exit(p, TODAY))
        self.assertNotIn("creditWatch", p)

    def test_short_covered_on_upgrade(self):
        set_ratings(dict(RATED, BBCO={"k": "co", "r": "BBB-", "n": 10}))
        p = self.pos(ticker="BBCO", direction="short", creditAtEntry={"r": "BB", "n": 12})
        self.assertIsNone(br.credit_exit(p, TODAY))
        self.assertEqual(br.credit_exit(p, "2026-11-19"), "credit-upgrade")

    def test_funds_options_and_old_positions(self):
        set_ratings(dict(RATED, IGCO={"k": "co", "r": "CCC", "n": 18}))
        self.assertIsNone(br.credit_exit(self.pos(etp={"lev": -3}), TODAY))
        self.assertIsNone(br.credit_exit(self.pos(instrument="option"), TODAY))
        set_ratings(dict(RATED))
        old = self.pos(creditAtEntry=None)
        self.assertIsNone(br.credit_exit(old, TODAY))                       # pre-z194: a baseline is taken
        self.assertEqual(old["creditBaseline"]["r"], "BBB")
        set_ratings(dict(RATED, IGCO={"k": "co", "r": "BB", "n": 12}))
        self.assertIsNone(br.credit_exit(old, "2026-11-19"))
        self.assertEqual(br.credit_exit(old, "2026-11-20"), "credit-downgrade")

    def test_review_only_changes_nothing(self):
        set_ratings(dict(RATED, IGCO={"k": "co", "r": "BB", "n": 12}))
        p = self.pos()
        self.assertIsNone(br.credit_exit(p, TODAY, review_only=True))
        self.assertNotIn("creditWatch", p)
        old = self.pos(creditAtEntry=None)
        br.credit_exit(old, TODAY, review_only=True)
        self.assertNotIn("creditBaseline", old)


class Rollup(unittest.TestCase):
    def test_credit_scores(self):
        st = {"trades": [
            {"status": "closed", "direction": "long", "creditAtEntry": {"n": 9}, "realizedPL": 50, "pnl": 50, "exitPrice": 1, "entryPrice": 1, "shares": 1},
            {"status": "closed", "direction": "long", "creditAtEntry": {"n": 10}, "realizedPL": -20, "pnl": -20, "exitPrice": 1, "entryPrice": 1, "shares": 1},
            {"status": "closed", "direction": "short", "creditAtEntry": {"n": 15}, "realizedPL": 30, "pnl": 30, "exitPrice": 1, "entryPrice": 1, "shares": 1},
            {"status": "open", "direction": "long", "creditAtEntry": {"n": 3}}]}
        br.learning_rollups(st)
        cs = {(x["bucket"], x["direction"]): x for x in st["creditScores"]}
        self.assertEqual(cs[("BBB", "long")]["trades"], 2)
        self.assertEqual(cs[("B", "short")]["trades"], 1)
        self.assertNotIn(("AAA-AA", "long"), cs)


if __name__ == "__main__":
    unittest.main()
