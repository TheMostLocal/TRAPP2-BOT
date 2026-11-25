# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   tests/test_opt_quotes.py
#  z208 - no quotes, no volatility: TRAPP2-OPTIONS' 0327 ET 11-24 broad pass wrote chains with bid = ask = 0 and Yahoo's
#  placeholder IVs (1e-05, 1/64 ...); its per-name signal read them as "calm IV" and the runner added +0.5 to 1,435
#  names. A signals.json row with a placeholder IV is no reading; a chain's ATM contract without a quote has no IV.
#  Run: python -m unittest discover -s tests
# ============================================================
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("RUNNER_DRY_RUN", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import bot_runner as br  # noqa: E402

TODAY = (br.NOW_UTC or __import__("datetime").datetime.now(__import__("datetime").timezone.utc)).strftime("%Y-%m-%d")


class Placeholders(unittest.TestCase):
    def test_values(self):
        for v in (0.015625, 0.0156, 0.0078, 0.0625, 0.03125, 1e-05, 0.0, None, float("nan"), 0.02):
            self.assertTrue(br._placeholder_iv(v), v)
        for v in (0.0576, 0.22, 0.4512, 1.16):
            self.assertFalse(br._placeholder_iv(v), v)

    def test_live_iv_needs_a_quote(self):
        self.assertEqual(br._live_iv({"iv": 0.31, "bid": 1.2, "ask": 1.3}), 0.31)
        self.assertEqual(br._live_iv({"iv": 0.31, "bid": 0, "ask": 0.05}), 0.31)
        self.assertIsNone(br._live_iv({"iv": 0.31, "bid": 0, "ask": 0}))
        self.assertIsNone(br._live_iv({"iv": 0.0156, "bid": 1.2, "ask": 1.3}))
        self.assertIsNone(br._live_iv(None))


class Signals(unittest.TestCase):
    def setUp(self):
        self._doc, self._chain = br._OPT_SIGNALS.get("doc"), br._load_option_chain

    def tearDown(self):
        br._OPT_SIGNALS["doc"], br._load_option_chain = self._doc, self._chain

    def test_a_placeholder_row_is_no_reading(self):
        br._OPT_SIGNALS["doc"] = {"AA": [0.0078, 0.0156, 0.0078, 0.5, 30, TODAY], "A": [0.3562, -0.0368, -0.0697, 0.0, 23, TODAY]}
        self.assertIsNone(br._opt_signal_row("AA"))
        self.assertEqual(br._opt_signal_row("A")["iv"], 0.3562)

    def test_a_chain_without_quotes_gives_no_signal(self):
        br._OPT_SIGNALS["doc"] = {}
        dead = {"spot": 100.0, "expiries": [{"dte": 30, "calls": [{"strike": 100.0, "iv": 0.0156, "bid": 0, "ask": 0}],
                                             "puts": [{"strike": 100.0, "iv": 0.0156, "bid": 0, "ask": 0}]}]}
        br._load_option_chain = lambda t: dead
        self.assertIsNone(br.extract_options_signal("ZZZ"))
        live = {"spot": 100.0, "expiries": [{"dte": 30, "calls": [{"strike": 100.0, "iv": 0.30, "bid": 4.0, "ask": 4.2}],
                                             "puts": [{"strike": 100.0, "iv": 0.34, "bid": 4.1, "ask": 4.3}]}]}
        br._load_option_chain = lambda t: live
        s = br.extract_options_signal("ZZZ")
        self.assertEqual((s["iv"], s["skew"]), (0.32, 0.04))


if __name__ == "__main__":
    unittest.main()
