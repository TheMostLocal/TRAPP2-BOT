# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   tests/test_seasoning.py
#  z195: a new listing (Anthropic's ANT, IPO 11-19) waits for RUNNER_MIN_HISTORY_SESSIONS daily closes before the bot
#  opens a position in it. Run: python -m unittest discover -s tests
# ============================================================
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("RUNNER_DRY_RUN", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import bot_runner as br  # noqa: E402


class Seasoning(unittest.TestCase):
    def setUp(self):
        self.real = br.load_history
        self.min = br.MIN_HISTORY_SESSIONS
        self.hist = {}
        br.load_history = lambda t, repo: self.hist.get(t)

    def tearDown(self):
        br.load_history = self.real
        br.MIN_HISTORY_SESSIONS = self.min

    def test_ipo_day_and_first_weeks_wait(self):
        br.MIN_HISTORY_SESSIONS = 20
        self.hist = {"ANT": [83.0], "NEWCO": [10.0] * 19, "AAPL": [200.0] * 300}
        self.assertIn("1 session of price history", br.seasoning_gate("ANT", "TRAPP2"))
        self.assertIn("19 sessions", br.seasoning_gate("NEWCO", "TRAPP2"))
        self.assertIsNone(br.seasoning_gate("AAPL", "TRAPP2"))
        self.assertIn("0 sessions", br.seasoning_gate("NOHIST", "TRAPP2"))       # no file yet: wait too

    def test_twenty_closes_clear_it_and_zero_turns_it_off(self):
        br.MIN_HISTORY_SESSIONS = 20
        self.hist = {"ANT": [83.0] * 20}
        self.assertIsNone(br.seasoning_gate("ANT", "TRAPP2"))
        br.MIN_HISTORY_SESSIONS = 0
        self.hist = {"ANT": [83.0]}
        self.assertIsNone(br.seasoning_gate("ANT", "TRAPP2"))

    def test_unknown_repo_is_not_judged(self):
        br.MIN_HISTORY_SESSIONS = 20
        self.assertIsNone(br.seasoning_gate("ANT", None))

    def test_default(self):
        self.assertEqual(self.min, int(os.environ.get("RUNNER_MIN_HISTORY_SESSIONS") or 20))


if __name__ == "__main__":
    unittest.main()
