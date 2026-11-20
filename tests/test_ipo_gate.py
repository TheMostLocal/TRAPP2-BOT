# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   tests/test_ipo_gate.py
#  z197: the IPO calendar in every entry - no new long into a major lock-up release (3 sessions before to 1 after),
#  a smaller long in the syndicate-initiation window after the research quiet period, ipoAtEntry on the trade;
#  shorts and exits never blocked; rule-defined releases are context only.
#  Run: python -m unittest discover -s tests
# ============================================================
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("RUNNER_DRY_RUN", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import bot_runner as br  # noqa: E402

DOC = {"asOf": "2026-11-19T14:00:00Z", "events": [
    {"ticker": "ANT", "date": "2026-11-29", "kind": "quiet-period", "text": "quiet period ends"},
    {"ticker": "ANT", "date": "2027-02-03", "kind": "lock-up", "major": True, "shares": 1612000000.0},
    {"ticker": "ANT", "date": "2027-11-19", "kind": "lock-up", "major": False},
    {"ticker": "SPCX", "rule": "after the Q3 results", "kind": "lock-up", "major": True},
    {"ticker": "SPCX", "date": "2026-12-09", "kind": "lock-up", "major": True}],
    "ipos": {"ANT": {"name": "Anthropic", "firstTrade": "2026-11-19", "price": 83.0,
                     "underwriters": {"lead": ["MS", "GS", "JPM"], "count": 22}, "fees": {"owner": "~$550M (owner, 11-19)"}},
             "SPCX": {"name": "SpaceX", "firstTrade": "2026-06-12", "price": 135.0}}}


def setUpModule():
    br._IPO.update(loaded=False, events=[], ipos={}, asOf=None)


def tearDownModule():
    br._IPO.update(loaded=False, events=[], ipos={}, asOf=None)


class Load(unittest.TestCase):
    def setUp(self):
        self.real = br.fetch_json
        br._IPO.update(loaded=False, events=[], ipos={}, asOf=None)

    def tearDown(self):
        br.fetch_json = self.real

    def test_loads_and_caches(self):
        calls = []
        br.fetch_json = lambda url, timeout=30: calls.append(url) or DOC
        br.load_ipo()
        br.load_ipo()
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].endswith("/TRAPP2-ANALYTICS/main/data/web/ipo.json"))
        self.assertEqual(len(br._IPO["events"]), 5)

    def test_missing_file_means_no_guard(self):
        br.fetch_json = lambda url, timeout=30: None
        br.load_ipo()
        self.assertIsNone(br.ipo_gate("ANT", "long", "2027-02-03"))
        self.assertEqual(br.ipo_size_mult("ANT", False, "2026-11-30"), (1.0, None))
        self.assertIsNone(br.ipo_card("ANT", "2026-11-30"))


class Gate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        br._IPO.update(loaded=True, events=DOC["events"], ipos=DOC["ipos"], asOf=DOC["asOf"])

    def test_sessions_signed(self):
        # 2027-02-03 is a Wednesday: Fri 01-29 -> Mon, Tue, Wed still to come
        self.assertEqual(br._ipo_sessions("2027-02-03", "2027-01-29"), -3)
        self.assertEqual(br._ipo_sessions("2027-02-03", "2027-02-03"), 0)
        self.assertEqual(br._ipo_sessions("2027-02-03", "2027-02-05"), 2)
        # Thanksgiving (Thu 11-26-2026) is not a session: Wed 11-25 -> Fri 11-27 is one
        self.assertEqual(br._ipo_sessions("2026-11-25", "2026-11-27"), 1)
        # an event on a Sunday lands on Monday; Friday is the session before it
        self.assertEqual(br._ipo_sessions("2026-11-29", "2026-11-27"), -1)
        self.assertEqual(br._ipo_sessions("2026-11-29", "2026-11-30"), 0)
        # research may publish the session AFTER a weekday quiet-period end
        self.assertEqual(br._ipo_sessions("2026-12-02", "2026-12-02", strictly_after=True), -1)
        self.assertEqual(br._ipo_sessions("2026-12-02", "2026-12-03", strictly_after=True), 0)
        self.assertIsNone(br._ipo_sessions("someday", "2026-11-27"))

    def test_lockup_window_blocks_new_longs_only(self):
        self.assertIsNone(br.ipo_gate("ANT", "long", "2027-01-28"))          # 4 sessions out
        self.assertIn("lock-up", br.ipo_gate("ANT", "long", "2027-01-29"))    # 3 sessions out
        self.assertIn("1,612M shares", br.ipo_gate("ANT", "long", "2027-02-03"))
        self.assertIn("ago", br.ipo_gate("ANT", "long", "2027-02-04"))        # the session after
        self.assertIsNone(br.ipo_gate("ANT", "long", "2027-02-05"))
        self.assertIsNone(br.ipo_gate("ANT", "short", "2027-02-03"))          # shorts never blocked
        self.assertIsNone(br.ipo_gate("ANT", "long", "2027-11-19"))
        self.assertIsNone(br.ipo_gate("NVDA", "long", "2027-02-03"))          # other names untouched

    def test_rule_defined_release_is_context_not_gate(self):
        self.assertIsNone(br.ipo_gate("SPCX", "long", "2026-11-19"))
        self.assertIn("lock-up", br.ipo_gate("SPCX", "long", "2026-12-04"))   # dated SPCX release (Wed 12-09) 3 out

    def test_quiet_period_window_sizes_longs(self):
        m, why = br.ipo_size_mult("ANT", False, "2026-11-30")                 # quiet period ended Sun 11-29
        self.assertEqual(m, br.IPO_QUIET_SIZE)
        self.assertIn("syndicate", why)
        self.assertEqual(br.ipo_size_mult("ANT", True, "2026-11-30"), (1.0, None))
        self.assertEqual(br.ipo_size_mult("ANT", False, "2026-12-04")[0], br.IPO_QUIET_SIZE)   # 5th session
        self.assertEqual(br.ipo_size_mult("ANT", False, "2026-12-07")[0], 1.0)   # 6th session after
        self.assertEqual(br.ipo_size_mult("ANT", False, "2026-11-27")[0], 1.0)   # before it ends

    def test_card(self):
        c = br.ipo_card("ANT", "2026-11-30")
        self.assertEqual(c["leadUnderwriters"], ["MS", "GS", "JPM"])
        self.assertTrue(c["conflictedWindow"])
        self.assertFalse(c["lockupGate"])
        self.assertEqual((c["next"]["kind"], c["next"]["sessionsTo"]), ("quiet-period", 0))   # initiations day
        c2 = br.ipo_card("ANT", "2026-12-01")
        self.assertEqual(c2["next"]["date"], "2027-02-03")
        self.assertTrue(c2["next"]["major"])
        self.assertGreater(c2["next"]["sessionsTo"], 40)
        self.assertIsNone(br.ipo_card("NVDA", "2026-11-30"))


if __name__ == "__main__":
    unittest.main()
