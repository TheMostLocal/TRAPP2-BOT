# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   tests/test_unwind.py
#  z190: a short sale the rules never allowed (a fund / ETP) is unwound at the next pass - through the runner's own
#  position review, on the four 11-16 positions as the journal holds them. Run: python -m unittest discover -s tests
# ============================================================
import copy
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("RUNNER_DRY_RUN", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import bot_runner as br  # noqa: E402

TODAY = "2026-11-17"


def short(tk, px, sh):
    return {"ticker": tk, "direction": "short", "instrument": "shares", "status": "open", "entryPrice": px,
            "entryDate": "2026-11-16", "entryAt": "2026-11-16T15:05:01+00:00", "shares": sh, "notional": round(px * sh, 2),
            "horizonClass": "medium", "horizonDaysPlanned": 10, "stopPrice": round(px * 1.08, 4), "targetPrice": round(px * 0.85, 4),
            "lastPrice": px}


def urow(tk, px, repo, **kw):
    r = {"ticker": tk, "price": px, "priceLocal": px, "currency": "USD", "fxRate": 1.0, "scan": True, "asOf": None,
         "assetClass": "Equity", "isetf": "false", "isfund": "", "sector": "Unknown", "name": tk, "repo": repo, "marketcap": None}
    r.update(kw)
    return r


class Unwind(unittest.TestCase):
    def setUp(self):
        self.state = {"bankroll": 110_742.34, "trades": [short("GLD", 356.015, 8.7931), short("SOXL", 49.625, 17.0613),
                                                       short("TAN", 40.71, 56.472), short("WGMI", 24.73, 99.2379),
                                                       short("BEAM", 22.08, 50.0563)]}
        self.universe = {"GLD": urow("GLD", 355.0, "TRAPP2-1"), "SOXL": urow("SOXL", 49.9, "TRAPP2-1"),
                         "TAN": urow("TAN", 40.5, "TRAPP2-1"), "WGMI": urow("WGMI", 24.6, "TRAPP2-1"),
                         "BEAM": urow("BEAM", 22.3, "TRAPP2-2", sector="Healthcare", marketcap=2.3e9, marketcapUSD=2.3e9)}

    def run_review(self, state, review_only=False):
        return br.manage_open_positions(state, self.universe, TODAY, ctx=None, review_only=review_only)

    def test_fund_shorts_are_unwound_inside_the_minimum_hold(self):
        st = copy.deepcopy(self.state)
        self.run_review(st)
        by = {t["ticker"]: t for t in st["trades"]}
        for tk in ("GLD", "SOXL", "TAN", "WGMI"):
            self.assertEqual(by[tk]["status"], "closed", tk)
            self.assertEqual(by[tk]["exitReason"], "not-shortable", tk)
        self.assertEqual(by["BEAM"]["status"], "open")          # a $2.3B stock: a short the rules allow, inside its hold
        # realized P/L lands in the bankroll: (entry - exit) x shares for each short
        want = sum((p["entryPrice"] - self.universe[p["ticker"]]["price"]) * p["shares"]
                   for p in self.state["trades"] if p["ticker"] != "BEAM")
        self.assertAlmostEqual(st["bankroll"] - self.state["bankroll"], want, delta=abs(want) * 0.02 + 5)  # fees

    def test_switch_off_keeps_them(self):
        st = copy.deepcopy(self.state)
        old = br.UNWIND_UNSHORTABLE
        try:
            br.UNWIND_UNSHORTABLE = False
            self.run_review(st)
        finally:
            br.UNWIND_UNSHORTABLE = old
        self.assertTrue(all(t["status"] == "open" for t in st["trades"]))

    def test_after_close_review_only_flags(self):
        st = copy.deepcopy(self.state)
        self.run_review(st, review_only=True)
        self.assertTrue(all(t["status"] == "open" for t in st["trades"]))

    def test_a_restored_etf_flag_is_enough(self):
        # once TRAPP2-1's profiles come back, the ETF flag alone says it - from any book
        st = copy.deepcopy(self.state)
        self.universe["GLD"] = urow("GLD", 355.0, "TRAPP2", isetf="true", name="SPDR Gold Shares")
        self.run_review(st)
        self.assertEqual({t["ticker"]: t["status"] for t in st["trades"]}["GLD"], "closed")


if __name__ == "__main__":
    unittest.main()
