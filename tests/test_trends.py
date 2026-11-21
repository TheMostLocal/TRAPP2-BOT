# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   tests/test_trends.py
#  z201: today's trends (TRAPP2-ANALYTICS data/trends/latest.json) in every entry - one x0.9 trim on a new equity long
#  while the market reads sell-off / heavy sell-off or the S&P 500 is in a correction / bear market; shorts, inverse
#  funds, non-equity funds, crypto and futures untouched; trendsAtEntry on the trade (the S&P run counting today's
#  provisional close); stale or missing: context only.
#  Run: python -m unittest discover -s tests
# ============================================================
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("RUNNER_DRY_RUN", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import bot_runner as br  # noqa: E402

TODAY = (br.NOW_UTC or datetime.now(timezone.utc)).strftime("%Y-%m-%d")


def doc(state="heavy sell-off", overlay="correction", on=5, date=TODAY, run=("down", 6, "2026-11-19"), now="red",
        final=False, chg=-2.14):
    read = {"state": state, "overlay": overlay, "on": on, "of": 7,
            "label": f"{state}{' · ' + overlay if overlay else ''}", "provisional": now is not None}
    spx = {"sym": "^GSPC", "date": date, "close": 7022.45, "prevClose": 7175.83, "chg": chg, "final": final,
           "open": 7150.39, "gap": -0.355, "fromOpen": -1.789, "high52": {"dd": -10.2},
           "streak": {"dir": run[0], "n": run[1], "from": "2026-11-12", "through": run[2]} if run else None}
    if now and not final:
        spx["provisional"] = {"now": now}
    return {"build": "z201", "date": date, "state": "final" if final else "intraday", "etTime": "13:18", "read": read,
            "indices": {"^GSPC": spx}, "vol": {"^VIX": {"close": 32.31}},
            "internals": {"pctUp": 46.8, "ew": -0.915, "cw": -2.218}}


def reset():
    br._TREND.update(loaded=False, doc=None, stale=True, age=None)


def setUpModule():
    reset()


def tearDownModule():
    reset()


class Base(unittest.TestCase):
    def setUp(self):
        self.real = br.fetch_json
        reset()

    def tearDown(self):
        br.fetch_json = self.real
        reset()

    def load(self, d):
        br.fetch_json = lambda url, timeout=30: d
        return br.load_trends()


class Load(Base):
    def test_reads_the_doc_once_from_analytics(self):
        seen = []
        br.fetch_json = lambda url, timeout=30: seen.append(url) or doc()
        br.load_trends()
        br.load_trends()
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].endswith("/TRAPP2-ANALYTICS/main/data/trends/latest.json"))
        self.assertFalse(br._TREND["stale"])
        self.assertEqual(br._TREND["age"], 0)

    def test_missing_or_malformed_is_context_free(self):
        for bad in (None, [], {"read": "x"}, {"indices": {}}):
            reset()
            self.load(bad)
            self.assertIsNone(br._TREND["doc"])
            self.assertEqual(br.trend_size_mult(), (1.0, None))
            self.assertIsNone(br.trend_card())

    def test_stale_doc_is_context_only(self):
        old = ((br.NOW_UTC or datetime.now(timezone.utc)) - timedelta(days=6)).strftime("%Y-%m-%d")
        self.load(doc(date=old))
        self.assertTrue(br._TREND["stale"])
        self.assertEqual(br.trend_size_mult(), (1.0, None))
        self.assertTrue(br.trend_card()["stale"])                     # still recorded, marked stale

    def test_future_dated_doc_is_not_trusted(self):
        fut = ((br.NOW_UTC or datetime.now(timezone.utc)) + timedelta(days=2)).strftime("%Y-%m-%d")
        self.load(doc(date=fut))
        self.assertTrue(br._TREND["stale"])


class Size(Base):
    def test_sell_off_trims_new_equity_longs_once(self):
        self.load(doc())
        m, why = br.trend_size_mult(False, None, None)
        self.assertEqual(m, 0.9)
        self.assertIn("heavy sell-off · correction", why)
        self.assertIn("5 of 7", why)
        lev3 = {"lev": 3.0, "underlying": "equity"}
        self.assertEqual(br.trend_size_mult(False, lev3, None)[0], 0.9)           # a levered long equity fund: same one trim

    def test_never_touches_shorts_inverse_or_non_equity(self):
        """Effective direction decides (a short in a bear fund is a long on the index - trimmed like one)."""
        self.load(doc())
        self.assertEqual(br.trend_size_mult(True, None, None), (1.0, None))        # a short
        self.assertEqual(br.trend_size_mult(False, {"lev": -3.0, "underlying": "equity"}, None), (1.0, None))   # bear fund long
        self.assertEqual(br.trend_size_mult(True, {"lev": -1.0, "underlying": "equity"}, None)[0], 0.9)    # short a bear fund = an effective long
        self.assertEqual(br.trend_size_mult(False, {"lev": 2.0, "underlying": "rates"}, None), (1.0, None))     # not an equity read
        self.assertEqual(br.trend_size_mult(False, None, {"assetClass": "crypto"}), (1.0, None))
        self.assertEqual(br.trend_size_mult(False, None, {"assetClass": "future"}), (1.0, None))

    def test_states(self):
        cases = [("calm", None, 1.0), ("under pressure", None, 1.0), ("sell-off", None, 0.9),
                 ("heavy sell-off", None, 0.9), ("under pressure", "correction", 0.9), ("calm", "bear market", 0.9)]
        for state, overlay, want in cases:
            reset()
            self.load(doc(state=state, overlay=overlay))
            self.assertEqual(br.trend_size_mult()[0], want, (state, overlay))

    def test_knob_off(self):
        self.load(doc())
        real = br.TREND_SIZE
        try:
            br.TREND_SIZE = 1.0
            self.assertEqual(br.trend_size_mult(), (1.0, None))
        finally:
            br.TREND_SIZE = real


class Card(Base):
    def test_card_counts_today_provisionally(self):
        self.load(doc())
        c = br.trend_card()
        self.assertEqual((c["state"], c["overlay"], c["on"], c["of"]), ("heavy sell-off", "correction", 5, 7))
        self.assertEqual((c["run"], c["runThrough"], c["runNow"]), (-6, "2026-11-19", -7))   # red now: would be 7
        self.assertEqual((c["gap"], c["fromOpen"], c["dd52"], c["vix"]), (-0.355, -1.789, -10.2, 32.31))
        self.assertEqual((c["pctUp"], c["ew"], c["cw"], c["session"], c["spxFinal"]), (46.8, -0.915, -2.218, "intraday", False))
        reset()
        self.load(doc(now="green", chg=0.2))
        self.assertEqual(br.trend_card()["runNow"], 1)                            # green now: the run would end
        reset()
        self.load(doc(final=True, run=("down", 7, TODAY)))
        c = br.trend_card()
        self.assertEqual((c["run"], c["runNow"], c["spxFinal"]), (-7, -7, True))  # after the close: today is in the run
        reset()
        self.load(doc(run=None, now=None))
        self.assertEqual((br.trend_card()["run"], br.trend_card()["runNow"]), (0, 0))


class Wired(unittest.TestCase):
    """The runner applies it at both sizing paths (risk book and notional), records it on the trade, loads it per session."""
    SRC = (Path(__file__).resolve().parent.parent / "pipeline" / "bot_runner.py").read_text(encoding="utf-8")

    def test_wiring(self):
        self.assertEqual(self.SRC.count('tmult, twhy = trend_size_mult(is_short, c.get("etp"), alt)'), 2)
        self.assertEqual(self.SRC.count('size_why = f"{size_why} · trends x{tmult:.2f} ({twhy})"'), 2)
        self.assertIn('"trendsAtEntry": trend_card(),', self.SRC)
        self.assertIn("load_trends()                                   # z201", self.SRC)


if __name__ == "__main__":
    unittest.main()
