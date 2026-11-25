# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   tests/test_flows.py
#  z206: who was forced to trade - TRAPP2-ANALYTICS' trends doc carries "flows" (the most-shorted 50, 12-1 momentum,
#  the week's losers vs winners, high vs low beta, each in sd of its year, and the day's label). The bot records it on
#  every trade (trendsAtEntry.flows) and opens no short sale into a crowded short: 20 %+ of the float sold short, 10 %+
#  while the tape shows shorts being covered (a short-covering rally today, or the session after one) - a put instead.
#  The market tape (pipeline/market_tape.py) reads TRAPP2's SPY / HYG when the macro book can't be read.
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
import market_tape as mt  # noqa: E402
import risk_book as rb  # noqa: E402

TODAY = (br.NOW_UTC or datetime.now(timezone.utc)).strftime("%Y-%m-%d")


def doc(label="short-covering rally", flags=None, after=None, fdate=TODAY, proxy=None):
    fl = {"date": fdate, "label": label, "z": {"si": 0.5, "mom": 0.4, "rev": 3.4, "beta": 3.6},
          "flags": flags if flags is not None else {"bounce": True, "riskOn": True}, "provisional": True, "source": "tapes"}
    if after:
        fl["after"] = {"prior": after, "priorChg": 4.73, "chg": -1.2}
    d = {"build": "z206", "date": TODAY, "state": "intraday", "etTime": "11:15",
         "read": {"state": "sell-off", "overlay": None, "on": 3, "of": 6, "label": "sell-off"},
         "indices": {"^GSPC": {"sym": "^GSPC", "date": TODAY, "chg": 4.7, "final": False, "streak": None}},
         "vol": {}, "internals": {"pctUp": 83, "ew": 2.9, "cw": 4.7}, "flows": fl}
    if proxy:
        d["proxy"] = {"^GSPC": proxy}
    return d


def stock(**kw):
    u = {"ticker": "XYZ", "price": 50.0, "assetClass": "Equity", "isetf": "false", "isfund": "", "marketcap": 10e9,
         "marketcapUSD": 10e9, "currency": "USD", "dv20USD": 50e6}
    u.update(kw)
    return u


class Base(unittest.TestCase):
    def setUp(self):
        self.real = br.fetch_json
        br._TREND.update(loaded=False, doc=None, stale=True, age=None, evidence=None)

    def tearDown(self):
        br.fetch_json = self.real
        br._TREND.update(loaded=False, doc=None, stale=True, age=None, evidence=None)

    def load(self, d):
        br.fetch_json = lambda url, timeout=30: d
        return br.load_trends()


class Crowding(unittest.TestCase):
    def test_standing_rule_and_the_covering_window(self):
        self.assertEqual(rb.shortable(stock())[0], True)                            # no short interest known: as before
        self.assertEqual(rb.shortable(stock(shortPctFloat=0.12))[0], True)          # 12 %: fine on a normal day
        ok, why = rb.shortable(stock(shortPctFloat=0.12), covering=True)            # ... not while shorts are being covered
        self.assertFalse(ok)
        self.assertIn("12% of the float already sold short (>= 10% while shorts are being covered)", why)
        ok, why = rb.shortable(stock(shortPctFloat=0.25))                           # 25 %: a crowded short any day
        self.assertFalse(ok)
        self.assertIn("(>= 20%) - a crowded short", why)
        self.assertTrue(rb.shortable(stock(shortPctFloat=1.49))[0])                 # > 100 %: a data check, not a reading
        self.assertTrue(rb.shortable(stock(shortPctFloat=0.05), covering=True)[0])

    def test_earlier_rules_still_first(self):
        ok, why = rb.shortable(stock(price=8.0, shortPctFloat=0.3))
        self.assertFalse(ok)
        self.assertNotIn("crowded", why)

    def test_knobs(self):
        k = rb.knobs()["shortCrowding"]
        self.assertEqual((k["maxShortPctFloat"], k["whileCovering"]), (rb.MAX_SHORT_SI, rb.SQUEEZE_SHORT_SI))
        self.assertEqual(k["finra"]["maxShortPctShares"], rb.MAX_SHORT_SO)                      # z207: FINRA first


class Flows(Base):
    def test_card_and_covering_window(self):
        self.load(doc())
        c = br.flows_card()
        self.assertEqual((c["label"], c["flags"], c["after"], c["provisional"]), ("short-covering rally", ["bounce", "riskOn"], None, True))
        self.assertTrue(br.covering_now())
        self.assertEqual(br.trend_card()["flows"]["label"], "short-covering rally")

    def test_the_session_after_is_still_the_window(self):
        self.load(doc(label="risk-off rotation", flags={"riskOff": True}, after="2026-11-23"))
        self.assertTrue(br.covering_now())
        self.assertEqual(br.flows_card()["after"], "2026-11-23")

    def test_squeeze_flag_counts_and_other_days_do_not(self):
        self.load(doc(label="rotation under a quiet index", flags={"squeeze": True}))
        self.assertTrue(br.covering_now())
        self.setUp()
        self.load(doc(label="de-grossing", flags={"unwind": True}))
        self.assertFalse(br.covering_now())

    def test_stale_absent_or_other_session(self):
        self.assertIsNone(br.flows_card())
        self.assertFalse(br.covering_now())
        self.load(doc(fdate="2026-01-02"))                                          # a flows block from another session
        self.assertIsNone(br.flows_card())
        self.setUp()
        d = doc()
        d.pop("flows")
        self.load(d)                                                                # a pre-z206 doc
        self.assertIsNone(br.flows_card())
        self.assertIsNone(br.trend_card()["flows"])

    def test_proxy_shows_in_the_card(self):
        self.load(doc(proxy="SPY"))
        self.assertEqual(br.trend_card()["proxy"], "SPY")


class Wired(unittest.TestCase):
    def test_wiring(self):
        src = Path(br.__file__).read_text()
        self.assertIn('rb.shortable(u, covering=covering_now())', src)
        self.assertIn('"shortPctFloat": _f(r.get("shortPctFloat"))', src)
        self.assertIn('"flows": flows_card()', src)


class TapeFallback(unittest.TestCase):
    def test_macro_book_out_reads_trapp2(self):
        ds = [f"2026-{m:02d}-{d:02d}" for m in range(1, 12) for d in range(1, 29)][:280]
        spy = [{"date": d, "close": 600 - i * 0.1, "volume": 8e7} for i, d in enumerate(ds)]
        hyg = [{"date": d, "close": 80.0, "volume": 1e7} for d in ds]

        def fetch(url):
            if "/TRAPP2-1/" in url:
                return None
            if url.endswith("/TRAPP2/main/data/history/SPY.json"):
                return spy
            if url.endswith("/TRAPP2/main/data/history/HYG.json"):
                return hyg
            return None
        H = mt.load_histories(fetch)
        self.assertEqual(H["src"], {"spy": "TRAPP2 SPY", "hyg": "TRAPP2 HYG", "spx": "SPY (TRAPP2)"})
        self.assertEqual(H["spx"], H["spy"])
        self.assertEqual(H["vix"], ([], [], []))


if __name__ == "__main__":
    unittest.main()
