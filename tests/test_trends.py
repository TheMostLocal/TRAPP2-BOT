# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   tests/test_trends.py
#  z201: today's trends (TRAPP2-ANALYTICS data/trends/latest.json) in every entry - one x0.9 trim on a new equity long
#  while the market reads sell-off / heavy sell-off or the S&P 500 is in a correction / bear market; shorts, inverse
#  funds, non-equity funds, crypto and futures untouched; trendsAtEntry on the trade (the S&P run counting today's
#  provisional close); stale or missing: context only.
#  z202: the record decides the trim (trend_evidence: sell-off longs vs the rest, by the read at the entry day's open,
#  per-day Welch t), the long side's net beta cap x0.75 in a sell-off, trendScores; pipeline/market_tape.py = quant/tape.py.
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
    br._TREND.update(loaded=False, doc=None, stale=True, age=None, evidence=None)
    br._TAPE.update(loaded=True, H=None, memo={})          # z202: no network in tests - each test sets its own tape


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


# ------------------------------------------------------------------------------------------------- z202 ----------
from datetime import date as _date  # noqa: E402
import market_tape as mt  # noqa: E402
import risk_book as rb  # noqa: E402


def _weekdays(n, end=_date(2026, 11, 20)):
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d -= timedelta(days=1)
    return list(reversed(out))


def tape_hist():
    """300 sessions climbing to a high, then 7 straight lower closes (-11 %), VIX 15 -> 36, HYG -2 % (as the quant test)."""
    ds = _weekdays(300)
    cs, vs, px = [], [], 5000.0
    for i in range(293):
        px *= 1.0015 if i % 3 else 0.9995
        cs.append(round(px, 2))
        vs.append(4e9 + (i % 5) * 1e7)
    for k in range(7):
        px *= 0.983
        cs.append(round(px, 2))
        vs.append(5e9 + k * 2e8)
    vix = [15.0] * 293 + [17, 19, 22, 25, 28, 32, 36]
    hyg = [80.0] * 294 + [79.8, 79.5, 79.2, 78.9, 78.6, 78.3]
    rows = lambda vals, vols=None: [{"date": d, "close": v, "volume": (vols[i] if vols else 0)} for i, (d, v) in enumerate(zip(ds, vals))]
    return {"spx": mt.series(rows(cs, vs)), "vix": mt.series(rows(vix)), "hyg": mt.series(rows(hyg)), "spy": mt.series(rows([c / 10 for c in cs]))}, ds


class Tape(unittest.TestCase):
    def test_state_as_known_at_the_open(self):
        H, ds = tape_hist()
        self.assertEqual(mt.state_on(ds[-8], H)["state"], "calm")
        m = mt.state_on("2026-11-23", H)
        self.assertEqual((m["state"], m["run"], m["asOf"], m["of"]), ("selloff", -7, "2026-11-20", 6))
        self.assertIsNone(mt.state_on("2026-12-15", H), "the history stops weeks earlier")
        self.assertEqual(mt.series({"error": "rate limited"}), ([], [], []), "an error page is no history")


def _trade(day, ret, direction="long", ticker="AAA", etp=None, instrument="shares", n=1000.0):
    pl = n * ret / 100
    return {"id": f"{ticker}-{day}-{ret}", "ticker": ticker, "status": "closed", "direction": direction, "instrument": instrument,
            "entryDate": day, "exitDate": day, "notional": n, "etp": etp, "realizedPL": pl, "pnl": pl, "exitPL": pl, "trims": []}


class Evidence(unittest.TestCase):
    def setUp(self):
        reset()
        self.H, self.ds = tape_hist()
        br._TAPE.update(loaded=True, H=self.H, memo={})
        self.sell = [d for d in self.ds if (mt.state_on(d, self.H) or {}).get("state") == "selloff"]
        self.calm = [d for d in self.ds[-200:-10] if (mt.state_on(d, self.H) or {}).get("state") == "calm"]
        self.real_pl = br._trade_pl
        br._trade_pl = lambda t: round(float(t.get("realizedPL") or 0), 2)      # the journal's P/L (trade_sync arithmetic aside)

    def tearDown(self):
        br._trade_pl = self.real_pl
        reset()

    def journal(self, sell_ret, calm_ret, n_sell_days=None, per_day=3, spread=0.5):
        import random
        r = random.Random(4)
        tr = []
        sell = (self.sell * 3)[: n_sell_days or len(self.sell)]
        for k, d in enumerate(sell):
            for j in range(per_day):
                tr.append(_trade(d, sell_ret + r.gauss(0, spread), ticker=f"S{k}{j}"))
        for k, d in enumerate(self.calm[:12]):
            for j in range(per_day):
                tr.append(_trade(d, calm_ret + r.gauss(0, spread), ticker=f"C{k}{j}"))
        return {"trades": tr}

    def test_not_yet_with_few_sell_off_days(self):
        self.assertTrue(0 < len(self.sell) < 8, "a few sell-off sessions (the real journal had 4 entry days)")
        ev = br.trend_evidence(self.journal(-2.0, 0.5))
        self.assertEqual(ev["verdict"], "not yet")
        self.assertIn("of 8 days needed", ev["why"])
        self.assertEqual(ev["selloff"]["days"], len(self.sell))

    def test_worse_better_and_no_difference(self):
        # more sell-off days than the tape has: label the calm ones as sell-offs through the memo (the rule is what's tested)
        for d in self.calm[12:24]:
            br._TAPE["memo"][d] = {"state": "selloff"}
        self.sell = self.calm[12:24]
        worse = br.trend_evidence(self.journal(-2.0, 0.6, per_day=2))
        self.assertEqual(worse["verdict"], "worse", worse)
        self.assertLess(worse["t"], -2)
        better = br.trend_evidence(self.journal(2.0, -0.4, per_day=2))
        self.assertEqual(better["verdict"], "better", better)
        same = br.trend_evidence(self.journal(0.3, 0.3, per_day=2, spread=1.5))
        self.assertEqual(same["verdict"], "no difference", same)

    def test_only_equity_longs_count(self):
        d = self.sell[0]
        st = {"trades": [_trade(d, -5, direction="short"),                                          # a short
                         _trade(d, -5, etp={"lev": -3.0, "underlying": "equity"}),                  # long a bear fund = short
                         _trade(d, -5, ticker="BTC-USD"), _trade(d, -5, ticker="CL=F"),             # crypto, futures
                         _trade(d, -5, etp={"lev": 2.0, "underlying": "rates"}),                    # not an equity read
                         _trade(d, -5, instrument="option"),                                        # options
                         _trade(d, -5, direction="short", etp={"lev": -1.0, "underlying": "equity"}),  # short a bear fund = long
                         _trade(d, -5, ticker="000660.KS"),                                         # a foreign listing: its own market
                         _trade(d, 1.0)]}
        ev = br.trend_evidence(st)
        self.assertEqual(ev["selloff"]["trades"], 2)

    def test_a_bad_record_is_capped(self):
        d = self.sell[0]
        ev = br.trend_evidence({"trades": [_trade(d, 403.0), _trade(d, -1.0)]})
        self.assertAlmostEqual(ev["selloff"]["meanPct"], (50.0 - 1.0) / 2, places=3)

    def test_trend_scores_rollup(self):
        st = self.journal(-2.0, 0.5)
        st["trades"].append(_trade(self.sell[0], 3.0, direction="short", ticker="SHRT"))
        br.learning_rollups(st)
        rows = {(r["read"], r["direction"]): r for r in st["trendScores"]}
        self.assertEqual(rows[("selloff", "long")]["trades"], 3 * len(self.sell))
        self.assertEqual(rows[("selloff", "short")]["trades"], 1)
        self.assertLess(rows[("selloff", "long")]["avgReturnPct"], 0)
        self.assertGreater(rows[("calm", "long")]["avgReturnPct"], 0)
        self.assertEqual([r["read"] for r in st["trendScores"]][0], "calm", "calm first, sell-off last")
        self.assertEqual(st["trendEvidence"]["verdict"], "not yet")


class EarnedTrim(Base):
    def test_the_record_moves_the_trim(self):
        self.load(doc())
        br._TREND["evidence"] = {"verdict": "not yet"}
        m, why = br.trend_size_mult()
        self.assertEqual(m, 0.9)
        self.assertIn("a prior - the record: not yet", why)
        br._TREND["evidence"] = {"verdict": "worse"}
        m, why = br.trend_size_mult()
        self.assertEqual(m, 0.8)
        self.assertIn("sell-off longs did worse", why)
        br._TREND["evidence"] = {"verdict": "better"}
        self.assertEqual(br.trend_size_mult(), (1.0, None), "the record says buying sell-offs paid: no trim")
        br._TREND["evidence"] = {"verdict": "worse"}
        self.assertEqual(br.trend_size_mult(True), (1.0, None), "shorts never touched")

    def test_beta_scale(self):
        self.load(doc())
        self.assertEqual(br.trend_beta_scale(), (0.75, "S&P 500 heavy sell-off · correction"))
        reset()
        self.load(doc(state="calm", overlay=None))
        self.assertEqual(br.trend_beta_scale(), (1.0, None))
        reset()
        old = ((br.NOW_UTC or datetime.now(timezone.utc)) - timedelta(days=6)).strftime("%Y-%m-%d")
        self.load(doc(date=old))
        self.assertEqual(br.trend_beta_scale(), (1.0, None), "stale: no change")


class BetaCap(unittest.TestCase):
    def test_long_side_tightens_shorts_pass(self):
        b = rb.Book(100000, regime_dialed=False)
        b.add("HELD", "Technology", "long", 30000, 30000, beta_v=1.0)            # +30 % net beta
        ok, _ = b.check("NEW", "Industrials", "long", 8000, 8000, 1.0, None, 100)
        self.assertTrue(ok, "+38 % is inside the normal +-45 %")
        b.long_beta_scale, b.long_beta_why = 0.75, "S&P 500 heavy sell-off · correction"
        ok, why = b.check("NEW", "Industrials", "long", 8000, 8000, 1.0, None, 100)
        self.assertFalse(ok)
        self.assertIn("long side +34% - S&P 500 heavy sell-off", why)
        ok, _ = b.check("HEDGE", "Industrials", "short", 8000, -8000, 1.0, None, 100)
        self.assertTrue(ok, "a short shrinks the exposure - it passes")
        self.assertEqual(b.summary()["longBetaScale"], 0.75)
        s = rb.Book(100000, regime_dialed=False)
        s.long_beta_scale = 0.75
        s.add("SHORTS", "Technology", "short", 40000, -40000, beta_v=1.0)      # -40 % net beta
        ok, _ = s.check("MORE", "Energy", "short", 4000, -4000, 1.0, None, 100)
        self.assertTrue(ok, "-44 %: the short side's cap is unchanged")


class Wired202(unittest.TestCase):
    SRC = (Path(__file__).resolve().parent.parent / "pipeline" / "bot_runner.py").read_text(encoding="utf-8")

    def test_wiring(self):
        self.assertIn("book.long_beta_scale, book.long_beta_why = trend_beta_scale()", self.SRC)
        self.assertIn('_TREND["evidence"] = trend_evidence(state)', self.SRC)
        self.assertIn('state["trendScores"] = ', self.SRC)
        self.assertIn('state["trendEvidence"] = trend_evidence(state)', self.SRC)


class FormulaMcap(unittest.TestCase):
    """z202: the bot's market cap = price x the share count TRAPP2-ANALYTICS data/sec/shares.json trusts."""
    DOC = {"date": "2026-11-21", "tickers": {"AAPL": {"s": 14600000000, "p": "sec", "st": "verified"},
                                             "SPLT": {"s": 1000000000, "p": "yahoo", "st": "check"},
                                             "NEWSPLIT": {"s": 100000000, "p": "sec", "st": "verified"},
                                             "FOREIGN": {"s": 7e8, "p": "sec", "st": "verified"}}}

    def test_apply(self):
        rows = {"AAPL": {"price": 342.56, "marketcap": 4978144288451.0, "currency": "USD"},
                "SPLT": {"price": 50.0, "marketcap": 5e10, "currency": "USD"},
                "NEWSPLIT": {"price": 10.0, "marketcap": 1e10, "currency": "USD"},      # a 10:1 split since the file
                "FOREIGN": {"price": 20.0, "marketcap": 1.4e10, "currency": "EUR"},
                "NOFILE": {"price": 5.0, "marketcap": 5e8, "currency": "USD"}}
        out = br.apply_formula_mcap(rows, self.DOC)
        self.assertEqual(out, {"sec": 1, "yahoo": 1, "moved": 1})
        a = rows["AAPL"]
        self.assertAlmostEqual(a["marketcap"], 342.56 * 14600000000)
        self.assertEqual((a["marketcapUSD"], a["marketcapYahoo"], a["mcapSource"]), (a["marketcap"], 4978144288451.0, "sec:verified"))
        self.assertEqual(rows["NEWSPLIT"]["marketcap"], 1e10, "Yahoo's count moved > 10 % since the file: Yahoo's kept")
        self.assertNotIn("marketcapYahoo", rows["NEWSPLIT"])
        self.assertEqual(rows["FOREIGN"]["marketcap"], 1.4e10)
        self.assertEqual(rows["NOFILE"]["marketcap"], 5e8)
        self.assertEqual(br.apply_formula_mcap({"X": {"price": 1, "marketcap": 1}}, {"nothing": 1}), {"sec": 0, "yahoo": 0, "moved": 0})

    def test_split_check_reads_yahoo(self):
        src = (Path(__file__).resolve().parent.parent / "pipeline" / "bot_runner.py").read_text(encoding="utf-8")
        self.assertIn('mc_now, mc_prev = _f(u.get("marketcapYahoo") or u.get("marketcap")), _f(pos.get("lastMarketCap"))', src)
        self.assertIn('pos["lastMarketCap"] = _f(u.get("marketcapYahoo") or u.get("marketcap"))', src)
        self.assertIn("apply_formula_mcap(rows)", src)


if __name__ == "__main__":
    unittest.main()
