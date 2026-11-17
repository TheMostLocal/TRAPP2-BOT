# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   tests/test_etp.py
#  z192: a leveraged / inverse ETP - its exposure, its signals (fund-clean, in its own direction) and its management
#  (a short-horizon, session-by-session signal trade; SQQQ / SPXS re-planned). Run: python -m unittest discover -s tests
# ============================================================
import copy
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("RUNNER_DRY_RUN", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import bot_runner as br  # noqa: E402
import risk_book as rb  # noqa: E402

TODAY = "2026-11-17"


class Exposure(unittest.TestCase):
    def test_known_and_named_products(self):
        e = lambda t, n, etf="true": rb.etp_exposure({"ticker": t, "name": n, "isetf": etf})
        self.assertEqual(e("SQQQ", "")["lev"], -3)                                  # the table, names blank or not
        self.assertEqual(e("SPXS", "Direxion Daily S&P 500 Bear 3X Shares")["lev"], -3)
        self.assertEqual(e("XQID", "ProShares UltraShort QQQ")["lev"], -2)
        self.assertEqual(e("XSH", "ProShares Short S&P500")["lev"], -1)
        self.assertEqual(e("XUPRO", "ProShares UltraPro S&P500")["lev"], 3)
        self.assertEqual(e("PBRG", "Leverage Shares 2X Long PBR Daily ETF", "false")["lev"], 2)
        self.assertEqual(e("XTBF", "ProShares Short 20+ Year Treasury")["underlying"], "rates")
        self.assertEqual(e("XNUG", "Direxion Daily Gold Miners Index Bull 2X Shares")["underlying"], "equity")
        for t, n in (("VCSH", "Vanguard Short-Term Corporate Bond ETF"), ("JPST", "JPMorgan Ultra-Short Income ETF"),
                     ("SPY", "State Street SPDR S&P 500 ETF Trust"), ("GLD", "SPDR Gold Shares")):
            self.assertIsNone(e(t, n), t)                                            # durations and plain funds
        for t, n in (("BBW", "Build-A-Bear Workshop, Inc."), ("UCTT", "Ultra Clean Holdings, Inc.")):
            self.assertIsNone(e(t, n, "false"), t)                                   # companies

    def test_drag(self):
        # -3x at a 3.9 % daily product vol (1.3 % underlying): 6 x 0.013^2 = 0.101 % a session
        self.assertAlmostEqual(rb.etp_drag(-3, 0.039), 6 * 0.013 ** 2, places=9)
        self.assertAlmostEqual(rb.etp_drag(3, 0.039), 3 * 0.013 ** 2, places=9)
        self.assertIsNone(rb.etp_drag(-3, None))


def closes_up(n=260, start=20.0, step=0.004):
    out, p = [], start
    for i in range(n):
        p *= 1 + step + (0.01 if i % 2 else -0.01)
        out.append(round(p, 4))
    return out


class Signals(unittest.TestCase):
    def setUp(self):
        self._lh, self._oc = br.load_history, br._load_option_chain
        br.load_history = lambda tk, repo: closes_up()
        br._load_option_chain = lambda tk: None
        self.ctx = {"weights": {}, "weight_mods": {}, "peer_ranks": {}, "mode": "choppy", "quad": None, "grades": {},
                    "xa": {"indices": 0.6, "safeHaven": 0.2}, "opt_market": {"marketTrend": 0.4}, "fed": None,
                    "patterns": None, "today": TODAY, "regime_agnostic": True}

    def tearDown(self):
        br.load_history, br._load_option_chain = self._lh, self._oc

    def row(self, tk, name, **kw):
        r = {"ticker": tk, "name": name, "price": 40.0, "repo": "TRAPP2-1", "sector": "Unknown", "isetf": "true", "isfund": "",
             "dividend_yield": 7.78, "beta": -3.4, "marketcap": None}
        r.update(kw)
        return r

    def test_a_bear_fund_reads_the_market_upside_down_and_has_no_fundamentals(self):
        sc = br.score_symbol("SQQQ", self.row("SQQQ", "ProShares UltraPro Short QQQ"), self.ctx)
        c = sc["comps"]
        for k in ("fundamentals", "health", "researchGrade", "peerGrade", "regimeGrade", "fed", "catalyst"):
            self.assertNotIn(k, c, k)
        self.assertAlmostEqual(c["crossAsset"], -(0.6 - 0.5 * 0.2), places=3)        # risk-on = a headwind for SQQQ
        self.assertAlmostEqual(c["optionsMarket"], -0.4, places=3)
        self.assertEqual(sc["etp"]["lev"], -3)

    def test_a_bull_fund_keeps_the_market_signs(self):
        c = br.score_symbol("TQQQ", self.row("TQQQ", "ProShares UltraPro QQQ", beta=3.5), self.ctx)["comps"]
        self.assertAlmostEqual(c["crossAsset"], 0.6 - 0.5 * 0.2, places=3)
        self.assertAlmostEqual(c["optionsMarket"], 0.4, places=3)

    def test_a_rates_product_takes_no_equity_market_signal(self):
        c = br.score_symbol("TMF", self.row("TMF", "Direxion Daily 20+ Year Treasury Bull 3X Shares"), self.ctx)["comps"]
        self.assertNotIn("crossAsset", c)
        self.assertNotIn("optionsMarket", c)

    def test_a_plain_etf_has_no_company_signals_but_its_market_ones(self):
        c = br.score_symbol("XLE", self.row("XLE", "Energy Select Sector SPDR Fund", sector="Energy", beta=1.0), self.ctx)["comps"]
        self.assertNotIn("fundamentals", c)
        self.assertIn("trend", c)


def held(tk, px, **kw):
    t = {"id": f"{tk}-1", "ticker": tk, "name": kw.pop("name", tk), "direction": "long", "instrument": "shares", "status": "open",
         "entryPrice": px, "entryDate": "2026-11-04", "entryAt": "2026-11-04T15:05:01+00:00", "shares": 100.0,
         "notional": round(px * 100, 2), "horizonType": "core", "horizonClass": "long", "horizonDaysPlanned": 180,
         "minHoldDays": 15, "stopPrice": round(px * 0.75, 4), "targetPrice": None, "style": "core", "lastPrice": px,
         "plannedExitBy": "2027-07-14"}
    t.update(kw)
    return t


def urow(tk, px, name):
    return {"ticker": tk, "price": px, "priceLocal": px, "currency": "USD", "fxRate": 1.0, "scan": True, "asOf": None,
            "assetClass": "Equity", "isetf": "true", "isfund": "", "sector": "", "name": name, "repo": "TRAPP2-1", "marketcap": None}


class Management(unittest.TestCase):
    def setUp(self):
        self._lh = br.load_history
        br.load_history = lambda tk, repo: closes_up()

    def tearDown(self):
        br.load_history = self._lh

    def test_sqqq_core_plan_is_replanned_and_its_gain_taken_at_the_short_target(self):
        st = {"bankroll": 110_000.0, "trades": [held("SQQQ", 40.345)]}
        uni = {"SQQQ": urow("SQQQ", 45.84, "ProShares UltraPro Short QQQ")}
        br.manage_open_positions(st, uni, TODAY, ctx=None)
        p = st["trades"][0]
        self.assertEqual((p["horizonClass"], p["horizonType"], p["minHoldDays"]), ("short", "swing", 2))
        self.assertEqual(p["etp"]["replannedFrom"]["minHoldDays"], 15)
        self.assertEqual((p["status"], p["exitReason"]), ("closed", "target"))          # +13.6 % >= the +12 % target
        self.assertGreater(p["realizedPL"], 0)

    def test_a_faded_signal_closes_it_after_the_minimum_hold_and_not_inside_it(self):
        sc0 = br.score_symbol
        try:
            br.score_symbol = lambda tk, u, ctx: {"comps": {"trend": 0.05}, "signed": 0.05, "etp": rb.etp_exposure(u)}
            st = {"bankroll": 110_000.0, "trades": [held("SPXS", 27.13, entryDate="2026-11-06")]}
            uni = {"SPXS": urow("SPXS", 27.01, "Direxion Daily S&P 500 Bear 3X Shares")}
            br.manage_open_positions(st, uni, TODAY, ctx={"stub": True})
            self.assertEqual(st["trades"][0]["exitReason"], "signal-faded")                # 0.05 < 0.10, past 2 sessions
            st2 = {"bankroll": 110_000.0, "trades": [held("SPXS", 27.13, entryDate="2026-11-16", entryAt="2026-11-16T15:05:01+00:00")]}
            br.manage_open_positions(st2, uni, TODAY, ctx={"stub": True})
            self.assertEqual(st2["trades"][0]["status"], "open")                            # inside its minimum hold
            br.score_symbol = lambda tk, u, ctx: {"comps": {"trend": 0.4}, "signed": 0.4, "etp": rb.etp_exposure(u)}
            st3 = {"bankroll": 110_000.0, "trades": [held("SPXS", 27.13, entryDate="2026-11-06")]}
            br.manage_open_positions(st3, uni, TODAY, ctx={"stub": True})
            self.assertEqual(st3["trades"][0]["status"], "open")                            # a live signal keeps it
        finally:
            br.score_symbol = sc0

    def test_replan_happens_once(self):
        st = {"bankroll": 110_000.0, "trades": [held("SPXS", 27.13, entryDate="2026-11-06")]}
        uni = {"SPXS": urow("SPXS", 27.01, "Direxion Daily S&P 500 Bear 3X Shares")}
        br.manage_open_positions(st, uni, TODAY, ctx=None)
        first = copy.deepcopy(st["trades"][0]["etp"])
        br.manage_open_positions(st, uni, TODAY, ctx=None)
        self.assertEqual(st["trades"][0]["etp"]["replannedFrom"], first["replannedFrom"])
        self.assertEqual(st["trades"][0]["status"], "open")                                # -0.4 %: inside stop and target

    def test_a_stock_is_not_touched(self):
        st = {"bankroll": 110_000.0, "trades": [held("PG", 150.0, name="Procter & Gamble")]}
        uni = {"PG": dict(urow("PG", 151.0, "The Procter & Gamble Company"), isetf="false", repo="TRAPP2")}
        br.manage_open_positions(st, uni, TODAY, ctx=None)
        p = st["trades"][0]
        self.assertEqual((p["horizonClass"], p["minHoldDays"], p["status"]), ("long", 15, "open"))
        self.assertNotIn("etp", p)


if __name__ == "__main__":
    unittest.main()
