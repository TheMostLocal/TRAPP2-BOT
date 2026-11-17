# ============================================================
#  >>> DESTINATION: TheMostLocal/TRAPP2-BOT
#  >>> FILE PATH:   tests/test_learning.py
#  z192: the bot learns from every closed trade - retrain() replays the journal (the app's botRetrain rule), stamps each
#  trade's lesson and rolls up signal / style scores. Run: python -m unittest discover -s tests
# ============================================================
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("RUNNER_DRY_RUN", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import bot_runner as br  # noqa: E402


def trade(tk, pl, comps, i, direction="long", exit_reason="target", **kw):
    t = {"id": f"{tk}-{i}", "ticker": tk, "name": kw.pop("name", tk), "status": "closed", "direction": direction,
         "instrument": "shares", "entryPrice": 100.0, "exitPrice": 100.0 + pl / 10.0, "shares": 10.0, "notional": 1000.0,
         "realizedPL": pl, "pnl": pl, "entryDate": f"2026-10-{i:02d}", "exitDate": f"2026-11-{i:02d}",
         "exitAt": f"2026-11-{i:02d}T15:00:00Z", "exitReason": exit_reason, "components": comps, "style": kw.pop("style", "momentum")}
    t.update(kw)
    return t


class Retrain(unittest.TestCase):
    def test_a_win_trusts_the_signals_that_leaned_its_way(self):
        st = {"trades": [trade("AAA", 50, {"trend": 0.8, "momentum": -0.4}, 1)]}
        br.retrain(st, None, "t")
        w = st["learnedWeights"]
        self.assertAlmostEqual(w["trend"], 1 + 0.05 * 0.8, places=6)        # agreed with a winner: +5 % x |0.8|
        self.assertAlmostEqual(w["momentum"], 1 - 0.05 * 0.4, places=6)     # leaned against a winner: -5 % x |0.4|
        self.assertEqual(st["trades"][0]["lesson"]["verdict"], "win")

    def test_a_loss_damps_them_and_a_short_reads_signs_the_other_way(self):
        st = {"trades": [trade("BBB", -30, {"trend": -1.0}, 1, direction="short")]}
        br.retrain(st, None, "t")
        self.assertAlmostEqual(st["learnedWeights"]["trend"], 0.95, places=6)   # a bearish signal on a losing short

    def test_flat_and_signal_free_trades_are_learned_from_but_move_nothing(self):
        st = {"trades": [trade("CCC", 0, {"trend": 0.9}, 1), trade("DDD", 40, {}, 2)]}
        lr = br.retrain(st, None, "t")
        self.assertEqual(st["learnedWeights"], {})
        self.assertEqual((lr["flat"], lr["noSignals"], lr["taught"]), (1, 1, 0))
        self.assertIn("lesson", st["trades"][1])                       # every closed trade gets its lesson

    def test_forced_exits_teach_at_half_weight(self):
        st = {"trades": [trade("EEE", 20, {"trend": 1.0}, 1, exit_reason="not-shortable")]}
        br.retrain(st, None, "t")
        self.assertAlmostEqual(st["learnedWeights"]["trend"], 1 + 0.05 * 0.5, places=6)

    def test_regime_signals_learn_only_from_a_trusted_right_call(self):
        rc = {"dialedIn": True}
        st = {"trades": [trade("F1", 50, {"regimeGrade": 1.0}, 1),                                   # no context: untrusted
                         trade("F2", 50, {"regimeGrade": 1.0}, 2, regimeContext=rc, regimeCallCorrect=False),
                         trade("F3", 50, {"regimeGrade": 1.0}, 3, regimeContext=rc, regimeCallCorrect=True)]}
        br.retrain(st, None, "t")
        self.assertAlmostEqual(st["learnedWeights"]["regimeGrade"], 1.05, places=6)   # only F3 taught it

    def test_replay_is_idempotent_and_bounded(self):
        st = {"trades": [trade(f"G{i}", 10, {"trend": 1.0}, i % 28 + 1) for i in range(200)]}
        br.retrain(st, None, "t")
        w1 = dict(st["learnedWeights"])
        br.retrain(st, None, "t")
        self.assertEqual(w1, st["learnedWeights"])
        self.assertEqual(w1["trend"], br.LEARN_MAX)
        st2 = {"trades": [trade(f"H{i}", -10, {"trend": 1.0}, i % 28 + 1) for i in range(200)]}
        br.retrain(st2, None, "t")
        self.assertEqual(st2["learnedWeights"]["trend"], br.LEARN_MIN)

    def test_trims_count_toward_the_verdict(self):
        t = trade("III", 0, {"trend": 0.5}, 1)
        t["realizedPL"] = t["pnl"] = -5
        t["trims"] = [{"date": "2026-10-20", "shares": 2, "price": 120, "realized": 40.0, "tier": 1}]
        st = {"trades": [t]}
        br.retrain(st, None, "t")
        self.assertEqual(st["trades"][0]["lesson"]["verdict"], "win")     # -5 at the exit, +40 locked by the trim

    def test_open_positions_do_not_teach(self):
        t = trade("JJJ", 99, {"trend": 1.0}, 1)
        t["status"] = "open"
        st = {"trades": [t]}
        br.retrain(st, None, "t")
        self.assertEqual(st["learnedWeights"], {})

    def test_the_weights_object_is_updated_in_place(self):
        lw = {"old": 2.0}
        st = {"learnedWeights": lw, "trades": [trade("KKK", 10, {"trend": 1.0}, 1)]}
        br.retrain(st, None, "t")
        self.assertIs(st["learnedWeights"], lw)                       # a pass's ctx holds this same dict
        self.assertNotIn("old", lw)


class LegacyComponents(unittest.TestCase):
    """Trades recorded before z192: a fund's company signals are dropped, an ETP's market signals put in its direction."""

    def test_sqqq_as_recorded_on_11_04(self):
        t = trade("SQQQ", 120, {"trend": 0.593, "momentum": 0.336, "fundamentals": 0.5, "health": 0.5, "crossAsset": 0.273}, 1,
                  name="ProShares UltraPro Short QQQ")
        comps, notes = br._lesson_components(t, {"SQQQ": {"ticker": "SQQQ", "name": "ProShares UltraPro Short QQQ", "isetf": "true"}})
        self.assertNotIn("fundamentals", comps)
        self.assertNotIn("health", comps)
        self.assertAlmostEqual(comps["crossAsset"], -0.273)            # risk-on tape = against a bear fund
        self.assertAlmostEqual(comps["trend"], 0.593)                 # its own chart stays as read
        self.assertTrue(notes)

    def test_spxs_market_signals_supported_its_long(self):
        t = trade("SPXS", 50, {"trend": 0.521, "crossAsset": -1, "optionsMarket": -0.444, "fundamentals": 0.45, "health": 0.45}, 1)
        comps, _ = br._lesson_components(t, None)                      # the table knows SPXS even without a universe row
        self.assertEqual((comps["crossAsset"], comps["optionsMarket"]), (1.0, 0.444))
        self.assertNotIn("fundamentals", comps)

    def test_a_plain_etf_only_loses_company_signals(self):
        t = trade("XLE", 10, {"trend": 0.4, "fundamentals": 0.3, "crossAsset": 0.2}, 1)
        comps, _ = br._lesson_components(t, {"XLE": {"ticker": "XLE", "name": "Energy Select Sector SPDR Fund", "isetf": "true"}})
        self.assertEqual(comps, {"trend": 0.4, "crossAsset": 0.2})

    def test_a_company_is_untouched_and_z192_trades_are_taught_as_recorded(self):
        c = {"trend": 0.4, "fundamentals": 0.3, "crossAsset": 0.2}
        self.assertEqual(br._lesson_components(trade("AAPL", 10, dict(c), 1, name="Apple Inc."), None)[0], c)
        t = trade("SQQQ", 10, {"crossAsset": -0.3, "trend": 0.4}, 1, signalBasis=br.SIGNAL_BASIS)
        self.assertEqual(br._lesson_components(t, None)[0], {"crossAsset": -0.3, "trend": 0.4})


class Rollups(unittest.TestCase):
    def test_signal_and_style_scores_come_from_the_journal(self):
        st = {"trades": [trade("A", 50, {"trend": 0.8}, 1, style="momentum"), trade("B", -20, {"trend": 0.6}, 2, style="momentum"),
                         trade("C", 30, {"trend": -0.5}, 3, style="core"), trade("D", 0, {"trend": 0.9}, 4, style="core")]}
        br.retrain(st, None, "t")
        br.learning_rollups(st, None)
        s = {x["signal"]: x for x in st["signalScores"]}["trend"]
        self.assertEqual((s["trades"], s["leanN"], s["winRateWhenLeaning"]), (3, 2, 0.5))   # the flat one is not decisive
        self.assertAlmostEqual(s["currentWeight"], st["learnedWeights"]["trend"], places=3)
        by = {x["style"]: x for x in st["styleScores"]}
        self.assertEqual((by["momentum"]["trades"], by["momentum"]["winRate"], by["momentum"]["totalPnl"]), (2, 0.5, 30.0))
        self.assertEqual((by["core"]["trades"], by["core"]["winRate"]), (2, 1.0))


if __name__ == "__main__":
    unittest.main()
