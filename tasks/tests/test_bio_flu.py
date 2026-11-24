"""Influenza forecasting task: weighted interval score, protocol and contract checks."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import bio_flu as fl  # noqa: E402
from autoresearch_suite.bio_physics_common import data_dir  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402

PREPARED = (data_dir() / fl.DATA).is_file()


def pinball_mean(observation, quantiles, levels=fl.QUANTILES):
    """Independent reference: WIS equals twice the mean quantile (pinball) loss over the levels."""
    losses = [2 * ((level - (observation < q)) * (observation - q)) for level, q in zip(levels, quantiles)]
    return float(np.mean(losses))


class WeightedIntervalScoreTests(unittest.TestCase):
    def test_quantile_levels_are_the_flusight_set(self):
        self.assertEqual(len(fl.QUANTILES), 23)
        self.assertEqual(fl.QUANTILES[11], 0.5)
        self.assertTrue(all(b > a for a, b in zip(fl.QUANTILES, fl.QUANTILES[1:])))

    def test_point_forecast_reduces_to_absolute_error(self):
        constant = np.full(23, 7.0)
        self.assertAlmostEqual(fl.weighted_interval_score(10.0, constant), 3.0, places=10)
        self.assertAlmostEqual(fl.weighted_interval_score(7.0, constant), 0.0, places=12)

    def test_matches_the_mean_pinball_loss(self):
        rng = np.random.default_rng(5)
        normal = np.array([np.sqrt(2) * _erfinv(2 * p - 1) for p in fl.QUANTILES])
        for _ in range(50):
            centre, spread = rng.uniform(0, 500), rng.uniform(1, 200)
            quantiles = centre + spread * normal
            observation = float(rng.uniform(0, 800))
            self.assertAlmostEqual(fl.weighted_interval_score(observation, quantiles),
                                   pinball_mean(observation, quantiles), places=6)

    def test_sharper_forecast_scores_better_when_correct(self):
        levels = np.array(fl.QUANTILES)
        wide = 100 + 200 * (levels - 0.5)
        tight = 100 + 20 * (levels - 0.5)
        self.assertLess(fl.weighted_interval_score(100.0, tight), fl.weighted_interval_score(100.0, wide))
        self.assertGreater(fl.weighted_interval_score(400.0, tight), fl.weighted_interval_score(400.0, wide))


def _erfinv(x):
    from math import log, sqrt
    a = 0.147
    ln = log(1 - x * x)
    first = 2 / (np.pi * a) + ln / 2
    return float(np.sign(x) * sqrt(sqrt(first * first - ln / a) - first))


class ContractTests(unittest.TestCase):
    def test_valid_forecast_is_accepted_and_sorted(self):
        matrix = [[float(i) for i in range(23)] for _ in range(4)]
        out = fl.check_forecast({"01": matrix, "02": matrix}, ["01", "02"])
        self.assertEqual(out["01"].shape, (4, 23))

    def test_invalid_forecasts_are_rejected(self):
        good = [[float(i) for i in range(23)] for _ in range(4)]
        bad_shape = [[float(i) for i in range(23)] for _ in range(3)]
        negative = [[-1.0] + [float(i) for i in range(22)] for _ in range(4)]
        decreasing = [[float(23 - i) for i in range(23)] for _ in range(4)]
        for answer in ({"01": good}, {"01": bad_shape, "02": good}, {"01": negative, "02": good},
                       {"01": decreasing, "02": good}, [good, good]):
            with self.assertRaises(CandidateError):
                fl.check_forecast(answer, ["01", "02"])

    def test_season_assignment(self):
        self.assertEqual(fl.season_of("2024-11-02"), "2024-2025")
        self.assertEqual(fl.season_of("2025-04-26"), "2024-2025")
        self.assertEqual(fl.season_of("2025-07-05"), "2024-2025")


@unittest.skipUnless(PREPARED, "prepared data missing; run scripts/prepare_bio_physics.py --only bio_flu")
class PreparedDataTests(unittest.TestCase):
    def test_protocol_covers_the_active_season_and_excludes_the_national_series(self):
        payload = fl.load()
        indices, codes = fl.protocol(payload, "2024-2025")
        self.assertEqual(len(indices), 26)
        self.assertEqual(len(codes), 52)
        self.assertNotIn("US", codes)
        months = {int(payload["dates"][i][5:7]) for i in indices}
        self.assertTrue(months <= set(fl.SEASON_MONTHS))

    def test_history_stops_at_the_reference_date(self):
        payload = fl.load()
        indices, codes = fl.protocol(payload, "2024-2025")
        first = indices[0]
        self.assertEqual(len(payload["series"][codes[0]][:first + 1]), first + 1)
        self.assertLess(payload["dates"][first], payload["dates"][first + 1])

    def test_series_are_weekly_and_non_negative(self):
        payload = fl.load()
        dates = payload["dates"]
        import datetime as dt
        steps = {(dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days for a, b in zip(dates, dates[1:])}
        self.assertEqual(steps, {7})
        for code, series in payload["series"].items():
            self.assertTrue(all(v is None or v >= 0 for v in series), code)


if __name__ == "__main__":
    unittest.main()
