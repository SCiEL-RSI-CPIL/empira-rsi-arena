"""Mauna Loa CO2 forecasts: output validation, what each forecast origin sees and the seed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import earth_co2 as co2  # noqa: E402
from autoresearch_suite import earth_common as ec  # noqa: E402
from autoresearch_suite import earth_sandbox as es  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import module_from, seed_source  # noqa: E402

# The seed is the package's: the generator module that defines it in the source repository is not in this library.
CO2_SEED = seed_source("science-earth-mauna-loa-co2-forecast")

SPY = '''
import numpy as np


class Spy:
    def forecast(self, history):
        return np.full(12, 300.0 + len(history["year"]) / 10.0)


def fit_co2_model(training):
    assert training["year"][-1] == 1999 and training["month"][-1] == 12
    return Spy()
'''


class UnitTests(unittest.TestCase):
    def test_validation(self):
        np.testing.assert_array_equal(co2.validate(0, np.full(12, 400.0)), np.full(12, 400.0))
        for bad in (np.full(11, 400.0), np.full(12, 700.0), np.full(12, np.nan)):
            with self.assertRaises(CandidateError):
                co2.validate(0, bad)


@unittest.skipUnless((ec.data_dir() / co2.DATA).is_file(), "run scripts/prepare_earth.py first")
class HeldOutTests(unittest.TestCase):
    def test_each_origin_sees_months_up_to_it(self):
        data = co2.load()
        setup, steps, origins = co2.protocol(data)
        outputs = es.run_online(module_from(SPY), "fit_co2_model", setup, steps, co2.validate)
        self.assertEqual([round((o[0] - 300.0) * 10) for o in outputs], [t + 1 for t in origins])
        self.assertEqual((data["year"][origins[0]], data["month"][origins[0]]), (2000, 1))

    def test_seed(self):
        value, details = co2.score(module_from(CO2_SEED))
        self.assertLess(value, 1.0)
        self.assertEqual(details["origins"], 300)


if __name__ == "__main__":
    unittest.main()
