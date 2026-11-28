"""Nuclear-mass extrapolation task: table parsing, held-out split and scoring."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import phys_nuclear as nu  # noqa: E402
from autoresearch_suite.bio_physics_common import data_dir  # noqa: E402

PREPARED = (data_dir() / nu.DATA).is_file()
HYDROGEN, NEUTRON = 7288.971064, 8071.31806

SAMPLE_2020 = """header
1N-Z    N    Z   A  EL    O     MASS EXCESS           BINDING ENERGY/A        BETA-DECAY ENERGY               ATOMIC MASS
                                   (keV)                  (keV)                    (keV)                        (micro-u)
0  4   30   26   56 Fe        -60608.0480     0.2814     8790.3547    0.0050  B-  -4196.6283     0.2815   55 934935.537      0.302
  -1   13   14   27 Si  -nn   -12384.501     0.115      8124.3360    0.0043  B- -4812.360     0.115     26 986704.690      0.124
  12   57   45  102 Rh   x    -86775#       300#       8577#        3#      B-      *                101 906844#     322#
"""


class ParsingTests(unittest.TestCase):
    def test_fixed_width_parser_reads_experimental_and_estimated_rows(self):
        rows = nu.parse_table(SAMPLE_2020, (28, 42), (42, 54))
        self.assertEqual(rows[(30, 26)], (56, "Fe", "-60608.0480", "0.2814"))
        self.assertEqual(rows[(13, 14)][0], 27)
        self.assertIn("#", rows[(57, 45)][2])
        self.assertEqual(nu._number(rows[(57, 45)][2]), -86775.0)

    def test_rms(self):
        self.assertAlmostEqual(nu.rms([3.0, -4.0]), (12.5) ** 0.5)


@unittest.skipUnless(PREPARED, "prepared data missing; run scripts/prepare_bio_physics.py --only phys_nuclear")
class PreparedDataTests(unittest.TestCase):
    def test_split_sizes_and_disjointness(self):
        payload = nu.load()
        training = {(r[0], r[1]) for r in payload["training"]}
        targets = {(r[0], r[1]) for r in payload["targets"]}
        self.assertEqual(len(payload["training"]), 2408)
        self.assertEqual(len(payload["targets"]), 71)
        self.assertFalse(training & targets)
        self.assertTrue(all(n >= 8 and z >= 8 for n, z in training | targets))

    def test_iron_56_binding_energy_matches_the_tabulated_value(self):
        # AME2016 tabulates B/A = 8790.354 keV for 56Fe; recompute it from the parsed mass excess.
        payload = nu.load()
        row = next(r for r in payload["training"] if r[0] == 30 and r[1] == 26)
        binding_per_nucleon = (26 * HYDROGEN + 30 * NEUTRON - row[4]) / 56
        self.assertAlmostEqual(binding_per_nucleon, 8790.35, delta=0.05)

    def test_ame2016_estimates_are_a_strong_reference(self):
        payload = nu.load()
        errors = [r[6] - r[4] for r in payload["targets"] if r[6] is not None]
        self.assertEqual(len(errors), 69)
        self.assertAlmostEqual(nu.rms(errors), 213.4, delta=0.5)


if __name__ == "__main__":
    unittest.main()
