"""RNA structure task: WUSS and dot-bracket parsing, projection, F1 and data integrity."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import bio_rna as rna  # noqa: E402
from autoresearch_suite.bio_physics_common import data_dir  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402

PREPARED = (data_dir() / rna.DATA).is_file()


class ParsingTests(unittest.TestCase):
    def test_consensus_brackets(self):
        self.assertEqual(rna.consensus_pairs("<<<...>>>"), [(0, 8), (1, 7), (2, 6)])
        self.assertEqual(rna.consensus_pairs("<<..>>[[..]]"), [(0, 5), (1, 4), (6, 11), (7, 10)])
        self.assertEqual(rna.consensus_pairs(":::,,,___"), [])

    def test_dot_bracket_round_trip(self):
        self.assertEqual(rna.dot_bracket_pairs("((....))"), [(0, 7), (1, 6)])
        self.assertEqual(rna.dot_bracket_pairs("........"), [])
        for bad in ("(((...", "...)))", "((..))x", "[[..]]"):
            with self.assertRaises(CandidateError):
                rna.dot_bracket_pairs(bad)

    def test_projection_drops_gapped_and_noncanonical_pairs(self):
        # Consensus pairs columns (0,9) and (1,8); the sequence has a gap at column 1.
        pairs = rna.consensus_pairs("<<......>>")
        sequence, projected = rna.project("G-AAAAAUUC", pairs)
        self.assertEqual(sequence, "GAAAAAUUC")
        self.assertEqual(projected, [(0, 8)])
        # A non-canonical pair (A opposite A) is dropped.
        _, none_kept = rna.project("A........A", pairs)
        self.assertEqual(none_kept, [])

    def test_projection_translates_thymine_and_enforces_minimum_span(self):
        sequence, projected = rna.project("GcaugT", rna.consensus_pairs("<....>"))
        self.assertEqual(sequence, "GCAUGU")
        self.assertEqual(projected, [(0, 5)])
        _, short = rna.project("GAAC", rna.consensus_pairs("<..>"))
        self.assertEqual(short, [])

    def test_f1_values(self):
        self.assertEqual(rna.f1([(0, 9), (1, 8)], [(0, 9), (1, 8)]), (1.0, 1.0, 1.0))
        score, sensitivity, precision = rna.f1([(0, 9), (1, 8)], [(0, 9)])
        self.assertAlmostEqual(sensitivity, 0.5)
        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(score, 2 / 3)
        self.assertEqual(rna.f1([(0, 9)], [])[0], 0.0)
        self.assertEqual(rna.f1([(0, 9)], [(2, 7)])[0], 0.0)


@unittest.skipUnless(PREPARED, "prepared data missing; run scripts/prepare_bio_physics.py --only bio_rna")
class PreparedDataTests(unittest.TestCase):
    def test_instances_are_clean_and_diverse(self):
        instances = rna.load_instances()
        self.assertGreaterEqual(len(instances), 60)
        self.assertGreaterEqual(len({i["family"] for i in instances}), 30)
        sequences = [i["sequence"] for i in instances]
        self.assertEqual(len(set(sequences)), len(sequences))
        for item in instances:
            self.assertTrue(set(item["sequence"]) <= set("ACGU"), item["family"])
            self.assertTrue(rna.MIN_LENGTH <= len(item["sequence"]) <= rna.MAX_LENGTH, item["family"])
            self.assertGreaterEqual(len(item["pairs"]), rna.MIN_PAIRS, item["family"])

    def test_reference_pairs_are_canonical_nested_and_in_range(self):
        for item in rna.load_instances():
            sequence, pairs = item["sequence"], [tuple(p) for p in item["pairs"]]
            seen = set()
            for i, j in pairs:
                self.assertTrue(0 <= i < j < len(sequence))
                self.assertGreaterEqual(j - i, 4)
                self.assertIn((sequence[i], sequence[j]), rna.CANONICAL)
                self.assertNotIn(i, seen)
                self.assertNotIn(j, seen)
                seen.update((i, j))
            for a, b in pairs:  # consensus pairs are nested, so no two pairs cross
                for c, d in pairs:
                    self.assertFalse(a < c < b < d)

    def test_excluded_families_are_absent(self):
        families = {i["family"].lower() for i in rna.load_instances()}
        for word in rna.EXCLUDED:
            self.assertFalse(any(word in family for family in families), word)


if __name__ == "__main__":
    unittest.main()
