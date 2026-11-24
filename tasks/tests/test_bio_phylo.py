"""Phylogeny task: simulator determinism, Newick parsing, splits and Robinson-Foulds distance."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import bio_phylo as ph  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402


class NewickTests(unittest.TestCase):
    def test_splits_of_a_hand_checked_tree(self):
        # ((t1,t2),(t3,t4),t5,t0) on six taxa: the non-trivial splits are {1,2} and {3,4}.
        self.assertEqual(ph.parse_newick("((t1,t2),(t3,t4),t5,t0);", 6),
                         {frozenset({1, 2}), frozenset({3, 4})})

    def test_branch_lengths_and_whitespace_are_tolerated(self):
        first = ph.parse_newick("((t1:0.1,t2:0.2):0.05,(t3,t4),t5,t0);", 6)
        second = ph.parse_newick(" ( ( t1 , t2 ) , ( t3 , t4 ) , t5 , t0 ) ; ", 6)
        self.assertEqual(first, second)

    def test_complementary_clades_give_the_same_split(self):
        left = ph.parse_newick("((t0,t1),(t2,t3,t4,t5));", 6)
        right = ph.parse_newick("((t2,t3,t4,t5),(t0,t1));", 6)
        self.assertEqual(left, right)
        self.assertEqual(left, {frozenset({2, 3, 4, 5})})

    def test_invalid_trees_are_rejected(self):
        for text, taxa in (("((t1,t2),(t3,t4),t5;", 6), ("((t1,t2),(t3,t4),t5,t0,t6);", 6),
                           ("((t1,t2),(t3,t4),t0);", 6), ("((t1,t1),(t3,t4),t5,t0);", 6),
                           ("((x1,t2),(t3,t4),t5,t0);", 6), ("", 6), ("(t0);", 6), (42, 6)):
            with self.assertRaises(CandidateError):
                ph.parse_newick(text, taxa)


class DistanceTests(unittest.TestCase):
    def test_robinson_foulds_bounds(self):
        a, b = {frozenset({1, 2})}, {frozenset({1, 2})}
        self.assertEqual(ph.robinson_foulds(a, b), 0.0)
        self.assertEqual(ph.robinson_foulds(a, {frozenset({3, 4})}), 1.0)
        self.assertAlmostEqual(ph.robinson_foulds({frozenset({1, 2}), frozenset({3, 4})}, {frozenset({1, 2})}), 1 / 3)
        self.assertEqual(ph.robinson_foulds(set(), set()), 0.0)


class SimulatorTests(unittest.TestCase):
    def test_instances_are_deterministic(self):
        first = ph.simulate(8, 200, 0.05, 0.15, 7)
        second = ph.simulate(8, 200, 0.05, 0.15, 7)
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_sequences_and_splits_are_well_formed(self):
        for taxa in (8, 12, 16):
            sequences, splits = ph.simulate(taxa, 300, 0.05, 0.15, taxa)
            self.assertEqual(set(sequences), {f"t{i}" for i in range(taxa)})
            self.assertTrue(all(len(s) == 300 and set(s) <= set(ph.BASES) for s in sequences.values()))
            self.assertEqual(len(splits), taxa - 3)  # an unrooted binary tree has n-3 internal splits
            for split in splits:
                self.assertNotIn(0, split)
                self.assertTrue(2 <= len(split) <= taxa - 2)

    def test_true_tree_scores_zero_distance(self):
        _, splits = ph.simulate(10, 300, 0.05, 0.15, 3)
        self.assertEqual(ph.robinson_foulds(splits, splits), 0.0)

    def test_closer_sequences_for_shorter_branches(self):
        short = ph.simulate(8, 2000, 0.01, 0.01, 5)[0]
        long = ph.simulate(8, 2000, 0.3, 0.3, 5)[0]

        def divergence(sequences):
            a, b = sequences["t0"], sequences["t1"]
            return sum(1 for x, y in zip(a, b) if x != y) / len(a)

        self.assertLess(divergence(short), divergence(long))

    def test_instance_table_is_varied(self):
        self.assertEqual(len(ph.INSTANCES), 20)
        self.assertGreaterEqual(len({spec[0] for spec in ph.INSTANCES}), 5)
        self.assertEqual(len({spec[4] for spec in ph.INSTANCES}), 20)  # distinct seeds


if __name__ == "__main__":
    unittest.main()
