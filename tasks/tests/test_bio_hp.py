"""HP lattice folding: energy recomputation, contract enforcement and benchmark integrity."""
import itertools
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import bio_hp  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402


def brute_force_energy(sequence, dimension):
    """Slowly enumerate every self-avoiding walk and return the minimum energy (tiny sequences only)."""
    units = [tuple(1 if a == k else 0 for a in range(dimension)) for k in range(dimension)]
    units += [tuple(-x for x in u) for u in units]
    best = 0
    for steps in itertools.product(range(len(units)), repeat=len(sequence) - 1):
        point, path = (0,) * dimension, [(0,) * dimension]
        for s in steps:
            point = tuple(a + b for a, b in zip(point, units[s]))
            path.append(point)
        if len(set(path)) != len(path):
            continue
        index = {p: i for i, p in enumerate(path)}
        contacts = 0
        for i, p in enumerate(path):
            if sequence[i] != "H":
                continue
            for u in units[:dimension]:
                j = index.get(tuple(a + b for a, b in zip(p, u)))
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
        best = min(best, -contacts)
    return best


class HPEnergyTests(unittest.TestCase):
    def test_energy_matches_a_hand_checked_conformation(self):
        # A 2x2 block of H residues: one non-chain contact between residues 0 and 3.
        self.assertEqual(bio_hp.energy("HHHH", [[0, 0], [1, 0], [1, 1], [0, 1]], 2), -1)
        # A straight chain has no non-chain contacts.
        self.assertEqual(bio_hp.energy("HHHH", [[0, 0], [1, 0], [2, 0], [3, 0]], 2), 0)
        # P residues never contribute: the only non-chain neighbour pair here is P-P.
        self.assertEqual(bio_hp.energy("PHHP", [[0, 0], [1, 0], [1, 1], [0, 1]], 2), 0)
        # H residues three apart around a square do contact each other.
        self.assertEqual(bio_hp.energy("HPPH", [[0, 0], [1, 0], [1, 1], [0, 1]], 2), -1)

    def test_energy_agrees_with_brute_force_enumeration(self):
        for sequence in ("HHPHH", "HPHPHH", "HHPPHH"):
            for dimension in (2, 3):
                conformations = []
                units = [tuple(1 if a == k else 0 for a in range(dimension)) for k in range(dimension)]
                units += [tuple(-x for x in u) for u in units]
                for steps in itertools.product(range(len(units)), repeat=len(sequence) - 1):
                    point, path = (0,) * dimension, [(0,) * dimension]
                    for s in steps:
                        point = tuple(a + b for a, b in zip(point, units[s]))
                        path.append(point)
                    if len(set(path)) == len(path):
                        conformations.append([list(p) for p in path])
                observed = min(bio_hp.energy(sequence, c, dimension) for c in conformations)
                self.assertEqual(observed, brute_force_energy(sequence, dimension), (sequence, dimension))

    def test_invalid_conformations_are_rejected(self):
        with self.assertRaises(CandidateError):  # self-intersecting
            bio_hp.energy("HHHH", [[0, 0], [1, 0], [0, 0], [0, 1]], 2)
        with self.assertRaises(CandidateError):  # not a unit step
            bio_hp.energy("HHHH", [[0, 0], [2, 0], [2, 1], [2, 2]], 2)
        with self.assertRaises(CandidateError):  # wrong shape
            bio_hp.energy("HHHH", [[0, 0], [1, 0], [1, 1]], 2)
        with self.assertRaises(CandidateError):  # non-integer coordinates
            bio_hp.energy("HHHH", [[0, 0], [0.5, 0], [1, 0], [1, 1]], 2)
        with self.assertRaises(CandidateError):  # nonfinite
            bio_hp.energy("HHHH", [[0, 0], [1, 0], [1, 1], [float("nan"), 1]], 2)


class HPBenchmarkTests(unittest.TestCase):
    def test_benchmark_sequences_have_the_published_lengths(self):
        lengths_2d = [20, 25, 36, 48, 50, 60, 64, 85, 100, 100]
        self.assertEqual([len(s) for _, s, _ in bio_hp.BENCHMARK_2D], lengths_2d)
        self.assertEqual([e for _, _, e in bio_hp.BENCHMARK_2D], [-9, -8, -14, -23, -21, -36, -42, -53, -48, -50])
        lengths_3d = [48] * 10 + [64, 67, 88, 103, 124, 136]
        self.assertEqual([len(s) for _, s, _ in bio_hp.BENCHMARK_3D], lengths_3d)
        self.assertEqual([e for _, _, e in bio_hp.BENCHMARK_3D][:10], [-32, -34, -34, -33, -32, -32, -32, -31, -34, -33])
        for name, sequence, best in bio_hp.BENCHMARK_2D + bio_hp.BENCHMARK_3D:
            self.assertEqual(set(sequence), {"H", "P"}, name)
            self.assertLess(best, 0, name)

    def test_run_length_expansion(self):
        self.assertEqual(bio_hp.expand("HP2H3"), "HPPHHH")
        self.assertEqual(bio_hp.expand("P8H2"), "PPPPPPPPHH")
        self.assertEqual(len(bio_hp.expand("HPH2P2H4PH3P2H2P2HPH3PHPH2P2H2P3HP8H2")), 48)

    def test_scoring_ratio_and_details(self):
        straight = {2: [[[i, 0] for i in range(len(s))] for _, s, _ in bio_hp.BENCHMARK_2D],
                    3: [[[i, 0, 0] for i in range(len(s))] for _, s, _ in bio_hp.BENCHMARK_3D]}
        for dimension in (2, 3):
            value, details = bio_hp.evaluate_conformations(dimension, straight[dimension])
            self.assertEqual(value, 0.0)  # straight chains have energy 0
            self.assertEqual(details["total_energy"], 0)
            self.assertEqual(details["matched_best_known"], 0)
        # A conformation list of the wrong length violates the contract.
        with self.assertRaises(CandidateError):
            bio_hp.evaluate_conformations(2, straight[2][:3])


if __name__ == "__main__":
    unittest.main()
