"""Exact reference, independent small-instance, and malformed-certificate checks."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'suite/_lib'))
from autoresearch_suite import runtime, science_prize as verify


def known_witness():
    # Golomb (1970), as attributed by JSP-000301; mathematical data, not new work.
    return {'n': 12167, 'factorizations': [[[23, 3]], [[2, 3], [3, 2], [13, 2]]]}


def cerny(n):
    # Classical Černý family, independently encoded from its definition.
    return [[(i + 1) % n for i in range(n)], list(range(n - 1)) + [0]]


class PowerfulCertificates(unittest.TestCase):
    def test_published_counterexample_and_factors(self):
        score, details = verify.powerful_counterexample(known_witness())
        self.assertEqual(score, 1)
        self.assertEqual(details['factorizations'], [[[23, 3]], [[2, 3], [3, 2], [13, 2]]])
        self.assertFalse(details['lean_verified'])
        self.assertEqual(23**3 + 1, 2**3 * 3**2 * 13**2)
        self.assertTrue(110**2 < 12167 < 12168 < 111**2)

    def test_prime_check_against_independent_sieve(self):
        sieve = [False, False] + [True] * 4999
        for p in range(2, len(sieve)):
            if sieve[p]:
                for multiple in range(2 * p, len(sieve), p):
                    sieve[multiple] = False
        self.assertEqual([verify.prime(n) for n in range(len(sieve))], sieve)

    def test_wrong_incomplete_composite_and_repeated_factors_rejected(self):
        cases = []
        for factors in ([[[23, 2]], [[2, 3], [3, 2], [13, 2]]],
                        [[[23, 3]], [[2, 3], [39, 2]]],
                        [[[23, 3]], [[2, 1], [2, 2], [3, 2], [13, 2]]],
                        [[[23, 3]], [[2, 3], [3, 2]]]):
            candidate = known_witness()
            candidate['factorizations'] = factors
            cases.append(candidate)
        # Both powerful, but one square: not a counterexample.
        cases.append({'n': 8, 'factorizations': [[[2, 3]], [[3, 2]]]})
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(runtime.CandidateError):
                verify.powerful_counterexample(candidate)

    def test_types_ranges_and_extra_claims_rejected(self):
        for value in [True, 12167.0, '12167', -1, 0, 10**12, 10**100]:
            candidate = known_witness()
            candidate['n'] = value
            with self.subTest(value=value), self.assertRaises(runtime.CandidateError):
                verify.powerful_counterexample(candidate)
        for value in [None, [], {}, {**known_witness(), 'lean_verified': True}]:
            with self.assertRaises(runtime.CandidateError):
                verify.powerful_counterexample(value)


class ResetThreshold(unittest.TestCase):
    def test_classical_family_reaches_conjectured_bound(self):
        for n in range(2, 13):
            transitions = cerny(n)
            score, details = verify.reset_threshold(transitions, n)
            self.assertEqual(score, (n - 1)**2)
            states = set(range(n))
            for letter in details['shortest_reset_word']:
                states = {transitions[letter][state] for state in states}
            self.assertEqual(len(states), 1)

    def test_exact_bfs_agrees_with_independent_word_enumeration(self):
        rng = random.Random(1729)
        for _ in range(30):
            n = 3
            transitions = [[rng.randrange(n) for _ in range(n)] for _ in range(2)]
            reference = None
            # Every shortest simple path in the 7-state nonempty subset graph
            # has <=6 edges. This bound does not assume Černý's conjecture.
            for length in range(1, 7):
                for word in itertools.product(range(2), repeat=length):
                    destinations = list(range(n))
                    for letter in word:
                        destinations = [transitions[letter][state] for state in destinations]
                    if len(set(destinations)) == 1:
                        reference = length
                        break
                if reference is not None:
                    break
            if reference is None:
                with self.assertRaisesRegex(runtime.CandidateError, 'not synchronizing'):
                    verify.reset_threshold(transitions, n)
            else:
                self.assertEqual(verify.reset_threshold(transitions, n)[0], reference)

    def test_renaming_states_and_letters_preserves_threshold(self):
        n = 7
        perm = [3, 1, 6, 0, 5, 2, 4]
        columns = [[0] * n for _ in range(2)]
        for letter, column in enumerate(cerny(n)):
            for state, target in enumerate(column):
                columns[letter][perm[state]] = perm[target]
        self.assertEqual(verify.reset_threshold(columns[::-1], n)[0], 36)

    def test_malformed_and_non_synchronizing_automata_rejected(self):
        for value in [[], [[0, 1]], [[0, 1], [1, 0]], [[0, 2], [0, 0]],
                      [[0, True], [0, 0]], [[0.0, 1], [0, 0]], [[0, 1], [0]]]:
            with self.subTest(value=value), self.assertRaises(runtime.CandidateError):
                verify.reset_threshold(value, 2)


class MaximalSidon(unittest.TestCase):
    def test_published_example_is_maximal_but_not_maximum(self):
        # Dias da Silva & Nathanson (2009), section 1, [1,7].
        self.assertEqual(verify.maximal_sidon([1, 3, 4], 7)[0], 3)
        self.assertEqual(verify.maximal_sidon([1, 2, 5, 7], 7)[0], 4)

    def test_every_small_set_agrees_with_independent_difference_characterization(self):
        def sidon(values):
            differences = [b - a for a, b in itertools.combinations(sorted(values), 2)]
            return len(differences) == len(set(differences))
        n = 8
        for mask in range(1 << n):
            values = [i + 1 for i in range(n) if mask >> i & 1]
            maximal = sidon(values) and all(not sidon(values + [x])
                                            for x in range(1, n + 1) if x not in values)
            if maximal:
                self.assertEqual(verify.maximal_sidon(values, n)[0], len(values))
            else:
                with self.assertRaises(runtime.CandidateError):
                    verify.maximal_sidon(values, n)

    def test_diagonal_collisions_and_nonmaximal_sets_rejected(self):
        for values in [[1, 2, 3], [1, 3], [], [1, 1, 3], [0, 3, 4],
                       [1, 3, 8], [True, 3, 4], [1.0, 3, 4]]:
            with self.subTest(values=values), self.assertRaises(runtime.CandidateError):
                verify.maximal_sidon(values, 7)
        self.assertEqual(verify.maximal_sidon([1], 1)[0], 1)


class PrizeIntegration(unittest.TestCase):
    def test_every_seed_runs_as_a_standalone_evaluator(self):
        records = json.loads((ROOT / 'catalog/science/prize.json').read_text())
        self.assertEqual(len(records), 3)
        for record in records:
            directory = ROOT / 'suite' / record['id']
            result = subprocess.run([sys.executable, str(directory / 'evaluator.py'),
                                     str(directory / 'initial_program.py')],
                                    cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(result.stdout)
            self.assertIs(value['valid'], True)
            self.assertIs(value['lean_verified'], False)
            self.assertEqual(value['combined_score'], value['score'] * (-1 if value['direction'] == 'minimize' else 1))


if __name__ == '__main__':
    unittest.main()
