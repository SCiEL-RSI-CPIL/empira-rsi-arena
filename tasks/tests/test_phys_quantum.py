"""Circuit synthesis and product-formula tasks: exact builders, contracts and order of accuracy."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

NEEDS_SCIPY = unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is not installed")
if importlib.util.find_spec("scipy"):  # the evaluator module imports scipy.linalg when it is loaded
    from autoresearch_suite import phys_quantum as q  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402


@NEEDS_SCIPY
class CircuitBuilderTests(unittest.TestCase):
    def test_targets_are_unitary_and_budgets_are_positive(self):
        self.assertEqual(len(q.TARGETS), 7)
        for name, qubits, target, budget, reason in q.TARGETS:
            size = 2 ** qubits
            self.assertEqual(target.shape, (size, size), name)
            self.assertTrue(np.allclose(target.conj().T @ target, np.eye(size), atol=1e-12), name)
            self.assertGreaterEqual(budget, 1, name)
            self.assertTrue(reason)

    def test_generic_unitaries_are_reproducible_and_not_sparse(self):
        first, second = q.generic_unitary(2, 1), q.generic_unitary(2, 1)
        self.assertTrue(np.array_equal(first, second))
        self.assertNotEqual(round(float(np.abs(first).min()), 6), 0.0)

    def test_cnot_circuit_reproduces_the_cnot_target(self):
        matrix, cnots = q.build_circuit([["cx", 0, 1]], 2, 1)
        self.assertEqual(cnots, 1)
        self.assertAlmostEqual(q.distance(q.CNOT, matrix), 0.0, places=12)

    def test_three_cnots_and_rotations_give_swap_exactly(self):
        # SWAP = CNOT(0,1) CNOT(1,0) CNOT(0,1): an independent hand-written decomposition.
        gates = [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
        matrix, cnots = q.build_circuit(gates, 2, 3)
        self.assertEqual(cnots, 3)
        self.assertAlmostEqual(q.distance(q.SWAP, matrix), 0.0, places=12)

    def test_distance_ignores_a_global_phase(self):
        self.assertAlmostEqual(q.distance(q.SWAP, np.exp(1j * 0.7) * q.SWAP), 0.0, places=12)
        self.assertGreater(q.distance(q.SWAP, q.CNOT), 0.0)

    def test_u3_is_unitary_for_sampled_angles(self):
        for angles in ((0.0, 0.0, 0.0), (1.1, -2.0, 0.3), (math.pi, math.pi / 2, -1.0)):
            gate = q.u3(*angles)
            self.assertTrue(np.allclose(gate.conj().T @ gate, np.eye(2), atol=1e-12))

    def test_invalid_circuits_are_rejected(self):
        for gates, budget in ((["cx"], 3), ([["swap", 0, 1]], 3), ([["cx", 0, 0]], 3), ([["cx", 0, 5]], 3),
                              ([["u3", 0, 0.0, 0.0]], 3), ([["cx", 0, 1], ["cx", 1, 0]], 1),
                              ([["u3", 0, float("nan"), 0.0, 0.0]], 3), ([["u3", True, 0.0, 0.0, 0.0]], 3)):
            with self.assertRaises(CandidateError):
                q.build_circuit(gates, 2, budget)
        with self.assertRaises(CandidateError):
            q.build_circuit([["cx", 0, 1]] * 10, 2, 10, gate_limit=5)


@NEEDS_SCIPY
class SplittingTests(unittest.TestCase):
    def test_hamiltonian_halves_are_hermitian_and_add_to_the_chain(self):
        a, b = q.heisenberg_groups(4, 0.8, 0.3)
        for part in (a, b):
            self.assertTrue(np.allclose(part, part.conj().T, atol=1e-12))
        self.assertEqual(a.shape, (16, 16))
        self.assertGreater(np.linalg.norm(a + b, 2), 0.0)

    def test_commuting_generators_are_split_exactly(self):
        diagonal_a = np.diag([0.3, -1.2, 0.5, 2.0]).astype(complex)
        diagonal_b = np.diag([1.0, 0.25, -0.75, 0.5]).astype(complex)
        errors = q.splitting_error(np.array([1.0]), np.array([1.0]), (diagonal_a, diagonal_b), (0.5, 1.0))
        self.assertLess(max(errors), 1e-12)

    def test_strang_splitting_has_third_order_local_error(self):
        groups = q.heisenberg_groups(4, 0.8, 0.3)
        errors = q.splitting_error(np.array([0.5, 0.5]), np.array([1.0, 0.0]), groups, (0.02, 0.04, 0.08))
        ratios = [errors[i + 1] / errors[i] for i in range(2)]
        for ratio in ratios:  # doubling the step multiplies a third-order error by about eight
            self.assertGreater(ratio, 6.5)
            self.assertLess(ratio, 9.5)

    def test_first_order_splitting_has_second_order_local_error(self):
        groups = q.heisenberg_groups(4, 0.8, 0.3)
        errors = q.splitting_error(np.array([1.0]), np.array([1.0]), groups, (0.02, 0.04, 0.08))
        ratios = [errors[i + 1] / errors[i] for i in range(2)]
        for ratio in ratios:  # doubling the step multiplies a second-order error by about four
            self.assertGreater(ratio, 3.2)
            self.assertLess(ratio, 4.8)


if __name__ == "__main__":
    unittest.main()
