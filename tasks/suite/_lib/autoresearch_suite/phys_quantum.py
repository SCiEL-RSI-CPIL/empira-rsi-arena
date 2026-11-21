"""Exact quantum circuit synthesis and product-formula design.

``unitary_synthesis`` asks for circuits of single-qubit rotations and CNOTs that reproduce
fixed target unitaries within published CNOT budgets: one CNOT for CNOT, two for iSWAP,
three for SWAP and for a generic two-qubit unitary (Vidal and Dawson 2004; Vatan and
Williams 2004), six for the Toffoli gate (optimal, Shende and Markov 2009) and twenty for
a generic three-qubit unitary (quantum Shannon decomposition, Shende, Bullock and Markov
2006). Every budget is achievable, so the distance can reach zero.

``trotter_splitting`` asks for the coefficients of a product formula
exp(-i t (A + B)) ~ prod_k exp(-i a_k t A) exp(-i b_k t B) for a Heisenberg spin chain
split into even and odd bonds. The evaluator builds every factor and the exact matrix
exponential and measures the spectral-norm error, the quantity bounded in the Trotter-error
literature (Childs, Su, Tran, Wiebe and Zhu 2021).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .bio_physics_common import as_array, run_sandboxed
from .runtime import CandidateError, require

PAULI = {"x": np.array([[0, 1], [1, 0]], complex), "y": np.array([[0, -1j], [1j, 0]]),
         "z": np.array([[1, 0], [0, -1]], complex), "i": np.eye(2, dtype=complex)}


def _hermitian(size, seed):
    """A fixed Hermitian matrix with small rational entries (no random number generator involved)."""
    k = np.arange(size)
    real = np.cos(seed * (k[:, None] + 1) * (k[None, :] + 1) / 7.0)
    imaginary = np.sin(seed * (k[:, None] + 2) * (k[None, :] + 3) / 11.0)
    matrix = real + 1j * (imaginary - imaginary.T) / 2
    return (matrix + matrix.conj().T) / 2


def generic_unitary(qubits, seed):
    """U = exp(-i H) for the fixed Hermitian H above: a dense unitary with no special structure."""
    return sla.expm(-1j * _hermitian(2 ** qubits, seed))


def _two_qubit(matrix):
    return np.array(matrix, dtype=complex)


SWAP = _two_qubit([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CNOT = _two_qubit([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
ISWAP = _two_qubit([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
TOFFOLI = np.eye(8, dtype=complex)
TOFFOLI[[6, 7]] = TOFFOLI[[7, 6]]

# (name, qubits, target unitary, CNOT budget, published justification of the budget)
TARGETS = (
    ("cnot", 2, CNOT, 1, "one CNOT by definition"),
    ("iswap", 2, ISWAP, 2, "iSWAP needs exactly two CNOTs (Vatan and Williams 2004)"),
    ("swap", 2, SWAP, 3, "SWAP needs exactly three CNOTs (Vatan and Williams 2004)"),
    ("generic_two_qubit_a", 2, generic_unitary(2, 1), 3, "three CNOTs suffice for every two-qubit unitary"),
    ("generic_two_qubit_b", 2, generic_unitary(2, 2), 3, "three CNOTs suffice for every two-qubit unitary"),
    ("toffoli", 3, TOFFOLI, 6, "six CNOTs are optimal for the Toffoli gate (Shende and Markov 2009)"),
    ("generic_three_qubit", 3, generic_unitary(3, 3), 20, "twenty CNOTs suffice for every three-qubit unitary"),
)


def u3(theta, phi, lam):
    """The standard single-qubit rotation U3(theta, phi, lambda)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def _apply_single(state_matrix, gate, qubit, qubits):
    operator = np.array([[1]], dtype=complex)
    for q in range(qubits):
        operator = np.kron(operator, gate if q == qubit else PAULI["i"])
    return operator @ state_matrix


def _apply_cnot(matrix, control, target, qubits):
    size = 2 ** qubits
    index = np.arange(size)
    control_bit = (index >> (qubits - 1 - control)) & 1
    flipped = index ^ (control_bit << (qubits - 1 - target))
    return matrix[flipped, :]


def build_circuit(gates, qubits, cnot_budget, gate_limit=400):
    """Build the unitary of a validated gate list; returns (unitary, cnot count)."""
    require(isinstance(gates, (list, tuple)), "circuit: expected a list of gates")
    require(len(gates) <= gate_limit, f"circuit: at most {gate_limit} gates allowed, got {len(gates)}")
    matrix = np.eye(2 ** qubits, dtype=complex)
    cnots = 0
    for position, gate in enumerate(gates):
        require(isinstance(gate, (list, tuple)) and gate, f"gate {position}: expected a non-empty sequence")
        kind = gate[0]
        require(isinstance(kind, str), f"gate {position}: first entry must be 'u3' or 'cx'")
        if kind == "u3":
            require(len(gate) == 5, f"gate {position}: u3 takes (qubit, theta, phi, lam)")
            qubit = gate[1]
            angles = as_array(gate[2:], shape=(3,), name=f"gate {position} angles", low=-1e4, high=1e4)
            require(isinstance(qubit, int) and not isinstance(qubit, bool) and 0 <= qubit < qubits,
                    f"gate {position}: qubit index out of range")
            matrix = _apply_single(matrix, u3(*angles), qubit, qubits)
        elif kind == "cx":
            require(len(gate) == 3, f"gate {position}: cx takes (control, target)")
            control, target = gate[1], gate[2]
            require(all(isinstance(q, int) and not isinstance(q, bool) and 0 <= q < qubits for q in (control, target)),
                    f"gate {position}: qubit index out of range")
            require(control != target, f"gate {position}: control and target must differ")
            cnots += 1
            require(cnots <= cnot_budget, f"circuit: more than {cnot_budget} CNOT gates")
            matrix = _apply_cnot(matrix, control, target, qubits)
        else:
            raise CandidateError(f"gate {position}: unknown gate {kind!r} (allowed: u3, cx)")
    return matrix, cnots


def distance(target, circuit):
    """Phase-insensitive normalized distance: 0 exactly when the circuit equals the target up to phase."""
    size = target.shape[0]
    overlap = abs(np.trace(target.conj().T @ circuit)) / size
    return float(max(0.0, 1.0 - min(1.0, overlap)))


SYNTHESIS_DRIVER = '''

def _bp_synthesis_driver(problems):
    _bp_seed()
    return [_bp_plain(synthesize(problem)) for problem in problems]
'''


def public_target(name, qubits, target, budget):
    """What the candidate sees: the target unitary as nested [real, imaginary] rows, and its budget."""
    return {"name": name, "qubits": qubits, "cnot_budget": budget,
            "real": target.real.tolist(), "imaginary": target.imag.tolist()}


def synthesis_score(module, P):
    problems = [public_target(name, qubits, target, budget) for name, qubits, target, budget, _ in TARGETS]
    answers = run_sandboxed(module, SYNTHESIS_DRIVER, "_bp_synthesis_driver", [problems], required=("synthesize",),
                            cpu_seconds=int(P.get("cpu_seconds", 120)), memory_mb=2048)
    require(isinstance(answers, list) and len(answers) == len(TARGETS), f"expected {len(TARGETS)} circuits")
    rows, values = [], []
    for (name, qubits, target, budget, _), gates in zip(TARGETS, answers):
        matrix, cnots = build_circuit(gates, qubits, budget)
        value = distance(target, matrix)
        values.append(value)
        rows.append({"target": name, "qubits": qubits, "cnots": cnots, "cnot_budget": budget,
                     "distance": value, "exact": value < 1e-10, "gates": len(gates)})
    return float(np.mean(values)), {"targets": rows, "exact_targets": sum(r["exact"] for r in rows),
                                    "worst_distance": max(values)}


# --- product formulas -------------------------------------------------------

def _site_operator(label, site, sites):
    operator = np.array([[1]], dtype=complex)
    for q in range(sites):
        operator = np.kron(operator, PAULI[label] if q == site else PAULI["i"])
    return operator


def heisenberg_groups(sites=6, delta=0.8, field=0.3):
    """Even-bond and odd-bond halves of an open XXZ chain in a uniform field (A + B = H)."""
    size = 2 ** sites
    groups = [np.zeros((size, size), complex), np.zeros((size, size), complex)]
    for bond in range(sites - 1):
        term = sum(_site_operator(p, bond, sites) @ _site_operator(p, bond + 1, sites) * (delta if p == "z" else 1.0)
                   for p in ("x", "y", "z"))
        groups[bond % 2] += term
    for site in range(sites):
        groups[site % 2] += field * _site_operator("z", site, sites)
    return groups[0], groups[1]


SPLITTING_DRIVER = '''

def _bp_splitting_driver(stages):
    _bp_seed()
    return _bp_plain(coefficients(stages))
'''


def splitting_error(a, b, groups, times):
    """Spectral-norm error of the product formula against the exact exponential at each time."""
    A, B = groups
    wa, va = np.linalg.eigh(A)
    wb, vb = np.linalg.eigh(B)
    exact_generator = A + B
    we, ve = np.linalg.eigh(exact_generator)
    errors = []
    for t in times:
        product = np.eye(A.shape[0], dtype=complex)
        for coefficient_a, coefficient_b in zip(a, b):
            product = (va * np.exp(-1j * coefficient_a * t * wa)) @ va.conj().T @ product
            product = (vb * np.exp(-1j * coefficient_b * t * wb)) @ vb.conj().T @ product
        exact = (ve * np.exp(-1j * t * we)) @ ve.conj().T
        errors.append(float(np.linalg.norm(product - exact, 2)))
    return errors


def splitting_score(module, P):
    stages = int(P["stages"])
    times = tuple(P.get("times", (0.25, 0.5, 1.0)))
    answer = run_sandboxed(module, SPLITTING_DRIVER, "_bp_splitting_driver", [stages], required=("coefficients",),
                           cpu_seconds=int(P.get("cpu_seconds", 60)), memory_mb=2048)
    require(isinstance(answer, (list, tuple)) and len(answer) == 2, "coefficients() must return (a, b)")
    a = as_array(answer[0], shape=(stages,), name="coefficients a", low=-50.0, high=50.0)
    b = as_array(answer[1], shape=(stages,), name="coefficients b", low=-50.0, high=50.0)
    require(abs(a.sum() - 1.0) < 1e-9 and abs(b.sum() - 1.0) < 1e-9,
            "coefficients: both a and b must sum to 1 (consistency of the product formula)")
    groups = heisenberg_groups(int(P.get("sites", 6)), float(P.get("delta", 0.8)), float(P.get("field", 0.3)))
    errors = splitting_error(a, b, groups, times)
    value = float(np.mean(errors))
    return value, {"times": list(times), "errors": errors, "stages": stages,
                   "sum_a": float(a.sum()), "sum_b": float(b.sum()), "max_abs_coefficient": float(max(abs(a).max(), abs(b).max()))}
