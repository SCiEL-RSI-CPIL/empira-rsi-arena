# Physics: product-formula design for Heisenberg-chain time evolution

**Source collection:** [Physics research: source-based research tasks](../../papers/science-physics_research.md)\
**Task id:** `science-physics-heisenberg-product-formula`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Choose the coefficients of a twelve-stage product formula that approximates the time evolution of a six-site Heisenberg chain split into even and odd bonds. The evaluator builds every factor and the exact propagator and scores the mean spectral-norm error at three evolution times.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

coefficients(stages) -> (a, b), two lists of 12 finite numbers in [-50, 50] with sum(a) = sum(b) = 1 (consistency). The evaluator forms the product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k = 1..12, where A and B are the odd- and even-bond halves (each including half of the uniform field) of a six-site open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3. Score (minimize): mean over t = 0.25, 0.5 and 1.0 of the spectral norm ||U(t) - exp(-i t (A + B))||. coefficients runs in a sandboxed subprocess with a 60 s CPU budget

## State of the art

No published value for this chain and protocol. For context, the evaluator gives 0.046 for repeated Strang splitting (the seed) and 0.31 for Suzuki's fourth-order formula with five Strang substeps, which loses at the largest time because its large coefficients amplify the error at big steps.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Theory of Trotter error with commutator scaling](https://doi.org/10.1103/PhysRevX.11.011020)

Parameters: `{"stages": 12, "times": [0.25, 0.5, 1.0], "sites": 6, "delta": 0.8, "field": 0.3, "cpu_seconds": 60}`

## Labels

Scientific contribution: `method` (tier 2); artifact `construction`, domain `natural_science`, data `none`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `quantum_information`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Approximate the time evolution of a quantum spin chain by a product of exponentials of simpler terms with as little error as possible for a given number of factors.

Why it matters: Product formulas are the standard way to simulate quantum dynamics on digital quantum computers, and the error of a given formula at a given step size determines the circuit depth needed for a target accuracy; designing better coefficients is an active line of work in both quantum simulation and geometric numerical integration.

- [Theory of Trotter error with commutator scaling](https://doi.org/10.1103/PhysRevX.11.011020): Main research objective: Bound and understand the error of product formulas for simulating quantum dynamics. Task connection: The task measures exactly that error, in the spectral norm, for a fixed-stage formula on a Heisenberg chain.
- [Fractal decomposition of exponential operators with applications to many-body theories and Monte Carlo simulations](https://doi.org/10.1016/0375-9601(90)90962-N): Main research objective: Construct higher-order product formulas for exponential operators. Task connection: Provides the classical constructions a candidate has to improve on at the measured step sizes.

The cited papers' main goal is the accuracy of product formulas for exponential operators, which the evaluator computes exactly for this chain.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- The Hamiltonian, the even/odd splitting and the error norm are defined in the evaluator; every factor is built from an eigendecomposition of A and B, so the product and the reference exponential are exact up to floating-point error. The stage count is fixed, which fixes the work per step: the task is to design better coefficients, as in the Trotter-error and optimized-splitting literature. Errors are measured at step sizes with t||A + B|| of order ten, where asymptotic order conditions alone do not determine the best scheme; results are specific to this chain.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:scipy`

## Sources

- paper: https://doi.org/10.1103/PhysRevX.11.011020 (Childs, Su, Tran, Wiebe and Zhu (2021), Theory of Trotter error with commutator scaling: commutator bounds on the spectral-norm error of product formulas for quantum simulation.)
- paper: https://doi.org/10.1016/0375-9601(90)90962-N (Suzuki (1990), Fractal decomposition of exponential operators: the higher-order product formulas built from symmetric splittings.)
- paper: https://doi.org/10.1016/S0377-0427(01)00492-7 (Blanes and Moan (2002), Practical symplectic partitioned Runge-Kutta and Runge-Kutta-Nystrom methods: optimized splitting coefficients for a fixed number of stages.)

## Limitations

- The error is measured on one six-site chain and three evolution times; coefficients tuned here need not be optimal for other Hamiltonians, system sizes or step sizes.
- The fixed stage count fixes the cost per step only in this two-group splitting; circuit depth and hardware costs of a quantum implementation are not modelled.
- Errors are computed on the full 64-dimensional Hilbert space by dense linear algebra; this is exact for this size but does not scale to larger chains.
- No published value exists for this exact chain, stage count and set of times.
