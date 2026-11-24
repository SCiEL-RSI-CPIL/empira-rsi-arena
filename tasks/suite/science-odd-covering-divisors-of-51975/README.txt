# Odd covering systems with moduli dividing 51975 (fewest uncovered residues)

**Source collection:** [Problem-bank constructions: number theory and additive sets: source-based research tasks](../../papers/science-prize_h.md)\
**Task id:** `science-odd-covering-divisors-of-51975`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

construct() -> classes [[m, r], ...] with distinct moduli m > 1 dividing 51975 = 3^3 * 5^2 * 7 * 11 (hence odd) and 0 <= r < m. The score is the number of residues modulo 51975 covered by no class (exact). Minimize. The score 0 (a covering system with distinct odd moduli) is impossible for this modulus by the necessary condition of Berger, Felzenbaum and Fraenkel (1986). Every divisor of 51975 above 1 is available once. The minimum is positive: an odd covering with distinct moduli needs prod (p-1)/(p-2) - sum 1/(p-2) > 2 over the primes p dividing the least common multiple (Berger, Felzenbaum and Fraenkel), and for 3, 5, 7, 11 the left side is 86/45. The score measures how close systems with this modulus come to a covering; it cannot decide the odd covering problem. No minimum is documented.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct() -> classes [[m, r], ...] with distinct moduli m > 1 dividing 51975 = 3^3 * 5^2 * 7 * 11 (hence odd) and 0 <= r < m. The score is the number of residues modulo 51975 covered by no class (exact). Minimize. The score 0 (a covering system with distinct odd moduli) is impossible for this modulus by the necessary condition of Berger, Felzenbaum and Fraenkel (1986).

## State of the art

No documented minimum for moduli dividing 51975; the analogous all-divisor minimum is tabulated only for small odd moduli (OEIS A399152), always positive. For 51975 the minimum is positive by the necessary condition of Berger, Felzenbaum and Fraenkel (Acta Arith. 45 (1986) 375-379): prod (p-1)/(p-2) - sum 1/(p-2) > 2 over the prime divisors of the least common multiple, and the left side is 86/45 for 3, 5, 7, 11.

Reference value: no documented value for this finite instance (reviewed 2026-11-24).

Underlying problem: [Erdős Problem #7](https://www.erdosproblems.com/7), status: open ($25).

- [The Erdős–Selfridge problem with square-free moduli](https://doi.org/10.2140/ant.2021.15.609), P. Balister, B. Bollobás, R. Morris, J. Sahasrabudhe, M. Tiba (2021)
- [Covering systems with restricted divisibility](https://doi.org/10.1215/00127094-2019-0058), Robert D. Hough, Pace P. Nielsen (2019)
- [OEIS A399152: Minimum number of residues mod 2n+1 left uncovered by one class per divisor](https://oeis.org/A399152)

Parameters: `{"base": 51975}`

## Labels

Scientific contribution: `open_problem` (tier 1); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Decide whether a covering system with distinct odd moduli exists (the Erdős-Selfridge odd covering problem).

Why it matters: The odd covering problem (Erdős Problem #7) is among the best-known open questions on covering systems, with prizes offered by both Erdős and Selfridge.

- [The Erdős–Selfridge problem with square-free moduli](https://doi.org/10.2140/ant.2021.15.609): Main research objective: Decide whether a covering system with distinct odd moduli exists (the Erdős-Selfridge odd covering problem). Task connection: Every divisor of 51975 above 1 is available once. The minimum is positive: an odd covering with distinct moduli needs prod (p-1)/(p-2) - sum 1/(p-2) > 2 over the primes p dividing the least common multiple (Berger, Felzenbaum and Fraenkel), and for 3, 5, 7, 11 the left side is 86/45. The score measures how close systems with this modulus come to a covering; it cannot decide the odd covering problem. No minimum is documented.

Primary research directly studies this mathematical objective; the finite scope, reference value and rediscovery status are explicit.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Every divisor of 51975 above 1 is available once. The minimum is positive: an odd covering with distinct moduli needs prod (p-1)/(p-2) - sum 1/(p-2) > 2 over the primes p dividing the least common multiple (Berger, Felzenbaum and Fraenkel), and for 3, 5, 7, 11 the left side is 86/45. The score measures how close systems with this modulus come to a covering; it cannot decide the odd covering problem. No minimum is documented. Exact local certificates are not Lean proofs; the Mahler measure and reciprocal sums are reported as floats. Consult the independent oracle and malformed-input tests.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Sources

- paper: https://doi.org/10.2140/ant.2021.15.609 (Paul Balister, Béla Bollobás, Robert Morris, Julian Sahasrabudhe and Marius Tiba, The Erdős–Selfridge problem with square-free moduli (2021). Crossref records inspected (Balister, Bollobás, Morris, Sahasrabudhe and Tiba 2021; Hough and Nielsen 2019). Erdős Problem #7 page inspected: no covering exists with distinct odd squarefree moduli, some modulus must be divisible by 2 or 3, and an odd covering would need a least common multiple divisible by 9 or 15; existence is open (Erdős offered $25 for a proof of nonexistence, Selfridge $300 for an example). OEIS A399152 inspected: the analogous minimum using every divisor of an odd M is tabulated only for small M (e.g. 731 for M = 1225). Every divisor of 51975 above 1 is available once. The minimum is positive: an odd covering with distinct moduli needs prod (p-1)/(p-2) - sum 1/(p-2) > 2 over the primes p dividing the least common multiple (Berger, Felzenbaum and Fraenkel), and for 3, 5, 7, 11 the left side is 86/45. The score measures how close systems with this modulus come to a covering; it cannot decide the odd covering problem. No minimum is documented.)
- archive: https://www.erdosproblems.com/7 (Erdős Problem #7 (open ($25)); statement, history and current bounds reviewed 2026-09-24.)
- benchmark: https://github.com/TheJustinSunPrize/awards/blob/bcf1866ea9a4ae82b32bd95b82ffd812b9ee80d2/problems/catalog-0001-0100.md#JSP-000047 (JSP-000047; pinned discovery source, independently paraphrased. Primary mathematical attribution follows the papers above.)

## Limitations

- No documented minimum for this modulus (reference label none). The minimum is provably positive (Berger, Felzenbaum and Fraenkel 1986), and a positive minimum for one modulus says nothing about others.
- Independent Python implementation; no paper text, upstream evaluator, Lean proof or prize-bank content is redistributed. Candidate Python executes as trusted research code.
