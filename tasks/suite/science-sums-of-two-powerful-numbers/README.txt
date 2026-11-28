# Integers up to 10^18 with many representations as sums of two powerful numbers

**Source collection:** [Problem-bank constructions: hypergraphs, graph constructions and instance-solving methods: source-based research tasks](../../papers/science-prize_i.md)\
**Task id:** `science-sums-of-two-powerful-numbers`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

construct() -> {'N': N, 'pairs': [[a, b], ...]} with 2 <= N <= 10^18 and between 1 and 1000 distinct pairs a <= b, a + b = N (a list of more than 1000 pairs is rejected, so the score is at most 1000), where a and b are powerful (every prime factor appears squared; 1 counts as powerful), checked by exact factorization. The score is the number of pairs, a certified lower bound on the representation count. Maximize. The cap of 1000 listed pairs is attainable: integers below 10^18 with more than 1000 representations as a sum of two squares exist, and squares are powerful. So 1000 is the maximum score and the task is a capped search exercise, not evidence on the conjectured subpolynomial bound.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct() -> {'N': N, 'pairs': [[a, b], ...]} with 2 <= N <= 10^18 and between 1 and 1000 distinct pairs a <= b, a + b = N (a list of more than 1000 pairs is rejected, so the score is at most 1000), where a and b are powerful (every prime factor appears squared; 1 counts as powerful), checked by exact factorization. The score is the number of pairs, a certified lower bound on the representation count. Maximize.

## State of the art

The contract caps the score at 1000 pairs and the cap is attained: 331854559557386125 = 5^3 * 13^2 * 17 * 29 * 37 * 41 * 53 * 61 * 73 * 89 is a sum of two positive squares in 1536 ways, each a pair of powerful numbers (computed for this package).

Reference value: proven optimum `1000` (reviewed 2026-11-24).

Underlying problem: [Erdős Problem #943](https://www.erdosproblems.com/943), status: open.

- [Binary quadratic forms with large discriminants and sums of two squareful numbers](https://doi.org/10.1515/crll.2004.027), Valentin Blomer (2004)
- [OEIS A085252](https://oeis.org/A085252)

Parameters: `{"bound": 1000000000000000000}`

## Labels

Scientific contribution: `rediscovery` (tier 2); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Decide whether the number of representations of n as a sum of two powerful numbers is n^(o(1)) (Erdős Problem #943).

Why it matters: Sums of powerful numbers connect to ternary quadratic forms and the abc conjecture.

- [Binary quadratic forms with large discriminants and sums of two squareful numbers](https://doi.org/10.1515/crll.2004.027): Main research objective: Decide whether the number of representations of n as a sum of two powerful numbers is n^(o(1)) (Erdős Problem #943). Task connection: The cap of 1000 listed pairs is attainable: integers below 10^18 with more than 1000 representations as a sum of two squares exist, and squares are powerful. So 1000 is the maximum score and the task is a capped search exercise, not evidence on the conjectured subpolynomial bound.

Primary research directly studies this mathematical objective; the finite scope, reference value and rediscovery status are explicit.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- The cap of 1000 listed pairs is attainable: integers below 10^18 with more than 1000 representations as a sum of two squares exist, and squares are powerful. So 1000 is the maximum score and the task is a capped search exercise, not evidence on the conjectured subpolynomial bound. Exact local certificates are not Lean proofs; reciprocal sums are reported as floats. Consult the independent oracle and malformed-input tests.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Sources

- paper: https://doi.org/10.1515/crll.2004.027 (Valentin Blomer, Binary quadratic forms with large discriminants and sums of two squareful numbers (2004). Crossref record inspected (Blomer studies how many integers are sums of two squareful numbers). Erdős Problem #943 page inspected: is the number of representations n^(o(1)) for every n? OEIS A085252 inspected (counts for small n). No record values over a range are documented. The cap of 1000 listed pairs is attainable: integers below 10^18 with more than 1000 representations as a sum of two squares exist, and squares are powerful. So 1000 is the maximum score and the task is a capped search exercise, not evidence on the conjectured subpolynomial bound.)
- archive: https://www.erdosproblems.com/943 (Erdős Problem #943 (open); statement, history and current bounds reviewed 2026-09-24.)
- benchmark: https://github.com/TheJustinSunPrize/awards/blob/bcf1866ea9a4ae82b32bd95b82ffd812b9ee80d2/problems/catalog-0701-0800.md#JSP-000784 (JSP-000784; pinned discovery source, independently paraphrased. Primary mathematical attribution follows the papers above.)

## Limitations

- The score is capped at 1000 listed pairs and the cap is attainable, so the optimum (1000) is known; listed pairs certify a lower bound on the representation count.
- Independent Python implementation; no paper text, upstream evaluator, Lean proof or prize-bank content is redistributed. Candidate Python executes as trusted research code.
