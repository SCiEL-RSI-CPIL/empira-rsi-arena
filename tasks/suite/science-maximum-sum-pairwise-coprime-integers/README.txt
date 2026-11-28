# Maximum sum of pairwise-coprime interval integers

**Source collection:** [Problem-bank constructions: graphs and set systems: source-based research tasks](../../papers/science-prize_c.md)\
**Task id:** `science-maximum-sum-pairwise-coprime-integers`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

construct() -> nonempty distinct Python integers from 1..256. Every pair must have greatest common divisor one. Maximize their exact sum. Integer one is allowed once. The local inclusive interval [1,256] is G(257) under the source’s strict endpoint convention. The prime baseline is a local elementary construction; no asymptotic result or global characterization is claimed, and the optimal score 7372 is the published value a(256) of OEIS A186736.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct() -> nonempty distinct Python integers from 1..256. Every pair must have greatest common divisor one. Maximize their exact sum. Integer one is allowed once.

## State of the art

The maximum sum of pairwise-coprime integers in 1..256 is 7372 (OEIS A186736, a(256) in the b-file).

Reference value: proven optimum `7372` (reviewed 2026-11-20).

- [On two unconventional number theoretic functions and on some related problems](https://users.renyi.hu/~p_erdos/1984-16.pdf), Paul Erdős (1984)
- [OEIS A186736: Maximum sum of relatively prime integers no larger than n](https://oeis.org/A186736)

Parameters: `{"n": 256}`

## Labels

Scientific contribution: `rediscovery` (tier 2); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.

Underlying problem: Determine the maximum sum and structure of pairwise-coprime integer sets in a bounded interval.

Why it matters: Precise attribution is verified, but direct primary-goal evidence for this exact task is not established.


The exact problem and prior result are source-verified. Importance remains unreviewed because the inspected paper’s principal objectives concern other arithmetic functions; its related-problem discussion alone is insufficient under the dataset’s strict primary-goal rule.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Exact finite certificate validation; independently implemented from the cited mathematical definition. Resource bounds and source adaptation are recorded per task. No Lean verification or general theorem is claimed.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Sources

- paper: https://users.renyi.hu/~p_erdos/1984-16.pdf (Paul Erdős, On two unconventional number theoretic functions and on some related problems (1984 bibliography; scanned volume marked 1983). Page 120, equation (29), defines G(n) as the maximum sum of pairwise-coprime positive integers below n; equation (30) states an asymptotic result credited jointly to Erdős and van Lint.)
- benchmark: https://github.com/TheJustinSunPrize/awards/blob/bcf1866ea9a4ae82b32bd95b82ffd812b9ee80d2/problems/catalog-0701-0800.md#JSP-000730 (JSP-000730; pinned revision bcf1866ea9a4ae82b32bd95b82ffd812b9ee80d2. Independently paraphrased finite adaptation; catalog contributor claims are not accepted as proof.)

## Limitations

- Independent finite computational adaptation. Not a Lean proof, prize submission, general conjecture solution, or guarantee of novelty.
- The local inclusive interval [1,256] is G(257) under the source’s strict endpoint convention. The prime baseline is a local elementary construction; no asymptotic result or global characterization is claimed, and the optimal score 7372 is the published value a(256) of OEIS A186736.
- Candidate Python code is trusted code; the evaluator process is not a security sandbox.
- The exact problem and prior result are source-verified. Importance remains unreviewed because the inspected paper’s principal objectives concern other arithmetic functions; its related-problem discussion alone is insufficient under the dataset’s strict primary-goal rule.
