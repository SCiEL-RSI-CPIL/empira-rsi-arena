# Software engineering: mutation-adequate test generation for CPython's textwrap

**Source collection:** [Software engineering research: source-based research tasks](../../papers/science-software_engineering.md)\
**Task id:** `science-software-mutation-textwrap`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Generate at most 30 test calls for textwrap (wrap, fill, shorten, dedent, indent) that distinguish as many of its 132 first-order mutants as possible from the original module. A mutant is killed when some test's result or exception type differs; the score is the fraction killed.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

generate_tests(subject) -> list of tests. subject = {'subject': str, 'files': {path: source} (CPython 3.12.14), 'target': path of the mutated module, 'api': {call: [argument kinds]}, 'mutants': [{'unit', 'operator', 'line', 'start', 'end', 'replacement'}, ...], 'max_tests': 30}. A test is {'call': name, 'args': [values]}. Argument kinds: 'str' (at most 500 characters), 'int' (|n| <= 10000), 'opt_int' (such an int or None), 'bool', 'float' (finite, |x| <= 1e6), 'fraction' (a str of at most 30 characters for Fraction(), with an exponent of at most two digits), 'spec' (a format spec of at most 12 characters without runs of four digits), 'number' (a 'fraction' str, an int with |n| <= 1000 or a float), 'numbers' (a list of at most 40 values, each a 'fraction' str, an int with |n| <= 10000 or a float with |x| <= 1e6), 'pairs' (at most 20 [str, str] pairs of at most 100 characters each), {'options': {keyword: kind}} (keyword arguments), {'strings': n} (exactly n strings), {'choice': [allowed values]}. Calls: textwrap: wrap/fill/shorten(text, width, options) -> textwrap.<call>(text, width, **options), dedent(text), indent(text, prefix); urllib.parse: each call is the function of that name with the arguments in order (urlparse/urlsplit(url, scheme, allow_fragments), urljoin(base, url, allow_fragments), parse_qs/parse_qsl(qs, keep_blank_values, strict_parsing, max_num_fields, separator), quote/quote_plus(string, safe)), urlunparse/urlunsplit get the strings as a tuple, urlencode(pairs, doseq, safe) passes the pairs as tuples; fractions: parse(s) -> Fraction(s), make(n, d) -> Fraction(n, d), from_float(x), binary(a, op, b) applies the operator (or divmod, max) to a and b with str operands converted by Fraction, limit_denominator(s, m), round(s, ndigits) (round(Fraction(s)) when ndigits is None), format(s, spec) -> format(Fraction(s), spec), method(s, name) -> getattr(Fraction(s), name)(); statistics (str values become Fraction): summary(name, data) -> statistics.<name>(data), median_grouped(data, interval), quantiles(data, n, method), weighted(name, data, weights) -> fmean or harmonic_mean with weights, spread_given_center(name, data, c) -> variance/pvariance/stdev/pstdev(data, c), bivariate(name, x, y) -> covariance, correlation (linear or ranked) or linear_regression (proportional or not), normal(mu, sigma, method, x) -> NormalDist(mu, sigma).method(x) (quantiles(int(x)) for 1 <= x <= 1000, overlap(NormalDist(x, 1.0))), from_samples(data). A mutant is the target file with lines start..end replaced by 'replacement'; it is killed when some test's outcome differs from the original module's outcome, where the outcome is a hash of the repr of the result (generators consumed, at most 10,001 items) or the type name of the raised exception, with 0.2 s of CPU time per call; tests that time out on the original are ignored. Score: killed mutants / all mutants, maximised (equivalent mutants make 1.0 unreachable); the whole mutation analysis may take at most 240 s of CPU time, otherwise the evaluation is invalid. generate_tests() runs in a sandboxed subprocess (no files, network or subprocesses; 60 s CPU, 2 GB) and may load the files and mutants itself; it should be deterministic (seed any randomness, do not read clocks). Invalid tests make the evaluation invalid.

## State of the art

No published value exists for this mutant set. As a reference, 2,000 random calls from the seed's generator kill 69.7% of the mutants (computed with this evaluator).

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Mutation-Driven Generation of Unit Tests and Oracles](https://doi.org/10.1109/TSE.2011.93)

Parameters: `{"subject": "textwrap", "cpu_seconds": 60, "max_tests": 30}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `software_engineering`, data `public_benchmark`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `software_engineering`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Automated unit test generation with strong oracles: produce a small set of tests that detects as many seeded faults (mutants) of a module as possible.

Why it matters: Mutation score is the standard measure of a test suite's fault-detection ability; generating tests that kill mutants automates oracle and input design for unit testing.

- [Mutation-Driven Generation of Unit Tests and Oracles](https://doi.org/10.1109/TSE.2011.93): Main research objective: Generate unit tests and assertions that detect mutants of the class under test. Task connection: The task asks for tests whose observed outcomes distinguish mutants from the original module.
- [Constraint-based automatic test data generation](https://doi.org/10.1109/32.92910): Main research objective: Automatically generate test data that kills mutants by solving reachability, necessity and sufficiency constraints. Task connection: The task's objective is the share of mutants killed by generated test data.
- [An empirical study of automated unit test generation for Python](https://doi.org/10.1007/s10664-022-10248-w): Main research objective: Generate unit tests for Python modules automatically and assess them by coverage and mutation score. Task connection: The subjects are Python modules and tests are scored by mutation score.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Subjects are pure-Python CPython 3.12.14 modules (PSF licence). Mutants are generated by the evaluator's own AST operators (AOR, ROR, LCR, ASR, UOD, CRP, BCR inside function bodies), with non-compiling, duplicate and trivially compiler-equivalent mutants removed; the remaining equivalent mutants are unknown. Test oracles compare observable outcomes with the original module (regression oracles), as in automated test generators, instead of hand-written assertions.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/software/pymod/mutation-textwrap.json`

## Sources

- paper: https://doi.org/10.1109/TSE.2011.93 (Fraser and Zeller (TSE 2012): generating unit tests and oracles that kill mutants.)
- paper: https://doi.org/10.1109/32.92910 (DeMillo and Offutt (TSE 1991): constraint-based generation of test data that kills mutants.)
- paper: https://doi.org/10.1007/s10664-022-10248-w (Lukasczyk, Kroiss and Fraser (EMSE 2023): automated unit test generation for Python (Pynguin), assessed by coverage and mutation score.)
- paper: https://doi.org/10.1109/ICSE.2015.103 (Papadakis, Jia, Harman and Le Traon (ICSE 2015): trivial compiler equivalence for removing equivalent mutants.)
- archive: https://www.python.org/downloads/release/python-31214/ (CPython 3.12.14 source release (PSF licence): the mutated modules.)

## Limitations

- Mutants come from this evaluator's operator set (MutPy- and mutmut-style replacements inside function bodies); scores are not comparable with other mutation tools.
- Non-compiling, duplicate and trivially compiler-equivalent mutants are removed, but other equivalent mutants remain, so 1.0 is unreachable and the number of killable mutants is unknown.
- Oracles are regression oracles: a test kills a mutant when the hash of the result's repr or the exception type differs from the original's, which also counts differences a specification might allow.
- Reference computed here: 2,000 random calls from the seed's generator kill 69.7% of the mutants; the 30-test budget is this dataset's choice.
- The candidate sees the mutants and may run them, so the task measures test generation and selection against known mutants, not generalisation to unseen faults.
