# Economics: Efficient school choice under coarse priorities

**Source collection:** [Finance and economics research: source-based research tasks](../../papers/science-finance_expansion.md)\
**Task id:** `science-economics-school-choice-tie-breaking`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Assign students to schools when school priorities have many ties, improving on deferred acceptance with a single tie-breaking lottery (DA-STB) while keeping the matching weakly stable. The evaluator checks every student-school pair for blocking and scores the share of the possible reduction in total preference rank achieved on six synthetic markets.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

assign_students(problem) -> array of S ints: the school of each student (-1 = unassigned). problem = {'students': S, 'schools': C, 'list_length': L, 'preferences': (S, L) school indices from most to least preferred, 'priorities': (S, C) ints (higher = higher priority, equal = tie; walk-zone tiers and siblings), 'capacities': (C,), 'lottery': (S,) the single tie-breaking order used by the DA-STB benchmark (lottery[0] wins all ties)}. Valid only if every student gets a school on her list or nothing, capacities hold, and the matching is weakly stable: no student prefers a school that has a free seat or admits someone of strictly lower priority there. Six fixed instances (150 to 300 students). Score: (total rank under DA-STB - total rank of the candidate) / (total rank under DA-STB - proven minimum total rank over weakly stable matchings), summed over instances; rank 1 is a first choice and unassigned counts L + 1 (maximize; the DA-STB seed scores 0 and the optimum 1). The candidate runs in a separate sandboxed process: numpy, scipy, scikit-learn, pandas and the other installed packages are importable, while the evaluator's library, its data files, the network and subprocesses are not accessible; one thread and 3 GB of memory. It receives only the arguments described here, one call at a time: the arguments of a call are sent after the value of the previous call has been received. Arguments and return values travel as numbers, strings, lists, dicts with string keys and numeric numpy arrays, so a tuple arrives as a list

## State of the art

The minimum total rank over weakly stable matchings is proven for every instance (total 2319 versus 2916 under DA-STB), so the best possible score is 1; the DA-STB seed scores 0.

Reference value: proven optimum `1.0` (reviewed 2026-10-07).

- [What's the Matter with Tie-Breaking? Improving Efficiency in School Choice](https://doi.org/10.1257/aer.98.3.669)

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `finance`, data `synthetic`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `finance`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Assign students to public schools when priorities are coarse, so that the assignment is stable with respect to priorities yet wastes as little student welfare as possible through arbitrary tie-breaking.

Why it matters: Cities such as New York and Boston assign tens of thousands of students each year with deferred acceptance and lottery tie-breaking; the efficiency lost to tie-breaking affects many students and motivated changes to real mechanisms.

- [What's the Matter with Tie-Breaking? Improving Efficiency in School Choice](https://doi.org/10.1257/aer.98.3.669): Main research objective: Show that tie-breaking in deferred acceptance causes avoidable inefficiency and construct efficient weakly stable matchings. Task connection: The task scores how much of the inefficiency of DA-STB a candidate removes while keeping weak stability.
- [Strategy-proofness versus Efficiency in Matching with Indifferences: Redesigning the NYC High School Match](https://doi.org/10.1257/aer.99.5.1954): Main research objective: Quantify the efficiency cost of tie-breaking in the NYC high-school match and its trade-off with strategy-proofness. Task connection: The benchmark mechanism is DA with single tie-breaking, the mechanism studied for NYC.

Both papers have efficient and stable school assignment under coarse priorities as their main research question.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Efficiency loss of single tie-breaking in school choice with coarse priorities (Erdil and Ergin 2008; Abdulkadiroglu, Pathak and Roth 2009) on synthetic NYC-style instances. Reference checks: the minimum total rank over weakly stable matchings was proven for every instance with an integer program (HiGHS, gap 0, at most 4.2 s), the program matches brute-force enumeration on small instances, and both DA-STB and the optimal matchings pass the evaluator's exact blocking-pair check. Truthful preferences are assumed: the score ignores incentives, and no strategy-proof mechanism improves on DA-STB in general.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`

## Sources

- paper: https://doi.org/10.1257/000282803322157061 (Abdulkadiroglu and Sonmez (2003): school choice as mechanism design; student-proposing deferred acceptance and stability with respect to school priorities.)
- paper: https://doi.org/10.1257/aer.98.3.669 (Erdil and Ergin (2008): tie-breaking makes DA inefficient under coarse priorities; stable improvement cycles reach student-optimal weakly stable matchings.)
- paper: https://doi.org/10.1257/aer.99.5.1954 (Abdulkadiroglu, Pathak and Roth (2009): single versus multiple tie-breaking and the efficiency-strategy-proofness trade-off in the redesigned NYC high-school match.)
- paper: https://doi.org/10.1016/S0304-3975(01)00206-7 (Manlove, Irving, Iwama, Miyazaki and Morita (2002): optimization over weakly stable matchings with ties is NP-hard in general (motivates the integer-programming reference).)

## Limitations

- Markets are synthetic (random homes and schools on a unit square, school quality, distance and taste shocks, nested walk zones and 5% siblings); they mimic coarse NYC-style priorities but are not real application data.
- Preferences are taken as truthful: the score measures welfare given reported preferences, and mechanisms that exploit it can be manipulable; Abdulkadiroglu, Pathak and Roth show no strategy-proof mechanism dominates DA-STB.
- Total rank with unassigned students counted as list length + 1 is one welfare criterion among several (for example rank profiles or the number assigned); the proven optimum depends on that choice.
- Instances are small enough for exact optimization (150 to 300 students) and their optima are known, so a perfect score reproduces the integer-programming solution.
- The candidate runs in a sandboxed child process that never receives later data, realised outcomes or reference solutions; module-level code still runs once in the evaluator process when the shared template imports the candidate, and outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
