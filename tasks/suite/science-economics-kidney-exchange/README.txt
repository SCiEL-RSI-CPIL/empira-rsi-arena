# Economics: Kidney exchange clearing with two- and three-way cycles

**Source collection:** [Finance and economics research: source-based research tasks](../../papers/science-finance_expansion.md)\
**Task id:** `science-economics-kidney-exchange`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Choose vertex-disjoint two- and three-way exchange cycles in eight public kidney-exchange pools of 256 and 512 incompatible patient-donor pairs to maximize the number of transplants. The evaluator verifies every donor-patient compatibility on the chosen cycles and compares the transplant count with the proven optimum.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

select_exchanges(pool) -> list of cycles; each cycle is a list of 2 or 3 distinct pair indices [v1, ..., vk] such that the donor of v1 is compatible with the patient of v2, ..., and the donor of vk with the patient of v1, and no pair appears in two cycles. pool = {'name': str, 'pairs': n (256 or 512), 'edges': (E, 2) int array of compatible (donor_pair, patient_pair), 0-based, 'max_cycle_length': 3, 'time_limit_seconds': 60.0 (advisory)}. Called once for each of eight PrefLib kidney pools (00036-00000151..154 with 256 pairs, 00036-00000191..194 with 512 pairs, no altruists). Score: total transplants (pairs on the chosen cycles) divided by the total proven optimum, 1945 (maximize; 1 is optimal). The whole evaluation must finish within 600 s. The candidate runs in a separate sandboxed process: numpy, scipy, scikit-learn, pandas and the other installed packages are importable, while the evaluator's library, its data files, the network and subprocesses are not accessible; one thread and 3 GB of memory. It receives only the arguments described here, one call at a time: the arguments of a call are sent after the value of the previous call has been received. Arguments and return values travel as numbers, strings, lists, dicts with string keys and numeric numpy arrays, so a tuple arrives as a list

## State of the art

The optimum of every pool is proven by an exact integer program over all 2- and 3-cycles (optimal transplants 166, 175, 158, 145, 351, 337, 300 and 313; 1945 in total), so the best possible score is 1. The greedy two-cycle seed scores 0.861.

Reference value: proven optimum `1.0` (reviewed 2026-10-07).

- [Clearing algorithms for barter exchange markets: enabling nationwide kidney exchanges](https://doi.org/10.1145/1250910.1250954)
- [PrefLib dataset 00036 (kidney)](https://github.com/PrefLib/PrefLib-Data/tree/0ee5e34c15ed20d93d870f6b3905e0f664bdb8c0/datasets/00036%20-%20kidney)

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `finance`, data `public_benchmark`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `finance`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Organize exchanges among incompatible patient-donor pairs so that as many patients as possible receive a compatible living-donor kidney.

Why it matters: Kidney exchange increases the number of living-donor transplants for patients with willing but incompatible donors; clearing pools optimally under operational cycle limits is NP-hard and directly affects how many patients are transplanted.

- [Efficient Kidney Exchange: Coincidence of Wants in Markets with Compatibility-Based Preferences](https://doi.org/10.1257/aer.97.3.828): Main research objective: Design efficient kidney exchange with two- and three-way exchanges and quantify the gains from larger exchanges. Task connection: The task maximizes transplants with two- and three-way exchanges, the setting analysed in the paper.
- [Clearing algorithms for barter exchange markets: enabling nationwide kidney exchanges](https://doi.org/10.1145/1250910.1250954): Main research objective: Develop algorithms that clear large kidney exchange pools optimally under a cycle-length cap. Task connection: The task is the cycle-cap-3 clearing problem of the paper on public generated pools.

Both papers have kidney exchange design and clearing as their main research goal.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Cycle-cap-3 clearing problem of Abraham, Blum and Sandholm (2007) on public PrefLib pools generated with the Saidman et al. (2006) method. Reference check: the optimum of every pool was proven with HiGHS on the cycle formulation (all 2- and 3-cycles enumerated, relative MIP gap 0; 4 to 8 s for 256 pairs, 83 to 153 s for 512 pairs) and the optimal cycle sets pass the evaluator's independent edge-by-edge check. Chains started by altruistic donors, failure-aware weights and dynamic arrivals are not modelled.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/finance/kidney_preflib_pools.csv`

## Sources

- paper: https://doi.org/10.1257/aer.97.3.828 (Roth, Sonmez and Unver (2007): kidney exchange with two- and three-way exchanges among incompatible patient-donor pairs and the gains from three-way cycles.)
- paper: https://doi.org/10.1145/1250910.1250954 (Abraham, Blum and Sandholm (2007): the clearing problem with a cycle-length cap, its NP-hardness for caps of 3 or more, and exact integer-programming clearing for nationwide pools.)
- paper: https://doi.org/10.1097/01.tp.0000195775.77081.25 (Saidman et al. (2006): the pool generator (blood types, PRA levels, spousal donors) used for the PrefLib instances.)
- benchmark: https://github.com/PrefLib/PrefLib-Data/tree/0ee5e34c15ed20d93d870f6b3905e0f664bdb8c0/datasets/00036%20-%20kidney (PrefLib dataset 00036 (kidney), pinned commit: pools 00036-00000151..154 (256 pairs) and 00036-00000191..194 (512 pairs), no altruists.)

## Limitations

- The pools are synthetic (Saidman et al. generator, donated to PrefLib by John Dickerson), not registry data; PrefLib removed other faulty kidney instances in September 2026 and only pools without altruists are used here.
- Static, single-period clearing that maximizes the number of transplants: no altruist-initiated chains, no crossmatch-failure probabilities, no patient priorities or weights and no dynamic arrivals, all of which matter in practice.
- Every pool's optimum is known, so a score of 1 reproduces the integer-programming solution; the 600 s limit makes the task about fast clearing on the larger pools, and time-limited candidates can depend on hardware.
- The candidate runs in a sandboxed child process that never receives later data, realised outcomes or reference solutions; module-level code still runs once in the evaluator process when the shared template imports the candidate, and outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
