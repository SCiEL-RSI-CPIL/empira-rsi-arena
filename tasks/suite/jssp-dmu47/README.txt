# Job-shop scheduling dmu47

**Task id:** `jssp-dmu47`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `jssp` (suite/_lib/arnew)\
**Areas:** optimization, theoretical computer science

## Task

Schedule 20 jobs on 20 machines (job-shop benchmark instance dmu47). Every job is a sequence of 20 operations, one on each machine, that must run in the given order; an operation occupies its machine for its whole duration without interruption, and a machine processes one operation at a time. Minimise the makespan, the time at which the last operation ends.

## Candidate interface

solve(inst) -> starts, a list of n_jobs lists of non-negative integers: starts[j][k] is the start time of operation k of job j. inst has name, n_jobs, n_machines and jobs[j] = list of (machine, duration) pairs in processing order, machines numbered from 0. Operation k + 1 of a job may not start before operation k has ended, and two operations on one machine may not overlap (one may start at the moment the other ends). Score: the makespan, minimized.

## The instance

Demirkol, Mehta and Uzsoy (1998) benchmark instance dmu47 of the cscmax subfamily, 20 jobs x 20 machines.

## Why it matters

**Problem:** Find machine sequences, as start times, of minimum makespan for the open job-shop benchmark instance dmu47.

Job-shop scheduling is the canonical model of machine scheduling in manufacturing and is NP-hard in the strong sense; the instances of this family are still open, meaning the best known makespan is above the best proven lower bound, so a shorter schedule is a new best known upper bound.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [The Complexity of Flowshop and Jobshop Scheduling](https://doi.org/10.1287/moor.1.2.117): Establish the computational complexity of flow-shop and job-shop makespan scheduling, showing the job-shop problem NP-complete in the strong sense. The task is the makespan job-shop problem on the fixed benchmark instance dmu47.
- [Deterministic job-shop scheduling: Past, present and future](https://doi.org/10.1016/S0377-2217(98)00113-1): Survey the formulations, solution methods and benchmark results for the deterministic job-shop scheduling problem. The task is the makespan job-shop problem on the fixed benchmark instance dmu47.
- [Combining hash-based Tabu search and frequent pattern mining for job-shop scheduling](https://doi.org/10.1080/24725854.2025.2532071): Solve the job-shop scheduling problem with a hash-based tabu search guided by frequent pattern mining. Shylo's page credits this algorithm (code HTSFP2024) with the best known makespan of dmu47; ScheduleOpt lists it as LHW2024.

## Record

Lower bound 3714, best known makespan 3939 (O. Shylo's bounds pages and the ScheduleOpt benchmark library, 2026-11-11).

**Record validation:** reproduced. Shylo's bounds page: lower bound 3684, upper bound 3939; ScheduleOpt jobshop table: lower bound 3714, upper bound 3939; the schedule published on Shylo's page, replayed by our verifier: makespan 3939.

## Data and requirements

dmu47 split from instance_data.txt of thomasWeise/jsspInstancesAndResults (819 KB, all instances in one file)

Requires: `py:numpy`, `file:data/arnew/jssp/dmu47.txt`. Run `python3 scripts/new_open_problems/prepare_new_data.py --family jssp` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- The bounds are those of 2026-11-11 (larger lower bound and smaller upper bound of the two tables); both tables change several times a year, so read them again before claiming a record.
- The record lies 6.06% above the lower bound.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
