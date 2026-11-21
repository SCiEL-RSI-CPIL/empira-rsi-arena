# Job-shop scheduling ta42

**Task id:** `jssp-ta42`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `jssp` (suite/_lib/arnew)\
**Areas:** optimization, theoretical computer science

## Task

Schedule 30 jobs on 20 machines (job-shop benchmark instance ta42). Every job is a sequence of 20 operations, one on each machine, that must run in the given order; an operation occupies its machine for its whole duration without interruption, and a machine processes one operation at a time. Minimise the makespan, the time at which the last operation ends.

## Candidate interface

solve(inst) -> starts, a list of n_jobs lists of non-negative integers: starts[j][k] is the start time of operation k of job j. inst has name, n_jobs, n_machines and jobs[j] = list of (machine, duration) pairs in processing order, machines numbered from 0. Operation k + 1 of a job may not start before operation k has ended, and two operations on one machine may not overlap (one may start at the moment the other ends). Score: the makespan, minimized.

## The instance

Taillard (1993) benchmark instance ta42, 30 jobs x 20 machines, random processing times.

## Why it matters

**Problem:** Find machine sequences, as start times, of minimum makespan for the open job-shop benchmark instance ta42.

Job-shop scheduling is the canonical model of machine scheduling in manufacturing and is NP-hard in the strong sense; the instances of this family are still open, meaning the best known makespan is above the best proven lower bound, so a shorter schedule is a new best known upper bound.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [The Complexity of Flowshop and Jobshop Scheduling](https://doi.org/10.1287/moor.1.2.117): Establish the computational complexity of flow-shop and job-shop makespan scheduling, showing the job-shop problem NP-complete in the strong sense. The task is the makespan job-shop problem on the fixed benchmark instance ta42.
- [Deterministic job-shop scheduling: Past, present and future](https://doi.org/10.1016/S0377-2217(98)00113-1): Survey the formulations, solution methods and benchmark results for the deterministic job-shop scheduling problem. The task is the makespan job-shop problem on the fixed benchmark instance ta42.
- [An extended Akers graphical method with a biased random-key genetic algorithm for job-shop scheduling](https://doi.org/10.1111/itor.12044): Solve the job-shop scheduling problem with a biased random-key genetic algorithm extended by an Akers graphical method. ScheduleOpt credits this paper (code GR2014) with the best known makespan of ta42; Shylo's page cites the related 2012 AT&T technical report (code GR2012) for dmu46.

## Record

Lower bound 1900, best known makespan 1937 (O. Shylo's bounds pages and the ScheduleOpt benchmark library, 2026-11-11).

**Record validation:** matches a pinned source. Shylo's bounds page: lower bound 1887, upper bound 1939; ScheduleOpt jobshop table: lower bound 1900, upper bound 1937; the schedule published on Shylo's page, replayed by our verifier: makespan 1939. Note: The smaller value 1937 is the one in the ScheduleOpt table, which publishes no schedule; the best schedule that could be replayed has makespan 1939.

## Data and requirements

ta42 split from instance_data.txt of thomasWeise/jsspInstancesAndResults (819 KB, all instances in one file)

Requires: `py:numpy`, `file:data/jssp/ta42.txt`. Run `python3 scripts/prepare_new_data.py --family jssp` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- The bounds are those of 2026-11-11 (larger lower bound and smaller upper bound of the two tables); both tables change several times a year, so read them again before claiming a record.
- The record lies 1.95% above the lower bound.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
