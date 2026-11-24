# CVRP X-n586-k159

**Task id:** `cvrplib-x-n586-k159`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `cvrp` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Route identical vehicles of capacity 28 from one depot to serve the 585 customers of the CVRPLIB X-set instance X-n586-k159. Every customer is visited exactly once, the total demand on a route may not exceed the capacity, and every route starts and ends at the depot. The number of routes is free (the k in the instance name does not limit it). Minimise the total cost: the sum over all routes of the Euclidean distances between consecutive stops, the legs from and to the depot included, each distance rounded to the nearest integer.

## Candidate interface

solve(inst) -> list of routes; a route is a list of customer ids from 1 to n in visiting order (the depot is node 0 and is not listed; empty routes are ignored). inst is a dict with n (number of customers), capacity, coords (array of shape (n + 1, 2), row 0 is the depot), demand (integer array of length n + 1, demand[0] = 0) and name. Score: the total cost, minimized.

## The instance

X-n586-k159 is an instance of the CVRPLIB X set with 585 customers and vehicle capacity 28, whose depot lies at (18, 761) of a grid with integer coordinates from 0 to 1000.

## Why it matters

**Problem:** Find a minimum-cost set of capacity-feasible vehicle routes from one depot that serves every customer of the CVRPLIB X instance X-n586-k159.

The capacitated vehicle routing problem is the core model of distribution logistics (delivery, collection and service fleets). It is NP-hard; the X instances of 100 to 1,000 customers were generated to cover the attributes that make it hard, and the 39 that CVRPLIB lists as not proven optimal are open, so a cheaper set of routes is a new best known solution of an open standard instance.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [The Truck Dispatching Problem](https://doi.org/10.1287/mnsc.6.1.80): Formulate and solve the routing of a fleet of capacitated trucks from a terminal to service stations, introducing the capacitated vehicle routing problem. The task's objective is the capacitated vehicle routing cost (rounded Euclidean distances, free number of routes) on the fixed CVRPLIB instance X-n586-k159.
- [A POPMUSIC matheuristic for the capacitated vehicle routing problem](https://doi.org/10.1016/j.cor.2021.105475): Improve solutions of large capacitated vehicle routing instances with a POPMUSIC matheuristic whose subproblems are solved by an exact branch-cut-and-price solver. The CVRPLIB update log of 2 Jul 2020 credits the current best known solution of X-n586-k159 to improved solutions by Eduardo Queiroga, Eduardo Uchoa and Ruslan Sadykov through a POPMUSIC heuristic using exact VRPSolver; the paper presents that method, and the task asks for a cost below it.

## Record

Best known solution 190316 (not proven optimal; CVRPLIB, checked 2026-11-11).

**Record validation:** reproduced. published routes evaluated by our verifier = 190316; equals the cost in the .sol file and in the registry.

## Data and requirements

CVRPLIB X instance file X-n586-k159.vrp (galgos.inf.puc-rio.br/cvrplib, download/instance/id)

Requires: `py:numpy`, `file:data/arnew/cvrp/X-n586-k159.vrp`. Run `python3 scripts/new_open_problems/prepare_new_data.py --family cvrp` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- The BKS is the cost of the solution file published by CVRPLIB on 2026-11-11; check for newer values before claiming a record.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
