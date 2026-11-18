# CVRP X-n303-k21

**Task id:** `cvrplib-x-n303-k21`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `cvrp` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Route identical vehicles of capacity 794 from one depot to serve the 302 customers of the CVRPLIB X-set instance X-n303-k21. Every customer is visited exactly once, the total demand on a route may not exceed the capacity, and every route starts and ends at the depot. The number of routes is free (the k in the instance name does not limit it). Minimise the total cost: the sum over all routes of the Euclidean distances between consecutive stops, the legs from and to the depot included, each distance rounded to the nearest integer.

## Candidate interface

solve(inst) -> list of routes; a route is a list of customer ids from 1 to n in visiting order (the depot is node 0 and is not listed; empty routes are ignored). inst is a dict with n (number of customers), capacity, coords (array of shape (n + 1, 2), row 0 is the depot), demand (integer array of length n + 1, demand[0] = 0) and name. Score: the total cost, minimized.

## The instance

X-n303-k21 is an instance of the CVRPLIB X set with 302 customers and vehicle capacity 794, whose depot lies at the centre (500, 500) of a grid with integer coordinates from 0 to 1000.

## Why it matters

**Problem:** Find a minimum-cost set of capacity-feasible vehicle routes from one depot that serves every customer of the CVRPLIB X instance X-n303-k21.

The capacitated vehicle routing problem is the core model of distribution logistics (delivery, collection and service fleets). It is NP-hard; the X instances of 100 to 1,000 customers were generated to cover the attributes that make it hard, and the 39 that CVRPLIB lists as not proven optimal are open, so a cheaper set of routes is a new best known solution of an open standard instance.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [The Truck Dispatching Problem](https://doi.org/10.1287/mnsc.6.1.80): Formulate and solve the routing of a fleet of capacitated trucks from a terminal to service stations, introducing the capacitated vehicle routing problem. The task's objective is the capacitated vehicle routing cost (rounded Euclidean distances, free number of routes) on the fixed CVRPLIB instance X-n303-k21.
- [Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood](https://doi.org/10.1016/j.cor.2021.105643): Present an open-source hybrid genetic search for the capacitated vehicle routing problem with the SWAP* neighbourhood and show that it improves the state of the art on CVRP benchmark instances. The CVRPLIB update log of 21 Sep 2020 credits the current best known solution of X-n303-k21 to Thibaut Vidal's modern implementation of hybrid genetic search for the CVRP; this paper presents Vidal's open-source HGS-CVRP, and the log entry does not say which version or paper the 2020 run used.

## Record

Best known solution 21736 (not proven optimal; CVRPLIB, checked 2026-11-11).

**Record validation:** reproduced. published routes evaluated by our verifier = 21736; equals the cost in the .sol file and in the registry.

## Data and requirements

CVRPLIB X instance file X-n303-k21.vrp (galgos.inf.puc-rio.br/cvrplib, download/instance/id)

Requires: `py:numpy`, `file:data/cvrp/X-n303-k21.vrp`. Run `python3 scripts/prepare_new_data.py --family cvrp` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- The BKS is the cost of the solution file published by CVRPLIB on 2026-11-11; check for newer values before claiming a record.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
