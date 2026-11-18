# CVRP XL-n1141-k112

**Task id:** `cvrplib-xl-n1141-k112`\
**Status:** ready\
**Direction:** minimize\
**Family:** `cvrp` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Route identical vehicles of capacity 761 from one depot to serve the 1140 customers of the CVRPLIB XL-set instance XL-n1141-k112. Every customer is visited exactly once, the total demand on a route may not exceed the capacity, and every route starts and ends at the depot. The number of routes is free (the k in the instance name does not limit it). Minimise the total cost: the sum over all routes of the Euclidean distances between consecutive stops, the legs from and to the depot included, each distance rounded to the nearest integer.

## Candidate interface

solve(inst) -> list of routes; a route is a list of customer ids from 1 to n in visiting order (the depot is node 0 and is not listed; empty routes are ignored). inst is a dict with n (number of customers), capacity, coords (array of shape (n + 1, 2), row 0 is the depot), demand (integer array of length n + 1, demand[0] = 0) and name. Score: the total cost, minimized.

## The instance

XL-n1141-k112 is an instance of the CVRPLIB XL set with 1,140 customers and vehicle capacity 761; its depot lies at the random position (370, 604), with a mix of random and clustered customers (3 seeds) and demands drawn uniformly from 50-100 (attributes from Table 1 of arXiv:2601.11467).

## Why it matters

**Problem:** Find a minimum-cost set of capacity-feasible vehicle routes from one depot that serves every customer of the CVRPLIB XL instance XL-n1141-k112, a large-scale instance with 1,000 to 10,000 customers.

Large-scale distribution routing with thousands of customers and routes of up to 200 stops is practical, and the XL instances were generated with the X-set scheme to cover its attributes, but they are beyond the reach of exact methods. In a 30-day community challenge from 12 January 2026 participating teams submitted 1,932 improved best known solutions, and the records kept moving afterwards, so a lower cost is a new best known solution of an open large-scale instance.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [The Truck Dispatching Problem](https://doi.org/10.1287/mnsc.6.1.80): Formulate and solve the routing of a fleet of capacitated trucks from a terminal to service stations, introducing the capacitated vehicle routing problem. The task's objective is the capacitated vehicle routing cost (rounded Euclidean distances, free number of routes) on the fixed CVRPLIB instance XL-n1141-k112.
- [AILS-II: An Adaptive Iterated Local Search Heuristic for the Large-Scale Capacitated Vehicle Routing Problem](https://doi.org/10.1287/ijoc.2023.0106): Solve large-scale capacitated vehicle routing instances with an adaptive iterated local search heuristic. Table 1 of the XL instance paper (arXiv:2601.11467) lists AILS-II as the method of the initial best known solution for 93 of the 100 XL instances, before the 2026 challenge improved them; the task asks for a cost below the current best known solution of XL-n1141-k112.
- [Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood](https://doi.org/10.1016/j.cor.2021.105643): Present an open-source hybrid genetic search for the capacitated vehicle routing problem with the SWAP* neighbourhood and show that it improves the state of the art on CVRP benchmark instances. The CVRPLIB update log credits many later XL records (Breunig, Seman, Bian) to runs that start from the published best known solution and re-solve subsets of routes with HGS-CVRP or a modified HGS; this paper presents the HGS-CVRP code, and it does not describe the XL records themselves.

## Record

Best known solution 95638 (CVRPLIB, checked 2026-11-11; these values still move: five improved between 2026-11-02 and 2026-11-11).

**Record validation:** reproduced. published routes evaluated by our verifier = 95638; equals the cost in the .sol file and in the registry.

## Data and requirements

CVRPLIB XL instance file XL-n1141-k112.vrp (galgos.inf.puc-rio.br/cvrplib, download/instance/id)

Requires: `py:numpy`, `file:data/cvrp/XL-n1141-k112.vrp`. Run `python3 scripts/prepare_new_data.py --family cvrp` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- The BKS is the cost of the solution file published by CVRPLIB on 2026-11-11; check for newer values before claiming a record.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
