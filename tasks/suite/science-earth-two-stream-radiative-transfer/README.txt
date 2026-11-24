# Earth science: two-stream closures for solar radiative transfer

**Source collection:** [Earth and environmental science research: source-based research tasks](../../papers/science-earth_research.md)\
**Task id:** `science-earth-two-stream-radiative-transfer`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Design the coefficients of a two-stream approximation for sunlight in a scattering and absorbing atmospheric layer as functions of single-scattering albedo, asymmetry parameter and solar zenith angle. The evaluator solves the two-stream equations for 12000 layers and scores the mean reflectance and transmittance error against an accurate 96-stream discrete-ordinate solution.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

two_stream_coefficients(omega, g, mu0) -> (gamma1, gamma2, gamma3, f): four float arrays (n,) for n layers given by float arrays omega (single-scattering albedo, 0.5 to 1, exactly 1 for conservative layers), g (Henyey-Greenstein asymmetry parameter, 0 to 0.95) and mu0 (cosine of the solar zenith angle, 0.05 to 1). For each layer the evaluator delta-scales with the forward fraction f (tau' = (1 - omega f) tau, omega' = omega (1 - f) / (1 - omega f)) and solves exactly the two-stream equations dFup/dtau' = gamma1 Fup - gamma2 Fdown - gamma3 omega' exp(-tau'/mu0) and dFdown/dtau' = gamma2 Fup - gamma1 Fdown + (1 - gamma3) omega' exp(-tau'/mu0) (Meador and Weaver 1980; unit beam irradiance, no diffuse light entering at the top, black surface; omega' is capped at 1 - 1e-10) for the layer's optical depth tau, on which the coefficients cannot depend. The candidate runs in a separate sandboxed process (numpy, scipy, scikit-learn and pandas importable; no file, network or subprocess access; one thread) with 60 s of CPU time for fitting and all predictions together; seed any randomness so that results are deterministic. The function is called once with the 3000 distinct (omega, g, mu0) triples of the design, and the coefficients returned for a triple are used at all four of its optical depths. Score: mean over 12000 fixed layers (3000 random triples, a quarter conservative, each at four optical depths, one drawn log-uniformly from each decade of 0.01-100) of |R - R_ref| + |T - T_ref|, where R is the reflectance, T the total (direct plus diffuse) transmittance and the references come from a 96-stream delta-M discrete-ordinate solution; lower is better. Coefficients must satisfy gamma1 > 0, gamma1 >= |gamma2|, 0 <= gamma3 <= 1 and 0 <= f < 1; NaN or wrong shapes are invalid

## State of the art

No published value exists for this case distribution. Published closures evaluated by the task's evaluator: delta-PIFM 0.01904, delta-Eddington 0.01917 (the seed), Eddington 0.01945, PIFM 0.01948, delta-quadrature 0.02072, quadrature 0.02320 (mean |dR| + |dT|); delta-Eddington errors reach 0.40 for grazing sun on thin forward-scattering layers.

Reference value: no documented value for this finite instance (reviewed 2026-11-24).

- [Two-Stream Approximations to Radiative Transfer in Planetary Atmospheres: A Unified Description of Existing Methods and a New Improvement](https://doi.org/10.1175/1520-0469(1980)037<0630:TSATRT>2.0.CO;2)

## Labels

Scientific contribution: `method` (tier 2); artifact `equation`, domain `natural_science`, data `synthetic`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `earth_and_environment`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Approximate multiple scattering of sunlight in atmospheric layers with two-stream equations accurately enough for weather and climate models.

Why it matters: Almost every weather and climate model computes solar heating with a two-stream approximation; its errors in cloud and aerosol reflectance and absorption feed directly into the simulated energy budget.

- [Two-Stream Approximations to Radiative Transfer in Planetary Atmospheres: A Unified Description of Existing Methods and a New Improvement](https://doi.org/10.1175/1520-0469(1980)037<0630:TSATRT>2.0.CO;2): Main research objective: Unify the existing two-stream approximations in one coefficient framework, assess their accuracy and propose an improved scheme. Task connection: The task asks for new coefficient functions in exactly this framework and scores them against an accurate multi-stream solution, as the paper does for its schemes.
- [Comparative Accuracy of Selected Multiple Scattering Approximations](https://doi.org/10.1175/1520-0469(1986)043<0784:CAOSMS>2.0.CO;2): Main research objective: Quantify the accuracy of two-stream and related approximations for reflection and transmission over a wide range of layer parameters. Task connection: The task's score is the same kind of accuracy assessment over optical depth, albedo, asymmetry and sun angle.

Both cited papers have the accuracy of two-stream radiative transfer approximations as their main goal, the problem the local task scores; no reproduction of their numbers is claimed.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Scores are recomputed from pinned public data prepared by scripts/prepare_earth.py; held-out data never enter the candidate's sandboxed process. The held-out observations are public, so a fixed candidate should be checked on later data before any scientific claim.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/earth/twostream/reference_flux_cases.json.gz`

## Sources

- paper: https://doi.org/10.1175/1520-0469(1980)037<0630:TSATRT>2.0.CO;2 (Meador and Weaver (1980): unified gamma1-gamma3 formulation of two-stream approximations, the coefficient tables of the Eddington and quadrature schemes, and their accuracy against exact solutions.)
- paper: https://doi.org/10.1175/1520-0469(1976)033<2452:TDEAFR>2.0.CO;2 (Joseph, Wiscombe and Weinman (1976): delta-Eddington approximation with forward fraction f = g^2 (the seed).)
- paper: https://doi.org/10.1175/1520-0469(1986)043<0784:CAOSMS>2.0.CO;2 (King and Harshvardhan (1986): comparative accuracy of two-stream and related approximations against exact reflectance and transmittance over optical depth, albedo, asymmetry and solar angle.)
- paper: https://doi.org/10.1364/AO.27.002502 (Stamnes et al. (1988): the discrete-ordinate method with delta-M scaling and stable eigen-solution that the reference solver follows.)

## Limitations

- Only a single homogeneous layer over a black surface with a Henyey-Greenstein phase function and unit beam irradiance is scored; multilayer atmospheres, surface reflection, real cloud or aerosol phase functions and thermal emission are not.
- The 12000 layers come from a fixed random design (seed 20261010; 3000 triples of albedo, asymmetry and sun angle, each at four optical depths); the generator and its seed are in the evaluator's source, so a candidate can rebuild the design, but it gains only the four depths at which each triple is scored. A closure tuned to this design may be less accurate for other distributions of optical depth, albedo, asymmetry or sun angle.
- The reference is a numerical solution (48 streams per hemisphere, delta-M), not an analytic one: it changes by about 1e-11 when the stream number is doubled and agreed with an independent Monte Carlo simulation to within its sampling error (about 6e-4) on the checked layers; conservative layers are computed with albedo 1 - 1e-10, and the stored fluxes are rounded to six decimals.
- Coefficients may depend on albedo, asymmetry and sun angle but not on optical depth; closures that use extra information (for example tabulated corrections) are allowed, so a better score is a more accurate two-stream approximation, not necessarily a cheaper or more physical one.
- The candidate runs in a sandboxed child process that never receives the reference fluxes, and no statement of the candidate is executed in the evaluator's process; outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
