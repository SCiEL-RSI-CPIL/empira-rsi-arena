# EoB-BBOB benchmark synthesis

**Paper:** Evolution of Benchmark: Black-Box Optimization Benchmark Design through Large Language Model (https://arxiv.org/pdf/2601.21877)\
**Task id:** `eob-eob-bbob-benchmark-synthesis`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Write a Python objective(X) function defining a deterministic, vectorised minimisation problem on [-5, 5]^10; scored by sqrt(LSI x min(1, ADC / 0.5)), combining landscape-feature similarity to the 24 BBOB functions with the spread of mean results of ten local optimisers (10 runs of 5,000 evaluations each; higher is better).

Benchmark named in the paper: COCO-BBOB / MetaBox. Paper metric: landscape similarity and algorithm distinction capability.

## Candidate interface

objective(X) -> float array of shape (n,): a deterministic, vectorised minimisation problem on the box [-5, 5]^10; X is a float array of shape (n, 10) whose rows are independent points (the candidate receives a copy). Values must be real numbers (strings, booleans and complex values are rejected), finite, with magnitude at most 1e100 and must not depend on the batch composition or on earlier calls; a constant function is rejected. Score (maximize): sqrt(LSI x min(1, ADC / 0.5)). LSI = 1 / (1 + d / sigma^2), where d is the Euclidean distance between the candidate's 16 landscape features (fixed 500-point Latin-hypercube sample; y-distribution, meta-model, local-rank, dispersion, nearest-better, information-content and fitness-distance features, scaled by their across-instance standard deviation) and the nearest of 24 BBOB anchors (each the centroid of five random instances of a BBOB function), and sigma^2 is the median distance from an instance to the centroid of the other instances of its function; LSI is also computed on a copy of the sample shifted by a tiny offset seeded by the SHA-256 of the candidate file, and the smaller value counts. ADC = standard deviation over 10 optimisers (DE/rand/1/bin, jDE, JADE, L-SHADE, constriction PSO, ring PSO, CLPSO, DMS-PSO, CMA-ES, sep-CMA-ES; 10 seeded runs of 5,000 evaluations each) of their mean final best value, after min-max normalisation of the 10 x 10 result matrix

Parameters: `{"runs": 10, "evals_per_dim": 500, "seed": 0, "samples": 500, "sample_seed": 2026}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `continuous_optimisation`, data `synthetic`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Clean-room recreation from the paper (Sections 3.2.2, 4.1 and 4.2): the EoB repository has no licence, so no EoB code or data were used; the scorer follows the paper's definitions.
- Candidate: one 10-dimensional function on [-5, 5]^10. EoB-BBOB consists of 24 10-dimensional functions; the paper does not state their domain ([-5, 5] is the BBOB box). EoB evolves a population of programs with an LLM and collects the 24 non-dominated functions over several runs (2-3 per its Section 4.2); this task scores a single function.
- LSI: Eq. 3 as printed (unsquared Euclidean distance divided by sigma^2). NeurELA, the paper's 16-dimensional landscape embedding, has no licence and is not used; 16 classical exploratory-landscape-analysis features (metaopt_ela.py) on a fixed 500-point Latin-hypercube sample replace it. Each of the 24 BBOB targets is the centroid of five instances' features computed with the suite's BBOB implementation (standard definitions with local random shifts and rotations, not COCO's official instances); features are scaled by their pooled across-instance standard deviation, and sigma^2 (not stated in the paper) is the median distance of an instance to the centroid of the other instances of its function (4.12), so a new instance of a BBOB function scores about 0.5. In the recorded check (validation/metaopt-checks.json), unseen BBOB instances (instance 9) scored 0.30-0.75 and were nearest to their own function in 20 of 24 cases; two non-BBOB functions scored 0.23 and 0.40, and a plain sphere and Rastrigin function with optima near the centre of the box scored 0.34 and 0.33: the BBOB instances place the optimum anywhere in [-4, 4]^10 and apply BBOB's transformations, and the features also reflect where the optimum lies in the box.
- ADC: Eq. 4, with the paper's 'min-max normalization on the Obj matrix' read as one normalisation over the whole 10 x 10 result matrix (runs x optimisers) and the population standard deviation across optimisers. The paper's pool (Vanilla DE, JDE21, MadDE, NL-SHADE-LBC, PSO, GLPSO, sDMS-PSO, SAHLPSO, CMA-ES, Sep-CMA-ES) is replaced by local implementations of DE/rand/1/bin, jDE, JADE, L-SHADE, constriction PSO, ring-topology PSO, CLPSO, DMS-PSO without its local search, CMA-ES and sep-CMA-ES, written from the original publications with default parameters, clipping to the box and fixed seeds. B = 10 runs as in the paper; the per-run budget of 5,000 evaluations (500 x dim) is a local choice because the paper does not state one.
- Scalarisation: EoB optimises LSI and ADC jointly (MOEA/D with PBI, theta = 5, 32 reference vectors); the single score sqrt(LSI x min(1, ADC / 0.5)) is a suite convention (0.5 is the largest possible standard deviation of values in [0, 1]).
- The paper's min-max normalisation makes ADC scale-free: a function on which all optimisers end within a small relative range can still reach a sizeable ADC (in the recorded check, a Schwefel-type function on which all ten optimisers ended between 3,570 and 3,670 scored ADC 0.16).
- Validity: outputs must be finite at every evaluated point; determinism and batch independence are checked only on the feature sample and 32 single rows, so the evaluator reports valid = None. The candidate is imported into the evaluator process, because the scorer calls it about 10,000 times (no held-out data are involved); module-level code can therefore replace the scoring functions (a candidate that does so scores 1.0), so scores are meaningful only for candidates that do not tamper with the evaluator. The BBOB definitions are public in the suite, so a candidate can copy a BBOB instance; because each target is a five-instance centroid, a copy scores like any new instance of that function rather than 1. The 500-point feature sample is fixed and public; the features are therefore also computed on a copy of the sample shifted by a tiny candidate-specific offset (1e-5 of the box width, seeded by the SHA-256 of the candidate file) and the smaller LSI counts, so special-casing the public sample points does not pay (unit test). A function that recomputes the offset from its own source text would not be detected, and the optimisers' seeds are public.
- Not verified against the paper: NeurELA, the paper's optimiser implementations and its LSI and ADC values are not available locally, so scores are not comparable with the paper's figures.
- Clean-room recreation from the EoB paper (no upstream code used): local ELA features replace NeurELA, a local 10-optimiser pool replaces the paper's, and the two objectives are combined into one scalar.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`

## Sources

- paper: https://arxiv.org/pdf/2601.21877 (Section 4.1–4.3)
- repository: https://github.com/MetaEvo/EoB/tree/5d9f30cb19a0231a74f58d17926cff828db6857f
- paper: https://hal.inria.fr/inria-00362633 (BBOB noiseless function definitions (Hansen et al., 2009) used for the 24 anchor functions of the landscape-similarity score)

## Limitations

- The inspected paper links no readable official repository; the official release was identified separately and is pinned (see below).
- Paper-provided anonymous code URL https://anonymous.4open.science/r/EoB could not be read: web tool inaccessible; direct HTTPS connection reset. No downloadable artifact verified.
- Execution interface inspected in pinned source. Full paper experiments have not been reproduced here.
