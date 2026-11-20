# Physics: broadband antireflection coating design

**Source collection:** [Physics research: source-based research tasks](../../papers/science-physics_research.md)\
**Task id:** `science-physics-broadband-antireflection-coating`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Design a stack of at most twenty low- and high-index layers (1.38 and 2.35) on glass that minimizes the mean reflectance between 400 and 800 nm at normal incidence, with at most 1000 nm of total coating. The evaluator checks the constraints and computes the reflectance spectrum exactly with the thin-film characteristic-matrix method.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

design(problem) -> (materials, thicknesses): a list of 'L'/'H' labels and a list of physical thicknesses in nm, layers ordered from the air side to the substrate. problem gives ambient_index 1.0, substrate_index 1.52, materials {'L': 1.38, 'H': 2.35} (lossless, non-dispersive), band_nm [400, 800], max_layers 20, max_total_nm 1000 and max_layer_nm 400; every thickness must lie in [0, max_layer_nm] and the sum must not exceed max_total_nm. design runs in a sandboxed subprocess with a 60 s CPU budget. Score (minimize): mean normal-incidence reflectance in percent over 401 equally spaced wavelengths in the band, computed exactly with the characteristic-matrix method (bare glass: 4.26 %)

## State of the art

No published value for this exact specification. For context, bare glass reflects 4.26 %, the single quarter-wave seed gives 1.58 %, and a multistart local optimization of alternating 14-layer stacks run during task construction reached 0.084 %.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Optimal single-band normal-incidence antireflection coatings](https://doi.org/10.1364/AO.35.000644)

Parameters: `{"band_nm": [400, 800], "points": 401, "max_layers": 20, "max_total_nm": 1000, "max_layer_nm": 400, "cpu_seconds": 60}`

## Labels

Scientific contribution: `method` (tier 2); artifact `construction`, domain `natural_science`, data `none`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `physics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Design a multilayer thin-film stack that minimizes the average reflectance of a glass surface over the visible band.

Why it matters: Antireflection coatings are needed on essentially every optical surface, and how low the average reflectance can go for a given number of layers, index contrast and total thickness is a long-studied design question with published limits.

- [Optimal single-band normal-incidence antireflection coatings](https://doi.org/10.1364/AO.35.000644): Main research objective: Determine how low the average reflectance of single-band antireflection coatings can be made as a function of bandwidth, index contrast and thickness. Task connection: The task poses this design problem for a fixed two-material system and band and scores the exact mean reflectance.
- [Application of the needle optimization technique to the design of optical coatings](https://doi.org/10.1364/AO.35.005493): Main research objective: Develop and apply a numerical method that designs optical coatings meeting a spectral target. Task connection: The task asks for exactly this kind of design procedure under an exact transfer-matrix objective.

The cited papers' main research goal is antireflection coating design and its attainable limits, which the evaluator measures exactly.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- The thin-film model is exact for homogeneous lossless layers at normal incidence; real coatings add dispersion, absorption, interface roughness and thickness errors, which are not modelled. The materials, band and limits are a local choice in the spirit of the single-band antireflection studies of Dobrowolski et al. (1996); no published value exists for this exact specification.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`

## Sources

- paper: https://doi.org/10.1364/AO.35.000644 (Dobrowolski, Tikhonravov, Trubetskov, Sullivan and Verly (1996), Optimal single-band normal-incidence antireflection coatings: how the lowest attainable average reflectance depends on bandwidth, index contrast and total optical thickness.)
- paper: https://doi.org/10.1364/AO.35.005493 (Tikhonravov, Trubetskov and DeBell (1996), Application of the needle optimization technique to the design of optical coatings: the standard numerical method for this design problem.)

## Limitations

- Layers are lossless and non-dispersive with fixed indices, and only normal incidence is scored; real coatings need dispersion data, absorption, oblique incidence and tolerance to thickness errors.
- The specification (two materials, 400-800 nm, 20 layers, 1000 nm) is a local choice in the spirit of Dobrowolski et al. (1996); no published optimum exists for it, so scores compare designs under this model only.
- The mean is taken over 401 equally spaced wavelengths, so a design could in principle exploit the sampling between grid points; the reported maximum reflectance helps to spot that.
