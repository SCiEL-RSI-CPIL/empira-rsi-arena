# Hard rule-chain

**Paper:** Discovering Reinforcement Learning Interfaces with Large Language Models (https://arxiv.org/pdf/2605.03408)\
**Task id:** `limen-hard-rule-chain`\
**Protocol:** `original`\
**Direction:** maximize\

## Task

Write JAX get_observation(state) and compute_reward(state, action, next_state) for a 13 x 13 four-room XLand-MiniGrid task (blue pyramid becomes a green ball; place it next to the yellow hex); LIMEN's evaluator trains PPO on them and scores the mean success rate within 400 steps over three 5M-step seeds, or a 3M-step cascade run's rate if below 1 % (higher is better).

Benchmark named in the paper: XLand-MiniGrid. Paper metric: mean success rate across three seeds.

## Candidate interface

get_observation(state) -> 1-D float array (length 1..512) and compute_reward(state, action, next_state) -> scalar float, both JAX-traceable (jit and vmap) functions of an XLand-MiniGrid 0.9.3 state: state.grid (H, W, 2) uint8 [tile id, colour id] indexed [row, col] (tiles: 0 empty, 1 floor, 2 wall, 3 ball, 4 square, 5 pyramid, 6 goal, 7 key, 8 locked door, 9 closed door, 10 open door, 11 hex, 12 star; colours: 1 red, 2 green, 3 blue, 4 purple, 5 yellow, 6 grey, 7 black (floor), 8 orange, 9 white, 10 brown, 11 pink), state.agent.position (2,) uint8 [row, col], state.agent.direction () uint8 (0 up, 1 right, 2 down, 3 left), state.agent.pocket (2,) uint8 [tile, colour] ([0, 0] = empty), state.step_num () and state.goal_encoding / state.rule_encoding; action is an int32 in 0..5 (forward, turn right, turn left, pick up, put down, toggle). compute_reward sees the true terminal state of an episode. As in LIMEN, the module may only import jax, jax.numpy (also jax.lax, jax.nn and math); in addition, attribute assignment, private or I/O attributes (load, save, config, debug, callbacks, experimental APIs), frame, code and traceback attributes (gi_frame, f_builtins, f_globals, co_code, tb_frame, ...), attributes named like system modules (os, sys, builtins, importlib, inspect, gc, ...), dunder names other than __name__, class patterns that read such attributes, and exec-like builtins are rejected, and an imported module may only be used as the base of an attribute access (jnp.where), not renamed or passed as a value. Task: in a 13 x 13 four-room grid (XLand-MiniGrid-R4-13x13) pick up the blue pyramid, which turns into a green ball in the pocket, then put the green ball down next to the yellow hex; the episode ends when the agent uses put-down while facing the green ball or the yellow hex and that tile is next to the other (a purple ball and an orange square are distractors with a dead-end rule); training episodes last at most 507 steps. Score (maximize): LIMEN's own CascadeEvaluator from Lossfunk/LIMEN@cf0b44da, unchanged, with the task's configuration file: crash filter (syntax, required functions, import, dry run and jit on a real state, observation length <= 512), then PPO from scratch (8,192 parallel environments, 16-step rollouts, one epoch of 16 minibatches, Adam lr 0.001 linearly decayed, gamma 0.99, GAE 0.95, clip 0.2, entropy 0.01, GRU actor-critic with 256-unit observation encoder and 512-unit GRU; LIMEN runs total steps // 131,072 PPO updates) trained on the candidate's observations and rewards. A 3,000,000-step run (22 updates) with seed 42 must reach 1 % success, otherwise its success rate is the score; candidates that pass are trained with seeds 42, 43 and 44 for 5,000,000 steps (38 updates, 4,980,736 steps) each and the score is the mean final success rate. Success means the episode ends (goal reached) before the 400-step evaluation horizon; the per-seed rate is measured on 50 evaluation episodes. CPU only, at most four threads; deterministic for a fixed candidate on the same platform and software stack

## Labels

Scientific contribution: `engineering` (tier 3); artifact `reward`, domain `machine_learning`, data `simulator`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented verifier for the cited task; the contract and task-specific evidence define which mathematical and experimental claims have been checked.

- Evaluator: LIMEN's own CascadeEvaluator (limen/evaluator.py at Lossfunk/LIMEN@cf0b44da, MIT) runs unchanged in a child process with LIMEN's configs/hard_rule_chain.yaml: crash filter (syntax, required functions, import, dry run and jit on a real state, observation length <= 512), a cascade run of 3,000,000 steps with seed 42 that must reach 1 % success (otherwise that run's success rate is the score, as LIMEN's archive records it), then PPO with seeds 42, 43 and 44 for 5,000,000 steps each (38 updates of 8,192 environments x 16 steps = 4,980,736 steps; LIMEN divides without rounding up) and the mean success rate over 50 evaluation episodes per seed with a 400-step horizon. LIMEN's LLM-driven search (prompts, MAP-Elites archive) is not part of the task; one interface is scored per evaluation. The 46 LIMEN files are fetched by scripts/prepare_jax_station.py --limen and checked one by one against catalog/jax_station-data-lock.json. This is LIMEN's in-search fitness (Section 5.3; the 'Cascade' and 'Full' budgets of the paper's Table 3); the paper's main results instead retrain the best interface of each method from scratch with 10 seeds at 6M steps (Section 5.5; Table 3, '10-seed Eval') and report 85 % for joint search on this task, so those values are not comparable with this score.
- Additions around the unchanged LIMEN code: the candidate source must pass a static policy before LIMEN sees it (imports limited to jax, jax.numpy, jax.lax, jax.nn and math, where LIMEN's own instructions allow jax and jax.numpy only; no attribute assignment; no private, file I/O, callback, configuration or experimental attributes; no exec-like builtins), and the child process has an audit hook (no subprocesses, sockets or ctypes; writes only in its private temporary directory, where LIMEN writes the candidate file). LIMEN's adapter.run_training is wrapped only to record per-run metrics, and the configuration's relative ruleset path is made absolute. The evaluator reads the candidate's source only (sandboxed.run); no statement of the candidate runs in the evaluator process. The interface is traced inside the child that trains the agent and measures its success, as in LIMEN itself; the static policy and audit hook keep honest interfaces from side effects but are not a proven security boundary.
- Hardware and software: CPU instead of the paper's single NVIDIA L4 GPU; jax/jaxlib 0.11.2, flax 0.12.10, optax 0.2.8, distrax 0.1.9 (tfp-nightly 0.26.0.dev20261010), xminigrid 0.9.3 (= dunnolab/xland-minigrid@0d306d14), numpy 2.5.3, Python 3.12; four XLA threads (NPROC/PJRT_NPROC=4) and PYTHONHASHSEED=0 in the child. LIMEN's code runs on these versions without changes.
- Seed: LIMEN's default observation (the adapter's flattened grid, position, direction, pocket and step counter) with a sparse reward of 1 on the step that completes the goal, i.e. the paper's Sparse baseline ('raw simulator observations with binary success reward'; the paper's Section 5.1 describes the default observation as a flattened 7 x 7 egocentric grid, XLand-MiniGrid's own view, whereas LIMEN's code uses this full-grid observation in its default and reward-only modes, and its default mode keeps XLand-MiniGrid's built-in reward, 1 - 0.9 x step / max_steps on completion); under random actions in 4,096 environments the reward fired exactly on XLand-MiniGrid's goal events (no misses, no false positives).
- Measured: the seed scores 0.000 (cascade run 0.00, below the 1 % threshold; 103 s, 311 CPU-seconds), as the paper's Sparse baseline fails here; LIMEN's published evolved interface (labelled 'Success rate: 76%') scores 0.707 (0.74, 0.74, 0.64), 5 points below the published value. Its full evaluation (cascade run of 3M steps plus 3 x 5M steps) took 645, 589, 640 s wall in three runs at load averages 6-20 (1831-1936 CPU-seconds, about 3 cores busy; peak memory 2.7 GB), with identical results. It fits the 900 s limit (child limit 840 s) with four cores available; on a busier machine a full evaluation of a passing interface can exceed it (one 5M-step seed took 262 s at load average 18 against 170-183 s at load 6-10) and then fails with a timeout. Interfaces rejected by the cascade take about 2 minutes.
- The seed scores 0, so search starts from a flat score until an interface passes the cascade; the paper reports mean success of 0.8 % for independently sampled LLM interfaces on this task and 69-76 % for its two successful search seeds.
- Determinism: repeated evaluations of the same candidate in the evaluator's child gave identical per-run success rates, returns and episode lengths (LIMEN's published interface, three times). Results depend on the numerical libraries and their thread settings: with VECLIB_MAXIMUM_THREADS (the thread count of Apple's Accelerate library) unset instead of 4, the same training of LIMEN's published easy interface gave a different final return on its 500,000-step run (6.149 instead of 6.143) and per-seed success rates differing by 0.02 (0.98 instead of 1.00 for seed 42). The evaluator fixes these variables, so scores are reproducible on the same platform and software stack but not across CPU types, BLAS/LAPACK builds or GPUs. valid is null: success rates are sampled estimates after stochastic training; near a success rate of 0.2 the standard error of the 150-episode mean is about 0.035.
- Pinned LIMEN evaluator (Lossfunk/LIMEN@cf0b44da, MIT) run unchanged in a child process on CPU JAX (jax 0.11.2, flax 0.12.10, distrax 0.1.9, xminigrid 0.9.3 = dunnolab/xland-minigrid@0d306d14), with LIMEN's configuration, seeds and cascade; the paper used one NVIDIA L4 GPU.

## Implementation credit

Implementation source: https://github.com/Lossfunk/LIMEN. Source path: `limen`.

See `problem.json` for immutable source receipts and component scope. Implementation provenance does not establish that this is the paper authors' experiment revision.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:jax`
- `py:flax`
- `py:distrax`
- `py:xminigrid`
- `py:openai`
- `py:dacite`
- `py:yaml`
- `py:psutil`
- `file:data/jax_station/limen-cf0b44da/limen/evaluator.py`

## Sources

- paper: https://arxiv.org/pdf/2605.03408 (Section 5.1)
- repository: https://github.com/Lossfunk/LIMEN/tree/cf0b44da37f7e7ee038920c860a8cb00c610c9af (configs/hard_rule_chain.yaml)
- repository: https://github.com/Lossfunk/LIMEN/tree/cf0b44da37f7e7ee038920c860a8cb00c610c9af (limen/evaluator.py (CascadeEvaluator), limen/train.py (PPO), limen/adapters/xminigrid.py, configs/hard_rule_chain.yaml, rulesets/hard_rule_chain.py and examples/evolved_interfaces/hard_rule_chain.py (MIT); run unchanged by the evaluator)
- repository: https://github.com/dunnolab/xland-minigrid/tree/0d306d1414dcb6e451358d94e7d9a91879462ee1 (XLand-MiniGrid 0.9.3 (Apache-2.0): environments, goal checks and rules; byte-identical to the PyPI wheel xminigrid==0.9.3)

## Limitations

- Execution interface inspected in pinned source. Full paper experiments have not been reproduced here.
