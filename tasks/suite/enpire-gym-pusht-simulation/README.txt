# Gym-PushT simulation

**Paper:** ENPIRE: Agentic Robot Policy Self-Improvement in the Real World (https://arxiv.org/pdf/2606.19980)\
**Task id:** `enpire-gym-pusht-simulation`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Write a Python class Policy whose act(observation) returns the pushing disc's target position in gym-pusht's PushT-v0 from the state vector (agent position, T-block position and angle); scored by the fraction of 100 fixed episodes (reset seeds 0-99, at most 300 steps each) in which the block covers more than 95% of the goal region (higher is better).

## Candidate interface

class Policy with __init__(self) and act(self, observation) -> (x, y): a heuristic controller for gym-pusht's PushT-v0 with obs_type='state'. observation is a float array [agent_x, agent_y, block_x, block_y, block_angle]: pixels in [0, 512] (block_x, block_y is the T's body origin, the midpoint of the outer edge of its bar) and the block angle in radians in [0, 2 pi). In its body frame (y up) the T is the union of the bar [-60, 60] x [0, 30] and the stem [-15, 15] x [30, 120]; the goal pose is fixed at (256, 256, pi / 4). The action is the target position (pixels) of the 15-pixel pushing disc, which gym-pusht's PD controller tracks for 0.1 s (10 physics steps) per step; actions are clipped to [0, 512] and must be finite. Each of 100 fixed episodes (gym-pusht reset seeds 0-99) is played by a fresh Policy instance in a fresh sandboxed process (NumPy available; no file, network or process access; 10 s CPU per episode) that receives only the observations; an episode ends on gym-pusht's own success test (coverage of the goal region above 0.95) or after 300 steps. Score: fraction of successful episodes (maximize)

Parameters: `{"episodes": 100, "first_seed": 0, "horizon": 300, "episode_cpu_seconds": 10}`

## Labels

Scientific contribution: `engineering` (tier 3); artifact `policy`, domain `robotics`, data `simulator`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Simulator: gym-pusht 0.1.6 from PyPI (Apache-2.0), unchanged, environment gym_pusht/PushT-v0 with obs_type='state' and its registered 300-step limit. It runs in a separate Python 3.12 environment, <task cache>/sim_control-env (pymunk 6.11.1, gymnasium 1.3.0, NumPy 2.5.3, shapely 2.1.2; created by scripts/prepare_sim_control.py --setup-environment), because gym-pusht calls pymunk's Space.add_collision_handler, which pymunk 7 removed; the shared suite environment was not modified. Keypoint and pixel observations are not offered. The paper does not say which observation its agents' controllers received (Section 3.1 describes the heuristics as synthesised perception and control tool calls), so the exact state vector, which needs no perception, is a local choice.
- Source protocol: in ENPIRE's simulated Push-T experiment (Section 3.1, Figure 5) coding agents learn heuristics, i.e. write non-learned controllers, through immutable Gym interfaces, and Claude Code and Codex reach 95% success in about two hours. The text inspected does not state how many episodes the simulation evaluation used (the autoresearch prompt of the real-robot stations, Figure 10, asks for 100% success over every window of 50 consecutive episodes), and the pinned ENPIRE release (NVlabs/ENPIRE@99ee90ac) has no gym-pusht entry point; the 100 evaluation episodes, the reset seeds 0-99 and the per-episode isolation are therefore local choices, and the agents' wall-clock research time is not measured. The earlier catalog reason (specialised hardware) did not apply to this simulation task.
- Score: fraction of the 100 episodes that reach gym-pusht's own success test (info['is_success']: the block covers more than 95% of the goal T); an episode ends at the first success or after 300 steps. Details report each episode's steps and maximum coverage. The paper measures task success as completing the task in one rollout given up to eight retries made after witnessing the previous failure (Section 3) and does not say whether its simulated Push-T used retries; here each seed is one episode without retries.
- Isolation: the simulator runs in a separate trusted server process; each episode is played by a fresh sandboxed child (software_sandbox: reads limited to the Python installation, no writes, network, subprocesses or ctypes; 10 s CPU and 1 GB per episode) that creates one Policy and receives only the 5-number observation of each step. Its target positions are validated (two finite numbers), clipped to [0, 512] and forwarded to the server, so the candidate cannot reset, inspect or modify the environment and never sees the coverage or the success flag; module state cannot carry over between episodes. The evaluator reads the candidate's source only (sandboxed.run); no statement of the candidate runs in the evaluator process.
- Seed: a greedy contact-pushing controller written for this task (one-step lookahead with a frictionless point-push model, a detour around the block, pushes fixed in world coordinates). It was tuned on reset seeds 1000-1099 and 2000-2099 (67% and 64% success) and scores 0.60 (60 of 100 episodes) on the evaluation seeds 0-99.
- gym-pusht 0.1.6 (Apache-2.0) runs unchanged in a separate Python environment (<task cache>/sim_control-env, pymunk 6.11.1 because gym-pusht calls the collision-handler API removed in pymunk 7); success is gym-pusht's own test. Episodes, seeds and the per-episode process isolation are local choices.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:../../.cache/sim_control-env/bin/python`

## Sources

- paper: https://arxiv.org/pdf/2606.19980 (Experimental tasks and supplementary experiments)
- repository: https://github.com/NVlabs/ENPIRE/tree/99ee90acf65b5b18957c8382ad580db999528be3
- benchmark: https://pypi.org/project/gym-pusht/0.1.6/ (gym_pusht/envs/pusht.py: PushT-v0 state observation, PD-controlled agent, coverage-based success (success_threshold 0.95); gym_pusht/__init__.py: max_episode_steps 300)
- repository: https://github.com/huggingface/gym-pusht (Apache-2.0 source of the Push-T simulator used unchanged)

## Limitations

- The inspected paper links no readable official repository; the official release was identified separately and is pinned (see below).
- Official NVIDIA project page inspected; no repository download link.
