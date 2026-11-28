# BallBalance

**Paper:** Eureka: Human-Level Reward Design via Coding Large Language Models (https://arxiv.org/pdf/2310.12931)\
**Task id:** `eureka-ballbalance`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Write a Python reward(observation, action, next_observation, info) for a CPU MuJoCo analogue of IsaacGymEnvs BallBalance (ball thrown onto a tilting tray); a linear policy is trained under it by the cross-entropy method, and the score is the policy's mean episode length in steps before the ball drops, at most 500, over 50 held-out throws (higher is better).

Benchmark named in the paper: Isaac Gym. Paper metric: task-specific fitness and human-normalized score.

## Candidate interface

reward(observation, action, next_observation, info) -> float: the per-step reward used to train a policy for a CPU MuJoCo analogue of IsaacGymEnvs BallBalance (a ball thrown onto a 0.5 m tray that tilts on a two-axis gimbal; one step is 0.01 s). observation and next_observation are float arrays [tilt_x, tilt_y, tilt_rate_x, tilt_rate_y, ball_x, ball_y, ball_z, ball_vx, ball_vy, ball_vz] (rad, rad/s, m, m/s; clipped to +-5) before and after the step; action holds the policy's 2 tilt-rate commands in [-1, 1]; info is a dict of the state after the step: ball_position (3), ball_velocity (3), tray_tilt (2), tray_tilt_velocity (2), tray_normal (3), tray_position (3, the fixed tray centre at height 0.559 m), ball_radius (0.1), progress (steps so far), max_episode_length (500) and terminated (True on the step the ball falls). The evaluator trains a linear policy tanh(W obs + b) with the cross-entropy method (80 generations of 24 candidates, each scored by its summed candidate reward on the same 4 throws, 4 elites) and reports the fitness of the final mean policy: the mean episode length in steps (until the ball's centre drops below 0.15 m, at most 500) over 50 held-out throws (maximize). The reward runs in a sandboxed process that only receives the recorded training transitions; it must be deterministic and return a finite number

Parameters: `{"task": "ballbalance", "iterations": 80, "population": 24, "elite": 4, "episodes": 4, "init_std": 0.1, "min_std": 0.02, "train_seed": 0, "eval_seed": 2026, "eval_episodes": 50, "cpu_seconds": 600, "wall_seconds": 840}`

## Labels

Scientific contribution: `engineering` (tier 3); artifact `reward`, domain `robotics`, data `simulator`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Clearly labelled proxy (protocol recreated): a CPU MuJoCo analogue of IsaacGymEnvs BallBalance as pinned in the Eureka repository (eureka-research/Eureka@9eee4280, isaacgymenvs/tasks/ball_balance.py and cfg/task/BallBalance.yaml; NVIDIA BSD-3-Clause headers; no code copied). Kept from the source: tray (radius 0.5 m, 0.02 m thick, density 100) at Isaac's height 0.559 m, ball (radius 0.1 m, density 200), the throw distribution of reset_idx, the control step (0.01 s), action integration (targets += 0.01 * 20 * action, actions clipped to [-1, 1]), observation clipping at 5, termination (ball centre below 0.15 m or 500 steps) and the task fitness of compute_bbot_reward (average episode length).
- Simplified mechanism: Isaac's tray stands on three two-link legs whose feet are held by PhysX rigid-body attractors; a direct MuJoCo translation with equality constraints was unstable in the triage pilot, so the tray tilts on a fixed-base two-axis gimbal driven by position actuators (kp 400, kv 20, tilt limit 30 degrees). The policy therefore has 2 actions (tilt rates) instead of 3 leg rates and a 10-dimensional observation (tilts, tilt rates, ball position and velocity) instead of 24 (leg joints and three force/torque sensors). A step on which MuJoCo reports a numerical instability ends the episode as a failure.
- Training: Eureka trains PPO (rl_games) for 3,000 iterations with 4,096 GPU environments per reward sample and ranks samples by the maximum consecutive_successes logged during training. Here a linear policy tanh(W obs + b) is trained by the cross-entropy method on one CPU: 80 generations of 24 parameter vectors drawn from a diagonal Gaussian (initial standard deviation 0.1, floor 0.02), each scored by its summed candidate reward on the same 4 throws of that generation, with the 4 best kept; the final mean policy is evaluated on 50 held-out throws and the mean episode length is the score (24 s for the seed and 55 s for the human reward on the development machine). Training seed 0 only: differences between candidates of less than about 90 steps are within training-seed noise. The paper scores each final reward by the average over 5 PPO runs of the maximum task fitness reached at 10 checkpoints and reports Isaac tasks as human normalized scores, (Method - Sparse) / |Human - Sparse| (its Section 4.2; intermediate rewards get 1 PPO run, Appendix D); here the raw fitness of the final policy of one CEM run is the score, so it is not on the paper's normalized scale.
- Noise gate (validation/sim_control-checks.json), released budget, training seeds 0-4, through the sandboxed scoring path: Eureka's human reward (compute_bbot_reward, ported to the analogue's state) reached 357.6 +- 87.5 steps (mean +- sample standard deviation), a zero reward 65.9 +- 32.5 and an alive bonus of 1 per step 146.0 +- 3.0; the human reward exceeds them by 4.4 and 3.4 pooled seed standard deviations. A policy that never tilts the tray keeps the ball for 227.8 steps on the same 50 throws: CEM under the zero reward or the alive bonus ends worse than doing nothing, CEM under the human reward ends better.
- Seed: the alive bonus (Eureka's sparse reward for this task, one unit per step), which scores 147.5 at training seed 0; the human reward scores 366.7 there.
- Isolation: the simulator, the CEM update and the fitness evaluation run in the evaluator process. The reward runs in a sandboxed child (software_sandbox: no file writes, reads outside the Python installation, network or subprocesses; CPU-time and memory limits) that receives, generation by generation, only the recorded raw states and actions of the training rollouts and returns one return per rollout; it never holds the simulator, the held-out throws or the fitness, so it can only rank the population. Sandboxed and in-process training give bit-identical policies (tests/test_sim_control_eureka.py). The evaluator reads the candidate's source only (sandboxed.run); no statement of the candidate runs in the evaluator process.
- Clearly labelled proxy: MuJoCo instead of Isaac Gym/PhysX, a gimbal tray instead of the three-leg robot and a CEM-trained linear policy instead of PPO with thousands of GPU environments; scores are comparable within the suite only.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:mujoco`

## Sources

- paper: https://arxiv.org/pdf/2310.12931 (Appendix B task table)
- repository: https://github.com/eureka-research/Eureka/tree/9eee42808b52d8abc14845d1547bfde886403ccc (eureka/cfg/env/ballbalance.yaml)
- repository: https://github.com/eureka-research/Eureka/blob/9eee42808b52d8abc14845d1547bfde886403ccc/isaacgymenvs/isaacgymenvs/tasks/ball_balance.py (_create_balance_bot_asset (geometry), reset_idx (throws), pre_physics_step (action integration), compute_bbot_reward (human reward, episode-length fitness))
- repository: https://github.com/eureka-research/Eureka/blob/9eee42808b52d8abc14845d1547bfde886403ccc/isaacgymenvs/isaacgymenvs/cfg/task/BallBalance.yaml (dt 0.01, maxEpisodeLength 500, actionSpeedScale 20, clipObservations 5, clipActions 1)

## Limitations

- Upstream commands and dependencies were inspected; costly training, paid APIs, and hardware experiments were not executed.
- Execution interface inspected in pinned source. Full paper experiments have not been reproduced here.
