# Crafter

**Paper:** Hierarchical Self-Improvement: A Framework for Task-Specific Evolvable Agent Harnesses (https://arxiv.org/pdf/2608.08466)\
**Task id:** `hsi-crafter`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Write a Python act(instruction, observation, valid_actions, history, llm) agent that picks each next text action in BALROG's Crafter, using a budgeted OpenRouter model client; scored by the mean fraction of Crafter's 22 achievements unlocked in 2 seeded episodes of at most 100 steps (higher is better).

Benchmark named in the paper: BALROG. Paper metric: episode progress / task success.

## Candidate interface

act(instruction: str, observation: str, valid_actions: list[str], history: list[tuple[str, str, float]], llm) -> str: the next action. `instruction` is the episode's fixed task text, `observation` the environment's latest text, `valid_actions` the actions the environment lists for this state (empty when it lists none), `history` the (action, observation, reward) triples of the current episode and `llm(prompt_or_messages, system=None, max_tokens=None, temperature=None, json_mode=False) -> str` a budgeted, cached OpenRouter chat client. act must return a str (anything else invalidates the evaluation); a crash or timeout ends the episode with the environment's score at that point; exhausting the model budget invalidates the evaluation. The agent runs in a sandboxed child process (one per episode; module state persists within an episode only) that sees nothing but these arguments. Environment: BALROG Crafter (precise locations as in HSI's config), 2 seeded episodes capped at 100 steps (early-game progression). Episode score: unlocked achievements / 22 (BALROG progression). Score: mean over the episodes; higher is better

Parameters: `{"adapter": "balrog", "runtime": "pinned", "suite": "crafter", "split": "all", "episodes_per_task": 2, "base_seed": 42, "max_steps": 100, "budget_usd": 0.5}`

## Labels

Scientific contribution: `benchmark` (tier 3); artifact `agent`, domain `games`, data `public_benchmark`, verification `sampled`, reference `none`, resources `model_api`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Step cap: HSI and BALROG play Crafter episodes of up to 2000 steps (HSI: 5 episodes x 3 repeats); here 2 episodes of at most 100 steps, so the task measures early-game progression only and scores are not comparable with 2000-step results. Every evaluation replays the same seeded episodes, so a search optimises on the scored episodes themselves; in HSI's in-distribution protocol every evaluation episode is generated from a newly sampled environment seed (its Section 4.1), so its test episodes are not the ones the harness was evolved on.
- Reproducibility: Crafter keeps the creatures of each chunk in Python sets ordered by memory address and picks the creature to despawn by index, so the same seed and actions gave different episodes in separate processes (checked); the evaluator replaces these sets with insertion-ordered sets, which makes a seeded episode reproducible without changing spawn or despawn probabilities.
- Environment: BALROG's own environment wrappers, instruction prompts, invalid-action handling (default action plus BALROG's feedback text) and progression statistics (balrog-ai/BALROG at commit b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5, MIT), run in the pinned Python 3.11 runtime agent_bench-env of the task cache (ARENA_TASK_CACHE_DIR, default ~/.cache/empira/tasks) created by `python scripts/prepare_agent_bench.py --setup-environment` (gym 0.23.0 as BALROG pins it, BALROG's Minigrid fork, baba-is-ai, crafter 1.8.3, balrog-nle 0.9.0 and BALROG's minihack fork at pinned commits).
- Protocol taken from HSI's BALROG evaluator (TailinZhou/hsi at commit 97a022b9a9c260c0498806fc9826ebf22e753ed9, src/benchmark/balrog/): the agent's reply is reduced to letters, spaces and colons before BALROG's validity check, the episode stops early when the last 12 valid actions form a strict period-2 or period-3 loop, and episode seeds are HSI's deterministic test seeds int(sha256(f'{base_seed}_{suite}/{task}_{episode}_0')[:8], 16). HSI's code was read to recover this protocol; the HSI repository has no LICENSE file, so none of its code is copied.
- Candidate: HSI evolves a per-step harness (prompts, hooks, memory) behind BALROG's agent call; here the candidate is the per-step policy act(...) with the same inputs (instruction prompt, current observation, BALROG's action list, the episode history) and a model client. HSI's harness also receives a task context (last reward, Crafter achievements); here that information is only available through the observations and history. HSI's lower-confidence-bound selection, repeats and evolution loop belong to the search method and are not part of the evaluator. HSI also requires every task interaction to go through the frozen backbone (evolution 'cannot replace the model with external search procedures or non-LLM policies', its Section 4.1); here act() may choose actions without calling the model.
- Model: the default client model is google/gemini-2.5-flash at temperature 0 (ARTASK_LLM_MODEL overrides it); the paper's backbone is DeepSeek-V4-Flash (deepseek-v4-flash-preview), run at task time with thinking disabled and temperature 0 (its Section 4.1 and Appendix B.1), whereas the local client sends no reasoning setting, so the provider's default applies; absolute scores are not comparable with the paper. Not verified: the paper's numbers were not reproduced.
- Isolation: the candidate runs in a resource-limited child process (software_sandbox) that cannot read files outside the Python installation; model calls and steps are forwarded to the evaluator, and the environment (including any gold plan or answer) runs in a separate server process, so the candidate cannot see labels or solutions.
- Model-driven agent evaluation: the score is an estimate on a fixed, seeded list of episodes and depends on the model (ARTASK_LLM_MODEL, default google/gemini-2.5-flash at temperature 0) and on the per-evaluation budget (task parameter budget_usd, overridden by ARTASK_LLM_BUDGET_USD); exhausting the budget makes the evaluation invalid. ARTASK_AGENT_EPISODES truncates the episode list for quick checks. The candidate runs in a resource-limited child process (one per episode) that receives only the instruction, observations, listed valid actions, its own history and a model client; the environment runs in a separate server process. Responses are cached under llm-cache of the task cache. `valid` is null because a sampled model score cannot certify a candidate.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `env:OPENROUTER_API_KEY`
- `network:openrouter`
- `file:../../.cache/agent_bench-env/bin/python`

## Sources

- paper: https://arxiv.org/pdf/2608.08466 (Section 4)
- repository: https://github.com/TailinZhou/hsi/tree/97a022b9a9c260c0498806fc9826ebf22e753ed9 (benchmark_config_goal/balrog_crafter/config.yaml)
- paper: https://arxiv.org/abs/2608.08466 (Sections 3.3.2 and 4.1; Table 1)
- repository: https://github.com/TailinZhou/hsi/tree/97a022b9a9c260c0498806fc9826ebf22e753ed9/benchmark_config_goal/balrog_crafter/config.yaml (episode counts, seeds, splits)
- repository: https://github.com/TailinZhou/hsi/tree/97a022b9a9c260c0498806fc9826ebf22e753ed9/src/benchmark/balrog/evaluator.py (episode loop: action cleaning, loop guard, seeds)
- repository: https://github.com/balrog-ai/BALROG/tree/b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5/balrog/environments (environment wrappers and instruction prompts (used unchanged))
- repository: https://pypi.org/project/crafter/1.8.3/ (Crafter environment (MIT))

## Limitations

- No root license was found in the inspected upstream tree; preserve upstream licensing conditions and do not assume redistribution rights.
- Execution interface inspected in pinned source. Full paper experiments have not been reproduced here.
