# ALFWorld

**Paper:** AgentSquare: Automatic LLM Agent Search in Modular Design Space (https://arxiv.org/pdf/2410.06153)\
**Task id:** `agentsquare-alfworld`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Write a Python act(instruction, observation, valid_actions, history, llm) agent that issues one text command per step in ALFWorld household games, using a budgeted OpenRouter model client; scored by the success rate over 12 fixed games (2 per task type from the 134 out-of-distribution games), at most 49 actions each (higher is better).

Benchmark named in the paper: ALFWorld. Paper metric: success rate.

## Candidate interface

act(instruction: str, observation: str, valid_actions: list[str], history: list[tuple[str, str, float]], llm) -> str: the next action. `instruction` is the episode's fixed task text, `observation` the environment's latest text, `valid_actions` the actions the environment lists for this state (empty when it lists none), `history` the (action, observation, reward) triples of the current episode and `llm(prompt_or_messages, system=None, max_tokens=None, temperature=None, json_mode=False) -> str` a budgeted, cached OpenRouter chat client. act must return a str (anything else invalidates the evaluation); a crash or timeout ends the episode with the environment's score at that point; exhausting the model budget invalidates the evaluation. The agent runs in a sandboxed child process (one per episode; module state persists within an episode only) that sees nothing but these arguments. Environment: ALFWorld text games (AlfredTWEnv's eval_out_of_distribution set, 134 games); a fixed sample of 2 games per task type (12 episodes), at most 49 actions each, no admissible-command list. Episode score: 1 when the game reports won, else 0. Score: success rate over the 12 episodes; higher is better

Parameters: `{"adapter": "alfworld", "runtime": "pinned", "per_type": 2, "seed": 0, "max_steps": 49, "budget_usd": 0.5}`

## Labels

Scientific contribution: `benchmark` (tier 3); artifact `agent`, domain `language`, data `public_benchmark`, verification `sampled`, reference `none`, resources `model_api`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Environment: ALFWorld 0.4.2 with textworld[pddl] 1.7.0 (MIT) in the pinned Python 3.11 runtime agent_bench-env of the task cache (ARENA_TASK_CACHE_DIR, default ~/.cache/empira/tasks); game files from ALFWorld's releases (json_2.1.1_json.zip of 0.2.2 and json_2.1.3_tw-pddl.zip of 0.4.2) selected by AlfredTWEnv.collect_game_files's rules, which yields exactly the 134 out-of-distribution games of AgentSquare's runner and of AgentBoard's ALFWorld test list. Each game is played with ALFWorld's own wrappers (AlfredDemangler without randomisation, AlfredInfos).
- Protocol from AgentSquare's tasks/alfworld/alfworld_run.py: the welcome banner is removed from the first observation, '>' and 'OK' are stripped from replies, a reply is cut at ', end' and its periods removed, 'think:' replies leave the game unchanged and are answered 'OK.', the admissible commands are not shown, and the loop runs range(1, 50), i.e. at most 49 actions. AgentSquare plays all 134 games; here a fixed sample of 2 games per task type (12) to bound the cost.
- Instruction: AgentSquare conveys the command syntax through ReAct's few-shot examples, which belong to the agent; here the episode instruction lists ALFWorld's command templates instead (no examples).
- Candidate: AgentSquare searches over agent module code (planning, reasoning, tool use, memory with a uniform interface) with module evolution, recombination and a performance predictor; those belong to the search method. Here the candidate is the agent program itself, act(...) called once per step; it may implement any modules internally. The AgentSquare repository has no licence, so its code is only read to recover the protocol; the environments come from the benchmarks' own releases.
- Model: AgentSquare runs GPT-3.5-turbo-0125 and GPT-4o; the default client model here is google/gemini-2.5-flash at temperature 0 (ARTASK_LLM_MODEL overrides it), so absolute scores are not comparable with the paper. Not verified: the paper's numbers were not reproduced.
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
- `file:data/agent_bench/alfworld/games.json`

## Sources

- paper: https://arxiv.org/pdf/2410.06153 (Appendix A.1)
- repository: https://github.com/tsinghua-fib-lab/AgentSquare/tree/8f5b3fe5d8a32f9b59d20370823bef2a2c86928c (tasks/alfworld/alfworld_run.py)
- paper: https://arxiv.org/abs/2410.06153 (Section 4.1 and Appendix A.1 (tasks and metrics))
- repository: https://github.com/tsinghua-fib-lab/AgentSquare/tree/8f5b3fe5d8a32f9b59d20370823bef2a2c86928c/tasks/alfworld/alfworld_run.py (eval_out_of_distribution, 134 games, step loop)
- benchmark: https://github.com/alfworld/alfworld/releases/tag/0.4.2 (json_2.1.3_tw-pddl.zip game files)
- benchmark: https://github.com/alfworld/alfworld/releases/tag/0.2.2 (json_2.1.1_json.zip trajectories (game selection))

## Limitations

- The 12 episodes are a fixed list of public benchmark items that every evaluation replays, and act may return actions without calling the model. A program that replays memorised answers or actions for these episodes is scored like an agent, so the score says nothing about other episodes of the benchmark.
