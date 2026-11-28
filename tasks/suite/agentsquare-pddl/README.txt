# PDDL

**Paper:** AgentSquare: Automatic LLM Agent Search in Modular Design Space (https://arxiv.org/pdf/2410.06153)\
**Task id:** `agentsquare-pddl`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Write a Python act(instruction, observation, valid_actions, history, llm) agent that plays AgentBoard's PDDL games (gripper, blockworld, barman, tyreworld) with one text action per step, using a budgeted OpenRouter model client; scored by the mean progress rate (fraction of goal literals reached) over 12 fixed problems, at most 30 steps each (higher is better).

Benchmark named in the paper: PDDL. Paper metric: progress rate.

## Candidate interface

act(instruction: str, observation: str, valid_actions: list[str], history: list[tuple[str, str, float]], llm) -> str: the next action. `instruction` is the episode's fixed task text, `observation` the environment's latest text, `valid_actions` the actions the environment lists for this state (empty when it lists none), `history` the (action, observation, reward) triples of the current episode and `llm(prompt_or_messages, system=None, max_tokens=None, temperature=None, json_mode=False) -> str` a budgeted, cached OpenRouter chat client. act must return a str (anything else invalidates the evaluation); a crash or timeout ends the episode with the environment's score at that point; exhausting the model budget invalidates the evaluation. The agent runs in a sandboxed child process (one per episode; module state persists within an episode only) that sees nothing but these arguments. Environment: AgentBoard's PDDL games (gripper, blockworld, barman, tyreworld; 60 problems), a fixed sample of 3 problems per game (12 episodes), at most 30 steps each; 'check valid actions' lists the grounded actions. Episode score: AgentBoard's progress rate (largest fraction of goal literals satisfied so far). Score: mean progress rate over the 12 episodes; higher is better

Parameters: `{"adapter": "pddl", "runtime": "pinned", "per_game": 3, "seed": 0, "max_steps": 30, "budget_usd": 0.5}`

## Labels

Scientific contribution: `benchmark` (tier 3); artifact `agent`, domain `language`, data `public_benchmark`, verification `sampled`, reference `none`, resources `model_api`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Protocol: AgentSquare lists PDDL among its tasks but its repository contains only AgentBoard's PDDL environment code and no runner, so AgentBoard's own PDDL protocol (tasks/pddl.py: 60 problems, progress rate, the 30-step limit of AgentSquare's AgentBoard run configuration) is used.
- Environment: AgentBoard's agentboard/environment/pddl_env/pddl_env.py with its vendored pddlgym (Apache-2.0 / MIT), fetched at the pinned commit and imported unchanged except for three shims: AgentBoard's registry and base class are replaced by trivial stand-ins, nltk.download is disabled (the punkt_tab tables are prepared offline; their licence is not stated) and pddlgym.make drops the disable_env_checker keyword that gym 0.23 does not accept.
- Sample: AgentBoard evaluates all 60 problems; here 3 problems per game (12) in a fixed SHA-256 order (seed 0). The instruction is the game's domain description from prompts/VanillaAgent/pddl_vanilla_prompt.json plus the goal; its few-shot examples belong to the agent and are not included.
- Candidate: AgentSquare searches over agent module code (planning, reasoning, tool use, memory with a uniform interface) with module evolution, recombination and a performance predictor; those belong to the search method. Here the candidate is the agent program itself, act(...) called once per step; it may implement any modules internally. The AgentSquare repository has no licence, so its code is only read to recover the protocol; the environments come from the benchmarks' own releases.
- Model: AgentSquare runs GPT-3.5-turbo-0125 and GPT-4o; the default client model here is google/gemini-2.5-flash at temperature 0 (ARTASK_LLM_MODEL overrides it), so absolute scores are not comparable with the paper. Not verified: the paper's numbers were not reproduced.
- Isolation: the candidate runs in a resource-limited child process (software_sandbox) that cannot read files outside the Python installation; model calls and steps are forwarded to the evaluator, and the environment (including any gold plan or answer) runs in a separate server process, so the candidate cannot see labels or solutions.
- Model-driven agent evaluation: the score is an estimate on a fixed, seeded list of episodes and depends on the model (ARTASK_LLM_MODEL, default google/gemini-2.5-flash at temperature 0) and on the per-evaluation budget (task parameter budget_usd, overridden by ARTASK_LLM_BUDGET_USD); exhausting the budget makes the evaluation invalid. ARTASK_AGENT_EPISODES truncates the episode list for quick checks. The candidate runs in a resource-limited child process (one per episode) that receives only the instruction, observations, listed valid actions, its own history and a model client; the environment runs in a separate server process. Responses are cached under llm-cache of the task cache. `valid` is null because a sampled model score cannot certify a candidate.
- Reward: the reward of each history triple is AgentBoard's running progress rate (largest fraction of goal literals satisfied so far), which AgentBoard logs but does not show to its agents; the goal literals themselves are part of the instruction.

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
- `file:data/agent_bench/agentboard/pddl_env/pddl_env.py`
- `file:data/agent_bench/agentboard/pddl-test.jsonl`

## Sources

- paper: https://arxiv.org/pdf/2410.06153 (Appendix A.1)
- repository: https://github.com/tsinghua-fib-lab/AgentSquare/tree/8f5b3fe5d8a32f9b59d20370823bef2a2c86928c
- paper: https://arxiv.org/abs/2410.06153 (Section 4.1 and Appendix A.1 (tasks and metrics))
- repository: https://github.com/tsinghua-fib-lab/AgentSquare/tree/8f5b3fe5d8a32f9b59d20370823bef2a2c86928c/tasks/sciworld/agentboard/environment/pddl_env (PDDL environment code only (no runner))
- repository: https://github.com/hkust-nlp/AgentBoard/tree/bb7255e2daf1989069a186dad9e53f70680961db/agentboard/tasks/pddl.py (episode protocol and progress rate)
- repository: https://github.com/hkust-nlp/AgentBoard/tree/bb7255e2daf1989069a186dad9e53f70680961db/agentboard/environment/pddl_env (PDDL environment and problems (used unchanged))
- benchmark: https://huggingface.co/datasets/hkust-nlp/agentboard/tree/0680548ef773505788d3cced65640a0fd511242f (data/pddl/test.jsonl (difficulty labels))

## Limitations

- The 12 episodes are a fixed list of public benchmark items that every evaluation replays, and act may return actions without calling the model. A program that replays memorised answers or actions for these episodes is scored like an agent, so the score says nothing about other episodes of the benchmark.
