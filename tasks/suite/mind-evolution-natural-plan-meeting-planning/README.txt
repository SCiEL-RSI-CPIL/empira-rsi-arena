# Natural Plan Meeting Planning

**Paper:** Evolving Deeper LLM Thinking (https://arxiv.org/pdf/2501.09891)\
**Task id:** `mind-evolution-natural-plan-meeting-planning`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Write a solve(question, llm) program that uses a budgeted OpenRouter LLM client to answer Natural Plan meeting-planning prompts (meet as many San Francisco friends as possible under availability and travel-time constraints); scored by success rate on a fixed 30-problem test sample, a plan succeeding when its simulated valid meetings match the golden plan's count (higher is better).

Benchmark named in the paper: Natural Plan Meeting Planning. Paper metric: success rate.

## Candidate interface

solve(question: str, llm) -> str. `question` is Natural Plan's 5-shot Meeting Planning prompt (meet as many friends as possible in San Francisco under availability, location and travel-time constraints; answer starting with 'SOLUTION:' in the shown step format). `llm(...)` is a budgeted, cached OpenRouter chat client. Plan checker (port of Natural Plan's evaluate_meeting_planning.py): the text after 'SOLUTION:' is split on periods and simulated step by step ('You start/travel/wait/meet ...'); the count of valid meetings before the first invalid step must equal that of the golden plan. Score: success rate on a fixed 30-problem sample of Mind Evolution's 500-problem test split; higher is better

Parameters: `{"benchmark": "meeting_planning", "samples": 30, "seed": 0}`

## Labels

Scientific contribution: `benchmark` (tier 3); artifact `agent`, domain `language`, data `public_benchmark`, verification `sampled`, reference `none`, resources `model_api`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Data: Natural Plan Meeting Planning (google-deepmind/natural-plan at commit ca76db336072ff8931db43bc1ca8d381038cf073, CC-BY-4.0 data). The test split follows Mind Evolution Appendix B: of the 100 problems at each number of people (1 to 10), the last 50 form the test set (500 problems). Each evaluation uses a fixed 30-problem sample.
- Reference check: the ported checker reproduces the official evaluate_meeting_planning.py result on all 1,000 shipped Gemini 1.5 Pro predictions (pred_5shot_pro; 0 disagreements, accuracy 0.384 with both) and scores every golden plan as solved. Local deviation: unknown places or people and truncated steps (KeyError, IndexError) end the simulation like the official ValueError handling instead of crashing the script.
- Mind Evolution scores plans with its own JSON-format evaluation function during search but reports Natural Plan's success criterion; this task uses the benchmark's text format and official criterion. The candidate is a free-form solve(question, llm) program instead of the paper's evolutionary search with Gemini 1.5 Flash and Pro; the default model is google/gemini-2.5-flash at temperature 0.
- Not verified: the paper's success rates were not reproduced; only the checker was verified against the official script.
- Mind Evolution receives textual feedback from programmatic evaluation functions (Appendix A.2) and structured constraints; the local candidate receives neither. Mind Evolution reports about US$0.52 and 5.35M input tokens per problem with Gemini 1.5 Flash; the local per-problem budget is far smaller.
- The candidate runs as trusted code in the evaluator's process, and the reference answers are stored in data/llm_bench of the task cache (ARENA_TASK_CACHE_DIR), so a candidate could read them; it can also replace the evaluator's scoring functions, raise the model client's budget attribute and read OPENROUTER_API_KEY from the environment, so the per-evaluation budget bounds an honest program only. Scores assume a candidate that does none of this.
- Model-driven evaluation: the score is an estimate on a fixed seeded sample and depends on the model (ARTASK_LLM_MODEL, default google/gemini-2.5-flash at temperature 0), the sample size (task parameter `samples`, overridden by ARTASK_LLM_SAMPLES) and the per-evaluation budget (ARTASK_LLM_BUDGET_USD, default 0.25). Responses are cached under llm-cache in the task cache (ARTASK_LLM_CACHE overrides) and shared by all candidates: a repeated call, also one at temperature > 0, returns the stored reply, and a cached reply costs nothing against the budget. `valid` is null because a sampled model score cannot certify a candidate.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `env:OPENROUTER_API_KEY`
- `network:openrouter`
- `file:data/llm_bench/meeting_planning_test.json`

## Sources

- paper: https://arxiv.org/pdf/2501.09891 (Section 4.3, Table 2, Appendix A.2 and Appendix B)
- benchmark: https://github.com/google-deepmind/natural-plan/tree/ca76db336072ff8931db43bc1ca8d381038cf073/data/meeting_planning.json (1,000 problems; test split = last 50 per number of people)
- repository: https://github.com/google-deepmind/natural-plan/tree/ca76db336072ff8931db43bc1ca8d381038cf073/evaluate_meeting_planning.py (official checker (ported))

## Limitations

- No official runnable repository was verified from the paper/project sources examined.
- Upstream commands and dependencies were inspected; costly training, paid APIs, and hardware experiments were not executed.
- No publicly accessible, paper-linked implementation with a verified immutable revision was established.
