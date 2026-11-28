# TravelPlanner

**Paper:** Evolving Deeper LLM Thinking (https://arxiv.org/pdf/2501.09891)\
**Task id:** `mind-evolution-travelplanner`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Write a solve(question, llm) program that uses a budgeted OpenRouter LLM client to turn a TravelPlanner query and its reference information into a day-by-day JSON plan; scored by success rate on a fixed 20-query validation sample, a plan succeeding only if it passes all eight commonsense and all five hard constraints (higher is better).

Benchmark named in the paper: TravelPlanner. Paper metric: success rate.

## Candidate interface

solve(question: str, llm) -> str. `question` is a TravelPlanner query, the query's reference information (TravelPlanner's sole-planning input) and the required plan format; `llm(...)` is a budgeted, cached OpenRouter chat client. The JSON list that ends last in the reply must hold one record per day with the keys days, current_city, transportation, breakfast, attraction, lunch, dinner and accommodation. A query is solved when the plan passes all eight commonsense and all five hard constraints of TravelPlanner's evaluation (Final Pass Rate). Score: success rate on a fixed 20-query sample of the 180 validation queries; higher is better

Parameters: `{"benchmark": "travelplanner", "samples": 20, "seed": 0}`

## Labels

Scientific contribution: `benchmark` (tier 3); artifact `agent`, domain `language`, data `public_benchmark`, verification `sampled`, reference `none`, resources `model_api`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Data: TravelPlanner validation set (osunlp/TravelPlanner at revision 8736504ecfc31b7f8b7e40122873c337e83fff7c, 180 queries with reference information, CC-BY-4.0) and the benchmark database of the TravelPlanner leaderboard space (osunlp/TravelPlannerLeaderboard at revision a57daf2e45155a5378dcf75ee9fe374e671ee048). Mind Evolution reports validation and test success rates; test labels are only scored by the official server, so the validation set is used. Each evaluation uses a fixed 20-query sample.
- Checker: a port of evaluation/commonsense_constraint.py and evaluation/hard_constraint.py (OSU-NLP-Group/TravelPlanner at commit e52c87f4ac348a3410c46dc3553c519db5ec5e23, MIT). With the full flight table, the port agrees with the official code on every constraint verdict and on the final pass for all 161 delivered plans of TravelPlanner's postprocess/example_evaluation.jsonl. Upstream checks that would raise on a malformed plan count as failed constraints.
- Flight table: restricted to the 1,698 flights of the routes and dates in the validation queries' reference information, which lists every flight a sole-planning plan is given. The official checker also accepts other existing flights whose number, origin and destination match (on any date); with the restricted table, 27 of the 161 example plans (produced in two-stage mode) receive a different sandbox verdict and none a different final verdict.
- Output format: TravelPlanner converts natural-language plans to JSON with GPT-4 before checking; here the candidate returns the JSON list itself (Mind Evolution also works with JSON plans). Candidate: a free-form solve(question, llm) program instead of Mind Evolution's evolutionary search with Gemini 1.5 Flash and Pro; the default model is google/gemini-2.5-flash at temperature 0. Not verified: the paper's success rates were not reproduced.
- Mind Evolution receives textual feedback from programmatic evaluation functions (Appendix A.2); the local candidate receives no evaluator feedback, no structured constraint JSON and no example plans from the training set. Mind Evolution reports about 3.10M input tokens per validation problem with Gemini 1.5 Flash; the local per-query budget is far smaller.
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
- `file:data/llm_bench/travelplanner_validation.json`
- `file:data/llm_bench/travel_flights.csv`
- `file:data/llm_bench/travel_accommodations.csv`
- `file:data/llm_bench/travel_restaurants.csv`
- `file:data/llm_bench/travel_attractions.csv`
- `file:data/llm_bench/travel_distance.csv`
- `file:data/llm_bench/travel_city_states.txt`
- `py:pandas`

## Sources

- paper: https://arxiv.org/pdf/2501.09891 (Section 4.1, Table 2, Appendix A.2 and Appendix B)
- benchmark: https://huggingface.co/datasets/osunlp/TravelPlanner/tree/8736504ecfc31b7f8b7e40122873c337e83fff7c (validation.csv (180 queries with reference information))
- archive: https://huggingface.co/spaces/osunlp/TravelPlannerLeaderboard/tree/a57daf2e45155a5378dcf75ee9fe374e671ee048/database (benchmark database used by the checks)
- repository: https://github.com/OSU-NLP-Group/TravelPlanner/tree/e52c87f4ac348a3410c46dc3553c519db5ec5e23/evaluation (commonsense and hard constraint checks (ported))

## Limitations

- No official runnable repository was verified from the paper/project sources examined.
- Upstream commands and dependencies were inspected; costly training, paid APIs, and hardware experiments were not executed.
- No publicly accessible, paper-linked implementation with a verified immutable revision was established.
