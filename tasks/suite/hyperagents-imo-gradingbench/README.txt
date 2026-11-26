# IMO-GradingBench

**Paper:** HyperAgents (https://arxiv.org/pdf/2603.19461)\
**Task id:** `hyperagents-imo-gradingbench`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Write a Python solve(question, llm) agent that, using a budgeted OpenRouter model client, grades a proposed proof of an IMO-level problem as correct, almost, partial or incorrect from the reference solution and grading guidelines; scored by exact-match accuracy against human grades on a fixed 30-item sample of a 100-item held-out IMO-GradingBench subset (higher is better).

## Candidate interface

solve(question: str, llm) -> str. `llm(prompt_or_messages, system=None, max_tokens=None, temperature=None, json_mode=False) -> str` is a budgeted, cached OpenRouter chat client. `question` gives an IMO-level problem, its reference solution, the grading guidelines and a proposed proof, and asks for one grade among correct, almost, partial and incorrect. Answer rule (this suite's answer extraction; the comparison follows Hyperagents' report.py): the final answer (the text after the last 'Answer:' marker, else the last \boxed{...}, else the last non-empty line), lower-cased and stripped of punctuation and asterisks, must equal the human grade. Score: accuracy on a fixed 30-item sample of Hyperagents' 100-item held-out IMO-GradingBench subset (normalized mean absolute error is reported alongside); higher is better

Parameters: `{"benchmark": "gradingbench", "samples": 30, "seed": 0}`

## Labels

Scientific contribution: `benchmark` (tier 3); artifact `agent`, domain `language`, data `public_benchmark`, verification `sampled`, reference `none`, resources `model_api`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Data: IMO-GradingBench of IMO-Bench (google-deepmind/superhuman at commit c1ee02e03d4cdb2ab21cd01ac927d895f5287fc8, the commit Hyperagents' setup pins; CC-BY-4.0). The 100-item held-out subset follows Hyperagents' domains/imo/curate_subsets.py: rows with 0, 1, 6 or 7 points, a pandas shuffle and a 300-row sample (random_state 42), then stratified train/val/test splits of 100 (scikit-learn, random_state 42). The resulting ids (35 incorrect, 35 correct, 24 partial, 6 almost) were computed here with pandas 2.2.3 and scikit-learn 1.9.1 and are fixed in llm_bench_sources.py; Hyperagents does not distribute its subset files, so identity with its split is not verified.
- Inputs: Hyperagents' imo_grading task input (problem, reference solution, grading guidelines and the student's proof). The question adds one instruction naming the four grades with paraphrased IMO-Bench definitions; Hyperagents' initial agent receives only the raw input fields.
- Metric: Hyperagents' report.py accuracy (lower-cased exact match of the grade). The normalized mean absolute error over IMO-Bench points (correct 7, almost 6, partial 1, incorrect 0; 7 for unparsable answers) is reported as normalized_mae.
- Candidate: Hyperagents evolves the code of a task agent and of the meta agent; here the candidate is a free-form solve(question, llm) program, default model google/gemini-2.5-flash at temperature 0. Not verified: the paper's grader accuracies were not reproduced; a fixed 30-item sample is evaluated.
- The paper describes the grader output as a numeric grade in {0, 1, 6, 7}; following Hyperagents' code (whose ground truth is the text Reward column), this task requires the word label and scores a numeric grade as wrong. The paper evaluates with o4-mini at temperature 1.0.
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
- `file:data/llm_bench/gradingbench_test.json`

## Sources

- paper: https://arxiv.org/pdf/2603.19461 (Section 4 and Appendix C)
- repository: https://github.com/facebookresearch/Hyperagents/tree/59a68f672dfb92c74aeb7e61535d776fb36e172d (domains/imo)
- benchmark: https://github.com/google-deepmind/superhuman/blob/c1ee02e03d4cdb2ab21cd01ac927d895f5287fc8/imobench/gradingbench.csv (IMO-GradingBench (1,000 human-graded proofs))
- repository: https://github.com/facebookresearch/Hyperagents/tree/59a68f672dfb92c74aeb7e61535d776fb36e172d/domains/imo/curate_subsets.py (held-out 100-item subset procedure)
