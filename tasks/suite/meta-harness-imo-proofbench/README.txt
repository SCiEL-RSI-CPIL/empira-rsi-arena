# IMO-ProofBench

**Paper:** Meta-Harness: End-to-End Optimization of Model Harnesses (https://arxiv.org/pdf/2603.28052)\
**Task id:** `meta-harness-imo-proofbench`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

IMO-ProofBench task evaluated in Meta-Harness: End-to-End Optimization of Model Harnesses.

Benchmark named in the paper: IMO-ProofBench. Paper metric: answer/proof accuracy.

## Candidate interface

solve(question: str, llm) -> str: a complete proof for an IMO-ProofBench problem (`question` is the problem statement and a request for a rigorous proof). `llm(...)` is a budgeted, cached OpenRouter chat client. A judge model grades the proof against the reference solution and the problem's grading guidelines on IMO-Bench's 0/1/6/7-point scale (first <points>N out of 7</points> block of its reply; unparsable grades count as 0). Score: mean points / 7 over a fixed seeded sample of the 60 problems; higher is better

Parameters: `{"samples": 20, "seed": 0, "judge_model": "google/gemini-2.5-flash", "judge_max_tokens": 8192, "budget_usd": 1.0}`

## Labels

Scientific contribution: `benchmark` (tier 3); artifact `agent`, domain `mathematics`, data `public_benchmark`, verification `sampled`, reference `none`, resources `model_api`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Data: IMO-ProofBench of IMO-Bench (google-deepmind/superhuman at commit c1ee02e03d4cdb2ab21cd01ac927d895f5287fc8, CC-BY-4.0): 30 basic and 30 advanced problems with reference solutions and grading guidelines, the IMO-ProofBench part of Meta-Harness's 200-problem evaluation set (its Table 11). Each evaluation uses a fixed 20-problem sample.
- Grading: a judge model applies a paraphrase of IMO-Bench's ProofAutoGrader prompt (rubric 7/6/1/0 points, reference solution, problem-specific guidelines, first <points> block; when the judge omits the tags its last 'N out of 7' phrase is used and counted as untagged_grades, and a grade that cannot be parsed counts as 0). ProofAutoGrader uses Gemini 2.5 Pro; the default judge here is google/gemini-2.5-flash for cost (set judge_model or ARTASK_LLM_JUDGE_MODEL to google/gemini-2.5-pro for the source's grader model). Scores are model-judged estimates; IMO-Bench reports that ProofAutoGrader correlates with human grades but recommends human verification.
- Metric: mean points / 7 (IMO-ProofBench's percentage of total points); the share of full-credit proofs is reported as pass_at_7. Meta-Harness reports 'accuracy' averaged over three samples with its own retrieval harness; its exact proof-scoring rule is not specified and its 535K-problem retrieval corpus is not provided, so the candidate is a free-form solve(question, llm) harness without retrieval.
- Cost: proofs are long, so the per-evaluation budget of the prover client is raised to USD 1.00 (parameter budget_usd); with gemini-2.5-flash as prover and judge a 20-problem evaluation costs roughly USD 0.4 to 0.7.
- Not verified: the paper's results were not reproduced.
- In the paper IMO-ProofBench is held out from the harness search (which uses 250 OlympiadBench and Omni-MATH problems) and evaluated with GPT-OSS-20B, GPT-5.4-nano, GPT-5.4-mini, Gemini-3.1-Flash-Lite and Gemini-3-Flash; only an aggregate over 200 problems is reported. Here its problems are the optimisation target, and there is no ProofBench-only paper number.
- The candidate runs as trusted code in the evaluator's process, and the reference answers are stored in data/llm_bench of the task cache (ARENA_TASK_CACHE_DIR), so a candidate could read them; it can also replace the evaluator's scoring functions, raise the model client's budget attribute and read OPENROUTER_API_KEY from the environment, so the per-evaluation budget bounds an honest program only. Scores assume a candidate that does none of this.
- Model-judged evaluation: proofs are graded by a judge model (ARTASK_LLM_JUDGE_MODEL, task default google/gemini-2.5-flash; IMO-Bench's ProofAutoGrader uses Gemini 2.5 Pro) on a fixed seeded sample; the prover model is ARTASK_LLM_MODEL (default google/gemini-2.5-flash). Responses are cached under llm-cache in the task cache (ARTASK_LLM_CACHE overrides) and shared by all candidates: a repeated call, also one at temperature > 0, returns the stored reply, and a cached reply costs nothing against the budget; `valid` is null.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `env:OPENROUTER_API_KEY`
- `network:openrouter`
- `file:data/llm_bench/proofbench.json`

## Sources

- paper: https://arxiv.org/pdf/2603.28052 (Section 4.2; 200 combined held-out problems)
- repository: https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact/tree/57fefdb2ff84af3fd81b69d67814acbe69bd0743
- benchmark: https://github.com/google-deepmind/superhuman/blob/c1ee02e03d4cdb2ab21cd01ac927d895f5287fc8/imobench/proofbench.csv (IMO-ProofBench: 60 problems with reference solutions and grading guidelines)
- paper: https://arxiv.org/abs/2511.01846 (Appendix B.5 ProofAutoGrader prompt and 0/1/6/7 rubric (paraphrased))
