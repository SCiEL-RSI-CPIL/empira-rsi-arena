# SingleEq

**Paper:** Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution (https://arxiv.org/pdf/2309.16797)\
**Task id:** `promptbreeder-singleeq`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Write a solve(question, llm) program that uses a budgeted OpenRouter LLM client to solve SingleEq problems; scored by accuracy on a fixed 30-problem sample of a deterministic test half, an answer counting when its final numeric literal matches the gold value within 1e-6 (higher is better).

Benchmark named in the paper: SingleEq. Paper metric: accuracy.

## Candidate interface

solve(question: str, llm) -> str. `question` is the benchmark problem text; `llm(prompt_or_messages, system=None, max_tokens=None, temperature=None, json_mode=False) -> str` is a budgeted, cached OpenRouter chat client. Answer rule: the evaluator takes the final answer (the text after the last 'Answer:' marker, else the last \boxed{...}, else the last non-empty line) and compares its last numeric literal (commas, $ and % ignored; a/b and \frac{a}{b} count as one literal) with the gold value; a match within 1e-6 (absolute or relative) scores 1, anything else 0. Score: accuracy on a fixed 30-problem sample of the deterministic SingleEq test half; higher is better

Parameters: `{"benchmark": "singleeq", "split": "test", "samples": 30, "seed": 0}`

## Labels

Scientific contribution: `benchmark` (tier 3); artifact `prompt`, domain `language`, data `public_benchmark`, verification `sampled`, reference `none`, resources `model_api`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Data: SingleEq (Koncel-Kedziorski et al., TACL 2015; 508 problems) as redistributed in the zero_shot_cot repository at commit 5ef330fcdeec0cd26aee27943504f91f8ec1c33c, fetched and SHA-256 checked by scripts/prepare_llm_bench.py. The repository declares no license; the data are not redistributed here.
- Split: Promptbreeder splits SingleEq into equal training and test halves without publishing the split. Here the halves come from a SHA-256 ordering of problem ids (254 test problems); each evaluation uses a fixed 30-problem sample of the test half (ARTASK_LLM_SAMPLES overrides the size).
- Candidate: a free-form solve(question, llm) program replaces Promptbreeder's evolved pair of task prompts; the default model is google/gemini-2.5-flash at temperature 0, not the paper's PaLM 2-L. Promptbreeder's answer extraction is not specified; the local numeric rule is stated in the contract.
- Not verified: the paper's accuracies were not reproduced and local scores are not comparable with them.
- Promptbreeder selects prompts by fitness on batches of 100 training-half problems and evaluates only the final best prompt on the test half; here candidates are optimised against the same 30-problem test sample that is reported and the training half is unused, so local scores are in-sample.
- Promptbreeder caps the Inducer at 30 and the Evaluator at 5 tokens (Appendix J.2) and uses its problem description only to seed and mutate prompts; the local candidate may make several unbounded model calls, and the task corresponds to neither the zero-shot nor the few-shot setting in particular (no training examples are supplied).
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
- `file:data/llm_bench/singleeq.json`

## Sources

- paper: https://arxiv.org/pdf/2309.16797 (Section 4 and Appendix I)
- benchmark: https://github.com/kojima-takeshi188/zero_shot_cot/tree/5ef330fcdeec0cd26aee27943504f91f8ec1c33c/dataset/SingleEq (questions.json (508 problems))

## Limitations

- No official runnable repository was verified from the paper/project sources examined.
- Upstream commands and dependencies were inspected; costly training, paid APIs, and hardware experiments were not executed.
- No publicly accessible, paper-linked implementation with a verified immutable revision was established.
