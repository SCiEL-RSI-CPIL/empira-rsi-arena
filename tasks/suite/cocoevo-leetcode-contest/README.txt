# LeetCode-Contest

**Paper:** CoCoEvo: Co-Evolution of Programs and Test Cases to Enhance Code Generation (https://arxiv.org/pdf/2502.10802)\
**Task id:** `cocoevo-leetcode-contest`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

LeetCode-Contest task evaluated in CoCoEvo: Co-Evolution of Programs and Test Cases to Enhance Code Generation.

Benchmark named in the paper: 80 weekly/biweekly contest problems. Paper metric: pass@1.

## Candidate interface

solve(question: str, llm) -> str. `question` is a LeetCode contest problem in CoCoEvo's format (function signature and statement with the examples removed) plus a request for the complete implementation; the last ```python block of the reply (else the whole reply) is executed. `llm(...)` is a budgeted, cached OpenRouter chat client. Each of the problem's tests calls the entry-point function in a fresh namespace of a resource-limited child process that sees only the call; the plain-data result must equal the expected value (Python == semantics). Score: pass@1, the fraction of a fixed 20-problem sample whose every test passes within 10 s of wall time per problem; higher is better

Parameters: `{"benchmark": "leetcode_contest", "samples": 20, "seed": 0, "time_limit": 10.0}`

## Labels

Scientific contribution: `benchmark` (tier 3); artifact `agent`, domain `software_engineering`, data `public_benchmark`, verification `sampled`, reference `none`, resources `model_api`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Data: LeetCode-Contest of the CoCoEvo authors (lbaf23/LeetCode-Contest at revision 05db6cfbad9648c312e0aff99144e21aa5781514, MIT): 80 weekly and biweekly contest problems released after March 2024 (20 easy, 39 medium, 21 hard) with 51,552 assert tests. The candidate sees the `prompt` field without examples, as in CoCoEvo.
- Reference check: the official solutions shipped with the dataset pass all 51,552 tests of all 80 problems in the local harness (4.8 s in total).
- Execution: CoCoEvo runs the asserts in five processes under a 2 s total limit; here one child process gets 10 s of wall time per problem (also the per-test limit). Each assert 'f(args) == literal' is split into the call, which runs in the child, and the expected literal, which stays in the evaluator; results are compared after plain-data encoding, so results that are not plain data (for example objects with a custom __eq__) fail where CoCoEvo's assert would accept them.
- Candidate: CoCoEvo co-evolves programs and test cases with an LLM; here the candidate is a free-form solve(question, llm) program that may generate, test and select code itself. The default model is google/gemini-2.5-flash at temperature 0. Not verified: the paper's pass@1 values were not reproduced; a fixed 20-problem sample is evaluated.
- The paper's four models (GPT-4o-mini, Qwen2.5-Coder-32B, Llama-3.1-70B and DeepSeek-V3) and its per-problem budget (100 code-generation and 10 test-generation calls) are not used; the local budget is about US$0.0125 per problem. The paper's leakage control (problems released after March 2024) does not hold for the 2025 default model. The paper averages five random tie-break repeats; here one deterministic submission per problem is scored.
- The candidate runs as trusted code in the evaluator's process, and the reference answers are stored in data/llm_bench of the task cache (ARENA_TASK_CACHE_DIR), so a candidate could read them; it can also replace the evaluator's scoring functions, raise the model client's budget attribute and read OPENROUTER_API_KEY from the environment, so the per-evaluation budget bounds an honest program only. Scores assume a candidate that does none of this.
- Model-driven evaluation: the score is an estimate on a fixed seeded sample and depends on the model (ARTASK_LLM_MODEL, default google/gemini-2.5-flash at temperature 0), the sample size (task parameter `samples`, overridden by ARTASK_LLM_SAMPLES) and the per-evaluation budget (ARTASK_LLM_BUDGET_USD, default 0.25). Responses are cached under llm-cache in the task cache (ARTASK_LLM_CACHE overrides) and shared by all candidates: a repeated call, also one at temperature > 0, returns the stored reply, and a cached reply costs nothing against the budget. `valid` is null because a sampled model score cannot certify a candidate.
- The generated code runs as a module named 'solution' (not '__main__'), so a demonstration under if __name__ == '__main__' is not executed, as in the source evaluator; whatever processes it starts are killed when the problem's tests end.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `env:OPENROUTER_API_KEY`
- `network:openrouter`
- `py:sortedcontainers`
- `file:data/llm_bench/leetcode_contest.json`

## Sources

- paper: https://arxiv.org/pdf/2502.10802 (Section IV-A, Table I)
- repository: https://github.com/lbaf23/llm-cocoevo/tree/504355b568b2a0cbd4054783c80bac9c76a4de6d
- benchmark: https://huggingface.co/datasets/lbaf23/LeetCode-Contest/tree/05db6cfbad9648c312e0aff99144e21aa5781514 (test.jsonl (80 problems, 51,552 tests))
- repository: https://github.com/lbaf23/llm-cocoevo/tree/504355b568b2a0cbd4054783c80bac9c76a4de6d/code_evaluator (assert-based pass@1 evaluation)

## Limitations

- The inspected paper does not link a runnable official implementation; the official release was identified separately and is pinned (see below).
- Official CoCoEvo release pinned; program/test co-evolution needs a code model endpoint and the separately downloaded contest/real-world datasets.
- Execution interface inspected in pinned source. Full paper experiments have not been reproduced here.
