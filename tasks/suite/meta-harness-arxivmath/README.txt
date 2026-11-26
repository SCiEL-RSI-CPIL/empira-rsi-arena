# ArXivMath

**Paper:** Meta-Harness: End-to-End Optimization of Model Harnesses (https://arxiv.org/pdf/2603.28052)\
**Task id:** `meta-harness-arxivmath`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

ArXivMath task evaluated in Meta-Harness: End-to-End Optimization of Model Harnesses.

Benchmark named in the paper: ArXivMath. Paper metric: answer/proof accuracy.

## Candidate interface

solve(question: str, llm) -> str. `llm(prompt_or_messages, system=None, max_tokens=None, temperature=None, json_mode=False) -> str` is a budgeted, cached OpenRouter chat client. `question` is an ArXivMath problem followed by 'Put your final answer within \boxed{}.' Answer rule: the content of the reply's last \boxed{...} (else the final answer text after the last 'Answer:' marker or the last line) is parsed as LaTeX and must be equivalent to the gold answer under Math-Verify 0.9.0 (symbolic and numeric equivalence; ordered tuples must match in order). Score: accuracy over all 40 ArXivMath December 2025 and January 2026 problems; higher is better

Parameters: `{"benchmark": "arxivmath", "samples": 40, "seed": 0, "budget_usd": 0.6}`

## Labels

Scientific contribution: `benchmark` (tier 3); artifact `agent`, domain `language`, data `public_benchmark`, verification `sampled`, reference `none`, resources `model_api`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Data: the ArXivMath December 2025 (17 problems) and January 2026 (23 problems) sets of MathArena (MathArena/arxivmath-1225 at revision 90c60aa1eadbb72168640169d82154b079cdeaac and MathArena/arxivmath-0126 at revision bac939ba603c93f905aa48072980ea681a2824cd, CC-BY-SA-4.0), the two ArXivMath parts of Meta-Harness's 200-problem IMO-level evaluation set (its Table 11). All 40 problems are evaluated.
- Metric: final-answer equivalence by Math-Verify (Apache-2.0) instead of MathArena's own answer checker; all 40 gold answers parse and verify against themselves, and no gold answer is accepted for a different problem. Meta-Harness reports accuracy averaged over three samples per problem; here one deterministic sample (temperature 0) is scored.
- Scope: Meta-Harness's math harnesses retrieve worked solutions from a 535K-problem corpus. That corpus is not provided, so the candidate is a free-form solve(question, llm) harness without retrieval (it may call the model several times and run code); the default model is google/gemini-2.5-flash, not the paper's models (GPT-OSS-20B, GPT-5.4-nano, GPT-5.4-mini, Gemini-3.1-Flash-Lite and Gemini-3-Flash). This mirrors the existing meta-harness-imo-answerbench task.
- Not verified: the paper's accuracies were not reproduced.
- In the paper ArXivMath is held out from the harness search (which uses 250 OlympiadBench and Omni-MATH problems), and only an aggregate over 200 problems from IMO-AnswerBench, IMO-ProofBench and ArXivMath is reported; here the 40 ArXivMath problems are the optimisation target, and there is no ArXivMath-only paper number.
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
- `file:data/llm_bench/arxivmath.json`
- `py:math_verify`

## Sources

- paper: https://arxiv.org/pdf/2603.28052 (Section 4.2; 200 combined held-out problems)
- repository: https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact/tree/57fefdb2ff84af3fd81b69d67814acbe69bd0743
- benchmark: https://huggingface.co/datasets/MathArena/arxivmath-1225/tree/90c60aa1eadbb72168640169d82154b079cdeaac (ArXivMath December 2025 (17 problems))
- benchmark: https://huggingface.co/datasets/MathArena/arxivmath-0126/tree/bac939ba603c93f905aa48072980ea681a2824cd (ArXivMath January 2026 (23 problems))
