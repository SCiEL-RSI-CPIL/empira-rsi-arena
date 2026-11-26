# Single-step code bug repair

**Paper:** Evolution through Large Models (https://arxiv.org/pdf/2206.08896)\
**Task id:** `elm-single-step-code-bug-repair`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Repair mutated 4-Parity and Quadratic Python programs in one model editing step; evaluate whether the repaired program computes the target function correctly.

Benchmark named in the paper: Single-step code bug repair. Paper metric: single-step repair success probability.

## Candidate interface

solve(question: str, llm) -> str. `question` shows one of ELM's seven buggy programs in ELM's prompt layout ('# A buggy implementation', the code, '# Fixed Bugs'), an attempt number and a request for the complete fixed implementation; `llm(...)` is a budgeted, cached OpenRouter chat client. The last ```python block of the reply (else the whole reply) is executed in a resource-limited child process: the repaired 4-Parity function must return the parity of all 16 inputs and the repaired Quadratic function must return a*x^2+b*x+c on 320 grid points. Score: the fraction of the 28 (program, attempt) pairs repaired in one step, an estimate of the single-step repair success probability; higher is better

Parameters: `{"benchmark": "elm_repair", "trials": 4, "samples": 28, "seed": 0}`

## Labels

Scientific contribution: `benchmark` (tier 3); artifact `agent`, domain `software_engineering`, data `public_benchmark`, verification `sampled`, reference `none`, resources `model_api`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Instances: ELM's Appendix A.1 gives the correct 4-Parity and Quadratic programs and the bug procedure; the seven buggy programs are rebuilt from it (4-Parity with 1 to 4 sum variables renamed from b to c and, for the fifth bug, modulus 3 instead of 2; Quadratic with the first one or two '+' operators replaced by '-'). No dataset or code was released; the unit tests are local (exhaustive 4-bit inputs and a 320-point grid of coefficients and x values) because the paper does not list its tests.
- Metric: ELM reports the success probability of one perturbation over many independent samples from a 300M-parameter diff model (and from prompted code models). Here each program gets 4 attempts whose prompts differ by the attempt number, so that a deterministic, cached model client still yields distinct samples; success means the repaired program passes all tests.
- Candidate: the mutation operator itself, a solve(question, llm) program (prompt and model call); the default model google/gemini-2.5-flash is far stronger than ELM's diff models, so the task is likely near saturation. Not verified: the paper's success curves were not reproduced.
- The paper reports success separately for each task and number of bugs; the local score pools all seven programs. Because the programs are fixed, a candidate can return the known correct 4-Parity or Quadratic code without acting as a mutation operator and score 1.0. The local prompt drops the paper's trailing 'def' completion prefix and asks, chat-style, for a python code block.
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

## Sources

- paper: https://arxiv.org/pdf/2206.08896 (Section 3.1 (Mutation through Diff, Figures 1-2) and Appendix A)
- paper: https://arxiv.org/abs/2206.08896 (Section 3.1 and Appendix A.1 (source code of 4-Parity and Quadratic and the bug procedure))

## Limitations

- No official runnable repository was verified from the paper/project sources examined.
- Upstream commands and dependencies were inspected; costly training, paid APIs, and hardware experiments were not executed.
- No publicly accessible, paper-linked implementation with a verified immutable revision was established.
