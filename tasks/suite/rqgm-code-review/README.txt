# Code review

**Paper:** The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators (https://arxiv.org/pdf/2606.26294)\
**Task id:** `rqgm-code-review`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Code review task evaluated in The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators.

Benchmark named in the paper: CRAVE. Paper metric: review classification accuracy.

## Candidate interface

solve(question: str, llm) -> str. `llm(prompt_or_messages, system=None, max_tokens=None, temperature=None, json_mode=False) -> str` is a budgeted, cached OpenRouter chat client. `question` gives a pull request (repository, title, description and the patch as a diff) and asks for the reviewer's verdict. Answer rule: the final answer (the text after the last 'Answer:' marker, else the last \boxed{...}, else the last non-empty line), upper-cased with every run of non-letters replaced by '_', must equal the gold verdict APPROVE or REQUEST_CHANGES. Score: accuracy on a fixed 30-item sample of CRAVE's 134-item test split; higher is better

Parameters: `{"benchmark": "crave", "samples": 30, "seed": 0}`

## Labels

Scientific contribution: `benchmark` (tier 3); artifact `agent`, domain `language`, data `public_benchmark`, verification `sampled`, reference `none`, resources `model_api`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Data: CRAVE (TuringEnterprises/CRAVE at revision 4d94975d6c9b70d7bcf611b15aadf3d89fe22870, MIT), the dataset's predefined test split (134 items from 67 pull requests in 16 repositories, each pull request once as APPROVE and once as REQUEST_CHANGES; balanced labels). RQGM lists a 100-item CRAVE test set (Table 3) whose selection is not published; it scores code-reviewer challengers against this held-out anchor but reports no held-out accuracy for code reviewers; each local evaluation uses a fixed 30-item sample of the test split.
- Inputs: repository, pull-request title, description and patch. CRAVE's `explanation` field (the reviewers' rationale, which reveals the verdict) and its LLM-written `hint` are withheld; RQGM does not specify which fields its reviewers receive.
- Metric: verdict accuracy, the quantity RQGM uses to anchor its co-evolved code reviewer. Candidate: RQGM evolves a reviewer agent (GPT-5.5 at low effort in the paper); here a free-form solve(question, llm) program, default model google/gemini-2.5-flash at temperature 0.
- The paper reports no CRAVE accuracy for its code reviewers (the reviewer accuracies it reports are for APReS paper reviewers), so there is no paper number to compare with.
- RQGM uses train feedback, a hidden validation utility and a separate held-out anchor; here the candidate is optimised by raw accuracy against the same 30-item sample it is scored on.
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
- `file:data/llm_bench/crave_test.json`

## Sources

- paper: https://arxiv.org/pdf/2606.26294 (Section 4, Domains and ground-truth anchors; Appendix C)
- benchmark: https://huggingface.co/datasets/TuringEnterprises/CRAVE/tree/4d94975d6c9b70d7bcf611b15aadf3d89fe22870 (test.parquet (134 items))

## Limitations

- arXiv HTML unavailable; full PDF retrieved and text extracted.
- No official public runnable implementation verified from the inspected paper. External benchmark code may exist separately.
- No publicly accessible, paper-linked implementation with a verified immutable revision was established.
