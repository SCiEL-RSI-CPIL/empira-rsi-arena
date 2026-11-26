# USPTO-50k

**Paper:** Meta Context Engineering via Agentic Skill Evolution (https://arxiv.org/pdf/2601.21557)\
**Paper:** Meta-Harness: End-to-End Optimization of Model Harnesses (https://arxiv.org/pdf/2603.28052)\
**Task id:** `mce-uspto-50k`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

USPTO-50k task evaluated in Meta Context Engineering via Agentic Skill Evolution.

Benchmark named in the paper: USPTO-50k. Paper metric: pass@1 exact-match accuracy.

## Candidate interface

solve(question: str, llm) -> str. `llm(prompt_or_messages, system=None, max_tokens=None, temperature=None, json_mode=False) -> str` is a budgeted, cached OpenRouter chat client. `question` gives the reaction type and the product SMILES (MCE's USPTO-50k format) and asks for the reactants as '.'-separated SMILES. Answer rule (MCE's USPTO check): the '.'-separated components of the final answer (the text after the last 'Answer:' marker, else the last \boxed{...}, else the last non-empty line), stripped and lower-cased, must equal the gold reactant set exactly; there is no SMILES canonicalisation. Score: accuracy on a fixed 30-item sample of MCE's 100-item USPTO-50k test sample; higher is better

Parameters: `{"benchmark": "uspto", "samples": 30, "seed": 0}`

## Labels

Scientific contribution: `benchmark` (tier 3); artifact `agent`, domain `language`, data `public_benchmark`, verification `sampled`, reference `none`, resources `model_api`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Related paper and local records

These records are grouped under this local package. Sharing a local evaluator does not establish identical paper experiments, datasets or candidate contracts. Configured upstream routes remain reachable with `run.py evaluate mce-uspto-50k --route upstream --paper <paper>`; local reference aliases retain their recorded native interfaces.

- Meta-Harness: USPTO-50k (`meta-harness-uspto-50k`; USPTO-50k, USPTO-50k)

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Data: MCE artifact (commit c63300daef93404e22e2569b3d08e72843e7123b), env/uspto/data/test.jsonl: 100 USPTO-50k test reactions stratified over the 10 reaction types by the MCE authors. The trailing 'Answer:' of MCE's question string is removed and a one-sentence task statement is prepended.
- Metric: MCE's exact set match of lower-cased SMILES strings. As in MCE, chemically identical reactants written as different SMILES strings score 0; the Jaccard partial credit MCE also logs is not part of the score.
- Candidate: MCE co-evolves context-engineering skills and learned context files that a fixed generator (DeepSeek-V3.1 in the paper) consumes, learning from 50 training and 30 validation instances (Appendix A.2). Here the candidate is a free-form solve(question, llm) program (any learned context must live in the program); the training split is not supplied by the evaluator, and the default model is google/gemini-2.5-flash at temperature 0.
- Not verified: the paper's accuracies were not reproduced; a fixed 30-item sample is evaluated.
- MCE evaluates on held-out test instances; here the candidate is optimised against the test items it is scored on. Meta-Harness uses USPTO-50k as one of its search datasets, framed as 180-class online classification with GPT-OSS-120B, and does not specify its matching rule. Lower-casing SMILES before matching conflates aromatic and aliphatic atoms, so the match is slightly more lenient than an exact string match.
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
- `file:data/llm_bench/uspto.json`

## Sources

- paper: https://arxiv.org/pdf/2601.21557 (Section 4.1 and Appendix A)
- repository: https://github.com/metaevo-ai/mce-artifact/tree/c63300daef93404e22e2569b3d08e72843e7123b (env/uspto/uspto_environment.py)
- paper: https://arxiv.org/pdf/2603.28052 (Section 4.1)
- repository: https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact/tree/57fefdb2ff84af3fd81b69d67814acbe69bd0743
- repository: https://github.com/metaevo-ai/mce-artifact/tree/c63300daef93404e22e2569b3d08e72843e7123b/env/uspto (data/test.jsonl and uspto_environment.py (reactant-set exact match))

## Limitations

- Execution interface inspected in pinned source. Full paper experiments have not been reproduced here.
