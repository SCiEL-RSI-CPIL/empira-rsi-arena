# HoVer

**Paper:** Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs (https://arxiv.org/pdf/2406.11695)\
**Paper:** GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning (https://arxiv.org/pdf/2507.19457)\
**Paper:** Reward-Free Evolving Agents via Pairwise Validator (https://arxiv.org/pdf/2607.14408)\
**Task id:** `mipro-hover`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Write a Python solve(claim, search, llm) program that retrieves, with at most 3 BM25 searches and 24 documents over 5.2 million 2017 Wikipedia abstracts and a budgeted OpenRouter model, the three gold supporting documents of a HoVer claim; scored by the fraction of claims whose gold titles were all retrieved, on a fixed 30-claim sample (higher is better).

Benchmark named in the paper: HoVer. Paper metric: Recall@21.

## Candidate interface

solve(claim: str, search, llm) -> any (the return value is not used). search(query: str, k: int = 7) -> list[str] returns up to k documents 'title | abstract' ranked by BM25 (bm25s, k1 0.9, b 0.4, English stopwords, Snowball stemming) over the 5.2 million 2017 Wikipedia abstracts; per claim at most 3 search calls, k from 1 to 10 per call and at most 24 documents in total (a call beyond the budget raises RuntimeError). llm(prompt_or_messages, system=None, max_tokens=None, temperature=None, json_mode=False) -> str is a budgeted, cached OpenRouter chat client. A claim is solved when the titles of all its gold supporting documents (three distinct Wikipedia articles) are among the documents that search returned for it (titles compared after NFD normalisation, lower-casing and removal of punctuation, articles and extra spaces). Score: fraction of solved claims on a fixed seeded sample of GEPA's HoVer test split; higher is better. The program runs in a sandbox without file or network access (standard library and numpy only)

Parameters: `{"samples": 30, "seed": 0, "budget_usd": 0.5}`

## Labels

Scientific contribution: `benchmark` (tier 3); artifact `agent`, domain `language`, data `public_benchmark`, verification `sampled`, reference `none`, resources `model_api`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Related paper and local records

These records are grouped under this local package. Sharing a local evaluator does not establish identical paper experiments, datasets or candidate contracts. Configured upstream routes remain reachable with `run.py evaluate mipro-hover --route upstream --paper <paper>`; local reference aliases retain their recorded native interfaces.

- GEPA: HoVer (`gepa-hover`; HoVer)
- Pairwise Validator: HoVer (`pairwise-validator-hover`; HoVer)

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Recreation of the HoVer multi-hop retrieval task as released in the GEPA artifact (gepa-ai/gepa-artifact at cbefbc1aa0f43dd39874ec4bf42211365dbda42e, MIT): the retriever is bm25s 0.2.12 (k1 0.9, b 0.4, English stopwords, PyStemmer 2.2.0.3 Snowball stemmer, the versions pinned by the artifact) over the 5,233,330 2017 Wikipedia abstracts of dspy/cache (wiki.abstracts.2017.tar.gz at revision ef6a5e72a98b47cef31574a400fea8fe149559a3), each rendered as 'title | abstract' and deduplicated by text as in the artifact's search() (top-k selection through jax.lax.top_k, as bm25s does when jax is installed; the artifact pins jax 0.6.0, this environment has 0.11.2); the metric is the artifact's discrete_retrieval_eval (every gold supporting title, normalised as dspy.evaluate.normalize_text, among the retrieved titles).
- Split: GEPA's HoVer data are the 3-document claims of the HoVer train release (6,084 claims), shuffled with random.Random(0) and split 40/40/20 into test/validation/train pools, with the test pool trimmed to 300 claims by random.Random(1).sample; this procedure is reimplemented over hover_train_release_v1.1.json (hover-nlp/hover at 39b84697) instead of the Hugging Face loading script, so identity with GEPA's 300 test claims rests on the same file order and Python's random module and was not checked against GEPA's own files. Each evaluation uses a fixed 30-claim sample of these 300 (ARTASK_LLM_SAMPLES overrides the size); candidates are optimised and scored on the same sample, so scores are in-sample.
- Program structure: GEPA (and MIPRO) optimise the prompts of a fixed DSPy program with 3 hops; here the whole program is the candidate, constrained to GEPA's retrieval budget: at most 3 search calls per claim, at most 10 documents per call and 24 in total (GEPA retrieves 7, 7 and 10). The documents returned by search are scored; the candidate's return value is ignored, so a program cannot claim titles it did not retrieve.
- MIPRO itself evaluated HoVer with its own Wikipedia retriever (not reproduced here), Recall@21 (3 hops of 7 passages), a Llama-3-8B task model and its own splits (500/500/1,520 claims restricted to 3 supporting facts, so about 17% of its test claims need only two documents, whereas the claims here have three gold documents); those settings are not reproduced and its numbers (39.0 test recall for MIPRO, Table 2) are not comparable with scores here. GEPA's reported HoVer scores (Qwen3-8B and GPT-4.1-mini task models) were not reproduced either. The third cited paper (Reward-Free Evolving Agents via Pairwise Validator) runs GEPA's HoVer setting with Qwen3-8B, Qwen3-32B, Gemma-4-E4B and gpt-oss-20b, 375 iterations, a 30-claim validation pool and a 50-claim test set (its Table 8); its HoVer scores (Table 2) were not reproduced either.
- Model: the default task model is google/gemini-2.5-flash at temperature 0 through OpenRouter, budget USD 0.5 per evaluation (ARTASK_LLM_BUDGET_USD overrides); responses are cached.
- Deviation: a search whose query bm25s cannot score (for example a query made only of stopwords) returns an empty list instead of failing the claim.
- Data: all 900 gold titles of the 300 test claims occur as document titles of the corpus. The prepared corpus and index take about 2.5 GB of disk (docs.jsonl 1.5 GB, bm25s arrays 1.0 GB, memory-mapped at evaluation; a search takes about 0.05 s); building them once took about 3 minutes on one core and several GB of memory. The HoVer claims are CC BY-SA 4.0; the abstracts are Wikipedia text (CC BY-SA) whose dspy/cache repository states no licence; nothing is redistributed (scripts/prepare_llm_tools.py hover).
- Model-driven evaluation: the score is an estimate on a fixed seeded sample and depends on the model (ARTASK_LLM_MODEL, default google/gemini-2.5-flash at temperature 0), the sample size (task parameter `samples`, overridden by ARTASK_LLM_SAMPLES) and the per-evaluation budget (task parameter `budget_usd`, overridden by ARTASK_LLM_BUDGET_USD). Responses are cached in llm-cache under the task cache (ARENA_TASK_CACHE_DIR, default ~/.cache/empira/tasks). `valid` is null because a sampled model score cannot certify a candidate. The candidate runs in a resource-limited child process without file, network or subprocess access (software_sandbox); task capabilities reach it only as proxy callables served by the evaluator.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `env:OPENROUTER_API_KEY`
- `network:openrouter`
- `py:bm25s`
- `py:Stemmer`
- `py:psutil`
- `file:data/llm_tools/hover/claims.json`
- `file:data/llm_tools/hover/index`

## Sources

- paper: https://arxiv.org/pdf/2406.11695 (Table 1 and Appendices A/E)
- repository: https://github.com/stanfordnlp/dspy/tree/40a6e168914a7b81a78b1a081d93f26de18c8d0d
- paper: https://arxiv.org/pdf/2507.19457 (Section 4 evaluation and benchmark appendix)
- repository: https://github.com/gepa-ai/gepa-artifact/tree/cbefbc1aa0f43dd39874ec4bf42211365dbda42e (gepa_artifact/benchmarks/hover)
- paper: https://arxiv.org/pdf/2607.14408 (Section 5.1 (Tasks), Table 2 and Appendix C (Table 8))
- repository: https://github.com/gepa-ai/gepa-artifact/tree/cbefbc1aa0f43dd39874ec4bf42211365dbda42e/gepa_artifact/benchmarks/hover (hover_program.py (bm25s retriever, HoverMultiHop), hover_utils.py (discrete_retrieval_eval), hover_data.py and benchmark.py (split))
- benchmark: https://github.com/hover-nlp/hover/blob/39b84697f196308f398a251a7aea9b82ae0f0562/data/hover/hover_train_release_v1.1.json (HoVer train release v1.1 (18,171 claims))
- archive: https://huggingface.co/dspy/cache/blob/ef6a5e72a98b47cef31574a400fea8fe149559a3/wiki.abstracts.2017.tar.gz (2017 Wikipedia abstracts (5,233,330 documents))

## Limitations

- Upstream commands and dependencies were inspected; costly training, paid APIs, and hardware experiments were not executed.
