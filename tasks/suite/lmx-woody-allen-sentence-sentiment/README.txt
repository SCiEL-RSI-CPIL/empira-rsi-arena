# Woody Allen sentence sentiment

**Paper:** Language Model Crossover: Variation through Few-Shot Prompting (https://arxiv.org/pdf/2302.12170)\
**Task id:** `lmx-woody-allen-sentence-sentiment`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Write a solve(question, llm) program that returns 1 to 300 one-line sentences rewriting the Woody Allen quote 'Life is divided into the horrible and the miserable.' with positive sentiment; scored by QD score, the sum over 30 niches of MiniLM embedding distance to the quote of the best positive probability from a RoBERTa sentiment classifier (higher is better).

Benchmark named in the paper: Woody Allen sentence sentiment. Paper metric: domain-specific fitness / QD score.

## Candidate interface

solve(question: str, llm) -> list[str]: an archive of 1 to 300 one-line sentences of 5 to 400 characters, each valid Unicode text (no unpaired surrogates) and at most 512 tokens for the sentiment classifier's tokenizer (400 characters of CJK text or of some emoji are more). `question` is the target quote; `llm(...)` is an optional budgeted OpenRouter client (it needs OPENROUTER_API_KEY only when called). Each sentence's fitness is the 'positive' probability of cardiffnlp/twitter-roberta-base-sentiment-latest and its niche is the Euclidean distance between its all-MiniLM-L6-v2 embedding and the quote's, clipped to [0, 1.25] and cut into 30 equal bins. Score: QD score, the sum over occupied niches of the best fitness (at most 30); higher is better

Parameters: `{"quote": "Life is divided into the horrible and the miserable.", "niches": 30, "max_distance": 1.25, "max_sentences": 300}`

## Labels

Scientific contribution: `benchmark` (tier 3); artifact `prompt`, domain `language`, data `none`, verification `simulation`, reference `none`, resources `unavailable`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Source: LMX (Meyerson et al.), Section 4.3 'Modifying Sentence Sentiment', and scripts/lmx_sentiment_demo.ipynb of jal278/lmx at commit b8fec5dd16451cf7afb6e02ac621b71d5bdfee7f (MIT). The target is the Woody Allen quote (negative_quotes[2] of the notebook): change its sentiment to positive while staying close to it.
- Fitness and niches follow the notebook's mapelites_sentiment_fitness and map_elites1d: positive-class probability of cardiffnlp/twitter-roberta-base-sentiment-latest (revision 3216a57f), Euclidean distance between all-MiniLM-L6-v2 embeddings (revision 1110a243) clipped to [0, 1.25] and cut into 30 bins, QD score = sum of the best fitness per occupied bin. The paper's text describes the span as 0 to 1.5; the pinned notebook code (1.25) is followed.
- Reference check (scripts/check_lmx_reference.py, receipt validation/lmx-reference-check.json): for the 49 sentences of the paper's printed Pareto fronts (Tables 3-5), the pinned models reproduce the printed positivity for 48 (largest difference 0.0017) and, for 43 (largest difference 0.019), the printed distance read as the Euclidean distance clipped at 1.25 and divided by 1.25 (inferred from the notebook's niche axis; the paper's text describes a 0-1.5 span); notebook-style calls (transformers sentiment pipeline with top_k=3, SentenceTransformer.encode) agree with the local functions within 6e-8. The paper gives no model revisions.
- Candidate: the paper runs MAP-Elites with LMX crossover (Pythia-1.4B, 2,500 evaluations) and reports the final QD score; here the candidate returns its archive directly (up to 300 one-line sentences), so the outer program search plays the role of LMX. Candidates may call the optional OpenRouter client. Not verified: the paper's QD-score curves were not reproduced.
- The optional generator defaults to google/gemini-2.5-flash, not the paper's Pythia-1.4B, and there is no fitness-evaluation budget comparable to the paper's 2,500 evaluations (the candidate may call the local scoring models itself within the timeout). The paper seeds every run with LMX offspring of all three quotes; the local candidate receives only the target quote.
- The two scoring models run locally on the CPU from pinned revisions fetched by `python scripts/prepare_llm_bench.py --models` (about 600 MB, SHA-256 checked); scoring is deterministic for a fixed archive.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:torch`
- `py:transformers`
- `py:sentence_transformers`
- `model:cardiffnlp/twitter-roberta-base-sentiment-latest@3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7`
- `model:sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`

## Sources

- paper: https://arxiv.org/pdf/2302.12170 (Section 4.3 and Appendix C.1 (Figure 16, Table 5))
- repository: https://github.com/jal278/lmx/tree/b8fec5dd16451cf7afb6e02ac621b71d5bdfee7f (scripts/lmx_sentiment_demo.ipynb)
- repository: https://github.com/jal278/lmx/blob/b8fec5dd16451cf7afb6e02ac621b71d5bdfee7f/scripts/lmx_sentiment_demo.ipynb (mapelites_sentiment_fitness and map_elites1d)
- repository: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest/tree/3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7 (sentiment classifier (fitness))
- repository: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 (sentence embedder (niche distance))

## Limitations

- Upstream commands and dependencies were inspected; costly training, paid APIs, and hardware experiments were not executed.
