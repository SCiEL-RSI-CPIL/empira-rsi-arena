# XLA/Enzyme-JAX e-graph extraction

**Paper:** Magellan: Autonomous Discovery of Novel Compiler Optimization Heuristics with AlphaEvolve (https://arxiv.org/pdf/2601.21096)\
**Task id:** `magellan-xla-enzyme-jax-e-graph-extraction`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

XLA/Enzyme-JAX e-graph extraction task evaluated in Magellan: Autonomous Discovery of Novel Compiler Optimization Heuristics with AlphaEvolve.

Benchmark named in the paper: XLA/Enzyme-JAX e-graph extraction. Paper metric: total extracted program cost.

## Candidate interface

extract(egraph) -> dict mapping e-class id (str) to the chosen e-node id (str). egraph is an egraph-serialize JSON object: nodes = {node id: {op, cost (float), eclass, children: [child e-node ids]}} where a child e-node id stands for that node's e-class, and root_eclasses = [e-class ids]. Every root and every e-class reachable from the roots through chosen e-nodes must be chosen, each chosen e-node must belong to its e-class and the chosen graph must be acyclic. Score (minimize): geometric mean over the e-graphs of the DAG cost (the chosen e-node cost of each reachable e-class, counted once) divided by a reference extractor's DAG cost; each extract call must finish within time_limit_s wall-clock seconds

Parameters: `{"graphs": ["vgg", "resnet50", "bert", "nasneta", "nasrnn"], "reference_name": "extraction-gym faster-greedy-dag extractor (commit 903ba0f818b50608fe20ae9e0f03c35cb27bc50a)", "reference_dag_cost": {"vgg": 4.850757016778516, "resnet50": 4.396866964176297, "bert": 0.82470399630256, "nasneta": 16.322293040515433, "nasrnn": 0.9633390190010687}, "bottom_up_dag_cost": {"vgg": 4.852382016833872, "resnet50": 4.396866964176297, "bert": 0.82470399630256, "nasneta": 18.80070003592664, "nasrnn": 0.9724250177441718}, "time_limit_s": 30.0}`

## Labels

Scientific contribution: `engineering` (tier 3); artifact `program`, domain `systems`, data `public_benchmark`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Clearly labelled proxy: Magellan extracts from e-graphs produced by Enzyme-JAX equality saturation of XLA programs; those e-graphs and Magellan's cost model are not public (the Enzyme-JAX equality-saturation artifact, Zenodo record 16916484, holds code only). This task uses the public Tensat e-graphs of extraction-gym (TASO tensor-graph rewrites of BERT, NASNet-A, NASRNN, ResNet-50 and VGG; the unpruned versions, which contain cycles) with their per-e-node costs, from github.com/egraphs-good/extraction-gym at commit 903ba0f818b50608fe20ae9e0f03c35cb27bc50a (MIT), fetched by scripts/prepare_or_csp.py and locked in catalog/or_csp-data-lock.json. Scores are not comparable with Magellan's results.
- Metric: Magellan minimises the total cost of the extracted program and reports a 7% improvement over manually designed extraction policies. Here the score is the geometric mean over the five e-graphs of candidate DAG cost / DAG cost of extraction-gym's faster-greedy-dag extractor (the best of its heuristic extractors on these e-graphs), so a value below 1 beats that heuristic; the reference costs come from running extraction-gym at the pinned commit, and its bottom-up costs are listed for context. Optimal (ILP) costs were not computed.
- Verification: check_extraction and dag_cost port ExtractionResult::check and ExtractionResult::dag_cost. On the 20 extractions of the bottom-up, faster-bottom-up, prio-queue and faster-greedy-dag extractors on the five e-graphs (choices dumped by a locally patched copy of extraction-gym), the port accepted every extraction and reproduced the Rust DAG cost bit-for-bit, and it rejects a missing root, a missing reachable e-class, an e-node outside its e-class and a cycle. The seed ports the bottom-up extractor and reproduces its choices exactly (score about 1.03).
- Each extract call has a 30 s wall-clock limit in the evaluator process; exceeding it makes the candidate invalid. The scorer checks and costs a freshly loaded copy of each e-graph and a plain copy of the returned mapping, so changing the e-graph it receives gains nothing; extract and the module's top-level code still run in the evaluator's process and can reach the scoring code there, so scores of untrusted candidates are not tamper-proof.
- The paper does not say whether the total cost it minimises is the cost of the extracted DAG or of the tree; scoring DAG cost here is an assumption.
- Validity check and DAG cost port extraction-gym's ExtractionResult::check and dag_cost (src/extract/mod.rs) and reproduce the Rust results bit-for-bit on extractions made by its own extractors.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/or_csp/egraph/tensat-vgg.json.gz`
- `file:data/or_csp/egraph/tensat-resnet50.json.gz`
- `file:data/or_csp/egraph/tensat-bert.json.gz`
- `file:data/or_csp/egraph/tensat-nasneta.json.gz`
- `file:data/or_csp/egraph/tensat-nasrnn.json.gz`

## Sources

- paper: https://arxiv.org/pdf/2601.21096 (Sections 3.1-3.3)
- benchmark: https://github.com/egraphs-good/extraction-gym/tree/903ba0f818b50608fe20ae9e0f03c35cb27bc50a (Tensat e-graphs, DAG-cost definition and the faster-greedy-dag reference extractor (proxy for Magellan's private e-graphs))

## Limitations

- No official public runnable implementation verified from the inspected paper. External benchmark code may exist separately.
- No publicly accessible, paper-linked implementation with a verified immutable revision was established.
