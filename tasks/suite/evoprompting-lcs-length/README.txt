# LCS Length

**Paper:** EvoPrompting: Language Models for Code-Level Neural Architecture Search (https://arxiv.org/pdf/2302.14838)\
**Task id:** `evoprompting-lcs-length`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Write build_processor(hidden_dim), returning the graph-neural-network processor (a PyTorch module) that the CLRS encode-process-decode network uses to learn the LCS-length dynamic program of CLRS-30. The network is trained for 1500 steps on 16-node instances; the score is the accuracy of the predicted back-pointer table on the 32 out-of-distribution 64-node test instances (higher is better).

Benchmark named in the paper: CLRS. Paper metric: OOD accuracy and parameter count.

## Candidate interface

build_processor(hidden_dim: int) -> torch.nn.Module (CPU, float32), called once with hidden_dim = 128. The module's forward(node_fts, edge_fts, graph_fts, adj, hidden) receives float32 tensors node_fts (B, N, H), edge_fts (B, N, N, H), graph_fts (B, H), adj (B, N, N) (0/1 adjacency with self-loops built from the input edges and pointer inputs/hints) and hidden (B, N, H) (the processor's previous node output, zeros at the first step), H = hidden_dim, and returns either the new node embeddings (B, N, H) or a pair (node embeddings (B, N, H), edge embeddings (B, N, N, H) or None); outputs must be finite float32 tensors. It is called once per algorithm step (hint frame), on N = 16 nodes during training and N = 64 at test (string matchers 20 and 80, optimal BST 17 and 65, segments intersect 4 and 4), so it must work for any N. At most 1,000,000 processor parameters. The program may import only torch, math, numpy, functools, itertools, typing, dataclasses, collections and operator; eval/exec/open/getattr-style dynamic access and dunder attributes (other than __init__) are rejected, and training runs in a separate process that, where the kernel permits it, cannot read the task cache holding the prepared data and its test outputs (on Linux with unprivileged user namespaces it runs in its own user, mount, PID, network and IPC namespaces in which the task cache is empty and every file system is read-only except its working directory; on macOS under sandbox-exec), and in which an audit hook also blocks Python-level file, network and process access; the result field os_sandbox names what was in force (linux-namespaces, sandbox-exec or audit-hook-only). With audit-hook-only the hook is defence in depth for honest program search, not a security boundary: native readers such as torch.from_file, and frame references to the trainer, are not blocked. The evaluator owns the CLRS encoders, decoders, losses and training (Adam, learning rate 1e-3, batch 32 trajectories, gradient clipping 1.0, the task's fixed number of steps on the CLRS-30 train split of 16-node trajectories with hints, best validation checkpoint) and scores the out-of-distribution test split: the mean over the algorithm's output probes of the CLRS metric (pointer accuracy, mask F1 or one-hot accuracy) on hard predictions, in [0, 1]. The score depends only on the fixed step budget, not on wall time. Training and prediction run on 4 CPU threads under safety limits: they must finish within parameter wall_seconds (1700 s) and stay under parameter memory_mb (resident memory of the training process); a candidate that exceeds either is invalid. Higher is better

Parameters: `{"algorithm": "lcs_length", "data_sha256": "bf8fbb6bca496b85d881f6318f642616591db283351855d6c50eb55a2f019832", "steps": 1500, "eval_every": 50, "val_samples": 32, "test_samples": 32, "eval_batch_size": 8, "checkpoint_steps": false, "train_seeds": [0], "wall_seconds": 1700, "threads": 4, "memory_mb": 3584}`

## Labels

Scientific contribution: `engineering` (tier 3); artifact `model`, domain `machine_learning`, data `public_benchmark`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Source protocol: EvoPrompting (Chen, Dohan and So, NeurIPS 2023, Section 4.2) evolved only the triplet-message function of a Triplet-GMPNN (JAX/Haiku) on three CLRS algorithms (articulation points, Graham scan, Kruskal's MST) with search fitness -(model size x best validation error) after 2000 training steps (AdamW on one P100 GPU, otherwise the single-task hyper-parameters of Ibarz et al. 2022), and then reported the out-of-distribution (OOD) test accuracy of five discovered models on all 30 algorithms (Appendix A.6, Table 3). The paper ran no search on this algorithm; its Table 3 entry evaluates models found on the other three.
- Objective: the score is the OOD test score itself, with no model-size term (the processor is capped at 1,000,000 parameters instead), so an optimiser selects on the test split that is reported. The candidate may replace the whole processor, a superset of the paper's triplet-only search space.
- Data: CLRS-30 v1.0.0, the archive read by clrs.create_dataset, converted locally by scripts/prepare_clrs.py and locked by content digest in catalog/clrs-data-lock.json: train split of 1000 trajectories with 16 nodes (all hint frames), validation 32 of 32 samples (16 nodes), OOD test 32 of 32 samples (64 nodes). Ibarz et al. and dm-clrs 2.x clrs/examples/run.py (whose single-task settings EvoPrompting followed) instead sample fresh training trajectories of sizes 4 to 16 with varied edge probabilities on the fly and validate on fresh 16-node samples; only their test split is this archive's. The archive states no separate data licence (the CLRS code is Apache-2.0); the files are generated locally and never redistributed.
- Noise: CLRS-30 v1.0.0 has only 32 OOD test samples for each graph algorithm (32 x 32 or 32 x 64 only for Kadane, quickselect, minimum, binary search, the string matchers and segments intersect), and Table 3 gives +- spreads of up to about 6 points for the Triplet-GMPNN (the paper does not say whether they are standard deviations or standard errors, nor over how many training runs); a single training run here is deterministic (training seed 0, 4 threads) but a different seed or a small architecture change can move the score by several points.
- Budget: 1500 Adam updates (learning rate 1e-3, batch 32 trajectories, gradient clipping 1.0, hidden size 128), validation every 50 updates with the best checkpoint restored, on the CPU with 4 threads. The score depends only on this step budget: the step count was chosen per algorithm from measured costs so that the seed's evaluation takes roughly 3 to 7 minutes on 4 threads of a quiet Apple M4 Max (2000 at most), and the 1700 s wall-clock limit (inside the 1800 s task timeout) is only a safety net for slow machines and slow candidates. Ibarz et al. train for 10,000 steps and EvoPrompting's search for 2000 (AdamW), so scores here are not comparable with the published GPU numbers and reward architectures that learn fast; for this algorithm the seed nevertheless comes close to them (seed 0.8368 against 86.05 % for EvoPrompting's Triplet-GMPNN and 80.51 % for Ibarz et al.).
- Test protocol as in CLRS: the model reads the inputs and the first hint frame and is unrolled for the ground-truth number of steps of each test trajectory; hard output predictions are scored with clrs evaluation.evaluate (pointer accuracy, mask F1 or one-hot accuracy, averaged over the output probes; permutation outputs decoded with 50 Sinkhorn iterations and scored as predecessor pointers).
- Reference checks: with the seed (the Triplet-GMPNN baseline) the PyTorch network has 270,419 parameters, exactly EvoPrompting's reported baseline model size for this algorithm (Table 3). Verification run (2026-10-11, 4 threads on an Apple M4 Max shared with other jobs, 1-minute load average 9.6): the seed scores 0.8368 in 619 s, and a processor without message passing (each node updated from its own features only) scores 0.8129 under the same budget. The margin over that processor (0.0239) is small, and its variation across training seeds was not measured. Table 3 lists 86.05% +- 0.65 for EvoPrompting's Triplet-GMPNN, 80.51% +- 1.84 for Ibarz et al. (2022) and 85.75% +- 0.80 for the best model (Baseline, 270,419 parameters); the paper states neither the training length nor the hardware behind the Table 3 values (its search trained each child model for 2000 steps on one P100 GPU), and the paper's numbers were not reproduced.
- Implementation: a PyTorch port of the dm-clrs 2.0.3 encode-process-decode network (encoders, decoders, losses, Sinkhorn permutation decoding, Triplet-GMPNN) with the single-task options of Ibarz et al. (2022) used by EvoPrompting (random pos, pred_h as input, permutation outputs, soft hint re-prediction without teacher forcing, hidden size 128); it was reviewed against the dm-clrs source but not numerically cross-checked against the original JAX/Haiku code. The training runs in a child process that receives the train split, the validation subset and only the inputs and first hint frames of the test subset; the test outputs are not sent to the child, and the task cache that holds the prepared data file is hidden from the child by the kernel where the platform permits it (Linux user namespaces, macOS sandbox-exec; result field os_sandbox), so a native reader finds no file there and an audit reproduction that scored 1.0 by reading it with torch.from_file and replacing the trainer's prediction through frame references now scores what its processor earns. Where neither mechanism is available (os_sandbox audit-hook-only: containers that forbid user namespaces, Linux before 5.12) that attack still works, and copies of the data outside the task cache are not hidden on any platform. The static source guard and the child's file, process and network policy are defence in depth for honest program search, not a security boundary: candidate code runs in the same child process as the trainer.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:torch`
- `file:data/clrs/lcs_length.npz`

## Sources

- paper: https://arxiv.org/pdf/2302.14838 (Section 4.2 and Appendix Table 3)
- repository: https://github.com/google-deepmind/clrs (dm-clrs 2.0.3 (Apache-2.0): probe specs, encoders, decoders, Triplet-GMPNN processor, losses and evaluation ported to PyTorch; README links the CLRS-30 archive)
- benchmark: https://storage.googleapis.com/dm-clrs/CLRS30_v1.0.0.tar.gz (CLRS-30 v1.0.0 dataset (85,628,509 bytes, SHA-256 7743b2f1...): train, validation and test splits of every algorithm; no separate data licence stated)
- paper: https://arxiv.org/abs/2205.15659 (CLRS benchmark: algorithms, probes, OOD protocol and metric (Velickovic et al., ICML 2022))
- paper: https://arxiv.org/abs/2209.11142 (Triplet-GMPNN processor and single-task training setup (Ibarz et al., LoG 2022))

## Limitations

- No official runnable repository was verified from the paper/project sources examined.
- Upstream commands and dependencies were inspected; costly training, paid APIs, and hardware experiments were not executed.
- No publicly accessible, paper-linked implementation with a verified immutable revision was established.
