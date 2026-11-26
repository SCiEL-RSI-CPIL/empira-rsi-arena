# CIFAR-10 decision-based adversarial attack

**Paper:** L-AutoDA: Leveraging Large Language Models for Automated Decision-based Adversarial Attacks (https://arxiv.org/pdf/2401.15335)\
**Task id:** `l-autoda-cifar-10-decision-based-adversarial-attack`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Write a Python draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams) function that proposes the next point of an L2 decision-based random-walk attack on a CIFAR-10 ResNet-18; scored by the mean L2 distance between 20 correctly classified test images and their adversarial examples after 2,500 queries per image (lower is better).

Benchmark named in the paper: CIFAR-10 decision-based adversarial attack. Paper metric: mean/median L2 adversarial perturbation versus query count.

## Candidate interface

draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams) -> x_new. org_img (the original image) and best_adv_img (the current adversarial example, misclassified by the target) are float32 numpy arrays of shape (3, 32, 32) with values in [0, 1]; std_normal_noise is a float64 array of the same shape of standard normal noise; hyperparams is a float64 array of shape (1,) holding the image's step size s (0.05 at the start; after every query p <- 0.95 p + 0.05 [proposal accepted] and s <- s * f(p) ** 0.1 with f(p) = 0.5 + 2p for p < 0.25 and 5/6 + 2p/3 otherwise). Return x_new, an array of shape (3, 32, 32); it is clipped to [0, 1], the target model (ResNet-18 on CIFAR-10) is queried once, and x_new replaces best_adv_img when it is misclassified and closer to org_img in L2. The function is called once per image per query; it may keep state across calls. Only numpy and the standard library are available (the program runs in a sandbox without file or network access). Score: mean L2 distance between the originals and the final adversarial examples after the task's query budget per image on its fixed set of correctly classified CIFAR-10 test images (pixel values in [0, 1]); lower is better

Parameters: `{"set": "heldout", "images": 20, "queries": 2500, "seed": 0, "s0": 0.05, "threads": 4}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `machine_learning`, data `public_benchmark`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Task: L-AutoDA (Guo et al., GECCO 2024 Companion) evolves the proposal function draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams) of an L2 decision-based random-walk attack; the paper's target is a CIFAR-10 ResNet-18 (Dadalto's edadaltocg/resnet18_cifar10, used here at revision ccacbae365101b9bfa45e1f17f4acac30c2601b2, MIT), its evolution fitness is the mean L2 distance on the first eight CIFAR-10 test images with at most 8,000 queries, and its test protocol uses 100 images per class at 2,500, 5,000 and 10,000 queries (Table 2).
- Framework: the evaluator implements Algorithm 1 as in the pinned repository's EvoAttack (pgg3/L-AutoDA at f35ca7915c60425e884325f08e9203091c39da82, no licence file): all images advance in lockstep, one query per image per step, proposals clipped to [0, 1], a proposal is accepted when misclassified and closer in L2 (compared with the repository's float32 norm computation), p <- 0.95 p + 0.05 [accepted] and s <- s f(p)^0.1 with the paper's piecewise-linear f; s starts at 0.05 as in the repository's code (the paper's text gives 0.001 as the initial value; the parameter s0 selects it). Every accepted example is re-verified with the model at the end (one that no longer classifies as adversarial falls back to its starting distance).
- Verification: (a) the rebuilt model classifies 94.98% of the 10,000 CIFAR-10 test images correctly, the accuracy on the model card (0.9498); (b) with the same model, starting points, noise stream and the seed proposal function, the evaluator's loop and the repository's EvoAttack (its code run unchanged through foolbox 3.3.4, with numpy.random.normal replaced by the evaluator's noise stream) return byte-identical adversarial examples for 4 images x 200 queries and 8 images x 1,500 queries (validation/llm_tools-checks.json). The paper's attack-success numbers were not reproduced.
- Reduced budget (deviation): the default fitness is the mean L2 over 20 correctly classified held-out test images (2 per class, a seeded sample of test indices >= 8) at 2,500 queries per image, the paper's smallest test budget, instead of 1,000 images; the paper's evolution protocol (the first eight test images that the model classifies correctly, 8,000 queries) is available with the parameters set='evolution', images=8, queries=8000. Scores are not comparable with the paper's tables.
- Wall clock (deviation): the repository aborts an attack after 120 s of wall-clock time, which makes its fitness depend on hardware speed; there is no such abort here, the score depends only on the query budget.
- Target model (deviation): the pinned repository attacks RobustBench's 'Standard' WRN-28-10 instead of the paper's ResNet-18; the paper's model is used here, rebuilt as a ResNet-18 with a 3x3 stem and no max-pool (timm's layout, strict state-dict load) behind CIFAR normalisation (mean 0.4914/0.4822/0.4465, std 0.2023/0.1994/0.2010), run on the CPU in float32 with 4 threads and 8-image chunks.
- Starting points (deviation): the paper starts from images with incorrect labels; here each attacked image starts from the L2-nearest of 10 seeded test images that the model assigns to a different class (fixed when the data are prepared; these selections cost no queries). The repository instead searches a blended-uniform-noise starting point.
- Noise (deviation): std_normal_noise comes from numpy's default_rng(seed) in the sandboxed child rather than the unseeded global generator, so an evaluation is deterministic on a fixed machine; proposals within floating-point rounding of the decision boundary may be classified differently on other hardware or thread counts.
- Sandbox: draw_proposals runs in a resource-limited child without file or network access (numpy and the standard library only) and sees the original images, its own proposals and which of them were accepted, never the labels or the model.
- Data: CIFAR-10 test images come from the binary archive at https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz (no licence stated; the page asks users to cite Krizhevsky 2009), downloaded and SHA-256 checked by scripts/prepare_llm_tools.py autoda, which stores only the attacked images and their starting points; nothing is redistributed.
- The target model runs on the CPU with 4 threads; the score depends only on the query budget, never on wall-clock time. Model decisions are deterministic on a fixed machine; proposals within floating-point rounding of the decision boundary may be classified differently on other hardware. The candidate runs in a resource-limited child process without file, network or subprocess access (software_sandbox); task capabilities reach it only as proxy callables served by the evaluator.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:torch`
- `py:psutil`
- `file:data/llm_tools/autoda/resnet18_cifar10.bin`
- `file:data/llm_tools/autoda/images.npy`
- `file:data/llm_tools/autoda/attack_sets.json`

## Sources

- paper: https://arxiv.org/pdf/2401.15335 (Section 5.1)
- repository: https://github.com/pgg3/L-AutoDA/tree/f35ca7915c60425e884325f08e9203091c39da82
- paper: https://arxiv.org/abs/2401.15335 (Section 5.1 (setup), Algorithm 1, Table 2)
- repository: https://github.com/pgg3/L-AutoDA/tree/f35ca7915c60425e884325f08e9203091c39da82 (core/attacks/EVOAttack_template.py (EvoAttack loop, f(p), 120 s abort), core/search_utils/task.py (draw_proposals contract))
- repository: https://huggingface.co/edadaltocg/resnet18_cifar10/tree/ccacbae365101b9bfa45e1f17f4acac30c2601b2 (pytorch_model.bin (44,774,261 bytes), config.json (normalisation))
- benchmark: https://www.cs.toronto.edu/~kriz/cifar.html (CIFAR-10 binary version, test_batch.bin)

## Limitations

- The paper and project sources examined link no runnable official repository; the release was identified separately and is pinned (see below).
- Upstream commands and dependencies were inspected; costly training, paid APIs, and hardware experiments were not executed.
- The matching pgg3/L-AutoDA repository has no README/license in the inspected tree; run_attack.py defaults to BoundaryAttack and evaluates only one batch, so it is not a paper evaluation command without changes.
- Execution interface inspected in pinned source. Full paper experiments have not been reproduced here.
