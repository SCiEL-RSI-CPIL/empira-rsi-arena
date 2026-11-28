"""L-AutoDA: evolved proposal functions for an L2 decision-based attack on a CIFAR-10 ResNet-18.

The candidate implements L-AutoDA's ``draw_proposals(org_img, best_adv_img,
std_normal_noise, hyperparams) -> x_new``.  The evaluator owns the random-walk
framework of the paper (Algorithm 1) and of the pinned repository's
``EvoAttack``: all attacked images advance in lockstep; at every step each
image gets one proposal (clipped to [0, 1]), the target model is queried once
per proposal, a proposal replaces the current adversarial example when it is
misclassified and closer to the original, and the per-image step size follows
``p <- 0.95 p + 0.05 [improved]``, ``s <- s * f(p) ** 0.1`` with f piecewise
linear (f(0) = 0.5, f(0.25) = 1, f(1) = 1.5).  The score is the mean L2
distance between the originals and the final adversarial examples (lower is
better), recomputed and re-verified with the target model by the evaluator.

The candidate runs in the sandbox child: it sees the original images (as the
attacker does in a decision-based attack), its own proposals and which of
them were accepted, never the labels or the model.  Noise comes from a
seeded generator, the budget is a fixed number of queries, and no wall-clock
limit enters the fitness.
"""
from __future__ import annotations

import base64
import json

import numpy as np

from . import llm_tools_common as common
from .runtime import CandidateError

SHAPE = (3, 32, 32)
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.201)
MODEL_FILE = "autoda/resnet18_cifar10.bin"

HARNESS = '''


def _artask_autoda_entry_(payload, oracle):
    import base64 as _b64
    import numpy as _np
    n, shape = int(payload["n"]), tuple(payload["shape"])

    def _decode(text):
        return _np.frombuffer(_b64.b64decode(text), dtype=_np.float32).reshape((n,) + shape).copy()

    originals, best = _decode(payload["originals"]), _decode(payload["starts"])
    rng = _np.random.default_rng(int(payload["noise_seed"]))
    step_sizes = _np.array(payload["s0"], dtype=_np.float64)
    function = globals().get("draw_proposals")
    if not callable(function):
        raise AttributeError("draw_proposals() is missing")
    for step in range(int(payload["steps"])):
        noise = rng.standard_normal((n,) + shape)
        proposals = _np.empty((n,) + shape, dtype=_np.float32)
        for i in range(n):
            out = _np.asarray(function(originals[i].copy(), best[i].copy(), noise[i].copy(), step_sizes[i:i + 1].copy()), dtype=_np.float64)
            if out.shape != shape:
                raise ValueError("draw_proposals returned shape %s, expected %s" % (out.shape, shape))
            proposals[i] = _np.clip(out, 0.0, 1.0)
        reply = oracle({"op": "query", "args": [step, _b64.b64encode(proposals.tobytes()).decode("ascii")], "kwargs": {}})
        accepted = _np.array(reply["accepted"], dtype=bool)
        best[accepted] = proposals[accepted]
        step_sizes = _np.array(reply["s"], dtype=_np.float64)
    return {"steps": int(payload["steps"])}
'''


def build_resnet18():
    """ResNet-18 with a 3x3 stem and no max-pool (the timm layout of edadaltocg/resnet18_cifar10)."""
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(cout)
            self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(cout)
            self.downsample = None
            if stride != 1 or cin != cout:
                self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return torch.relu(out + (x if self.downsample is None else self.downsample(x)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            widths, cin = (64, 128, 256, 512), 64
            for index, width in enumerate(widths, 1):
                stride = 1 if index == 1 else 2
                setattr(self, f"layer{index}", nn.Sequential(Block(cin, width, stride), Block(width, width, 1)))
                cin = width
            self.fc = nn.Linear(512, 10)
            self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1((x - self.mean) / self.std)))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            return self.fc(torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, 1), 1))

    return Net()


def load_model(threads=4):
    import torch
    torch.set_num_threads(int(threads))
    model = build_resnet18()
    state = torch.load(str(common.data_file(MODEL_FILE)), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def classify(model, images, batch=8):
    """Predicted labels of float32 images in [0, 1], shape (n, 3, 32, 32).

    Always in chunks of 8 images: on the CPU used for development the
    convolution path was several times faster per image below 16 images per
    batch, and a fixed chunking keeps the computation identical at every step.
    """
    import torch
    out = []
    with torch.inference_mode():
        for start in range(0, len(images), batch):
            out.append(model(torch.from_numpy(np.ascontiguousarray(images[start:start + batch], dtype=np.float32))).argmax(1).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def step_factor(p):
    """L-AutoDA's f(p): 0.5 + 2p below p = 0.25, 5/6 + 2p/3 above (f(0.25) = 1, f(1) = 1.5)."""
    return np.where(p < 0.25, 0.5 + 0.5 * p / 0.25, 1.0 + 0.5 * (p - 0.25) / 0.75)


def attack_set(name, count):
    """Original images (float32 in [0, 1]), labels and starting points of a prepared attack set."""
    meta = json.loads(common.data_file("autoda/attack_sets.json").read_text())
    if name not in meta["sets"]:
        raise CandidateError(f"unknown attack set {name!r}")
    chosen = meta["sets"][name][:count]
    if len(chosen) < count:
        raise CandidateError(f"attack set {name!r} has only {len(chosen)} images")
    images = np.load(common.data_file("autoda/images.npy"))
    as_float = lambda rows: images[rows].astype(np.float32) / np.float32(255.0)  # noqa: E731
    labels = np.array([row["label"] for row in chosen], dtype=np.int64)
    ids = {"test_indices": [row["index"] for row in chosen], "start_indices": [row["start_index"] for row in chosen]}
    return as_float([row["row"] for row in chosen]), labels, as_float([row["start_row"] for row in chosen]), ids


class Attack:
    """Evaluator side of the random walk: queries, acceptance, step sizes (the authority on every number)."""

    def __init__(self, model, originals, labels, starts, steps, s0=0.05):
        self.model, self.originals, self.labels, self.steps = model, originals, labels, int(steps)
        self.best = starts.copy()
        self.n = len(originals)
        self.p = np.zeros(self.n)
        self.s = np.full(self.n, float(s0))
        self.step = 0
        self.accepted = 0

    def distances(self, images):
        """L2 distances in float64 (reported scores)."""
        return np.linalg.norm((self.originals - images).reshape(self.n, -1).astype(np.float64), axis=1)

    def _closer(self, proposals):
        """The repository's acceptance test: float32 torch L2 norms, ((x0 - x).flatten(1) ** 2).sum(-1).sqrt()."""
        import torch
        originals = torch.from_numpy(self.originals)
        norm = lambda images: ((originals - torch.from_numpy(images)).flatten(1) ** 2).sum(dim=-1).sqrt()  # noqa: E731
        return (norm(proposals) < norm(self.best)).numpy()

    def query(self, step, encoded):
        if step != self.step or self.step >= self.steps:
            raise CandidateError("attack protocol violated (unexpected query)")
        if not isinstance(encoded, str):
            raise CandidateError("proposals must be transmitted as base64 float32")
        try:
            proposals = np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.float32).reshape(self.originals.shape).copy()
        except (ValueError, TypeError):
            raise CandidateError("malformed proposal batch") from None
        if not np.isfinite(proposals).all():
            raise CandidateError(f"draw_proposals returned a non-finite value at step {step}")
        np.clip(proposals, 0.0, 1.0, out=proposals)
        adversarial = classify(self.model, proposals) != self.labels
        improved = adversarial & self._closer(proposals)
        self.p = 0.95 * self.p + (1 - 0.95) * improved  # the repository's arithmetic (alpha_p = 0.95)
        self.s = self.s * np.power(step_factor(self.p), 0.1)
        self.best[improved] = proposals[improved]
        self.step += 1
        self.accepted += int(improved.sum())
        return {"accepted": improved.tolist(), "s": self.s.tolist()}


def evaluate(module, P):
    source = common.candidate_source(module, "draw_proposals")
    model = load_model(P.get("threads", 4))
    originals, labels, starts, ids = attack_set(P.get("set", "heldout"), int(P.get("images", 20)))
    if (classify(model, originals) != labels).any() or (classify(model, starts) == labels).any():
        raise RuntimeError("prepared attack set is inconsistent with the target model")
    attack = Attack(model, originals, labels, starts, int(P.get("queries", 2500)), float(P.get("s0", 0.05)))
    encode = lambda array: base64.b64encode(np.ascontiguousarray(array, dtype=np.float32).tobytes()).decode("ascii")  # noqa: E731
    payload = {"n": attack.n, "shape": list(SHAPE), "originals": encode(originals), "starts": encode(starts),
               "noise_seed": int(P.get("seed", 0)), "s0": attack.s.tolist(), "steps": attack.steps}
    ok, _, failure = common.call_candidate(source, "draw_proposals", [], handlers={"query": attack.query}, harness=HARNESS,
                                           harness_entry="_artask_autoda_entry_", payload=payload,
                                           cpu_seconds=int(P.get("cpu_seconds", 1200)), wall_seconds=int(P.get("wall_seconds", 1800)))
    if not ok:
        raise CandidateError(f"draw_proposals failed in the sandbox ({failure['kind']}): {failure['message']}")
    if attack.step != attack.steps:
        raise CandidateError(f"the attack stopped after {attack.step} of {attack.steps} steps")
    initial = attack.distances(starts)
    # Re-verification with the target model; an example that no longer classifies as adversarial (floating-point
    # nondeterminism at the decision boundary) falls back to its starting point's distance.
    reverified = classify(model, attack.best) != labels
    final = np.where(reverified, attack.distances(attack.best), initial)
    return float(final.mean()), {
        "images": attack.n, "queries_per_image": attack.steps, "median_l2": round(float(np.median(final)), 6),
        "initial_mean_l2": round(float(initial.mean()), 6), "success_rate_l2_below_0.5": round(float((final < 0.5).mean()), 4),
        "accepted_proposals": attack.accepted, "reverified_adversarial": int(reverified.sum()),
        "per_image_l2": [round(float(v), 6) for v in final], **ids}
