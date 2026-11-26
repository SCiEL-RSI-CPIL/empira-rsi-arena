"""Train a candidate CLRS processor for a fixed budget and predict the out-of-distribution test outputs.

Runs inside the sandboxed child process (clrs_worker.py). ``train_and_predict``
builds the CLRS network of :mod:`clrs_net` around ``build_processor(hidden)``,
trains it on the CLRS-30 train split (16 nodes) with the single-task
hyper-parameters of Ibarz et al. (2022) for ``steps`` Adam updates, scores the
validation split (16 nodes) every ``eval_every`` updates, restores the best
validation checkpoint and returns compact hard predictions (clrs_metrics) for
the test inputs. The test outputs never reach this process; the parent scores
the predictions.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from . import clrs_data as D
from . import clrs_metrics as M
from . import clrs_net
from . import clrs_specs as S
from .runtime import CandidateError, require

DEFAULTS = {"hidden": 128, "steps": 300, "batch_size": 32, "learning_rate": 1e-3, "grad_clip": 1.0,
            "eval_every": 50, "eval_batch_size": 32, "seed": 0, "threads": 4, "train_threads": None,
            "max_processor_parameters": 1_000_000, "checkpoint_steps": False}


def as_tensors(batch, torch):
    inputs = {p.name: torch.from_numpy(p.data) for p in batch.inputs}
    first = {p.name: torch.from_numpy(p.data[0]) for p in batch.hints}
    hints = {p.name: torch.from_numpy(p.data) for p in batch.hints}
    outputs = {p.name: torch.from_numpy(p.data) for p in batch.outputs}
    return inputs, first, hints, outputs, torch.from_numpy(batch.lengths)


def split_output(out, node_fts, torch):
    """Accept ``node`` or ``(node, edge_or_None)``; enforce float32 (B,N,H) and (B,N,N,H) shapes."""
    if isinstance(out, torch.Tensor):
        node, edge = out, None
    elif isinstance(out, (tuple, list)) and len(out) == 2:
        node, edge = out
    else:
        raise CandidateError("processor must return node embeddings or a (node_embeddings, edge_embeddings_or_None) pair")
    b, n, h = node_fts.shape
    require(isinstance(node, torch.Tensor) and tuple(node.shape) == (b, n, h) and node.dtype == torch.float32,
            f"node embeddings must be a float32 tensor of shape {(b, n, h)}")
    require(edge is None or (isinstance(edge, torch.Tensor) and tuple(edge.shape) == (b, n, n, h) and edge.dtype == torch.float32),
            f"edge embeddings must be None or a float32 tensor of shape {(b, n, n, h)}")
    return node, edge


def probe_processor(processor, hidden, torch):
    """Check the processor contract on a small random graph; return True when it emits edge embeddings."""
    generator = torch.Generator().manual_seed(1234)
    b, n, h = 2, 5, hidden
    args = (torch.randn(b, n, h, generator=generator), torch.randn(b, n, n, h, generator=generator),
            torch.randn(b, h, generator=generator), torch.ones(b, n, n), torch.zeros(b, n, h))
    was_training = processor.training
    processor.eval()
    try:
        with torch.no_grad():
            node, edge = split_output(processor(*args), args[0], torch)
    except CandidateError:
        raise
    except Exception as error:  # noqa: BLE001 - any failure of candidate code is a contract violation
        raise CandidateError(f"processor forward failed on a 5-node probe graph: {type(error).__name__}: {error}") from None
    finally:
        processor.train(was_training)
    require(bool(torch.isfinite(node).all()) and (edge is None or bool(torch.isfinite(edge).all())),
            "processor returned non-finite values on the probe graph")
    return edge is not None


class CheckedProcessor:
    """Calls the candidate and enforces the output contract (shapes, float32, finite values) at every step.

    Finiteness is checked here and not only on the probe graph: a processor that is finite on the 5-node
    probe and the 16-node training graphs but not on the larger test graphs would otherwise be scored on
    the argmax of non-finite logits instead of being rejected.
    """

    def __init__(self, processor, torch):
        self.processor, self.torch = processor, torch

    def __call__(self, node_fts, edge_fts, graph_fts, adj, hidden):
        node, edge = split_output(self.processor(node_fts, edge_fts, graph_fts, adj, hidden), node_fts, self.torch)
        finite = self.torch.isfinite
        require(bool(finite(node).all()) and (edge is None or bool(finite(edge).all())),
                f"processor returned non-finite values on a {node_fts.shape[1]}-node graph")
        return node, edge


def count_parameters(processor, max_parameters, torch, before_first_call=False):
    """Number of parameters registered on the processor, checked against the cap and the float32 rule.

    A module can create parameters in ``forward`` (torch.nn.LazyLinear, or by hand), which a count
    taken right after ``build_processor`` misses. The cap is therefore checked three times: before
    the first call (parameters that are still uninitialised are skipped there), after the probe call
    and after training and prediction.
    """
    parameters = list(processor.parameters())
    if before_first_call:
        parameters = [p for p in parameters if not isinstance(p, torch.nn.parameter.UninitializedParameter)]
    try:
        count = sum(p.numel() for p in parameters)
    except Exception as error:  # noqa: BLE001 - a parameter that is still uninitialised after a call
        raise CandidateError(f"processor parameters cannot be counted: {type(error).__name__}: {error}") from None
    require(count <= max_parameters, f"processor has {count} parameters; at most {max_parameters} are allowed")
    require(all(p.dtype == torch.float32 for p in parameters), "processor parameters must be float32")
    return count


def build_network(build_processor, algorithm, train, hidden, max_parameters, torch):
    require(callable(build_processor), "the program must define build_processor(hidden_dim)")
    try:
        processor = build_processor(hidden)
    except CandidateError:
        raise
    except Exception as error:  # noqa: BLE001
        raise CandidateError(f"build_processor raised {type(error).__name__}: {error}") from None
    require(isinstance(processor, torch.nn.Module), "build_processor must return a torch.nn.Module")
    count_parameters(processor, max_parameters, torch, before_first_call=True)
    edge_output = probe_processor(processor, hidden, torch)
    processor_parameters = count_parameters(processor, max_parameters, torch)
    sample = D.make_batch(train, algorithm, [0])
    dims = {p.name: p.data.shape[-1] for p in sample.inputs + sample.hints + sample.outputs if p.type == S.CATEGORICAL}
    net = clrs_net.Net(S.model_spec(algorithm), dims, hidden, processor, edge_output)
    net.call_processor = CheckedProcessor(processor, torch)
    return net, processor_parameters, edge_output


def batch_loss(net, spec, tensors, nodes, generator):
    inputs, first, hints, outputs, lengths = tensors
    out_preds, hint_steps = net(inputs, first, lengths, nodes, int(lengths.max()) - 1, generator)
    loss = sum(clrs_net.output_loss(spec[name][2], truth, out_preds[name], nodes) for name, truth in outputs.items())
    for name, truth in hints.items():
        loss = loss + clrs_net.hint_loss(spec[name][2], truth, [step[name] for step in hint_steps], lengths, nodes)
    return loss


def compact_outputs(algorithm, preds):
    """clrs decoders.postprocess(hard=True) of output logits -> compact integer predictions (clrs_metrics)."""
    result = {}
    for name, _, typ in M.outputs(algorithm):
        value = preds[name]
        if typ == S.SHOULD_BE_PERMUTATION:
            perm = clrs_net.log_sinkhorn(value, 50, 0.1, True).argmax(-1).numpy()
            result[name] = M.fuse_permutation(perm, preds[name + "_mask"].argmax(-1).numpy())
        elif typ == S.MASK:
            result[name] = (value > 0.0).numpy().astype(np.int8)
        elif typ in (S.MASK_ONE, S.CATEGORICAL, S.POINTER):
            result[name] = value.argmax(-1).numpy()
        else:
            raise ValueError(f"unsupported output type {typ}")
    return {name: np.asarray(value, dtype=M.COMPACT_DTYPES[S.SPECS[algorithm][name][2]]) for name, value in result.items()}


def predict(net, split, algorithm, batch_size, torch):
    """Compact hard predictions for every sample of a split, in the split's sample order."""
    count = len(split["lengths"])
    result = {}
    net.eval()
    try:
        with torch.no_grad():
            for chunk in D.length_batches(split["lengths"], np.arange(count), batch_size):
                batch = D.make_batch(split, algorithm, chunk, with_outputs=False)
                inputs, first, _, _, lengths = as_tensors(batch, torch)
                out, _ = net(inputs, first, lengths, batch.nb_nodes, int(lengths.max()) - 1, None, keep_hints=False)
                for name, value in compact_outputs(algorithm, out).items():
                    if name not in result:
                        result[name] = np.zeros((count,) + value.shape[1:], dtype=value.dtype)
                    result[name][chunk] = value
    finally:
        net.train()
    return result


def train_and_predict(build_processor, algorithm, data, **budget):
    """Train on data['train'], select on data['val'], predict data['test'] -> (predictions, details)."""
    unknown = set(budget) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown budget keys {sorted(unknown)}")
    P = {**DEFAULTS, **budget}
    # Training and validation run on small (16-node) graphs, where extra threads barely help; the larger test
    # graphs are predicted with all `threads`.
    torch.set_num_threads(int(P["train_threads"] or P["threads"]))
    torch.manual_seed(P["seed"])
    rng = np.random.RandomState(P["seed"])
    started = time.time()
    train, val, test = data["train"], data["val"], data["test"]
    val_truth = M.truth_arrays(algorithm, val)
    net, processor_parameters, edge_output = build_network(build_processor, algorithm, train, P["hidden"],
                                                           P["max_processor_parameters"], torch)
    net.checkpoint_steps = bool(P["checkpoint_steps"])
    parameters = list(net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=P["learning_rate"])
    generator = torch.Generator().manual_seed(P["seed"] + 1)
    best, losses, curve, val_seconds = (-math.inf, None, -1), [], [], 0.0
    try:
        for step in range(P["steps"]):
            indices = rng.choice(len(train["lengths"]), P["batch_size"], replace=True)
            batch = D.make_batch(train, algorithm, indices, rng=rng)
            loss = batch_loss(net, net.spec, as_tensors(batch, torch), batch.nb_nodes, generator)
            require(bool(torch.isfinite(loss)), f"training loss became non-finite at step {step}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, P["grad_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % P["eval_every"] == 0 or step == P["steps"] - 1:
                tick = time.time()
                val_score = M.score(algorithm, predict(net, val, algorithm, P["eval_batch_size"], torch), val_truth)[0]
                val_seconds += time.time() - tick
                curve.append([step, round(val_score, 6)])
                if val_score > best[0]:
                    best = (val_score, copy.deepcopy(net.state_dict()), step)
        trained = time.time()
        net.load_state_dict(best[1])
        torch.set_num_threads(int(P["threads"]))
        predictions = predict(net, test, algorithm, P["eval_batch_size"], torch)
        processor_parameters = max(processor_parameters,
                                   count_parameters(net.processor, P["max_processor_parameters"], torch))
    except (CandidateError, TimeoutError, KeyboardInterrupt, MemoryError):
        raise
    except Exception as error:  # noqa: BLE001 - failures inside candidate code during training or prediction
        raise CandidateError(f"training or prediction failed: {type(error).__name__}: {error}") from None
    finished = time.time()
    return predictions, {
        "best_val_score": best[0], "best_val_step": best[2], "val_curve": curve,
        "final_train_loss": float(np.mean(losses[-10:])), "steps": P["steps"], "batch_size": P["batch_size"],
        "hidden": P["hidden"], "processor_parameters": processor_parameters,
        "model_parameters": sum(p.numel() for p in parameters), "edge_embeddings": edge_output,
        "train_seconds": round(trained - started - val_seconds, 2), "val_seconds": round(val_seconds, 2),
        "test_seconds": round(finished - trained, 2), "threads": int(P["threads"]),
        "train_threads": int(P["train_threads"] or P["threads"]),
    }
