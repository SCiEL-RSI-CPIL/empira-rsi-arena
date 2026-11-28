"""PyTorch port of the CLRS encode-process-decode network (clrs/_src/{encoders,decoders,nets,losses}.py).

Only the processor is supplied by the candidate. Everything else follows the
single-task setting of Ibarz et al. (2022) in dm-clrs 2.0.3: linear encoders of
inputs and hints summed into node/edge/graph features, the processor called once
per hint step on (node, edge, graph features, adjacency, previous hidden), the
CLRS decoders on [node features, hidden, new hidden], hints encoded and decoded
with soft re-prediction (no teacher forcing after the first step), permutation
outputs decoded with Sinkhorn iterations, and the CLRS losses. Linear layers use
Haiku's default initialisation; scalar hint encoders use the 'xavier_on_scalars'
option. Differences from the JAX code are numerical only (float32 kernels,
random streams) except that output predictions are decoded only at the steps
where some trajectory ends (the only ones whose predictions are kept). Training
can recompute every algorithm step in the backward pass (``checkpoint_steps``;
identical values and gradients, much less memory for long trajectories).
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import clrs_specs as S

BIG = 1e6


def hk_linear(fan_in, fan_out, std=None):
    """nn.Linear with Haiku's default init: N(0, std^2) truncated at 2 std (std = 1/sqrt(fan_in)), zero bias."""
    layer = nn.Linear(fan_in, fan_out)
    std = 1.0 / math.sqrt(fan_in) if std is None else std
    with torch.no_grad():
        nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
        layer.bias.zero_()
    return layer


def one_hot(index, nodes):
    return F.one_hot(index.long(), nodes).float()


def log_sinkhorn(x, steps, temperature, zero_diagonal, uniform=None):
    """CLRS log-space Sinkhorn normalisation; Gumbel noise from ``uniform`` (same shape as x) when given (training)."""
    if uniform is not None:
        x = x - torch.log(-torch.log(uniform + 1e-12) + 1e-12)
    x = x / temperature
    if zero_diagonal:
        x = x - BIG * torch.eye(x.shape[-1])
    for _ in range(steps):
        x = torch.log_softmax(x, dim=-1)
        x = torch.log_softmax(x, dim=-2)
    return x


class Encoder(nn.Module):
    """One input or hint probe (clrs encoders.construct_encoders / accum_*)."""

    def __init__(self, stage, loc, typ, dims, hidden):
        super().__init__()
        self.loc, self.typ = loc, typ
        std = 1.0 / math.sqrt(hidden) if (stage == S.HINT and typ == S.SCALAR) else None
        fan_in = dims if typ == S.CATEGORICAL else 1
        self.first = hk_linear(fan_in, hidden, std)
        self.second = hk_linear(1, hidden) if (loc == S.EDGE and typ == S.POINTER) else None

    def _apply(self, data):
        return self.first(data) if self.typ == S.CATEGORICAL else self.first(data.unsqueeze(-1))

    def forward(self, data, feats, nodes):
        """Add this probe to feats = [node, edge, graph, adjacency]; pointer data are indices or soft rows."""
        pointer = self.typ in (S.POINTER, S.PERMUTATION_POINTER)
        if pointer and not torch.is_floating_point(data):
            data = one_hot(data, nodes)
        if self.loc == S.NODE and pointer:
            feats[3] = ((feats[3] + ((data + data.transpose(1, 2)) > 0.5).float()) > 0).float()
            feats[1] = feats[1] + self._apply(data)
        elif self.loc == S.EDGE:
            if self.typ == S.MASK:
                feats[3] = ((feats[3] + ((data + data.transpose(1, 2)) > 0.0).float()) > 0).float()
            if self.typ == S.POINTER:  # mean over senders / receivers of the (B, N, N, N) pointer encoding
                w1, w2 = self.first.weight[:, 0], self.second.weight[:, 0]
                feats[1] = (feats[1] + data.mean(1).unsqueeze(-1) * w1 + self.first.bias
                            + data.mean(2).unsqueeze(-1) * w2 + self.second.bias)
            else:
                feats[1] = feats[1] + self._apply(data)
        elif self.loc == S.NODE:
            feats[0] = feats[0] + self._apply(data)
        elif self.loc == S.GRAPH and self.typ != S.POINTER:
            feats[2] = feats[2] + self._apply(data)
        else:
            raise ValueError(f"unsupported probe location/type {self.loc}/{self.typ}")
        return feats


class Decoder(nn.Module):
    """One output or hint probe (clrs decoders.construct_decoders / _decode_*_fts)."""

    def __init__(self, loc, typ, dims, hidden, edge_dim):
        super().__init__()
        self.loc, self.typ = loc, typ
        self.inner_checkpoint = True  # set by Net.forward: False when whole steps are checkpointed
        node_dim, out = 3 * hidden, (dims if typ == S.CATEGORICAL else 1)
        if loc == S.NODE and typ in (S.POINTER, S.PERMUTATION_POINTER):
            self.layers = nn.ModuleList([hk_linear(node_dim, hidden), hk_linear(node_dim, hidden),
                                         hk_linear(edge_dim, hidden), hk_linear(hidden, 1)])
        elif loc == S.NODE:
            self.layers = nn.ModuleList([hk_linear(node_dim, out)])
        elif loc == S.EDGE and typ == S.POINTER:
            self.layers = nn.ModuleList([hk_linear(node_dim, hidden), hk_linear(node_dim, hidden), hk_linear(edge_dim, hidden),
                                         hk_linear(node_dim, hidden), hk_linear(hidden, 1)])
        elif loc == S.EDGE:
            self.layers = nn.ModuleList([hk_linear(node_dim, out), hk_linear(node_dim, out), hk_linear(edge_dim, out)])
        elif loc == S.GRAPH and typ != S.POINTER:
            self.layers = nn.ModuleList([hk_linear(node_dim, out), hk_linear(hidden, out)])
        else:
            raise ValueError(f"unsupported probe location/type {loc}/{typ}")

    def forward(self, h_t, e_t, graph_fts, noise=None):
        L = self.layers
        if self.loc == S.NODE and self.typ in (S.POINTER, S.PERMUTATION_POINTER):
            p_1, p_2, p_3 = L[0](h_t), L[1](h_t), L[2](e_t)
            p_e = p_2.unsqueeze(-2) + p_3  # (B, N, N, H)
            p_m = torch.maximum(p_1.unsqueeze(-2), p_e.transpose(1, 2))
            preds = L[3](p_m).squeeze(-1)
            if self.typ == S.PERMUTATION_POINTER:
                preds = log_sinkhorn(preds, 10, 0.1, True, noise)
            return preds
        if self.loc == S.NODE:
            preds = L[0](h_t)
            return preds if self.typ == S.CATEGORICAL else preds.squeeze(-1)
        if self.loc == S.EDGE:
            pred = L[0](h_t).unsqueeze(-2) + L[1](h_t).unsqueeze(-3) + L[2](e_t)
            if self.typ == S.POINTER:
                return _edge_pointer_logits(pred, L[3](h_t), L[4], self.inner_checkpoint)
            return pred if self.typ == S.CATEGORICAL else pred.squeeze(-1)
        pred = L[0](h_t.max(dim=-2).values) + L[1](graph_fts)
        return pred if self.typ == S.CATEGORICAL else pred.squeeze(-1)


def _edge_pointer_block(pred, second, weight, bias):
    return F.linear(torch.maximum(pred.unsqueeze(-2), second[:, None, None]), weight, bias).squeeze(-1)


def _edge_pointer_logits(pred, second, final, use_checkpoint=True, block_elements=1 << 24):
    """max(pred[b,i,j], second[b,k]) -> Linear: (B, N, N, N) logits, in row blocks to bound memory.

    During training each block is recomputed in the backward pass (checkpointed) unless the
    whole algorithm step is already checkpointed (``use_checkpoint=False``)."""
    batch, nodes, _, hidden = pred.shape
    rows = max(1, block_elements // max(1, batch * nodes * nodes * hidden))
    pieces = []
    for start in range(0, nodes, rows):
        block = pred[:, start:start + rows]
        if use_checkpoint and torch.is_grad_enabled():
            pieces.append(checkpoint(_edge_pointer_block, block, second, final.weight, final.bias, use_reentrant=False))
        else:
            pieces.append(_edge_pointer_block(block, second, final.weight, final.bias))
    return torch.cat(pieces, dim=1)


def soft_hint(typ, preds):
    """clrs decoders.postprocess(hard=False) for hints fed back to the encoders."""
    if typ == S.MASK:
        return torch.sigmoid(preds)
    if typ in (S.MASK_ONE, S.CATEGORICAL, S.POINTER):
        return torch.softmax(preds, dim=-1)
    return preds


class Net(nn.Module):
    """Encoders + candidate processor + decoders, unrolled over the hint steps of a batch."""

    def __init__(self, spec, dims, hidden, processor, edge_output):
        super().__init__()
        self.spec, self.hidden, self.processor = spec, hidden, processor
        edge_dim = 2 * hidden if edge_output else hidden
        self.encoders = nn.ModuleDict({name: Encoder(stage, loc, typ, dims.get(name, 1), hidden)
                                       for name, (stage, loc, typ) in spec.items() if stage in (S.INPUT, S.HINT)})
        self.decoders = nn.ModuleDict({name: Decoder(loc, typ, dims.get(name, 1), hidden, edge_dim)
                                       for name, (stage, loc, typ) in spec.items() if stage in (S.OUTPUT, S.HINT)})
        self.call_processor = None  # optional wrapper that checks the processor contract at every step
        self.checkpoint_steps = False

    def step(self, static, hints, hidden, nodes, noise, decode_outputs):
        """One processor step: encode the current hints, run the processor, decode hints (and outputs)."""
        call = self.call_processor or self.processor
        feats = list(static)
        for name, data in hints.items():
            feats = self.encoders[name](data, feats, nodes)
        node_fts, edge_fts, graph_fts, adj = feats
        new_hidden, new_edge = call(node_fts, edge_fts, graph_fts, adj, hidden)
        h_t = torch.cat([node_fts, hidden, new_hidden], dim=-1)
        e_t = edge_fts if new_edge is None else torch.cat([edge_fts, new_edge], dim=-1)
        hint_preds, candidates = {}, {}
        for name, decoder in self.decoders.items():
            if self.spec[name][0] == S.HINT:
                hint_preds[name] = decoder(h_t, e_t, graph_fts, noise.get(name))
            elif decode_outputs:
                candidates[name] = decoder(h_t, e_t, graph_fts, noise.get(name))
        return new_hidden, hint_preds, candidates

    def forward(self, inputs, first_hints, lengths, nodes, steps, generator=None, keep_hints=True):
        """Return (final output predictions, list of hint predictions per step).

        inputs / first_hints: {name: tensor}; first_hints hold the ground-truth hints of
        frame 0 (the only ground-truth hints the model reads). steps = max(lengths) - 1.
        generator: Gumbel noise for permutation outputs (training). keep_hints=False drops
        the per-step hint predictions (evaluation). With ``checkpoint_steps`` set, each
        training step is recomputed in the backward pass instead of storing its
        activations (same values and gradients, less memory, more time).
        """
        batch = lengths.shape[0]
        step_checkpoints = bool(self.checkpoint_steps and torch.is_grad_enabled())
        for decoder in self.decoders.values():
            decoder.inner_checkpoint = not step_checkpoints
        static = [torch.zeros(batch, nodes, self.hidden), torch.zeros(batch, nodes, nodes, self.hidden),
                  torch.zeros(batch, self.hidden), torch.eye(nodes).expand(batch, nodes, nodes).clone()]
        for name, data in inputs.items():
            static = self.encoders[name](data, static, nodes)
        hidden = torch.zeros(batch, nodes, self.hidden)
        hints, outputs, hint_steps = first_hints, None, []
        permutations = [name for name, (stage, _, typ) in self.spec.items() if stage == S.OUTPUT and typ == S.PERMUTATION_POINTER]
        for step in range(max(1, steps)):
            # A trajectory of length L keeps the output prediction of step L - 2 (step 0 when L = 1), so outputs
            # are decoded only at steps where some trajectory ends; the result is identical.
            decode = step == 0 or bool((lengths == step + 2).any())
            noise = {}
            if generator is not None and decode:
                noise = {name: torch.rand(batch, nodes, nodes, generator=generator) for name in permutations}
            if step_checkpoints:
                new_hidden, hint_preds, candidates = checkpoint(self.step, static, hints, hidden, nodes, noise, decode,
                                                                use_reentrant=False)
            else:
                new_hidden, hint_preds, candidates = self.step(static, hints, hidden, nodes, noise, decode)
            if outputs is None:
                outputs = candidates
            elif candidates:
                outputs = {name: _keep_done(lengths, step, candidates[name], value) for name, value in outputs.items()}
            if keep_hints:
                hint_steps.append(hint_preds)
            hints = {name: soft_hint(self.spec[name][2], value) for name, value in hint_preds.items()}
            hidden = new_hidden
        return outputs, hint_steps


def _keep_done(lengths, step, new, old):
    """clrs nets._is_not_done_broadcast: trajectories still running take the new prediction."""
    running = (lengths > step + 1).float().view(-1, *([1] * (new.dim() - 1)))
    return running * new + (1.0 - running) * old


# --- losses (clrs/_src/losses.py, full-sample training) -------------------------------------

def _bce(logits, truth):
    return torch.clamp(logits, min=0) - logits * truth + torch.log1p(torch.exp(-logits.abs()))


def output_loss(typ, truth, pred, nodes):
    if typ == S.SCALAR:
        return ((pred - truth) ** 2).mean()
    if typ == S.MASK:
        mask = (truth != S.MASKED).float()
        return (_bce(pred, truth) * mask).sum() / mask.sum()
    if typ in (S.MASK_ONE, S.CATEGORICAL):
        masked_truth = truth * (truth != S.MASKED).float()
        return -(masked_truth * torch.log_softmax(pred, dim=-1)).sum() / (truth == 1).sum()
    if typ == S.POINTER:
        return -(one_hot(truth, nodes) * torch.log_softmax(pred, dim=-1)).sum(-1).mean()
    if typ == S.PERMUTATION_POINTER:
        return -(truth * pred).sum(-1).mean()
    raise ValueError(typ)


def hint_loss(typ, truth, preds, lengths, nodes):
    """truth: (T, B, ...) ground truth; preds: list of T-1 predictions for frames 1..T-1."""
    truth, pred = truth[1:len(preds) + 1], torch.stack(preds)
    if typ == S.SCALAR:
        loss, mask = (pred - truth) ** 2, None
    elif typ == S.MASK:
        loss, mask = _bce(pred, truth), (truth != S.MASKED).float()
    elif typ == S.MASK_ONE:
        loss, mask = -(truth * torch.log_softmax(pred, dim=-1)).sum(-1, keepdim=True), None
    elif typ == S.CATEGORICAL:
        loss, mask = -(truth * torch.log_softmax(pred, dim=-1)).sum(-1), (truth == 1).any(-1).float()
    elif typ == S.POINTER:
        loss, mask = -(one_hot(truth, nodes) * torch.log_softmax(pred, dim=-1)).sum(-1), None
    else:
        raise ValueError(typ)
    if mask is None:
        mask = torch.ones_like(loss)
    steps = torch.arange(loss.shape[0]).view(-1, 1)
    running = (lengths.view(1, -1) > steps + 1).float().view(*running_shape(loss))
    mask = mask * running
    return (loss * mask).sum() / torch.clamp(mask.sum(), min=1e-12)


def running_shape(loss):
    return (loss.shape[0], loss.shape[1]) + (1,) * (loss.dim() - 2)
