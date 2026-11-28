"""Reference CLRS processors in PyTorch: MPNN and Triplet-GMPNN (clrs/_src/processors.py, class PGN).

``TripletGMPNN(h)`` is the processor of Ibarz et al. (2022) that EvoPrompting uses
as its baseline (processor_type='triplet_gmpnn': fully connected max aggregation,
message MLP [h, h], layer norm, gating, 8 triplet features); ``TripletGMPNN(h,
triplets=False, gated=False)`` is the CLRS 'mpnn' processor. The seed programs of
the CLRS tasks contain the same code. Interface (shared by candidates)::

    forward(node_fts (B,N,H), edge_fts (B,N,N,H), graph_fts (B,H), adj (B,N,N), hidden (B,N,H))
        -> (new_hidden (B,N,H), edge_embeddings (B,N,N,H) or None)

The triplet maximum over the first node index is taken over the terms that depend
on it only, then the remaining terms are added; this equals the CLRS expression.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def linear(fan_in, fan_out, bias=0.0):
    """Haiku-style initialisation: truncated normal (2 std, std = 1/sqrt(fan_in)) weights, constant bias."""
    layer = nn.Linear(fan_in, fan_out)
    std = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
        layer.bias.fill_(bias)
    return layer


class TripletGMPNN(nn.Module):
    def __init__(self, hidden, nb_triplet_fts=8, triplets=True, gated=True, use_ln=True):
        super().__init__()
        h = hidden
        self.m_1, self.m_2 = linear(2 * h, h), linear(2 * h, h)
        self.m_e, self.m_g = linear(h, h), linear(h, h)
        self.msg_mlp = nn.Sequential(linear(h, h), nn.ReLU(), linear(h, h))
        self.o1, self.o2 = linear(2 * h, h), linear(h, h)
        self.ln = nn.LayerNorm(h) if use_ln else None
        self.triplets = triplets
        if triplets:
            t = nb_triplet_fts
            self.t_1, self.t_2, self.t_3 = linear(2 * h, t), linear(2 * h, t), linear(2 * h, t)
            self.t_e_1, self.t_e_2, self.t_e_3 = linear(h, t), linear(h, t), linear(h, t)
            self.t_g, self.o3 = linear(h, t), linear(t, h)
        self.gated = gated
        if gated:
            self.gate1, self.gate2, self.gate3 = linear(2 * h, h), linear(h, h), linear(h, h, bias=-3.0)

    def triplet_messages(self, z, edge_fts, graph_fts):
        """o3(max_i tri[b, i, j, k]) with tri = t1_i + t2_j + t3_k + e1_ij + e2_ik + e3_jk + g -> (B, N, N, H)."""
        tri_1, tri_2, tri_3 = self.t_1(z), self.t_2(z), self.t_3(z)
        tri_e_1, tri_e_2, tri_e_3 = self.t_e_1(edge_fts), self.t_e_2(edge_fts), self.t_e_3(edge_fts)
        over_i = (tri_1[:, :, None, None] + tri_e_1[:, :, :, None] + tri_e_2[:, :, None, :]).amax(dim=1)
        rest = tri_2[:, :, None] + tri_3[:, None, :] + tri_e_3 + self.t_g(graph_fts)[:, None, None]
        return torch.relu(self.o3(over_i + rest))

    def forward(self, node_fts, edge_fts, graph_fts, adj, hidden):
        z = torch.cat([node_fts, hidden], dim=-1)
        msgs = (self.m_1(z)[:, None] + self.m_2(z)[:, :, None] + self.m_e(edge_fts)
                + self.m_g(graph_fts)[:, None, None])  # msgs[b, i, j]: from node i to node j
        msgs = self.msg_mlp(torch.relu(msgs))
        msgs = msgs.amax(dim=1)  # MPNN: fully connected, so the adjacency mask is all ones
        out = torch.relu(self.o1(z) + self.o2(msgs))
        if self.ln is not None:
            out = self.ln(out)
        if self.gated:
            gate = torch.sigmoid(self.gate3(torch.relu(self.gate1(z) + self.gate2(msgs))))
            out = out * gate + hidden * (1 - gate)
        edges = self.triplet_messages(z, edge_fts, graph_fts) if self.triplets else None
        return out, edges


def mpnn(hidden):
    return TripletGMPNN(hidden, triplets=False, gated=False)


def triplet_gmpnn(hidden):
    return TripletGMPNN(hidden)
