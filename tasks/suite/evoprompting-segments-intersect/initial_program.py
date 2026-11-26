"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

import torch
from torch import nn


def linear(fan_in, fan_out, bias=0.0):
    """Linear layer with Haiku's default initialisation (truncated normal, std 1/sqrt(fan_in)) and a constant bias."""
    layer = nn.Linear(fan_in, fan_out)
    std = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
        layer.bias.fill_(bias)
    return layer


class TripletGMPNN(nn.Module):
    """Triplet-GMPNN processor of Ibarz et al. (2022), the baseline of EvoPrompting's CLRS search: a fully
    connected max-aggregation MPNN with a two-layer message MLP, layer norm, an update gate and triplet edge
    messages (8 triplet features)."""

    def __init__(self, hidden, nb_triplet_fts=8):
        super().__init__()
        h, t = hidden, nb_triplet_fts
        self.m_1, self.m_2 = linear(2 * h, h), linear(2 * h, h)
        self.m_e, self.m_g = linear(h, h), linear(h, h)
        self.msg_mlp = nn.Sequential(linear(h, h), nn.ReLU(), linear(h, h))
        self.o1, self.o2 = linear(2 * h, h), linear(h, h)
        self.ln = nn.LayerNorm(h)
        self.t_1, self.t_2, self.t_3 = linear(2 * h, t), linear(2 * h, t), linear(2 * h, t)
        self.t_e_1, self.t_e_2, self.t_e_3 = linear(h, t), linear(h, t), linear(h, t)
        self.t_g, self.o3 = linear(h, t), linear(t, h)
        self.gate1, self.gate2, self.gate3 = linear(2 * h, h), linear(h, h), linear(h, h, bias=-3.0)

    def triplet_messages(self, z, edge_fts, graph_fts):
        # tri[b, i, j, k] = t1(z_i) + t2(z_j) + t3(z_k) + e1(e_ij) + e2(e_ik) + e3(e_jk) + g(graph);
        # the maximum over i is taken over the terms that depend on i, then o3 maps to (batch, nodes, nodes, hidden).
        over_i = (self.t_1(z)[:, :, None, None] + self.t_e_1(edge_fts)[:, :, :, None]
                  + self.t_e_2(edge_fts)[:, :, None, :]).amax(dim=1)
        rest = (self.t_2(z)[:, :, None] + self.t_3(z)[:, None, :] + self.t_e_3(edge_fts)
                + self.t_g(graph_fts)[:, None, None])
        return torch.relu(self.o3(over_i + rest))

    def forward(self, node_fts, edge_fts, graph_fts, adj, hidden):
        z = torch.cat([node_fts, hidden], dim=-1)
        # msgs[b, i, j] is the message from node i to node j; max over all senders i (adj is not used: fully connected).
        msgs = (self.m_1(z)[:, None] + self.m_2(z)[:, :, None] + self.m_e(edge_fts)
                + self.m_g(graph_fts)[:, None, None])
        msgs = self.msg_mlp(torch.relu(msgs)).amax(dim=1)
        out = self.ln(torch.relu(self.o1(z) + self.o2(msgs)))
        gate = torch.sigmoid(self.gate3(torch.relu(self.gate1(z) + self.gate2(msgs))))
        out = out * gate + hidden * (1 - gate)
        return out, self.triplet_messages(z, edge_fts, graph_fts)


def build_processor(hidden_dim):
    return TripletGMPNN(hidden_dim)
# EVOLVE-BLOCK-END
