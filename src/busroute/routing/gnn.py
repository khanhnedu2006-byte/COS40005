"""Bộ mã hoá đồ thị (GNN) và đầu policy/value cho PPO."""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphEncoder(nn.Module):
    """Encode node features using a stack of GraphSAGE convolutions.

    Args:
        in_dim: Input features per node.
        hidden_dim: Output features per node.
        num_layers: Number of graph convolution layers.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(SAGEConv(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return node embeddings for features [N, F] and edge indices [2, E]."""
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        return x


class PolicyValueHead(nn.Module):
    """Score candidate edges and estimate the value of the current state.

    Args:
        hidden_dim: Node embedding width and hidden layer width.
        edge_feat_dim: Number of features per candidate edge.
        global_dim: Number of global state features.
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int, global_dim: int) -> None:
        super().__init__()
        d = 2 * hidden_dim + edge_feat_dim + global_dim
        self.actor = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, cur_emb: torch.Tensor, cand_embs: torch.Tensor,
        cand_feats: torch.Tensor, global_feats: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked logits [K] and value [1] for K candidate edges.

        Node embeddings are [H] and [K, H], edge features are [K, E],
        global features are [G], and mask is boolean [K]. The caller must
        handle an all-false mask before sampling an action.
        """
        k = cand_embs.shape[0]
        z = torch.cat(
            [cur_emb.expand(k, -1), cand_embs, cand_feats, global_feats.expand(k, -1)],
            dim=-1,
        )
        logits = (
            self.actor(z).squeeze(-1).masked_fill(~mask, float("-inf"))
        )  # action mask
        value = self.critic(torch.cat([cur_emb, global_feats], dim=-1))
        return logits, value
