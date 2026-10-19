"""Bộ mã hoá đồ thị (GNN) và đầu policy/value cho PPO."""
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphEncoder(nn.Module):
    # Mã hoá mọi node một lần cho mỗi episode (đồ thị tĩnh), đặc trưng động đưa vào policy head
    def __init__(self, in_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(SAGEConv(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        return x


class PolicyValueHead(nn.Module):
    # Chấm điểm từng cạnh ứng viên: [emb node hiện tại, emb node đích, đặc trưng cạnh, đặc trưng toàn cục]
    def __init__(self, hidden_dim, edge_feat_dim, global_dim):
        super().__init__()
        d = 2 * hidden_dim + edge_feat_dim + global_dim
        self.actor = nn.Sequential(nn.Linear(d, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.critic = nn.Sequential(nn.Linear(hidden_dim + global_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, 1))

    def forward(self, cur_emb, cand_embs, cand_feats, global_feats, mask):
        k = cand_embs.shape[0]
        z = torch.cat([cur_emb.expand(k, -1), cand_embs, cand_feats, global_feats.expand(k, -1)], dim=-1)
        logits = self.actor(z).squeeze(-1).masked_fill(~mask, float("-inf"))   # action mask
        value = self.critic(torch.cat([cur_emb, global_feats], dim=-1))
        return logits, value
