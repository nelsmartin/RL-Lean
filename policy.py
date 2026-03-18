import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEncoder(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, tokens):
        emb = self.embedding(tokens)   # [seq_len, dim]
        return emb.mean(dim=0)         # [dim]


class PolicyNetwork(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.encoder = SimpleEncoder(embed_dim=embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_tokens, action_tokens_list):
        state_vec = self.encoder(state_tokens)

        scores = []
        for action_tokens in action_tokens_list:
            action_vec = self.encoder(action_tokens)
            x = torch.cat([state_vec, action_vec], dim=-1)
            scores.append(self.fc(x))

        scores = torch.stack(scores).squeeze(-1)  # [num_actions]
        return F.softmax(scores, dim=0)
