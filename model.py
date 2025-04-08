import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, embed: torch.Tensor, model_dims = 512, bottleneck_factor = 2, hash_rounds = 2):
        super().__init__()
        self.emb = embed
        emb_dim = embed.shape[-1]
        self.task_emb = nn.Linear(emb_dim, 2)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim // 2),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim),
        )

    def forward(self, t, l):
        if t.dtype.is_floating_point:
            content_emb = t
        else:
            content_emb = self.emb[t]

        task_emb = self.task_emb.weight[l]
        x = torch.cat([content_emb, task_emb], dim=-1)
        o = self.ffn(x)
        return o


