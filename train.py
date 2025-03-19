import torch
import polars as pl
import numpy as np
from torch import nn
import itertools as it
import torch.nn.functional as F
from rich.progress import track
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

tokens = pl.scan_parquet("tokens.parquet").select("tokens", "label").explode("tokens").collect()

def loader(tokens, bsize):
    for i in it.count(step=bsize):
        select = np.arange(i, i + bsize) % tokens.height
        chunk = torch.from_numpy(tokens[select].to_numpy().T).cuda()
        yield chunk

def mueller_hash(t):
    t = ((t >> 16) ^ t) * 0x45d9f3b
    t = ((t >> 16) ^ t) * 0x45d9f3b
    t = (t >> 16) ^ t
    return t

class BloomEmbed(nn.Module):
    def __init__(self, num, dim, k):
        super().__init__()
        self.vocab = nn.Linear(dim, num)
        self.num = num
        self.dim = dim
        self.k = k

    def forward(self, t):
        out = torch.zeros(t.shape + (self.dim,), device=t.device)
        for r in range(self.k):
            out += self.vocab.weight[mueller_hash(t + r) % self.num] / self.k


        return out

class Autoencoder(nn.Module):
    def __init__(self, vocab_size = 1024, model_dims = 512, bottleneck_factor = 2, hash_rounds = 2):
        super().__init__()
        self.emb = nn.Linear(model_dims, 65536)
        self.task_emb = nn.Linear(model_dims, 2)
        self.ffn = nn.Sequential(
            nn.Linear(model_dims * 2, model_dims // 2),
            nn.GELU(),
            nn.Linear(model_dims // 2, model_dims),
        )

    def forward(self, t, l):
        # content_emb = self.emb(t)
        content_emb = self.emb.weight[t]
        task_emb = self.task_emb.weight[l]
        x = torch.cat([content_emb, task_emb], dim=-1)
        o = self.ffn(x)
        return o

        

model = Autoencoder().cuda()
model.train()
embed = nn.Linear(512, 65536).cuda()
steps = 8192
optim = Adam(model.parameters(), lr=1e-3)
sched = OneCycleLR(optim, 1e-3, total_steps=steps)
for t, l in track(it.islice(loader(tokens, 1024), steps), total=steps):
    optim.zero_grad()
    logits = model(t, l) @ model.emb.weight.T # embed.weight.detach().T
    loss = F.cross_entropy(logits, t.long())
    # loss = torch.square(embed.weight[t] - model(t, l)).mean()
    loss.backward()
    optim.step()
    sched.step()
    print(f"{loss:.3f}" + " " * 32)

