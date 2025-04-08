import torch
import polars as pl
import numpy as np
from torch import nn
import itertools as it
import torch.nn.functional as F
from model import Autoencoder
from rich.progress import track
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

tokens = pl.scan_parquet("tokens.parquet").select("tokens", "label").explode("tokens").collect()

from xcodec2.vq.codec_decoder_vocos import CodecDecoderVocos

codec_state = dict()
ckpt = torch.load("model.ckpt", map_location="cpu", weights_only=False)
ckpt = ckpt["state_dict"]
for key, value in ckpt.items():
    if key.startswith('generator.'):
        new_key = key[len('generator.'):]
        codec_state[new_key] = value
codec = CodecDecoderVocos()
codec.load_state_dict(codec_state)
codec.eval()

embed = codec.quantizer.get_output_from_indices(torch.arange(65536)[:, None]).cuda()

def loader(tokens, bsize):
    for i in it.count(step=bsize):
        select = np.arange(i, i + bsize) % tokens.height
        chunk = torch.from_numpy(tokens[select].to_numpy().T).cuda()
        yield chunk

model = Autoencoder(embed.detach()).cuda()
model.train()
steps = 131072
optim = Adam(model.parameters(), lr=1e-3)
sched = OneCycleLR(optim, 1e-3, total_steps=steps)

def _logits(latents):
    return latents @ embed.detach().T # embed.weight.detach().T


def _recon_loss(t, l):
    return F.cross_entropy(_logits(model(t, l)), t.long())

def _cycle_loss(t, l):
    return F.cross_entropy(_logits(model(model(t, 1 - l), l)), t.long())

for t, l in track(it.islice(loader(tokens, 1024), steps), total=steps):
    optim.zero_grad()
    _logits(model(t, l))
    loss = 0.5 * _cycle_loss(t, l) + 0.5 * _recon_loss(t, l)
    # loss = torch.square(embed.weight[t] - model(t, l)).mean()
    loss.backward()
    optim.step()
    sched.step()
    # print(f"{inverse_rate:.3g}" + " " * 32)
    print(f"{loss:.3g}" + " " * 32)

