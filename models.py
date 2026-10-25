"""GPT-2 next-item policy, transformer reward model RM-A and batching helpers."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model

from common import BOS, N_SPECIAL, PAD
from sequences import context, pad


def gpt_config(vocab_size, cfg, max_len):
    d = cfg['dropout']
    return GPT2Config(vocab_size=vocab_size, n_positions=max_len + 1, n_embd=cfg['n_embd'],
                      n_layer=cfg['n_layer'], n_head=cfg['n_head'], resid_pdrop=d, embd_pdrop=d,
                      attn_pdrop=d, bos_token_id=BOS, eos_token_id=BOS, pad_token_id=PAD)


class Policy(nn.Module):
    """GPT-2 over item tokens; the item distribution is the next-token softmax."""

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.lm = GPT2LMHeadModel(conf)

    def forward(self, x):
        return self.lm(input_ids=x, attention_mask=(x != PAD).long()).logits

    def last_logits(self, x, lengths):
        h = self.lm.transformer(input_ids=x, attention_mask=(x != PAD).long()).last_hidden_state
        h = h[torch.arange(len(x), device=x.device), lengths - 1]
        return self.lm.lm_head(h)


class RewardModel(nn.Module):
    """RM-A: GPT-2 body (initialised from the policy) + MLP on (history, candidate).

    Two heads: logit of watched share and logit of "user returns within N days".
    """

    def __init__(self, conf, body_state=None, n_out=2):
        super().__init__()
        self.body = GPT2Model(conf)
        if body_state is not None:
            self.body.load_state_dict(body_state)
        d = conf.n_embd
        self.head = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, n_out))

    def _pair(self, h, e):
        return self.head(torch.cat([h, e, h * e], dim=-1))

    def positions(self, x):
        """Score every next token of a sequence batch: output [B, L-1, 2]."""
        h = self.body(input_ids=x, attention_mask=(x != PAD).long()).last_hidden_state
        return self._pair(h[:, :-1], self.body.wte(x[:, 1:]))

    def candidates(self, x, lengths, cand):
        """Score candidates [B, G] after each context: output [B, G, 2] logits."""
        h = self.body(input_ids=x, attention_mask=(x != PAD).long()).last_hidden_state
        h = h[torch.arange(len(x), device=x.device), lengths - 1]
        e = self.body.wte(cand)
        return self._pair(h[:, None].expand(-1, cand.shape[1], -1), e)


def mask_logits(logits, seen):
    """Forbid special tokens and already-watched items (seen: [B, H] padded with PAD)."""
    logits = logits.clone()
    logits[:, :N_SPECIAL] = float('-inf')
    logits.scatter_(1, seen, float('-inf'))
    return logits


def batch_inputs(data, users, ends, max_len, device, seen_cap=2000):
    ctx = [context(data, u, e, max_len) for u, e in zip(users, ends)]
    seen = [data['tok'][max(data['ptr'][u], e - seen_cap):e] for u, e in zip(users, ends)]
    seen = [s if len(s) else np.array([PAD]) for s in seen]
    x = torch.from_numpy(pad(ctx)).to(device)
    lengths = torch.tensor([len(c) for c in ctx], device=device)
    return x, lengths, torch.from_numpy(pad(seen)).to(device)


@torch.no_grad()
def recommend(model, data, users, ends, max_len, device, k=10, batch=512):
    """Top-k tokens per user from the masked next-item distribution."""
    model.eval()
    out = []
    for s in range(0, len(users), batch):
        x, lengths, seen = batch_inputs(data, users[s:s + batch], ends[s:s + batch], max_len, device)
        logits = mask_logits(model.last_logits(x, lengths), seen)
        out.append(logits.topk(k, dim=-1).indices.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, k), dtype=np.int64)


def load_policy(path, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = Policy(GPT2Config.from_dict(state['conf']))
    model.load_state_dict(state['model'])
    return model.to(device)


def save_policy(model, path, **extra):
    from common import atomic_save
    atomic_save({'conf': model.conf.to_dict(), 'model': model.state_dict(), **extra}, path)
