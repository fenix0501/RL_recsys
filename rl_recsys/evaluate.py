"""Shared evaluation: relevance, predicted reward under RM-A and RM-B, behaviour, multi-step."""
from __future__ import annotations

import numpy as np
import torch

from rl_recsys.common import N_SPECIAL
from rl_recsys.metrics import behaviour, relevance, tokens_to_items
from rl_recsys.models import batch_inputs, mask_logits


@torch.no_grad()
def rm_a_lists(rm_a, data, users, ends, top, cfg, device, batch=512):
    """RM-A sigmoid outputs [N, k, 2] for every recommended item given the real history."""
    out = []
    for s in range(0, len(users), batch):
        x, lengths, _ = batch_inputs(data, users[s:s + batch], ends[s:s + batch], cfg['max_len'], device)
        cand = torch.from_numpy(top[s:s + batch]).to(device)
        out.append(torch.sigmoid(rm_a.candidates(x, lengths, cand)).cpu().numpy())
    return np.concatenate(out)


def rm_b_metrics(b, dur):
    # watch hours under the held-out model: the honest check for watch-time rewards
    return {'rm_b_wp': float(b.mean()), 'rm_b_watch_h': float((b * dur).mean())}


def realized_hours(data, split, top_items, cap=5.0):
    """Duration-weighted hits: total_dur (h, capped) of recommended items whose user-item
    aggregate has its last watch in the target week, 0 otherwise.

    An aggregate relevance-times-duration proxy that uses no reward model. KION keeps only
    the user-item aggregate, so this is the whole accumulated duration, not watching done in
    that week.
    """
    if getattr(split, '_real', None) is None:
        rows = split.target_rows
        users = np.searchsorted(data['ptr'], rows, side='right') - 1
        hours = np.minimum(data['dur'][rows] / 3600, cap)
        split._real = {(int(u), int(i)): float(h) for u, i, h in zip(users, data['item'][rows], hours)}
    real = split._real
    return np.array([sum(real.get((int(u), int(i)), 0.0) for i in rec)
                     for u, rec in zip(split.users, top_items)])


def evaluate_top(top, data, split, rm_a, rm_b, cfg, device, sft_top=None, k=10):
    """One row of the comparison table for top-k token lists on a split."""
    items = tokens_to_items(data, top)
    rel = relevance(items, split.targets, k)
    rel['real_h10'] = realized_hours(data, split, items[:, :k])
    a = rm_a_lists(rm_a, data, split.users, split.end, top, cfg, device)
    dur = np.minimum(data['item_meta']['dur_h'].to_numpy()[top - N_SPECIAL], cfg['rl']['r3_cap_hours'])
    row = {**rel.mean().to_dict(),
           'rm_a_wp': float(a[..., 0].mean()), 'rm_a_return': float(a[..., 1].mean()),
           'rm_a_watch_h': float((a[..., 0] * dur).mean()),
           **rm_b_metrics(rm_b.score_lists(split.cutoff, split.users, top), dur),
           **behaviour(data, top, sft_top)}
    return row, rel


@torch.no_grad()
def multistep(policy, rm_a, rm_b, data, split, cfg, device, n_users=2000, steps=5, batch=256):
    """Greedy rollout: each generated item is appended to the context as if watched.

    Per step we record RM-A on the generated context (what an RL objective would
    see), RM-A on the real context (no fake items), RM-B (real features only),
    popularity and whether the item is in the user's real next-week targets.
    Control: the same number of items ranked once from the unchanged real context
    (rank t of a single top-k), which separates context drift from the plain decline
    of hit rate with rank.
    """
    policy.eval()
    idx = np.arange(min(n_users, len(split.users)))
    users, ends = split.users[idx], split.end[idx]
    pop = np.log1p(data['item_meta']['count'].to_numpy())
    rows = []
    gen_all, stat_all = [], []
    for s in range(0, len(idx), batch):
        x, lengths, seen = batch_inputs(data, users[s:s + batch], ends[s:s + batch], cfg['max_len'], device)
        x0, l0 = x.clone(), lengths.clone()
        gens, a_fake = [], []
        static = mask_logits(policy.last_logits(x0, l0), seen).topk(steps, -1).indices
        stat_all.append(static.cpu().numpy())
        for _ in range(steps):
            logits = mask_logits(policy.last_logits(x, lengths), seen)
            nxt = logits.argmax(-1)
            a_fake.append(torch.sigmoid(rm_a.candidates(x, lengths, nxt[:, None])[:, 0, 0]).cpu().numpy())
            gens.append(nxt.cpu().numpy())
            seen = torch.cat([seen, nxt[:, None]], 1)
            x, lengths = append_token(x, lengths, nxt, cfg['max_len'] + 1)
        gen = np.stack(gens, 1)
        gen_all.append(gen)
        a_real = torch.sigmoid(rm_a.candidates(x0, l0, torch.from_numpy(gen).to(device))[..., 0]).cpu().numpy()
        rows.append((np.stack(a_fake, 1), a_real))
    gen = np.concatenate(gen_all)
    a_fake = np.concatenate([r[0] for r in rows])
    a_real = np.concatenate([r[1] for r in rows])
    b = rm_b.score_lists(split.cutoff, users, gen)
    items = tokens_to_items(data, gen)
    hit = np.array([[i in split.targets[j] for i in items[j]] for j in range(len(idx))], dtype=float)
    stat = tokens_to_items(data, np.concatenate(stat_all))
    hit_static = np.array([[i in split.targets[j] for i in stat[j]] for j in range(len(idx))], dtype=float)
    out = []
    for t in range(steps):
        out.append({'step': t + 1, 'rm_a_generated_ctx': float(a_fake[:, t].mean()),
                    'rm_a_real_ctx': float(a_real[:, t].mean()), 'rm_b': float(b[:, t].mean()),
                    'log_popularity': float(pop[gen[:, t] - N_SPECIAL].mean()), 'hit_rate': float(hit[:, t].mean()),
                    'hit_rate_static_rank': float(hit_static[:, t].mean())})
    return out


def append_token(x, lengths, tok, max_width):
    """Append one token per row (right-padded batch); drop the oldest token when full."""
    if x.shape[1] < max_width:
        x = torch.cat([x, torch.zeros_like(x[:, :1])], 1)
    full = lengths >= x.shape[1]
    if full.any():
        x[full] = torch.roll(x[full], -1, dims=1)
        lengths = torch.where(full, lengths - 1, lengths)
    x[torch.arange(len(x), device=x.device), lengths] = tok
    return x, lengths + 1
