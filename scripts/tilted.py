#!/usr/bin/env python3
"""Closed-form KL-regularised optimum, no training: pi*(a|s) ∝ pi_ref(a|s) * exp(r(s,a) / beta).

Next-item recommendation is a one-step bandit, so the optimum of E[r] - beta * KL(pi || pi_ref)
can be computed exactly by scoring every item in the vocabulary with RM-A. Its top-10 is
argmax_a log pi_ref(a|s) + r(s,a) / beta. This is the ceiling any RL method with the same
reward and KL can reach, so comparing GRPO with it separates optimisation error from the
reward itself. beta here is in raw reward units (GRPO normalises advantages, so its beta is
not on the same scale; compare the frontiers, not the beta values).

    python scripts/tilted.py --out out_full
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from rl_recsys.common import N_SPECIAL, ROOT, log, pick_device

BETAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


@torch.no_grad()
def tilted_tops(sft, rm_a, data, split, cfg, device, hours, k=10, batch=32, rm_w=None, rm_w7=None):
    """Top-k of the tilted policy for every (reward, beta), plus beta=inf (SFT itself)."""
    from rl_recsys.models import batch_inputs, mask_logits
    sft.eval()
    E = rm_a.body.wte.weight  # [V, d]
    tops = {}
    for s in range(0, len(split.users), batch):
        x, lengths, seen = batch_inputs(data, split.users[s:s + batch], split.end[s:s + batch],
                                        cfg['max_len'], device)
        logp = torch.log_softmax(mask_logits(sft.last_logits(x, lengths), seen), -1)
        h = rm_a.body(input_ids=x, attention_mask=(x != 0).long()).last_hidden_state
        h = h[torch.arange(len(x), device=device), lengths - 1]
        H = h[:, None].expand(-1, E.shape[0], -1)
        Ee = E[None].expand(len(x), -1, -1)
        wp = torch.sigmoid(rm_a.head(torch.cat([H, Ee, H * Ee], -1))[..., 0])
        rewards = {'r1': wp, 'r3': wp * hours,
                   'r7a': wp * hours * (cfg['rl'].get('r7_alpha', 0.3) * logp).exp()}
        if rm_w is not None:
            ow = all_items(rm_w, x, lengths, device)
            rewards['r8'] = ow[..., 2] * ow[..., 0] * hours
            rewards['r8log'] = ow[..., 2] * torch.log1p(ow[..., 0] * hours)
        if rm_w7 is not None:
            ow = all_items(rm_w7, x, lengths, device)
            rewards['r9'] = ow[..., 2] * ow[..., 0] * hours
        for kind, r in rewards.items():
            for beta in BETAS:
                score = logp + r / beta  # -inf entries (seen, special) stay excluded
                tops.setdefault((kind, beta), []).append(score.topk(k, -1).indices.cpu().numpy())
    return {key: np.concatenate(v) for key, v in tops.items()}


def all_items(rm, x, lengths, device):
    """Sigmoid outputs of a reward model for every vocabulary item: [B, V, n_out]."""
    h = rm.body(input_ids=x, attention_mask=(x != 0).long()).last_hidden_state
    h = h[torch.arange(len(x), device=device), lengths - 1]
    E = rm.body.wte.weight
    H = h[:, None].expand(-1, E.shape[0], -1)
    Ee = E[None].expand(len(x), -1, -1)
    return torch.sigmoid(rm.head(torch.cat([H, Ee, H * Ee], -1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'out'))
    ap.add_argument('--config', default=str(ROOT / 'config.json'))
    args = ap.parse_args()
    out = Path(args.out)
    cfg = json.loads(Path(args.config).read_text())
    device = pick_device(cfg['device'])
    from rl_recsys.evaluate import evaluate_top
    from rl_recsys.models import load_policy, recommend
    from rl_recsys.reward import load_rm_a
    from rl_recsys.rm_b import RewardBClient
    from rl_recsys.sequences import make_split
    data = pickle.loads((out / 'data.pkl').read_bytes())
    sft = load_policy(out / 'sft.pt', device)
    rm_a = load_rm_a(out / 'rm_a.pt', device)
    rm_w = load_rm_a(out / 'rm_a_w.pt', device) if (out / 'rm_a_w.pt').exists() else None
    rm_w7 = load_rm_a(out / 'rm_a_w7.pt', device) if (out / 'rm_a_w7.pt').exists() else None
    rm_b = RewardBClient(out / 'data.pkl', out / 'rm_b.pkl')
    hours = torch.cat([torch.zeros(N_SPECIAL),
                       torch.tensor(data['item_meta']['dur_h'].to_numpy(), dtype=torch.float32)])
    hours = hours.clamp(max=cfg['rl']['r3_cap_hours']).to(device)
    for name in ['val', 'test']:
        split = make_split(data, name, cfg['max_eval_users'], cfg['seed'])
        sft_top = recommend(sft, data, split.users, split.end, cfg['max_len'], device)
        base, _ = evaluate_top(sft_top, data, split, rm_a, rm_b, cfg, device, sft_top)
        rows = [{'reward': 'none', 'beta': float('inf'), **base}]
        for (kind, beta), top in tilted_tops(sft, rm_a, data, split, cfg, device, hours, rm_w=rm_w, rm_w7=rm_w7).items():
            row, _ = evaluate_top(top, data, split, rm_a, rm_b, cfg, device, sft_top)
            rows.append({'reward': kind, 'beta': beta, **row})
            log(name, kind, beta, {c: round(row[c], 4) for c in ('ndcg10', 'real_h10', 'rm_b_wp', 'rm_b_watch_h', 'coverage', 'dur_h')})
        pd.DataFrame(rows).to_csv(out / f'tilted_{name}.csv', index=False)
    rm_b.close()


if __name__ == '__main__':
    main()
