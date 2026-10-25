#!/usr/bin/env python3
"""Optimisation gap of trained policies on their own objective, against the exact optimum.

For a reward r and the SFT reference, pi*_beta ∝ pi_ref * exp(r / beta) maximises
E_pi[r] - beta * KL(pi || pi_ref). Sweeping beta traces the best achievable E[r] at each KL
(the frontier). A trained stochastic policy gives one point (KL, E[r]); the vertical
distance to the frontier at the same KL is its optimisation gap. This is scale-free, so the
advantage normalisation inside GRPO does not matter. Everything is exact over the full
vocabulary, per test context, then averaged.

    python scripts/objective_gap.py --out out_full
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
from tilted import all_items

BETAS = [float(b) for b in np.geomspace(0.002, 20, 41)]  # dense: the gap is read off by interpolation
MODELS = {'grpo_r1_b2.0_fix': 'r1', 'grpo_r3_b2.0_fix': 'r3', 'grpo_r8_b1.0_fix': 'r8',
          'grpo_r8_b2.0_fix': 'r8', 'grpo_r9_b1.0_fix': 'r9', 'grpo_r9_b2.0_fix': 'r9',
          'ipo_t0.5_fix': 'r1', 'awr_fix': 'r1'}


def stats(logq, logp_ref, r):
    """Per-row E_q[r] and KL(q || ref) for distributions given as log-probs (-inf = excluded)."""
    q = logq.exp()
    ok = torch.isfinite(logq)
    er = (q * torch.where(ok, r, torch.zeros_like(r))).sum(-1)
    kl = (q * torch.where(ok, logq - logp_ref, torch.zeros_like(logq))).sum(-1)
    return er, kl


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'out'))
    ap.add_argument('--config', default=str(ROOT / 'config.json'))
    ap.add_argument('--users', type=int, default=5000)
    ap.add_argument('--models', help='name:reward,... instead of the default MODELS')
    ap.add_argument('--tag', default='', help='suffix for the output csv files')
    args = ap.parse_args()
    models = dict(m.split(':') for m in args.models.split(',')) if args.models else MODELS
    out = Path(args.out)
    cfg = json.loads(Path(args.config).read_text())
    device = pick_device(cfg['device'])
    from rl_recsys.models import batch_inputs, load_policy, mask_logits
    from rl_recsys.reward import load_rm_a
    from rl_recsys.sequences import make_split
    data = pickle.loads((out / 'data.pkl').read_bytes())
    split = make_split(data, 'test', cfg['max_eval_users'], cfg['seed'])
    users, ends = split.users[:args.users], split.end[:args.users]
    sft = load_policy(out / 'sft.pt', device).eval()
    rms = {'a': load_rm_a(out / 'rm_a.pt', device), 'w': load_rm_a(out / 'rm_a_w.pt', device),
           'w7': load_rm_a(out / 'rm_a_w7.pt', device)}
    pols = {m: load_policy(out / 'runs' / m / 'final.pt', device).eval()
            for m in models if (out / 'runs' / m / 'final.pt').exists()}
    hours = torch.cat([torch.zeros(N_SPECIAL), torch.tensor(data['item_meta']['dur_h'].to_numpy(),
                                                            dtype=torch.float32)])
    hours = hours.clamp(max=cfg['rl']['r3_cap_hours']).to(device)
    acc = {}
    for s in range(0, len(users), 32):
        x, lengths, seen = batch_inputs(data, users[s:s + 32], ends[s:s + 32], cfg['max_len'], device)
        logp_ref = torch.log_softmax(mask_logits(sft.last_logits(x, lengths), seen), -1)
        oa = all_items(rms['a'], x, lengths, device)
        ow = all_items(rms['w'], x, lengths, device)
        ow7 = all_items(rms['w7'], x, lengths, device)
        rewards = {'r1': oa[..., 0], 'r3': oa[..., 0] * hours,
                   'r8': ow[..., 2] * ow[..., 0] * hours, 'r9': ow7[..., 2] * ow7[..., 0] * hours}
        for kind, r in rewards.items():
            for beta in BETAS:
                logq = torch.log_softmax(logp_ref + r / beta, -1)
                er, kl = stats(logq, logp_ref, r)
                acc.setdefault(('pi*', kind, beta), []).append(torch.stack([er, kl], 1).cpu())
        for m, pol in pols.items():
            logq = torch.log_softmax(mask_logits(pol.last_logits(x, lengths), seen), -1)
            kind = models[m]
            er, kl = stats(logq, logp_ref, rewards[kind])
            acc.setdefault((m, kind, None), []).append(torch.stack([er, kl], 1).cpu())
            er0, _ = stats(logp_ref, logp_ref, rewards[kind])
            acc.setdefault(('sft', kind, None), []).append(torch.stack([er0, torch.zeros_like(er0)], 1).cpu())
    rows = [{'policy': p, 'reward': k, 'beta': b, 'E_r': float(v[:, 0].mean()), 'KL': float(v[:, 1].mean())}
            for (p, k, b), v in ((key, torch.cat(val)) for key, val in acc.items())]
    df = pd.DataFrame(rows)
    gaps = []
    for _, row in df[~df.policy.isin(['pi*', 'sft'])].iterrows():
        f = df[(df.policy == 'pi*') & (df.reward == row.reward)].sort_values('KL')
        best = float(np.interp(row.KL, f.KL, f.E_r))
        sft_er = float(df[(df.policy == 'sft') & (df.reward == row.reward)].E_r.iloc[0])
        gaps.append({'policy': row.policy, 'reward': row.reward, 'KL': row.KL, 'E_r': row.E_r,
                     'E_r_optimum_same_KL': best, 'E_r_sft': sft_er,
                     'share_of_optimal_gain': (row.E_r - sft_er) / (best - sft_er) if best > sft_er else np.nan})
    df.to_csv(out / f'objective_frontier{args.tag}.csv', index=False)
    pd.DataFrame(gaps).to_csv(out / f'objective_gap{args.tag}.csv', index=False)
    log(pd.DataFrame(gaps).round(4).to_string())


if __name__ == '__main__':
    main()
