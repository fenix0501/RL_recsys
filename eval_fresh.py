#!/usr/bin/env python3
"""One-off confirmation on fresh users: pre-declared final candidates, evaluated once.

The 10k-user test sample (and the 10k validation sample) were used repeatedly to choose
methods, rewards, beta, RMs and list depth, so their intervals ignore that selection. The
test week has many more users with targets that no evaluation has ever looked at; this
script scores only the candidates declared before running it, on all of those users, with
the same masking, metrics and paired bootstrap against SFT as the selection test.

    python eval_fresh.py --config config_full.json --out out_full
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import TASK2, log, pick_device

CANDIDATES = ['awr_hours_g1', 'crr_hours_g0.0_exp_b1', 'grpo_r9_b2.0_fix']


def fresh_split(data, cfg):
    from sequences import Split, make_split
    full = make_split(data, 'test', 0, cfg['seed'])
    used = make_split(data, 'test', cfg['max_eval_users'], cfg['seed']).users
    val = make_split(data, 'val', cfg['max_eval_users'], cfg['seed']).users
    keep = np.flatnonzero(~np.isin(full.users, np.union1d(used, val)))
    return Split(users=full.users[keep], end=full.end[keep], targets=[full.targets[i] for i in keep],
                 target_rows=full.target_rows, cutoff=full.cutoff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(TASK2 / 'config.json'))
    ap.add_argument('--out', default=str(TASK2 / 'out'))
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    device = pick_device(cfg['device'])
    from evaluate import realized_hours
    from gpt import popular_top
    from metrics import behaviour, relevance, tokens_to_items
    from models import load_policy, recommend
    data = pickle.loads((out / 'data.pkl').read_bytes())
    split = fresh_split(data, cfg)
    log(f'fresh test users: {len(split.users)}')
    models = {'popular': None, 'sft': out / 'sft.pt', **{m: out / 'runs' / m / 'final.pt' for m in CANDIDATES}}
    rows, per_user, sft_top = [], {}, None
    for name, path in models.items():
        top = popular_top(data, split, k=50) if path is None else \
            recommend(load_policy(path, device), data, split.users, split.end, cfg['max_len'], device, k=50)
        if name == 'sft':
            sft_top = top
        items = tokens_to_items(data, top)
        cols = {}
        for k in (10, 50):
            rel = relevance(items, split.targets, k)
            cols[f'ndcg{k}'], cols[f'recall{k}'] = rel['ndcg10'].to_numpy(), rel['recall10'].to_numpy()
            cols[f'real_h{k}'] = realized_hours(data, split, items[:, :k])
        beh = behaviour(data, top[:, :10], sft_top[:, :10] if sft_top is not None and name != 'popular' else None)
        rows.append({'model': name, **{c: float(v.mean()) for c, v in cols.items()},
                     **{c: beh[c] for c in ('coverage', 'dur_h', 'series_share')}})
        per_user[name] = np.stack([cols[c] for c in ('ndcg10', 'real_h10', 'ndcg50', 'recall50', 'real_h50')], 1)
        log(name, {c: round(v, 4) for c, v in rows[-1].items() if c != 'model'})
    table = pd.DataFrame(rows)
    rng = np.random.default_rng(cfg['seed'])
    base = per_user['sft']
    names = ['ndcg10', 'real_h10', 'ndcg50', 'recall50', 'real_h50']
    for i, name in enumerate(table['model']):
        delta = per_user[name] - base
        boot = np.stack([delta[rng.integers(0, len(delta), len(delta))].mean(0) for _ in range(500)])
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
        for j, c in enumerate(names):
            table.loc[i, f'delta_{c}'] = delta[:, j].mean()
            table.loc[i, f'{c}_ci_low'], table.loc[i, f'{c}_ci_high'] = lo[j], hi[j]
    table['n_users'] = len(split.users)
    table.to_csv(out / 'results_fresh_test.csv', index=False)
    log('wrote', out / 'results_fresh_test.csv')


if __name__ == '__main__':
    main()
