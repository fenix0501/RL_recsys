#!/usr/bin/env python3
"""Relevance at a longer cut-off (default @50) for the saved policies on the test week.

Same test users, masking and paired bootstrap against SFT as `run.py final_eval`, but only
relevance and the duration proxy, so it needs no reward model and no retraining.

    python eval_at50.py --config config_full.json --out out_full
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(TASK2 / 'config.json'))
    ap.add_argument('--out', default=str(TASK2 / 'out'))
    ap.add_argument('--k', type=int, default=50)
    ap.add_argument('--runs', help='comma-separated run names (default: every runs/*/final.pt)')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out, k = Path(args.out), args.k
    device = pick_device(cfg['device'])
    from evaluate import realized_hours
    from gpt import popular_top
    from metrics import relevance, tokens_to_items
    from models import load_policy, recommend
    from sequences import make_split
    data = pickle.loads((out / 'data.pkl').read_bytes())
    test = make_split(data, 'test', cfg['max_eval_users'], cfg['seed'])
    runs = args.runs.split(',') if args.runs else sorted(d.parent.name for d in (out / 'runs').glob('*/final.pt'))
    models = {'popular': None, 'sft': out / 'sft.pt', **{r: out / 'runs' / r / 'final.pt' for r in runs}}
    rows, per_user = [], {}
    for name, path in models.items():
        if path is None:
            top = popular_top(data, test, k=k)
        else:
            top = recommend(load_policy(path, device), data, test.users, test.end, cfg['max_len'], device, k=k)
        items = tokens_to_items(data, top)
        rel = relevance(items, test.targets, k)
        rel.columns = [f'recall{k}', f'ndcg{k}', f'map{k}']
        rel[f'real_h{k}'] = realized_hours(data, test, items)
        rows.append({'model': name, **rel.mean().to_dict()})
        per_user[name] = rel[[f'ndcg{k}', f'recall{k}', f'real_h{k}']].to_numpy()
        log(name, {c: round(v, 4) for c, v in rows[-1].items() if c != 'model'})
    table = pd.DataFrame(rows)
    rng = np.random.default_rng(cfg['seed'])
    base = per_user['sft']
    boot = rng.integers(0, len(base), (500, len(base)))
    for i, name in enumerate(table['model']):
        delta = per_user[name] - base
        lo, hi = np.quantile(delta[boot].mean(1), [0.025, 0.975], axis=0)
        for j, col in enumerate([f'ndcg{k}', f'recall{k}', f'real_h{k}']):
            table.loc[i, f'delta_{col}'] = delta[:, j].mean()
            table.loc[i, f'{col}_ci_low'], table.loc[i, f'{col}_ci_high'] = lo[j], hi[j]
    table.to_csv(out / f'results_test_at{k}.csv', index=False)
    log('wrote', out / f'results_test_at{k}.csv')


if __name__ == '__main__':
    main()
