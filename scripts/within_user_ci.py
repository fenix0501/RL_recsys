#!/usr/bin/env python3
"""Within-user Spearman of the watched-share head for RM-A, RM-P and the item-mean baseline on
the test-week holdout, with a user-bootstrap CI for the RM-P minus RM-A difference.

Both the target and the prediction are centred on each user's mean over that user's holdout
rows (users with >= 3 rows), then one Spearman is taken over all remaining rows.

    python scripts/within_user_ci.py --out out_full
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
from scipy.stats import spearmanr

from rl_recsys.common import ROOT, log, pick_device, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'config.json'))
    ap.add_argument('--out', default=str(ROOT / 'out'))
    ap.add_argument('--boot', type=int, default=200)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    device = pick_device(cfg['device'])
    from rl_recsys.reward import load_rm_a, rm_holdout
    from rl_recsys.rm_personal import load_rm_p, predict
    from rl_recsys.sequences import make_split
    data = pickle.loads((out / 'data.pkl').read_bytes())
    h = rm_holdout(data, make_split(data, 'test', cfg['max_eval_users'], cfg['seed']))
    df = pd.DataFrame({'u': h['users'], 'y': h['wp'], 'item_mean': h['item_mean'],
                       'rm_a': predict(load_rm_a(out / 'rm_a.pt', device), data, h, cfg, device)[:, 0],
                       'rm_p': predict(load_rm_p(out / 'rm_p.pt', data, device), data, h, cfg, device)[:, 0]})
    df = df[df.groupby('u')['u'].transform('size') >= 3].copy()
    for c in ('y', 'item_mean', 'rm_a', 'rm_p'):
        df[c] = df[c] - df.groupby('u')[c].transform('mean')
    users = df['u'].unique()
    groups = {u: g.index.to_numpy() for u, g in df.groupby('u')}
    rho = lambda d, c: float(spearmanr(d['y'], d[c])[0])
    res = {'n_users': int(len(users)), 'n_rows': int(len(df)),
           **{f'within_user_{c}': rho(df, c) for c in ('item_mean', 'rm_a', 'rm_p')}}
    rng = np.random.default_rng(cfg['seed'])
    diffs = []
    for _ in range(args.boot):
        idx = np.concatenate([groups[u] for u in rng.choice(users, len(users))])
        d = df.loc[idx]
        diffs.append(rho(d, 'rm_p') - rho(d, 'rm_a'))
    res['diff_rm_p_minus_rm_a'] = res['within_user_rm_p'] - res['within_user_rm_a']
    res['diff_ci'] = [float(x) for x in np.quantile(diffs, [0.025, 0.975])]
    log(res)
    write_json(out / 'within_user_ci.json', res)


if __name__ == '__main__':
    main()
