#!/usr/bin/env python3
"""Task 2 pipeline: data -> SFT GPT-2 -> RM-A / RM-B -> RL runs -> test evaluation.

Every stage writes its artefact into --out and is skipped when the artefact
exists, and RL runs resume from their last checkpoint, so re-running the same
command after an interruption continues where it stopped.

    python run.py                 # full pipeline (overnight)
    python run.py --smoke         # tiny end-to-end check, separate folder
    python run.py --runs grpo_r1_b0.05,awr   # only some RL runs
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from common import TASK2, log, pick_device, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(TASK2 / 'config.json'))
    ap.add_argument('--out', help='artefact folder (default out or out_smoke)')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--runs', help='comma-separated RL run names (default: all in config)')
    ap.add_argument('--final', help='comma-separated runs to evaluate on test (default: all finished)')
    ap.add_argument('--stop-after', choices=['data', 'sft', 'rm', 'rl'], help='stop after this stage')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out or TASK2 / ('out_smoke' if args.smoke else 'out')).resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg['device'])
    log(f'device={device} out={out} smoke={args.smoke}')
    started = time.time()

    from sequences import build, make_split
    data_path = out / 'data.pkl'
    if data_path.exists():
        data = pickle.loads(data_path.read_bytes())
    else:
        data = build(cfg, smoke=args.smoke)
        data_path.write_bytes(pickle.dumps(data))
    val = make_split(data, 'val', cfg['max_eval_users'], cfg['seed'])
    test = make_split(data, 'test', cfg['max_eval_users'], cfg['seed'])
    log(f'eval users: val {len(val.users)}, test {len(test.users)}')
    if args.stop_after == 'data':
        return

    from gpt import train_sft
    from models import load_policy, recommend
    if not (out / 'sft.pt').exists():
        train_sft(data, val, cfg, device, out, smoke=args.smoke)
    sft = load_policy(out / 'sft.pt', device)
    if args.stop_after == 'sft':
        return

    from reward import load_rm_a, rm_holdout, train_rm_a
    from rm_b import RewardBClient
    if not (out / 'rm_a.pt').exists():
        state = torch.load(out / 'sft.pt', map_location='cpu', weights_only=False)
        train_rm_a(data, state['model'], sft.conf, val, cfg, device, out, smoke=args.smoke)
    rm_a = load_rm_a(out / 'rm_a.pt', device)
    rm_ens = [rm_a]
    for k in range(1, cfg['rm_a'].get('ensemble', 1)):
        name = f'rm_a_e{k}'
        if not (out / f'{name}.pt').exists():
            state = torch.load(out / 'sft.pt', map_location='cpu', weights_only=False)
            train_rm_a(data, state['model'], sft.conf, val, cfg, device, out, smoke=args.smoke,
                       seed=cfg['seed'] + 100 * k, name=name, user_frac=0.8)
        rm_ens.append(load_rm_a(out / f'{name}.pt', device))
    rm_w = None
    if cfg['rm_a'].get('watch_head'):
        if not (out / 'rm_a_w.pt').exists():
            state = torch.load(out / 'sft.pt', map_location='cpu', weights_only=False)
            train_rm_a(data, state['model'], sft.conf, val, cfg, device, out, smoke=args.smoke,
                       name='rm_a_w', watch_head=True)
        rm_w = load_rm_a(out / 'rm_a_w.pt', device)
    rm_w7 = None
    if cfg['rm_a'].get('watch_window_days'):
        if not (out / 'rm_a_w7.pt').exists():
            state = torch.load(out / 'sft.pt', map_location='cpu', weights_only=False)
            train_rm_a(data, state['model'], sft.conf, val, cfg, device, out, smoke=args.smoke,
                       name='rm_a_w7', watch_head=True, watch_window=cfg['rm_a']['watch_window_days'])
        rm_w7 = load_rm_a(out / 'rm_a_w7.pt', device)
    rm_b = RewardBClient(data_path, out / 'rm_b.pkl')
    if not (out / 'reward_quality.json').exists():
        q_b = rm_b.fit(cfg, rm_holdout(data, val))
        q_a = torch.load(out / 'rm_a.pt', map_location='cpu', weights_only=False)['holdout']
        write_json(out / 'reward_quality.json', {'rm_a': q_a, 'rm_b': q_b,
                                                 'holdout': 'validation-week rows, in-vocabulary items'})
    if args.stop_after == 'rm':
        return

    import rl
    sft_val_top = recommend(sft, data, val.users, val.end, cfg['max_len'], device)
    specs = cfg['rl']['runs']
    if args.runs:
        names = set(args.runs.split(','))
        specs = [s for s in specs if s['name'] in names]
    if any(s.get('rm') == 'personal' for s in specs) and not (out / 'rm_p.pt').exists():
        from rm_personal import train_rm_p
        train_rm_p(data, out / 'sft.pt', val, cfg, device, out, smoke=args.smoke)
    for spec in specs:
        rl.run(spec, data, out / 'sft.pt', rm_a, rm_b, val, sft_val_top, cfg, device,
               out / 'runs' / spec['name'], smoke=args.smoke, rm_ens=rm_ens, rm_w=rm_w, rm_w7=rm_w7)
    if args.stop_after == 'rl':
        return

    final_eval(data, test, sft, rm_a, rm_b, cfg, device, out, args.smoke,
               only=set(args.final.split(',')) if args.final else None)
    rm_b.close()
    log(f'all done in {(time.time() - started) / 3600:.2f} h')


def final_eval(data, test, sft, rm_a, rm_b, cfg, device, out, smoke, only=None):
    """Same protocol for every model on the test week; written once at the end."""
    from evaluate import evaluate_top, multistep
    from gpt import popular_top
    from metrics import tokens_to_items
    from models import load_policy, recommend
    sft_top = recommend(sft, data, test.users, test.end, cfg['max_len'], device)
    models = {'popular': None, 'sft': sft}
    for d in sorted((out / 'runs').glob('*/final.pt')):
        if only is None or d.parent.name in only:
            models[d.parent.name] = d
    rows, per_user, ms, examples = [], {}, {}, {}
    for name, m in models.items():
        if name == 'popular':
            top = popular_top(data, test)
        else:
            policy = m if name == 'sft' else load_policy(m, device)
            top = recommend(policy, data, test.users, test.end, cfg['max_len'], device)
            ms[name] = multistep(policy, rm_a, rm_b, data, test, cfg, device,
                                 n_users=200 if smoke else cfg['multistep']['users'],
                                 steps=cfg['multistep']['steps'])
        row, rel = evaluate_top(top, data, test, rm_a, rm_b, cfg, device, sft_top)
        rows.append({'model': name, **row})
        per_user[name] = rel[['ndcg10', 'real_h10']].to_numpy()
        examples[name] = tokens_to_items(data, top[:5]).tolist()
        log('test', name, {k: round(v, 4) for k, v in row.items() if k in ('ndcg10', 'real_h10', 'rm_a_wp', 'rm_b_wp', 'rm_b_watch_h', 'coverage')})
    table = pd.DataFrame(rows)
    rng = np.random.default_rng(cfg['seed'])
    base = per_user['sft']
    boot = rng.integers(0, len(base), (500, len(base)))
    for i, name in enumerate(table['model']):
        delta = per_user[name] - base
        lo, hi = np.quantile(delta[boot].mean(1), [0.025, 0.975], axis=0)
        table.loc[i, 'delta_ndcg_vs_sft'] = delta[:, 0].mean()
        table.loc[i, 'delta_ci_low'], table.loc[i, 'delta_ci_high'] = lo[0], hi[0]
        table.loc[i, 'delta_real_h_vs_sft'] = delta[:, 1].mean()
        table.loc[i, 'real_h_ci_low'], table.loc[i, 'real_h_ci_high'] = lo[1], hi[1]
    table.to_csv(out / 'results_test.csv', index=False)
    write_json(out / 'multistep_test.json', ms)
    write_json(out / 'examples_test.json', {'users': data['user_ids'][test.users[:5]].tolist(), 'top10': examples})
    plot(table, out)
    log('wrote', out / 'results_test.csv')


def plot(table, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col in zip(axes, ['rm_b_wp', 'rm_b_watch_h']):
        ax.scatter(table['ndcg10'], table[col])
        for _, r in table.iterrows():
            ax.annotate(r['model'], (r['ndcg10'], r[col]), fontsize=7)
        ax.set_xlabel('NDCG@10 (test)')
        ax.set_ylabel(f'{col} (held-out RM-B)')
    fig.tight_layout()
    fig.savefig(out / 'tradeoff_test.png', dpi=130)


if __name__ == '__main__':
    main()
