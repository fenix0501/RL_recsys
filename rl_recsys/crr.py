"""Offline RL with a learned critic on logged returns: CRR and AWR with a value baseline.

Following "Integrating Offline RL with Transformers for Sequential Recommendation"
(GPT-2 actor + critic + CRR): a critic Q(s, a) is fitted on the logs, then the
SFT policy is fine-tuned by weighted cross-entropy on logged next items,

    L = -E[ f(A(s, a)) log pi(a | s) ],  A(s, a) = Q(s, a) - sum_a' pi(a' | s) Q(s, a'),

with f = exp(A / std_batch(A) / beta) (clipped at 20) or 1[A > 0], weights divided
by their batch mean as in AWR. Unlike the RM-based methods this
optimises a multi-step return of observed rewards, G_t = sum_k gamma^k r_{t+k}
over the user's next train-period views.

The critic is fitted by Monte-Carlo regression of Q(s_t, a_t) on G_t rather than
by TD with a target network: on fixed logs both estimate Q of the logging policy
(in-sample, SARSA-style), and MC has no bootstrapping across window boundaries.
Critic: GPT-2 body initialised from SFT, Q(s, a) = w_a . h(s) + b_a, so Q over the
whole vocabulary is one matmul and the expectation under pi is exact.

`awrv` is classic AWR (Peng et al. 2019): the same weighted cross-entropy with
A = G_t - V(s), V(s) = sum_a pi(a | s) Q(s, a).

    python -m rl_recsys.crr --out out_full --critics hours:0.9,wp50:0.9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from transformers import GPT2Config, GPT2Model

from rl_recsys.common import BOS, N_SPECIAL, PAD, append_jsonl, atomic_save, log, seed_all
from rl_recsys.sequences import pad, train_sequences


def step_rewards(data, kind, cap_hours):
    if kind == 'hours':
        return np.minimum(data['dur'] / 3600, cap_hours)
    if kind == 'wp':
        return data['wp'].astype(np.float64)
    if kind == 'wp50':
        return (data['wp'] > 0.5).astype(np.float64)
    if kind == 'full':
        return (data['wp'] >= 0.9).astype(np.float64)
    raise ValueError(kind)


def mc_returns(data, kind, gamma, cap_hours):
    """G_t over each user's train-period views, censored at the train cutoff (0 outside it)."""
    r = step_rewards(data, kind, cap_hours)
    G = np.zeros(len(r), dtype=np.float64)
    for _, a, b in train_sequences(data):
        acc = 0.0
        for t in range(b - 1, a - 1, -1):
            acc = r[t] + gamma * acc
            G[t] = acc
    return G.astype(np.float32)


class Critic(nn.Module):
    def __init__(self, conf, body_state=None):
        super().__init__()
        self.conf = conf
        self.body = GPT2Model(conf)
        if body_state is not None:
            self.body.load_state_dict(body_state)
        self.w = nn.Embedding(conf.vocab_size, conf.n_embd + 1)
        nn.init.normal_(self.w.weight, std=0.02)

    def hidden(self, x):
        return self.body(input_ids=x, attention_mask=(x != PAD).long()).last_hidden_state

    def q_of(self, h, a):
        w = self.w(a)
        return (h * w[..., :-1]).sum(-1) + w[..., -1]

    def q_all(self, x, lengths):
        h = self.hidden(x)[torch.arange(len(x), device=x.device), lengths - 1]
        return h @ self.w.weight[:, :-1].T + self.w.weight[:, -1]


def critic_name(kind, gamma):
    return f'critic_{kind}_g{gamma}'


def windows_of(data, max_len):
    """Train windows of event indices (-1 = BOS), as for RM-A, tagged with their user."""
    out, users = [], []
    for u, a, b in train_sequences(data):
        idx = np.r_[-1, np.arange(a, b)]
        for s in range(0, len(idx) - 1, max_len):
            piece = idx[s:min(s + max_len + 1, len(idx))]
            if len(piece) >= 2:
                out.append(piece)
                users.append(u)
    return out, np.array(users)


def train_critic(data, sft_path, kind, gamma, cfg, device, out_dir, smoke=False):
    c = cfg['crr']
    name = critic_name(kind, gamma)
    seed_all(cfg['seed'])
    G = mc_returns(data, kind, gamma, cfg['rl']['r3_cap_hours'])
    train_mask = data['day'] < data['cutoffs']['train']
    mu, sd = float(G[train_mask].mean()), float(G[train_mask].std() + 1e-6)
    state = torch.load(sft_path, map_location='cpu', weights_only=False)
    conf = GPT2Config.from_dict(state['conf'])
    body = {k[len('lm.transformer.'):]: v for k, v in state['model'].items() if k.startswith('lm.transformer.')}
    model = Critic(conf, body).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=0.01)
    windows, wu = windows_of(data, cfg['max_len'])
    rng = np.random.default_rng(cfg['seed'])
    held_users = rng.random(len(data['ptr']) - 1) < c['holdout_frac']
    hold = [w for w, u in zip(windows, wu) if held_users[u]]
    windows = [w for w, u in zip(windows, wu) if not held_users[u]]
    tok = np.r_[data['tok'], BOS]
    y_all = np.r_[(G - mu) / sd, 0.0].astype(np.float32)
    # item-mean return on the training users: how much of G is just "which item"
    lab_tr = np.concatenate([w[1:] for w in windows])
    item_mean = np.bincount(tok[lab_tr], weights=y_all[lab_tr], minlength=conf.vocab_size) / \
        np.maximum(np.bincount(tok[lab_tr], minlength=conf.vocab_size), 1)
    log(f'{name}: {len(windows)} windows, holdout {len(hold)}, G mean {mu:.3f} sd {sd:.3f}')

    def batches(ws, order):
        for s in range(0, len(order), c['batch_size']):
            part = [ws[i] for i in order[s:s + c['batch_size']]]
            idx = pad(part, value=len(tok) - 1)
            valid = pad([np.ones(len(w), bool) for w in part], value=False, dtype=bool)
            x = torch.from_numpy(np.where(valid, tok[idx], PAD)).to(device)
            lab = idx[:, 1:]
            m = valid[:, 1:] & (tok[lab] >= N_SPECIAL)
            yield x, lab, m

    best, bad = -2.0, 0
    for epoch in range(1 if smoke else c['epochs']):
        model.train()
        losses = []
        for x, lab, m in batches(windows, rng.permutation(len(windows))):
            q = model.q_of(model.hidden(x)[:, :-1], x[:, 1:])
            mt = torch.from_numpy(m).to(device)
            loss = ((q - torch.from_numpy(y_all[lab]).to(device)) ** 2)[mt].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        ys, ps, bs = [], [], []
        with torch.no_grad():
            for x, lab, m in batches(hold, np.arange(len(hold))):
                q = model.q_of(model.hidden(x)[:, :-1], x[:, 1:]).cpu().numpy()
                ys.append(y_all[lab][m])
                ps.append(q[m])
                bs.append(item_mean[tok[lab][m]])
        y, p, b = map(np.concatenate, (ys, ps, bs))
        row = {'epoch': epoch, 'train_mse': float(np.mean(losses)), 'holdout_mse': float(((y - p) ** 2).mean()),
               'spearman': float(spearmanr(y, p)[0]), 'baseline_item_mean_spearman': float(spearmanr(y, b)[0]),
               'baseline_item_mean_mse': float(((y - b) ** 2).mean()), 'n': int(len(y))}
        append_jsonl(out_dir / f'{name}_log.jsonl', row)
        log(name, row)
        if row['spearman'] > best:
            best, bad = row['spearman'], 0
            atomic_save({'conf': conf.to_dict(), 'model': model.state_dict(), 'mu': mu, 'sd': sd,
                         'kind': kind, 'gamma': gamma, 'holdout': row}, out_dir / f'{name}.pt')
        else:
            bad += 1
            if bad >= c['patience']:
                break
    return best


def get_critic(spec, data, sft_path, cfg, device, out_dir, smoke=False):
    """Critic for the spec (trained and cached on first use) and standardised MC returns."""
    kind, gamma = spec['critic_reward'], float(spec['gamma'])
    path = out_dir / f'{critic_name(kind, gamma)}.pt'
    if not path.exists():
        train_critic(data, sft_path, kind, gamma, cfg, device, out_dir, smoke)
    state = torch.load(path, map_location='cpu', weights_only=False)
    critic = Critic(GPT2Config.from_dict(state['conf']))
    critic.load_state_dict(state['model'])
    critic = critic.to(device).eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    G = (mc_returns(data, kind, gamma, cfg['rl']['r3_cap_hours']) - state['mu']) / state['sd']
    return critic, G


def crr_step(spec, policy, critic, G, awr, rng, data, cfg, device):
    from rl_recsys.rl import masked_logp
    rows, _ = awr
    i = rng.integers(0, len(rows), cfg['rl']['batch_size'])
    logp, (x, lengths) = masked_logp(policy, data, rows[i, 0], rows[i, 1], cfg, device)
    a = torch.from_numpy(rows[i, 2:3]).to(device)
    with torch.no_grad():
        q = critic.q_all(x, lengths)
        v = (logp.exp() * q).sum(-1)
        if spec['method'] == 'crr':
            adv = q.gather(1, a).squeeze(1) - v
        else:  # awrv: Monte-Carlo return minus the critic's state value
            adv = torch.from_numpy(G[rows[i, 1]]).to(device) - v
        z = adv / (adv.std() + 1e-6)
        if spec.get('f') == 'binary':
            w = (z > 0).float()
        else:
            w = torch.exp(z / spec['beta']).clamp(max=20)
        w = w / (w.mean() + 1e-8)
    nll = -logp.gather(1, a).squeeze(1)
    return {'loss': (w * nll).mean(), 'reward': float(G[rows[i, 1]].mean()), 'kl': 0.0,
            'adv_mean': float(adv.mean()), 'adv_std': float(adv.std()), 'pos_frac': float((adv > 0).float().mean())}


def main():
    from rl_recsys.common import ROOT, pick_device
    import pickle
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'config.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--critics', required=True, help='comma-separated reward:gamma, e.g. hours:0.9,wp50:0.9')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out).resolve()
    device = pick_device(cfg['device'])
    data = pickle.loads((out / 'data.pkl').read_bytes())
    for item in args.critics.split(','):
        kind, gamma = item.split(':')
        if (out / f'{critic_name(kind, float(gamma))}.pt').exists():
            log(f'{critic_name(kind, float(gamma))}: exists, skipping')
            continue
        train_critic(data, out / 'sft.pt', kind, float(gamma), cfg, device, out, smoke=args.smoke)


if __name__ == '__main__':
    main()
