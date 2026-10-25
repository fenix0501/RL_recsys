"""Stage 3: RL fine-tuning of the SFT policy.

GRPO (main): for each train-period context sample G items from the policy,
score them with RM-A, use group-normalised rewards as advantages, and add the
exact KL(pi || pi_ref) over the item vocabulary (one-step bandit, so KL is cheap
and exact). One gradient step per fresh batch, so the PPO ratio is 1 and no
clipping is needed.

Reward R6 is built to be hard to hack (see `robust_reward`): watch time rather
than watched share, pessimistic over an RM-A ensemble, anchored to relevance by
log pi_ref, and penalised for items the whole batch converges on.

Variants: `ptx_alpha` adds cross-entropy on real logged next items to GRPO
(PPO-ptx), keeping relevance anchored to the logs, not only to pi_ref; `bon` is
best-of-N distillation (sample N from the frozen SFT, SFT on the RM-best one).
Cheaper comparisons on logs only (no RM): AWR (watched-share weighted SFT) and
DPO/IPO on within-user pairs (high vs low watched share among the next few items).
All runs checkpoint every `save_every` steps and resume from the last checkpoint.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import N_SPECIAL, append_jsonl, atomic_save, log, seed_all
from evaluate import evaluate_top
from models import batch_inputs, mask_logits, recommend, save_policy
from sequences import train_sequences


class ContextSampler:
    """Random train-period prefixes [ptr[u], end) with at least one item."""

    def __init__(self, data, seed):
        seqs = train_sequences(data)
        self.u = np.array([s[0] for s in seqs])
        self.a = np.array([s[1] for s in seqs])
        self.b = np.array([s[2] for s in seqs])
        self.rng = np.random.default_rng(seed)

    def sample(self, n):
        i = self.rng.integers(0, len(self.u), n)
        ends = self.a[i] + 1 + (self.rng.random(n) * (self.b[i] - self.a[i])).astype(np.int64)
        return self.u[i], np.minimum(ends, self.b[i])


def masked_logp(model, data, users, ends, cfg, device):
    x, lengths, seen = batch_inputs(data, users, ends, cfg['max_len'], device)
    logits = mask_logits(model.last_logits(x, lengths), seen)
    return F.log_softmax(logits, -1), (x, lengths)


def exact_kl(logp, ref_logp):
    p = logp.exp()
    diff = torch.where(torch.isfinite(logp), logp - ref_logp, torch.zeros_like(logp))
    return (p * diff).sum(-1)


def reward(kind, rm_out, ref_logp_a, a, aux, rl_cfg):
    wp, ret = rm_out[..., 0], rm_out[..., 1]
    if kind == 'r1':
        return wp
    if kind == 'r2':
        return wp + rl_cfg['r2_lambda'] * ref_logp_a
    if kind == 'r3':
        return wp * aux['dur'][a].clamp(max=rl_cfg['r3_cap_hours'])
    if kind == 'r4':
        return ret
    if kind == 'r7a':
        # pi_ref ** alpha flattens the peaked next-item softmax before it multiplies watch hours
        return wp * aux['dur'][a].clamp(max=rl_cfg['r3_cap_hours']) * (rl_cfg['r7_alpha'] * ref_logp_a).exp()
    if kind in ('r7', 'r7h'):
        # P(watch | ctx) * E[share | watch]: pi_ref stands in for the chance the item is opened at all
        r = wp * ref_logp_a.exp()
        return r * aux['dur'][a].clamp(max=rl_cfg['r3_cap_hours']) if kind == 'r7h' else r
    if kind == 'r5':
        # watched more than this item is usually watched: removes the short-item advantage
        return wp - aux['item_mean'][a]
    raise ValueError(kind)


def dpo_pairs(data, margin, window=5):
    """(user, context end, chosen token, rejected token) from the next `window` train items."""
    out = []
    tok, wp = data['tok'], data['wp']
    for u, a, b in train_sequences(data):
        for t in range(a, b - 1):
            nxt = np.arange(t, min(t + window, b))
            nxt = nxt[tok[nxt] >= N_SPECIAL]
            if len(nxt) < 2:
                continue
            hi, lo = nxt[np.argmax(wp[nxt])], nxt[np.argmin(wp[nxt])]
            if wp[hi] - wp[lo] >= margin:
                out.append((u, t, tok[hi], tok[lo]))
    return np.array(out, dtype=np.int64).reshape(-1, 4)


def run(spec, data, sft_path, rm_a, rm_b, val, sft_val_top, cfg, device, run_dir, smoke=False, rm_ens=None, rm_w=None, rm_w7=None):
    from models import load_policy
    rl = cfg['rl']
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'final.pt').exists():
        log(f'{spec["name"]}: done, skipping')
        return
    seed = spec.get('seed', cfg['seed'])
    seed_all(seed)
    policy = load_policy(sft_path, device)
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=spec.get('lr', rl['lr']), weight_decay=0.0)
    steps = 20 if smoke else spec.get('steps', rl['steps'])
    every = 10 if smoke else rl['eval_every']
    save_every = 10 if smoke else rl['save_every']
    sampler = ContextSampler(data, seed)
    start = 0
    ckpt = run_dir / 'ckpt.pt'
    if ckpt.exists():
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        policy.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        sampler.rng.bit_generator.state = state['rng']
        start = state['step']
        log(f'{spec["name"]}: resumed at step {start}')
    pad_tok = lambda v: torch.cat([torch.zeros(N_SPECIAL, device=device),
                                   torch.tensor(v, dtype=torch.float32, device=device)])
    meta = data['item_meta']
    aux = {'dur': pad_tok(meta['dur_h'].to_numpy()),
           'item_mean': pad_tok(meta['mean_wp'].fillna(meta['mean_wp'].mean()).to_numpy()),
           'rm_ens': rm_ens or [rm_a], 'rm_w': rm_w, 'rm_w7': rm_w7}
    rm_reward = rm_a
    if spec.get('rm') == 'personal':
        # RM-P (rm_personal.py) gives the reward; validation logging keeps the shared RM-A
        from rm_personal import load_rm_p
        rm_reward = load_rm_p(run_dir.parent.parent / 'rm_p.pt', data, device)
        aux['rm_w7'] = rm_reward
    pairs = dpo_pairs(data, spec.get('margin', 0.5)) if spec['method'] == 'dpo' else None
    awr_rows = awr_rows_of(data) if spec['method'] in ('awr', 'crr', 'awrv') or spec.get('ptx_alpha') else None
    critic = G = None
    if spec['method'] in ('crr', 'awrv'):
        from crr import get_critic
        critic, G = get_critic(spec, data, sft_path, cfg, device, run_dir.parent.parent, smoke)
    if pairs is not None:
        log(f'DPO pairs: {len(pairs)}')
    vi = np.arange(min(rl['val_users'], len(val.users)))
    val_sub = type(val)(users=val.users[vi], end=val.end[vi], targets=[val.targets[i] for i in vi],
                        target_rows=val.target_rows, cutoff=val.cutoff)
    if start == 0:
        log_val(spec, 0, ref, data, val_sub, rm_a, rm_b, sft_val_top[vi], cfg, device, run_dir, {})
    t0 = time.time()
    hist = []
    for step in range(start + 1, steps + 1):
        # dropout during RL makes sampling noisy and puts a floor under KL; old runs kept it on
        policy.train(spec.get('rl_dropout', True))
        if spec['method'] == 'grpo':
            stats = grpo_step(spec, policy, ref, rm_reward, sampler, data, aux, cfg, device)
            if spec.get('ptx_alpha'):
                ptx = log_nll(policy, awr_rows, sampler.rng, data, cfg, device).mean()
                stats['loss'] = stats['loss'] + spec['ptx_alpha'] * ptx
                stats['ptx_nll'] = ptx.item()
        elif spec['method'] == 'bon':
            stats = bon_step(spec, policy, ref, rm_a, sampler, data, aux, cfg, device)
        elif spec['method'] == 'dpo':
            stats = dpo_step(spec, policy, ref, pairs, sampler.rng, data, cfg, device)
        elif spec['method'] == 'awr':
            stats = awr_step(spec, policy, awr_rows, sampler.rng, data, cfg, device)
        elif spec['method'] in ('crr', 'awrv'):
            from crr import crr_step
            stats = crr_step(spec, policy, critic, G, awr_rows, sampler.rng, data, cfg, device)
        else:
            raise ValueError(spec['method'])
        opt.zero_grad(set_to_none=True)
        stats.pop('loss').backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        hist.append(stats)
        if step % every == 0 or step == steps:
            train = {k: float(np.mean([h[k] for h in hist])) for k in hist[0]}
            train['sec_per_step'] = (time.time() - t0) / len(hist)
            hist, t0 = [], time.time()
            log_val(spec, step, policy, data, val_sub, rm_a, rm_b, sft_val_top[vi], cfg, device, run_dir, train)
        if step % save_every == 0 or step == steps:
            atomic_save({'model': policy.state_dict(), 'opt': opt.state_dict(), 'step': step,
                         'rng': sampler.rng.bit_generator.state}, ckpt)
    save_policy(policy, run_dir / 'final.pt', spec=spec, steps=steps)
    log(f'{spec["name"]}: finished')


def log_val(spec, step, policy, data, val, rm_a, rm_b, sft_top, cfg, device, run_dir, train):
    top = recommend(policy, data, val.users, val.end, cfg['max_len'], device)
    row, _ = evaluate_top(top, data, val, rm_a, rm_b, cfg, device, sft_top)
    row = {'run': spec['name'], 'step': step, **{f'train_{k}': v for k, v in train.items()},
           **{f'val_{k}': v for k, v in row.items()}}
    append_jsonl(run_dir / 'log.jsonl', row)
    log(spec['name'], {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()
                       if k in ('step', 'train_reward', 'train_kl', 'val_ndcg10', 'val_rm_a_wp', 'val_rm_b_wp',
                                'val_coverage', 'val_jaccard_vs_sft', 'train_sec_per_step')})


def grpo_step(spec, policy, ref, rm_a, sampler, data, aux, cfg, device):
    rl = cfg['rl']
    B, G = rl['batch_size'], rl['group_size']
    users, ends = sampler.sample(B)
    logp, (x, lengths) = masked_logp(policy, data, users, ends, cfg, device)
    for m in (rm_a, aux.get('rm_w'), aux.get('rm_w7')):
        if hasattr(m, 'set_users'):
            m.set_users(users, ends)
    with torch.no_grad():
        ref_logp, _ = masked_logp(ref, data, users, ends, cfg, device)
        a = torch.multinomial(logp.exp(), G, replacement=True)
        out = torch.sigmoid(rm_a.candidates(x, lengths, a))
        if spec['reward'] == 'r6':
            r, parts = robust_reward(spec, x, lengths, a, ref_logp, aux, rl)
        elif spec['reward'] in ('r8', 'r8log', 'r9'):
            r, parts = expected_watch_reward(x, lengths, a, aux, rl, spec['reward'])
        else:
            r, parts = reward(spec['reward'], out, ref_logp.gather(1, a), a, aux, rl), {}
        adv = r - r.mean(1, keepdim=True)
        # 'group' divides by each group's std (amplifies near-ties); 'batch' uses one batch-wide scale
        adv = adv / ((r.std() if spec.get('adv_norm', 'group') == 'batch' else r.std(1, keepdim=True)) + 1e-4)
    pg = -(adv * logp.gather(1, a)).mean()
    kl = exact_kl(logp, ref_logp).mean()
    loss = pg + spec['beta'] * kl
    ent = -(logp.exp() * torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))).sum(-1).mean()
    return {'loss': loss, 'reward': r.mean().item(), 'rm_wp': out[..., 0].mean().item(),
            'kl': kl.item(), 'entropy': ent.item(), **parts}


def dpo_step(spec, policy, ref, pairs, rng, data, cfg, device):
    p = pairs[rng.integers(0, len(pairs), cfg['rl']['batch_size'])]
    logp, _ = masked_logp(policy, data, p[:, 0], p[:, 1], cfg, device)
    with torch.no_grad():
        ref_logp, _ = masked_logp(ref, data, p[:, 0], p[:, 1], cfg, device)
    c = torch.from_numpy(p[:, 2:3]).to(device)
    r = torch.from_numpy(p[:, 3:4]).to(device)
    margin = (logp.gather(1, c) - ref_logp.gather(1, c)) - (logp.gather(1, r) - ref_logp.gather(1, r))
    if spec.get('loss') == 'ipo':
        # IPO: regress the log-ratio margin to 1/(2 tau) instead of pushing it to infinity
        loss = ((margin - 1 / (2 * spec['beta'])) ** 2).mean()
    else:
        loss = -F.logsigmoid(spec['beta'] * margin).mean()
    kl = exact_kl(logp.detach(), ref_logp).mean()
    return {'loss': loss, 'reward': (margin > 0).float().mean().item(), 'kl': kl.item()}


def awr_rows_of(data):
    """Train-period (user, context end, next token, watched share) rows."""
    rows = []
    for u, a, b in train_sequences(data):
        t = np.arange(a, b)
        t = t[data['tok'][t] >= N_SPECIAL]
        rows.append(np.stack([np.full(len(t), u), t, data['tok'][t]], 1))
    rows = np.concatenate(rows)
    return rows, data['wp'][rows[:, 1]]


def awr_step(spec, policy, awr, rng, data, cfg, device):
    rows, wp = awr
    i = rng.integers(0, len(rows), cfg['rl']['batch_size'])
    logp, _ = masked_logp(policy, data, rows[i, 0], rows[i, 1], cfg, device)
    if spec.get('weight') == 'hours':
        # weight logged next items by the real hours watched instead of the watched share
        h = np.minimum(data['dur'][rows[i, 1]] / 3600, cfg['rl']['r3_cap_hours'])
        w = (1 + h) ** spec.get('gamma', 1.0)
    elif spec.get('weight') in ('wp50', 'full'):
        # binary observed reward: watched more than half / watched to the end
        r = wp > 0.5 if spec['weight'] == 'wp50' else wp >= 0.9
        w = np.exp((r[i] - r.mean()) / spec['temperature'])
    else:
        w = np.exp((wp[i] - wp.mean()) / spec['temperature'])
    w = torch.from_numpy(np.minimum(w / w.mean(), 20)).float().to(device)
    nll = -logp.gather(1, torch.from_numpy(rows[i, 2:3]).to(device)).squeeze(1)
    return {'loss': (w * nll).mean(), 'reward': float(wp[i].mean()), 'kl': 0.0}


def log_nll(policy, awr, rng, data, cfg, device):
    """Per-row NLL of real logged next items (the PPO-ptx anchor)."""
    rows, _ = awr
    i = rng.integers(0, len(rows), cfg['rl']['batch_size'])
    logp, _ = masked_logp(policy, data, rows[i, 0], rows[i, 1], cfg, device)
    return -logp.gather(1, torch.from_numpy(rows[i, 2:3]).to(device)).squeeze(1)


def bon_step(spec, policy, ref, rm_a, sampler, data, aux, cfg, device):
    """Best-of-N distillation: N samples from frozen SFT, SFT on the RM-best one."""
    rl = cfg['rl']
    users, ends = sampler.sample(rl['batch_size'])
    logp, (x, lengths) = masked_logp(policy, data, users, ends, cfg, device)
    with torch.no_grad():
        ref_logp, _ = masked_logp(ref, data, users, ends, cfg, device)
        a = torch.multinomial(ref_logp.exp(), spec.get('n', rl['group_size']), replacement=True)
        out = torch.sigmoid(rm_a.candidates(x, lengths, a))
        r = reward(spec['reward'], out, ref_logp.gather(1, a), a, aux, rl)
        best = a.gather(1, r.argmax(1, keepdim=True))
    loss = -logp.gather(1, best).mean()
    return {'loss': loss, 'reward': r.max(1).values.mean().item(),
            'kl': exact_kl(logp.detach(), ref_logp).mean().item()}


def robust_reward(spec, x, lengths, a, ref_logp, aux, rl):
    """R6 = pessimistic watch hours + relevance anchor - batch-collapse penalty.

    * watch hours (share x estimated length, capped) instead of share: short items lose their edge;
    * mean - kappa * std over the RM-A ensemble: items the RMs disagree on are not worth chasing;
    * + lam * log pi_ref(a): rewards stay tied to what the user would plausibly pick;
    * - gamma * log(1 + times a was sampled in this batch): converging on the same items for
      everyone stops paying off.
    """
    hours = aux['dur'][a].clamp(max=rl['r3_cap_hours'])
    preds = torch.stack([torch.sigmoid(m.candidates(x, lengths, a)[..., 0]) for m in aux['rm_ens']])
    watch = preds * hours
    pess = watch.mean(0) - spec.get('kappa', 1.0) * (watch.std(0) if len(preds) > 1 else 0)
    rel = spec.get('lam', 0.1) * ref_logp.gather(1, a)
    counts = torch.bincount(a.ravel(), minlength=ref_logp.shape[1]).float()
    pop = spec.get('gamma', 0.1) * torch.log1p(counts[a])
    r = pess + rel - pop
    return r, {'r6_watch': watch.mean().item(), 'r6_std': watch.std(0).mean().item() if len(preds) > 1 else 0.0,
               'r6_rel': rel.mean().item(), 'r6_pop': pop.mean().item()}


def expected_watch_reward(x, lengths, a, aux, rl, kind='r8'):
    """R8 = P(watches | history, item) * E[share | watched] * length = expected watch hours.

    Short items lose (few hours), long but unlikely items lose (low P(watch)); both factors come
    from the RM-A variant with the watch head (rm_a_w.pt).
    """
    model = aux['rm_w7'] if kind == 'r9' else aux['rm_w']  # r9: watch head trained on a 7-day window
    out = torch.sigmoid(model.candidates(x, lengths, a))
    p_watch, share = out[..., 2], out[..., 0]
    hours = share * aux['dur'][a].clamp(max=rl['r3_cap_hours'])
    # r8log: concave value, an extra hour of a long item is worth less than the first one
    r = p_watch * (torch.log1p(hours) if kind == 'r8log' else hours)
    return r, {'r8_p_watch': p_watch.mean().item(), 'r8_share': share.mean().item()}
