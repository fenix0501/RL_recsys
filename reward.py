"""Stage 2: reward models.

RM-A (used by RL): GPT-2 body initialised from SFT + MLP, two heads:
  * watched share of the candidate (BCE on watched_pct/100),
  * the user comes back with another view within `return_days` (for reward R4).
RM-B (evaluation only, never seen by RL) lives in rm_b.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import BOS, N_SPECIAL, PAD, UNK, append_jsonl, atomic_save, log, seed_all
from models import RewardModel, batch_inputs
from rm_b import reward_quality, safe_auc
from sequences import pad, train_sequences

# ---------------------------------------------------------------- labels

def return_labels(data, days):
    """1 if the user has a view on a later day within `days`; with the last day it covers."""
    ptr, day = data['ptr'], data['day']
    nxt = np.full(len(day), np.iinfo(np.int64).max // 2, dtype=np.int64)
    for u in range(len(ptr) - 1):
        d = day[ptr[u]:ptr[u + 1]]
        pos = np.searchsorted(d, d, side='right')
        ok = pos < len(d)
        nxt[ptr[u]:ptr[u + 1]][ok] = d[pos[ok]]
    return (nxt <= day + days).astype(np.float32)


def rm_windows(data, max_len):
    """Train-period windows with aligned event indices (-1 for BOS)."""
    out = []
    for u, a, b in train_sequences(data):
        idx = np.r_[-1, np.arange(a, b)]
        for s in range(0, len(idx) - 1, max_len):
            piece = idx[s:min(s + max_len + 1, len(idx))]
            if len(piece) >= 2:
                out.append(piece)
    return out


# ---------------------------------------------------------------- RM-A

def train_rm_a(data, policy_state, conf, val, cfg, device, out_dir, smoke=False, seed=None,
               name='rm_a', user_frac=1.0, watch_head=False, watch_window=0):
    """Extra ensemble members use another seed and a bootstrap-style subset of users."""
    r = cfg['rm_a']
    seed = cfg['seed'] if seed is None else seed
    seed_all(seed)
    ret = return_labels(data, r['return_days'])
    ret_ok = data['day'] + r['return_days'] < data['cutoffs']['train']
    body = {k[len('lm.transformer.'):]: v for k, v in policy_state.items() if k.startswith('lm.transformer.')}
    model = RewardModel(conf, body, n_out=3 if watch_head else 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=r['lr'], weight_decay=0.01)
    windows = rm_windows(data, cfg['max_len'])
    if user_frac < 1:
        keep = np.random.default_rng(seed).random(len(windows)) < user_frac
        windows = [w for w, k in zip(windows, keep) if k]
    holdout = rm_holdout(data, val)
    tok = np.r_[data['tok'], BOS]  # index -1 -> BOS
    wp = np.r_[data['wp'], 0.0].astype(np.float32)
    rt = np.r_[ret, 0.0].astype(np.float32)
    rok = np.r_[ret_ok, False]
    wend = np.r_[window_ends(data, watch_window, data['cutoffs']['train']), len(tok) - 1] if watch_window else None
    rng = np.random.default_rng(seed)
    best, bad = -1.0, 0
    for epoch in range(1 if smoke else r['max_epochs']):
        model.train()
        losses = []
        order = rng.permutation(len(windows))
        for s in range(0, len(order), r['batch_size']):
            idx = pad([windows[i] for i in order[s:s + r['batch_size']]], value=len(tok) - 1)
            valid_pos = pad([np.ones(len(windows[i]), bool) for i in order[s:s + r['batch_size']]],
                            value=False, dtype=bool)
            x = torch.from_numpy(np.where(valid_pos, tok[idx], PAD)).to(device)
            out = model.positions(x)
            lab = idx[:, 1:]
            m_wp = valid_pos[:, 1:] & (tok[lab] >= N_SPECIAL)
            m_rt = m_wp & rok[lab]
            y_wp = torch.from_numpy(wp[lab]).to(device)
            y_rt = torch.from_numpy(rt[lab]).to(device)
            m1 = torch.from_numpy(m_wp).to(device)
            m2 = torch.from_numpy(m_rt).to(device)
            l1 = F.binary_cross_entropy_with_logits(out[..., 0][m1], y_wp[m1])
            l2 = F.binary_cross_entropy_with_logits(out[..., 1][m2], y_rt[m2]) if m2.any() else 0 * l1
            loss = l1 + 0.5 * l2
            if watch_head:
                pos = None
                if wend is not None:
                    # positive = a random item the user watched within `watch_window` days, not only the next one
                    j = lab + (rng.random(lab.shape) * np.maximum(wend[lab] - lab, 1)).astype(np.int64)
                    pos = torch.from_numpy(tok[np.minimum(j, len(tok) - 1)]).to(device)
                loss = loss + watch_loss(model, x, m1, r.get('watch_negatives', 8), device, pos)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(l1.item())
        pred = rm_a_predict(model, data, holdout, cfg, device)
        q = reward_quality(holdout['wp'], pred[:, 0], holdout['item_mean'])
        q['return_auc'] = safe_auc(ret[holdout['rows']], pred[:, 1])
        if watch_head:
            q['watch_auc'] = watch_auc(model, data, holdout, cfg, device)
        row = {'epoch': epoch, 'train_bce_wp': float(np.mean(losses)), **q}
        append_jsonl(out_dir / f'{name}_log.jsonl', row)
        log(name, row)
        if q['spearman'] > best:
            best, bad = q['spearman'], 0
            atomic_save({'conf': conf.to_dict(), 'model': model.state_dict(), 'holdout': q, 'n_out': 3 if watch_head else 2}, out_dir / f'{name}.pt')
        else:
            bad += 1
            if bad >= r['patience']:
                break
    return best


def rm_holdout(data, val):
    """Validation-week rows (in vocabulary) with the pre-week history as context."""
    users, ends, rows = [], [], []
    h_end = dict(zip(val.users, val.end))
    row_user = np.searchsorted(data['ptr'], val.target_rows, side='right') - 1
    keep = data['tok'][val.target_rows] >= N_SPECIAL
    for r, u in zip(val.target_rows[keep], row_user[keep]):
        users.append(u)
        ends.append(h_end[u])
        rows.append(r)
    rows = np.array(rows)
    train_mask = data['day'] < data['cutoffs']['train']
    item_mean = pd.Series(data['wp'][train_mask]).groupby(data['tok'][train_mask]).mean()
    return {'users': np.array(users), 'ends': np.array(ends), 'rows': rows,
            'tok': data['tok'][rows], 'wp': data['wp'][rows],
            'item_mean': item_mean.reindex(data['tok'][rows]).fillna(data['wp'][train_mask].mean()).to_numpy()}


@torch.no_grad()
def rm_a_predict(model, data, h, cfg, device, batch=1024):
    """Sigmoid outputs [N, 2] for (user history, candidate token) rows."""
    model.eval()
    out = []
    for s in range(0, len(h['rows']), batch):
        x, lengths, _ = batch_inputs(data, h['users'][s:s + batch], h['ends'][s:s + batch], cfg['max_len'], device)
        cand = torch.from_numpy(h['tok'][s:s + batch]).to(device)[:, None]
        out.append(torch.sigmoid(model.candidates(x, lengths, cand)[:, 0]).cpu().numpy())
    return np.concatenate(out)


def load_rm_a(path, device):
    from transformers import GPT2Config
    state = torch.load(path, map_location='cpu', weights_only=False)
    n_out = state['model']['head.2.weight'].shape[0]  # 3 when the watch head is present
    model = RewardModel(GPT2Config.from_dict(state['conf']), n_out=n_out)
    model.load_state_dict(state['model'])
    return model.to(device).eval()


def watch_loss(model, x, mask, n_neg, device, pos_tok=None):
    """Head 2: P(user watches this item next | history), real next items vs uniform random items.

    Unlike the softmax pi_ref, a per-item binary classifier gives a soft, calibrated-looking
    score that is not dominated by the few top items, so it can multiply a value estimate.
    """
    h = model.body(input_ids=x, attention_mask=(x != PAD).long()).last_hidden_state[:, :-1]
    pos = model._pair(h, model.body.wte(x[:, 1:] if pos_tok is None else pos_tok))[..., 2]
    neg_tok = torch.randint(N_SPECIAL, model.body.wte.num_embeddings, (*x[:, 1:].shape, n_neg), device=device)
    neg = model._pair(h[:, :, None].expand(-1, -1, n_neg, -1), model.body.wte(neg_tok))[..., 2]
    logits = torch.cat([pos[..., None], neg], -1)[mask]
    labels = torch.zeros_like(logits)
    labels[:, 0] = 1
    return F.binary_cross_entropy_with_logits(logits, labels)


@torch.no_grad()
def watch_auc(model, data, h, cfg, device, n_neg=4, batch=1024):
    """AUC of the watch head: real validation-week items vs random items for the same users."""
    model.eval()
    scores, labels = [], []
    rng = np.random.default_rng(0)
    for s in range(0, len(h['rows']), batch):
        x, lengths, _ = batch_inputs(data, h['users'][s:s + batch], h['ends'][s:s + batch], cfg['max_len'], device)
        pos = h['tok'][s:s + batch][:, None]
        neg = rng.integers(N_SPECIAL, model.body.wte.num_embeddings, (len(pos), n_neg))
        cand = torch.from_numpy(np.concatenate([pos, neg], 1)).to(device)
        out = model.candidates(x, lengths, cand)[..., 2].cpu().numpy()
        scores.append(out.ravel())
        labels.append(np.tile(np.r_[1, np.zeros(n_neg)], len(pos)))
    model.train()
    return safe_auc(np.concatenate(labels), np.concatenate(scores))


def window_ends(data, days, cutoff_day):
    """Per event, exclusive end of its `days`-day window, never past the cutoff.

    Clipping at the cutoff keeps window positives inside the train period; near the cutoff
    the window is simply shorter (an incomplete horizon), it never reaches validation rows.
    """
    ptr, day = data['ptr'], data['day']
    out = np.empty(len(day), dtype=np.int64)
    for u in range(len(ptr) - 1):
        d = day[ptr[u]:ptr[u + 1]]
        stop = np.searchsorted(d, cutoff_day, side='left')
        out[ptr[u]:ptr[u + 1]] = ptr[u] + np.minimum(np.searchsorted(d, d + days, side='right'), stop)
    return out
