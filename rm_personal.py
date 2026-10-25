"""RM-A with personal features (RM-P): does the user's own history add a personal reward signal?

RM-A sees only the item tokens of the history, so it cannot know how much of each item the
user actually watched, and its prediction is mostly item-level. RM-P adds point-in-time user
features computed from the watched shares and hours of the user's earlier views (mean share,
hours, share on series vs films, share of full watches, ...) and their interactions with the
candidate's length, type and item mean share, on top of the RM-A pair features.

Same heads and training as RM-A with the watch head on a 7-day window (heads: watched share,
return within 7 days, P(watch)), so one model serves R1/R3/R5 (head 0) and R9 (heads 2 and 0).
Callers give the users before scoring: `set_users(rm, users, ends)` (no-op for other models).

Also fits an isotonic calibration of the watched-share head on the validation holdout and
reports the expected calibration error before and after, for RM-A and RM-P.

    python rm_personal.py --config config_full.json --out out_full
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import BOS, N_SPECIAL, PAD, append_jsonl, atomic_save, log, seed_all
from models import RewardModel, batch_inputs
from sequences import pad

N_UF, N_IF = 8, 3


class UserFeatures:
    """Point-in-time user features for a history [ptr[u], end), from prefix sums over events."""

    def __init__(self, data, cap_hours=3.0, prior=3.0):
        meta = data['item_meta']
        tok, wp = data['tok'], data['wp'].astype(np.float64)
        vocab = tok >= N_SPECIAL
        dur_tok = np.r_[np.zeros(N_SPECIAL), meta['dur_h'].to_numpy()]
        ser_tok = np.r_[np.zeros(N_SPECIAL), meta['is_series'].to_numpy()]
        series = ser_tok[tok] * vocab
        cols = np.stack([np.ones(len(tok)), wp, np.minimum(data['dur'] / 3600, cap_hours),
                         np.where(vocab, dur_tok[tok], np.nan), series, wp * series,
                         wp * (1 - series), (wp >= 0.9).astype(float), vocab.astype(float), 1 - series], 1)
        cols[:, 3] = np.where(np.isnan(cols[:, 3]), 0, cols[:, 3])
        self.C = np.vstack([np.zeros((1, cols.shape[1])), np.cumsum(cols, 0)])
        self.ptr = data['ptr']
        self.event_user = np.searchsorted(self.ptr, np.arange(len(tok)), side='right') - 1
        train = data['day'] < data['cutoffs']['train']
        self.g_wp, self.g_h = float(wp[train].mean()), float(cols[train, 2].mean())
        self.g_dur = float(dur_tok[tok[train & vocab]].mean())
        self.prior = prior
        self.mu, self.sd = 0.0, 1.0
        sample = np.flatnonzero(train)[::50]
        f = self.raw(self.event_user[sample], sample)
        self.mu, self.sd = f.mean(0), f.std(0) + 1e-6

    def raw(self, users, ends):
        s = self.C[ends] - self.C[self.ptr[users]]
        n, k = s[:, 0], self.prior
        mean = lambda a, b, g: (a + k * g) / (b + k)
        ser_n, film_n = s[:, 4], s[:, 9]
        return np.stack([np.log1p(n), mean(s[:, 1], n, self.g_wp), mean(s[:, 2], n, self.g_h),
                         mean(s[:, 3], s[:, 8], self.g_dur), mean(s[:, 4], n, 0.34),
                         mean(s[:, 5], ser_n, self.g_wp), mean(s[:, 6], film_n, self.g_wp),
                         mean(s[:, 7], n, 0.3)], 1)

    def __call__(self, users, ends):
        return ((self.raw(np.asarray(users), np.asarray(ends)) - self.mu) / self.sd).astype(np.float32)

    def at_events(self, idx):
        """Features of the history before each event index (for training on all positions)."""
        return self(self.event_user[idx], idx)


def item_features(data):
    meta = data['item_meta']
    f = np.stack([np.log(meta['dur_h'].to_numpy()), meta['is_series'].to_numpy(),
                  meta['mean_wp'].fillna(meta['mean_wp'].mean()).to_numpy()], 1)
    f = (f - f.mean(0)) / (f.std(0) + 1e-6)
    return np.vstack([np.zeros((N_SPECIAL, N_IF)), f]).astype(np.float32)


class PersonalRewardModel(RewardModel):
    def __init__(self, conf, body_state=None, n_out=3, item_feats=None):
        super().__init__(conf, body_state, n_out)
        d = conf.n_embd
        self.head = nn.Sequential(nn.Linear(3 * d + N_UF + N_IF + N_UF * N_IF, 2 * d), nn.GELU(), nn.Linear(2 * d, n_out))
        self.register_buffer('item_feats', torch.zeros(conf.vocab_size, N_IF) if item_feats is None else torch.from_numpy(item_feats))
        self.uf_fn, self._uf = None, None

    def set_users(self, users, ends):
        self._uf = torch.from_numpy(self.uf_fn(users, ends)).to(self.item_feats.device)

    def _pair_p(self, h, tok, uf):
        e, ci = self.body.wte(tok), self.item_feats[tok]
        uf = uf.expand(*ci.shape[:-1], N_UF)
        inter = (uf[..., :, None] * ci[..., None, :]).flatten(-2)
        return self.head(torch.cat([h, e, h * e, uf, ci, inter], -1))

    def positions(self, x, uf_pos):
        h = self.body(input_ids=x, attention_mask=(x != PAD).long()).last_hidden_state
        return self._pair_p(h[:, :-1], x[:, 1:], uf_pos)

    def candidates(self, x, lengths, cand):
        assert self._uf is not None and len(self._uf) == len(x), 'call set_users(users, ends) before scoring'
        h = self.body(input_ids=x, attention_mask=(x != PAD).long()).last_hidden_state
        h = h[torch.arange(len(x), device=x.device), lengths - 1]
        return self._pair_p(h[:, None].expand(-1, cand.shape[1], -1), cand, self._uf[:, None])

    def all_items(self, x, lengths):
        V = self.item_feats.shape[0]
        cand = torch.arange(V, device=x.device)[None].expand(len(x), -1)
        return self.candidates(x, lengths, cand)


def set_users(rm, users, ends):
    if rm is not None and hasattr(rm, 'set_users'):
        rm.set_users(users, ends)


def load_rm_p(path, data, device):
    from transformers import GPT2Config
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = PersonalRewardModel(GPT2Config.from_dict(state['conf']), n_out=state['n_out'])
    model.load_state_dict(state['model'])
    model.uf_fn = UserFeatures(data)
    return model.to(device).eval()


def train_rm_p(data, sft_path, val, cfg, device, out_dir, smoke=False, name='rm_p'):
    from reward import return_labels, rm_holdout, rm_windows, window_ends
    from rm_b import reward_quality, safe_auc
    from transformers import GPT2Config
    r = cfg['rm_a']
    seed_all(cfg['seed'])
    state = torch.load(sft_path, map_location='cpu', weights_only=False)
    conf = GPT2Config.from_dict(state['conf'])
    body = {k[len('lm.transformer.'):]: v for k, v in state['model'].items() if k.startswith('lm.transformer.')}
    model = PersonalRewardModel(conf, body, n_out=3, item_feats=item_features(data)).to(device)
    model.uf_fn = uf = UserFeatures(data)
    opt = torch.optim.AdamW(model.parameters(), lr=r['lr'], weight_decay=0.01)
    ret = return_labels(data, r['return_days'])
    ret_ok = data['day'] + r['return_days'] < data['cutoffs']['train']
    windows = rm_windows(data, cfg['max_len'])
    holdout = rm_holdout(data, val)
    tok = np.r_[data['tok'], BOS]
    wp = np.r_[data['wp'], 0.0].astype(np.float32)
    rt = np.r_[ret, 0.0].astype(np.float32)
    rok = np.r_[ret_ok, False]
    wend = np.r_[window_ends(data, r['watch_window_days'], data['cutoffs']['train']), len(tok) - 1]
    n_neg = r.get('watch_negatives', 8)
    rng = np.random.default_rng(cfg['seed'])
    best, bad = -1.0, 0
    for epoch in range(1 if smoke else r['max_epochs']):
        model.train()
        losses = []
        order = rng.permutation(len(windows))
        for s in range(0, len(order), r['batch_size']):
            part = [windows[i] for i in order[s:s + r['batch_size']]]
            idx = pad(part, value=len(tok) - 1)
            valid = pad([np.ones(len(w), bool) for w in part], value=False, dtype=bool)
            x = torch.from_numpy(np.where(valid, tok[idx], PAD)).to(device)
            lab = idx[:, 1:]
            labc = np.minimum(lab, len(tok) - 2)
            ufp = torch.from_numpy(uf.at_events(labc.ravel()).reshape(*lab.shape, N_UF)).to(device)
            h = model.body(input_ids=x, attention_mask=(x != PAD).long()).last_hidden_state[:, :-1]
            out = model._pair_p(h, x[:, 1:], ufp)
            m_wp = valid[:, 1:] & (tok[lab] >= N_SPECIAL)
            m1 = torch.from_numpy(m_wp).to(device)
            m2 = torch.from_numpy(m_wp & rok[lab]).to(device)
            l1 = F.binary_cross_entropy_with_logits(out[..., 0][m1], torch.from_numpy(wp[lab]).to(device)[m1])
            l2 = F.binary_cross_entropy_with_logits(out[..., 1][m2], torch.from_numpy(rt[lab]).to(device)[m2]) if m2.any() else 0 * l1
            # watch head: a random item from the next 7 days vs uniform random items
            j = lab + (rng.random(lab.shape) * np.maximum(wend[lab] - lab, 1)).astype(np.int64)
            pos_tok = torch.from_numpy(tok[np.minimum(j, len(tok) - 1)]).to(device)
            pos = model._pair_p(h, pos_tok, ufp)[..., 2]
            neg_tok = torch.randint(N_SPECIAL, conf.vocab_size, (*lab.shape, n_neg), device=device)
            neg = model._pair_p(h[:, :, None].expand(-1, -1, n_neg, -1), neg_tok, ufp[:, :, None])[..., 2]
            logits = torch.cat([pos[..., None], neg], -1)[m1]
            labels = torch.zeros_like(logits)
            labels[:, 0] = 1
            loss = l1 + 0.5 * l2 + F.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(l1.item())
        pred = predict(model, data, holdout, cfg, device)
        q = reward_quality(holdout['wp'], pred[:, 0], holdout['item_mean'])
        q['return_auc'] = safe_auc(ret[holdout['rows']], pred[:, 1])
        q['watch_auc'] = watch_auc_p(model, data, holdout, cfg, device)
        row = {'epoch': epoch, 'train_bce_wp': float(np.mean(losses)), **q}
        append_jsonl(out_dir / f'{name}_log.jsonl', row)
        log(name, row)
        if q['spearman'] > best:
            best, bad = q['spearman'], 0
            atomic_save({'conf': conf.to_dict(), 'model': model.state_dict(), 'holdout': q, 'n_out': 3}, out_dir / f'{name}.pt')
        else:
            bad += 1
            if bad >= r['patience']:
                break
    return best


@torch.no_grad()
def predict(model, data, h, cfg, device, batch=1024):
    """Sigmoid outputs [N, n_out] for (user history, candidate) rows; works for RM-A and RM-P."""
    model.eval()
    out = []
    for s in range(0, len(h['rows']), batch):
        x, lengths, _ = batch_inputs(data, h['users'][s:s + batch], h['ends'][s:s + batch], cfg['max_len'], device)
        set_users(model, h['users'][s:s + batch], h['ends'][s:s + batch])
        cand = torch.from_numpy(h['tok'][s:s + batch]).to(device)[:, None]
        out.append(torch.sigmoid(model.candidates(x, lengths, cand)[:, 0]).cpu().numpy())
    model.train()
    return np.concatenate(out)


@torch.no_grad()
def watch_auc_p(model, data, h, cfg, device, n_neg=4, batch=1024):
    from rm_b import safe_auc
    model.eval()
    scores, labels = [], []
    rng = np.random.default_rng(0)
    for s in range(0, len(h['rows']), batch):
        x, lengths, _ = batch_inputs(data, h['users'][s:s + batch], h['ends'][s:s + batch], cfg['max_len'], device)
        set_users(model, h['users'][s:s + batch], h['ends'][s:s + batch])
        pos = h['tok'][s:s + batch][:, None]
        neg = rng.integers(N_SPECIAL, model.body.wte.num_embeddings, (len(pos), n_neg))
        cand = torch.from_numpy(np.concatenate([pos, neg], 1)).to(device)
        scores.append(model.candidates(x, lengths, cand)[..., 2].cpu().numpy().ravel())
        labels.append(np.tile(np.r_[1, np.zeros(n_neg)], len(pos)))
    model.train()
    return safe_auc(np.concatenate(labels), np.concatenate(scores))


def ece(y, p, bins=15):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, bins - 1)
    return float(sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any()))


def personal_signal(y, p, tok):
    """Within-item Spearman: y and the prediction both centred on their per-item mean over the
    holdout rows (items seen at least twice), so item-level effects cannot drive the correlation."""
    import pandas as pd
    from scipy.stats import spearmanr
    df = pd.DataFrame({'y': y, 'p': p, 't': tok})
    df = df[df.groupby('t')['t'].transform('size') >= 2]
    g = df.groupby('t')
    return float(spearmanr(df['y'] - g['y'].transform('mean'), df['p'] - g['p'].transform('mean'))[0])


def within_user(y, p, users, min_rows=3):
    """Spearman after centring y and the prediction on each user's mean (users with >= min_rows
    holdout rows): only differences between items for the same user count, which is what a
    reward has to rank; a user-level offset (some users finish everything) cannot help."""
    import pandas as pd
    from scipy.stats import spearmanr
    df = pd.DataFrame({'y': y, 'p': p, 'u': users})
    df = df[df.groupby('u')['u'].transform('size') >= min_rows]
    g = df.groupby('u')
    return float(spearmanr(df['y'] - g['y'].transform('mean'), df['p'] - g['p'].transform('mean'))[0])


def calibration_report(data, val, test, rms, cfg, device):
    """Isotonic calibration of the watched-share head fitted on the validation week, checked on the test week."""
    from reward import rm_holdout
    from sklearn.isotonic import IsotonicRegression
    hv, ht = rm_holdout(data, val), rm_holdout(data, test)
    rep = {}
    for name, m in rms.items():
        pv, pt = predict(m, data, hv, cfg, device)[:, 0], predict(m, data, ht, cfg, device)[:, 0]
        iso = IsotonicRegression(out_of_bounds='clip').fit(pv, hv['wp'])
        from rm_b import reward_quality
        q = reward_quality(ht['wp'], pt, ht['item_mean'])
        rep[name] = {'test_spearman': q['spearman'], 'test_item_mean_spearman': q['baseline_item_mean_spearman'],
                     'test_mae': q['mae'], 'test_within_item_spearman': personal_signal(ht['wp'], pt, ht['tok']),
                     'test_within_user_spearman': within_user(ht['wp'], pt, ht['users']),
                     'test_within_user_item_mean_spearman': within_user(ht['wp'], ht['item_mean'], ht['users']),
                     'test_ece': ece(ht['wp'], pt), 'test_ece_isotonic': ece(ht['wp'], iso.predict(pt)),
                     'test_mae_isotonic': float(np.abs(ht['wp'] - iso.predict(pt)).mean())}
        log('calibration', name, {k: round(v, 4) for k, v in rep[name].items()})
    return rep


def main():
    from common import TASK2, pick_device, write_json
    from reward import load_rm_a
    from sequences import make_split
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(TASK2 / 'config.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out).resolve()
    device = pick_device(cfg['device'])
    data = pickle.loads((out / 'data.pkl').read_bytes())
    val = make_split(data, 'val', cfg['max_eval_users'], cfg['seed'])
    test = make_split(data, 'test', cfg['max_eval_users'], cfg['seed'])
    if not (out / 'rm_p.pt').exists():
        train_rm_p(data, out / 'sft.pt', val, cfg, device, out, smoke=args.smoke)
    rms = {'rm_a': load_rm_a(out / 'rm_a.pt', device), 'rm_a_w7': load_rm_a(out / 'rm_a_w7.pt', device),
           'rm_p': load_rm_p(out / 'rm_p.pt', data, device)}
    write_json(out / 'rm_p_quality.json', calibration_report(data, val, test, rms, cfg, device))


if __name__ == '__main__':
    main()
