"""User sequences, item vocabulary and the leak-free weekly split for task 2.

Uses the pinned KION loader (data.py) and the same calendar weeks as task 1:
model fitting uses events before the validation week (2021-08-09); validation
targets are 08-09..08-15 and test targets 08-16..08-22. KION rows are
user-item aggregates with a day-level last_watch_dt, so the order inside one
day is unknown; we break same-day ties with a fixed random key.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rl_recsys.common import BOS, N_SPECIAL, PAD, UNK, log

EPOCH = pd.Timestamp('1970-01-01')


def to_day(ts):
    return int((pd.Timestamp(ts) - EPOCH) / pd.Timedelta(days=1))


def build(cfg, smoke=False):
    from rl_recsys.data import load_raw, split_dates
    events, users, items = load_raw()
    dates = split_dates(events)
    train_end, val_end = dates['validation']
    test_end = dates['test'][1]
    rng = np.random.default_rng(cfg['seed'])
    eligible = np.sort(events.loc[events['last_watch_dt'] < train_end, 'user_id'].unique())
    n_users = 3000 if smoke else cfg['n_users']
    chosen = np.sort(rng.permutation(eligible)[:min(n_users, len(eligible))])
    ev = events[events['user_id'].isin(chosen) & (events['last_watch_dt'] < test_end)].copy()
    ev['watched_pct'] = ev['watched_pct'].fillna(0).clip(0, 100) / 100
    ev['tie'] = rng.random(len(ev))
    ev = ev.sort_values(['user_id', 'last_watch_dt', 'tie'], kind='stable').reset_index(drop=True)

    train = ev[ev['last_watch_dt'] < train_end]
    counts = train['item_id'].value_counts()
    min_count = 2 if smoke else cfg['min_item_count']
    vocab_items = np.sort(counts[counts >= min_count].index.to_numpy())
    token_of = pd.Series(np.arange(len(vocab_items)) + N_SPECIAL, index=vocab_items)
    tok = ev['item_id'].map(token_of).fillna(UNK).astype(np.int64).to_numpy()

    user_ids, starts = np.unique(ev['user_id'].to_numpy(), return_index=True)
    ptr = np.r_[starts, len(ev)].astype(np.int64)
    data = {
        'user_ids': user_ids, 'ptr': ptr, 'tok': tok,
        'item': ev['item_id'].to_numpy(np.int64), 'wp': ev['watched_pct'].to_numpy(np.float32),
        'dur': ev['total_dur'].to_numpy(np.float64),
        'day': ((ev['last_watch_dt'] - EPOCH) // pd.Timedelta(days=1)).to_numpy(np.int64),
        'vocab_items': vocab_items,
        'cutoffs': {'train': to_day(train_end), 'val': to_day(val_end), 'test': to_day(test_end)},
        'item_meta': item_meta(train, items, vocab_items),
        'users_meta': users[users['user_id'].isin(user_ids)].reset_index(drop=True),
        'items_meta_raw': items,
    }
    log(f'sequences: {len(user_ids)} users, {len(ev)} events, vocab {len(vocab_items)} items; '
        f'OOV share {np.mean(tok == UNK):.3f}')
    return data


def item_meta(train, items, vocab_items):
    """Static per-token descriptors from the train period, used for behaviour analysis."""
    g = train.groupby('item_id')
    meta = pd.DataFrame({'item_id': vocab_items})
    meta['count'] = meta['item_id'].map(g.size()).fillna(0).to_numpy()
    meta['mean_wp'] = meta['item_id'].map(g['watched_pct'].mean()).to_numpy()
    meta['dur_h'] = meta['item_id'].map(estimated_duration_h(train)).to_numpy()
    info = items.set_index('item_id')
    meta['is_series'] = meta['item_id'].map(info['content_type'].eq('series')).fillna(False).astype(float).to_numpy()
    meta['genres'] = meta['item_id'].map(info['genres']).fillna('').to_numpy()
    meta['title'] = meta['item_id'].map(info['title']).fillna('?').to_numpy()
    meta['dur_h'] = meta['dur_h'].fillna(meta['dur_h'].median())
    return meta


def estimated_duration_h(rows):
    """KION has no item length: estimate it as watch seconds / watched share."""
    ok = rows[(rows['watched_pct'] >= 0.2) & (rows['total_dur'] > 0)]
    return (ok['total_dur'] / ok['watched_pct']).groupby(ok['item_id']).median().clip(60, 50 * 3600) / 3600


@dataclass
class Split:
    users: np.ndarray        # user indices into data['user_ids']
    end: np.ndarray          # exclusive event index: history = [ptr[u], end)
    targets: list            # list of sets of raw item ids in the target week
    target_rows: np.ndarray  # event indices of target-week rows (for RM holdout)
    cutoff: int              # first day of the target week


def history_end(data, cutoff_day):
    """Per user, index of the first event at or after the cutoff."""
    ptr, day = data['ptr'], data['day']
    out = np.empty(len(ptr) - 1, dtype=np.int64)
    for u in range(len(out)):
        out[u] = ptr[u] + np.searchsorted(day[ptr[u]:ptr[u + 1]], cutoff_day, side='left')
    return out


def make_split(data, name, max_users, seed):
    """Warm users with at least one target in the week ending at `name`'s cutoff."""
    start = data['cutoffs']['train'] if name == 'val' else data['cutoffs']['val']
    stop = data['cutoffs'][name]
    ptr = data['ptr']
    h_end = history_end(data, start)
    t_end = history_end(data, stop)
    has_hist = h_end > ptr[:-1]
    has_target = t_end > h_end
    users = np.flatnonzero(has_hist & has_target)
    if max_users and len(users) > max_users:
        users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))
    targets = [set(data['item'][h_end[u]:t_end[u]].tolist()) for u in users]
    rows = np.concatenate([np.arange(h_end[u], t_end[u]) for u in users]) if len(users) else np.array([], int)
    return Split(users=users, end=h_end[users], targets=targets, target_rows=rows, cutoff=start)


def train_sequences(data):
    """Token sequences of every user strictly before the train cutoff."""
    h_end = history_end(data, data['cutoffs']['train'])
    ptr = data['ptr']
    return [(u, ptr[u], h_end[u]) for u in range(len(ptr) - 1) if h_end[u] > ptr[u]]


def context(data, u, end, max_len):
    """Model input for the history [ptr[u], end): BOS + items, last max_len tokens."""
    seq = data['tok'][data['ptr'][u]:end]
    if len(seq) < max_len:
        return np.r_[BOS, seq]
    return seq[-max_len:]


def chunks(data, max_len):
    """Training windows: BOS-prefixed head, then non-overlapping tails of max_len."""
    out = []
    for u, a, b in train_sequences(data):
        seq = np.r_[BOS, data['tok'][a:b]]
        wp = np.r_[0.0, data['wp'][a:b]].astype(np.float32)
        for s in range(0, len(seq) - 1, max_len):
            piece = slice(s, min(s + max_len + 1, len(seq)))
            if piece.stop - piece.start >= 2:
                out.append((seq[piece], wp[piece]))
    return out


def pad(seqs, value=PAD, dtype=np.int64):
    width = max(len(s) for s in seqs)
    arr = np.full((len(seqs), width), value, dtype=dtype)
    for i, s in enumerate(seqs):
        arr[i, :len(s)] = s
    return arr
