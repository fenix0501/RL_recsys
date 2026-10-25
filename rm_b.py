"""RM-B: LightGBM watched-share model used only for evaluation (RL never sees it).

Features are computed strictly before the cutoff: item popularity and mean
watched share, estimated length, content type, user activity and genre/type
affinity. If RL raises RM-A but not RM-B, the policy exploits RM-A rather than
finding better items.

LightGBM and torch ship separate OpenMP runtimes that crash together on macOS,
so RM-B runs in its own worker process (this file as a script, no torch) and the
pipeline talks to it through `RewardBClient`.
"""
from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import N_SPECIAL, log
from sequences import estimated_duration_h, history_end


def safe_auc(y, p):
    from sklearn.metrics import roc_auc_score  # lazy: sklearn bundles its own libomp
    y = np.asarray(y) >= 0.5
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float('nan')


def reward_quality(y, pred, item_mean):
    return {'mae': float(np.abs(y - pred).mean()), 'spearman': float(spearmanr(y, pred)[0]),
            'auc_wp50': safe_auc(y, pred), 'n': int(len(y)),
            'baseline_item_mean_mae': float(np.abs(y - item_mean).mean()),
            'baseline_item_mean_spearman': float(spearmanr(y, item_mean)[0]),
            'baseline_item_mean_auc_wp50': safe_auc(y, item_mean)}



class RewardB:
    """LightGBM watched-share model on point-in-time features."""

    def __init__(self, data):
        self.data = data
        items = data['items_meta_raw'].set_index('item_id')
        genres = items['genres'].fillna('').str.split(', ')
        names = sorted({g for gs in genres for g in gs if g})
        col = {g: i for i, g in enumerate(names)}
        self.item_index = pd.Series(np.arange(len(items)), index=items.index)
        G = np.zeros((len(items), len(names)), dtype=np.float32)
        for r, gs in enumerate(genres):
            for g in gs:
                if g:
                    G[r, col[g]] = 1
        self.G = G
        self.static = pd.DataFrame({
            'is_series': items['content_type'].eq('series').astype(float),
            'release_year': items['release_year'], 'age_rating': items['age_rating'],
            'for_kids': items['for_kids'].astype(float), 'n_genres': G.sum(1)}, index=items.index)
        um = data['users_meta'].set_index('user_id')
        self.users_static = pd.DataFrame(
            {c: um[c].astype('category').cat.codes for c in ['age', 'income', 'sex']} | {'kids_flg': um['kids_flg']},
            index=um.index).reindex(data['user_ids'])
        self.tables = {}
        self.model = None

    def _tables(self, cutoff):
        if cutoff in self.tables:
            return self.tables[cutoff]
        d = self.data
        m = d['day'] < cutoff
        item = pd.Series(d['item'][m])
        wp = pd.Series(d['wp'][m])
        g = wp.groupby(item.to_numpy())
        rec = m & (d['day'] >= cutoff - 7)
        it = pd.DataFrame({'item_count': np.log1p(g.size()), 'item_mean_wp': (g.sum() + 0.46 * 20) / (g.size() + 20),
                           'item_count_7d': np.log1p(pd.Series(d['item'][rec]).value_counts())})
        rows = pd.DataFrame({'item_id': d['item'][m], 'watched_pct': d['wp'][m], 'total_dur': d['dur'][m]})
        it['dur_h'] = estimated_duration_h(rows)
        it = it.reindex(self.static.index).join(self.static)
        it['item_count'] = it['item_count'].fillna(0)
        it['item_count_7d'] = it['item_count_7d'].fillna(0)
        h_end = history_end(d, cutoff)
        ptr = d['ptr']
        user_of = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
        um = m
        n = np.bincount(user_of[um], minlength=len(ptr) - 1)
        wsum = np.bincount(user_of[um], weights=d['wp'][um], minlength=len(ptr) - 1)
        rows_item = self.item_index.reindex(d['item'][um]).to_numpy()
        known = ~np.isnan(rows_item)
        ser = np.zeros(um.sum())
        ser[known] = self.static['is_series'].to_numpy()[rows_item[known].astype(int)]
        sser = np.bincount(user_of[um], weights=ser, minlength=len(ptr) - 1)
        last = np.full(len(ptr) - 1, np.nan)
        has = h_end > ptr[:-1]
        last[has] = cutoff - d['day'][h_end[has] - 1]
        P = np.zeros((len(ptr) - 1, self.G.shape[1]), dtype=np.float32)
        np.add.at(P, user_of[um][known], self.G[rows_item[known].astype(int)])
        P /= np.maximum(P.sum(1, keepdims=True), 1)
        us = pd.DataFrame({'user_n': np.log1p(n), 'user_mean_wp': np.where(n > 0, wsum / np.maximum(n, 1), np.nan),
                           'user_series_share': np.where(n > 0, sser / np.maximum(n, 1), np.nan),
                           'days_since_last': last})
        us = pd.concat([us, self.users_static.reset_index(drop=True)], axis=1)
        self.tables[cutoff] = (it, us, P)
        return self.tables[cutoff]

    def features(self, cutoff, users, items):
        it, us, P = self._tables(cutoff)
        f = pd.concat([us.iloc[users].reset_index(drop=True), it.reindex(items).reset_index(drop=True)], axis=1)
        ii = self.item_index.reindex(items).to_numpy()
        aff = np.full(len(items), np.nan)
        ok = ~np.isnan(ii)
        for s in range(0, len(items), 200_000):
            sl = np.flatnonzero(ok[s:s + 200_000]) + s
            Gi = self.G[ii[sl].astype(int)]
            aff[sl] = (P[users[sl]] * Gi).sum(1) / np.maximum(Gi.sum(1), 1)
        f['genre_affinity'] = aff
        f['type_affinity'] = np.where(f['is_series'] == 1, f['user_series_share'], 1 - f['user_series_share'])
        return f

    def week_rows(self, cutoff):
        d = self.data
        m = (d['day'] >= cutoff) & (d['day'] < cutoff + 7)
        rows = np.flatnonzero(m)
        users = np.searchsorted(d['ptr'], rows, side='right') - 1
        h_end = history_end(d, cutoff)
        keep = h_end[users] > d['ptr'][users]
        return rows[keep], users[keep]

    def fit(self, cfg, holdout):
        import lightgbm as lgb
        r = cfg['rm_b']
        train_cut = self.data['cutoffs']['train']
        X, y, grp = [], [], []
        for w in range(1, r['train_weeks'] + 1):
            c = train_cut - 7 * w
            rows, users = self.week_rows(c)
            X.append(self.features(c, users, self.data['item'][rows]))
            y.append(self.data['wp'][rows])
            grp.append(users)
        X, y, grp = pd.concat(X, ignore_index=True), np.concatenate(y), np.concatenate(grp)
        es = np.isin(grp, np.random.default_rng(cfg['seed']).choice(np.unique(grp), len(np.unique(grp)) // 10, replace=False))
        self.model = lgb.LGBMRegressor(objective='cross_entropy', n_estimators=r['n_estimators'],
                                       learning_rate=r['learning_rate'], num_leaves=r['num_leaves'],
                                       min_child_samples=r['min_child_samples'], n_jobs=r['n_jobs'],
                                       random_state=cfg['seed'], verbose=-1)
        self.model.fit(X[~es], y[~es], eval_X=(X[es],), eval_y=(y[es],),
                       callbacks=[lgb.early_stopping(30, verbose=False)])
        pred = self.predict_rows(train_cut, holdout['users'], holdout['tok'])
        q = reward_quality(holdout['wp'], pred, holdout['item_mean'])
        q['train_rows'] = int((~es).sum())
        q['best_iteration'] = int(self.model.best_iteration_ or r['n_estimators'])
        log('RM-B', q)
        return q

    def predict_rows(self, cutoff, users, tokens):
        items = self.data['vocab_items'][np.asarray(tokens) - N_SPECIAL]
        return self.model.predict(self.features(cutoff, np.asarray(users), items))

    def score_lists(self, cutoff, users, top):
        """Mean predicted share for each user's list top [N, k]."""
        k = top.shape[1]
        return self.predict_rows(cutoff, np.repeat(users, k), top.ravel()).reshape(-1, k)


class RewardBClient:
    """Same interface as RewardB, executed in a torch-free subprocess."""

    def __init__(self, data_path, model_path):
        self.model_path = Path(model_path)
        self.proc = subprocess.Popen([sys.executable, __file__, str(data_path), str(model_path)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _call(self, *req):
        pickle.dump(req, self.proc.stdin)
        self.proc.stdin.flush()
        ok, value = pickle.load(self.proc.stdout)
        if not ok:
            raise RuntimeError(f'RM-B worker failed: {value}')
        return value

    def fit(self, cfg, holdout):
        return self._call('fit', cfg, holdout)

    def predict_rows(self, cutoff, users, tokens):
        return self._call('predict_rows', cutoff, users, tokens)

    def score_lists(self, cutoff, users, top):
        return self._call('score_lists', cutoff, users, top)

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait()


def worker(data_path, model_path):
    import traceback
    out = sys.stdout.buffer
    sys.stdout = sys.stderr  # keep the pickle channel clean of prints
    data = pickle.loads(Path(data_path).read_bytes())
    rm = RewardB(data)
    if Path(model_path).exists():
        rm.model = pickle.loads(Path(model_path).read_bytes())
    while True:
        try:
            req = pickle.load(sys.stdin.buffer)
        except EOFError:
            return
        try:
            value = getattr(rm, req[0])(*req[1:])
            if req[0] == 'fit':
                Path(model_path).write_bytes(pickle.dumps(rm.model))
            res = (True, value)
        except Exception:
            res = (False, traceback.format_exc())
        pickle.dump(res, out)
        out.flush()


if __name__ == '__main__':
    worker(*sys.argv[1:3])
