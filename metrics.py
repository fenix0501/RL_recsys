"""Relevance and behaviour metrics on top-k lists (same protocol for every model)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import N_SPECIAL


def tokens_to_items(data, top):
    return data['vocab_items'][top - N_SPECIAL]


def relevance(top_items, targets, k=10):
    """Per-user Recall@k (hits / |targets|, as in task 1), NDCG@k and MAP@k."""
    disc = 1 / np.log2(np.arange(k) + 2)
    rows = []
    for rec, true in zip(top_items[:, :k], targets):
        hit = np.fromiter((i in true for i in rec), dtype=float, count=len(rec))
        n = len(true)
        ideal = disc[:min(n, k)].sum()
        prec = np.cumsum(hit) / np.arange(1, len(hit) + 1)
        rows.append((hit.sum() / n, (hit * disc).sum() / ideal, (prec * hit).sum() / min(n, k)))
    return pd.DataFrame(rows, columns=['recall10', 'ndcg10', 'map10'])


def behaviour(data, top, ref_top=None):
    """Diversity, popularity and content descriptors of the recommended lists."""
    meta = data['item_meta']
    idx = top - N_SPECIAL
    counts = np.bincount(idx.ravel(), minlength=len(meta)).astype(float)
    srt = np.sort(counts)
    n = len(srt)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(srt) / (n * srt.sum())) if srt.sum() else 0.0
    out = {
        'coverage': float((counts > 0).mean()),
        'gini': gini,
        'top10_items_share': float(np.sort(counts)[::-1][:10].sum() / counts.sum()),
        'log_popularity': float(np.log1p(meta['count'].to_numpy()[idx]).mean()),
        'item_mean_wp': float(np.nanmean(meta['mean_wp'].to_numpy()[idx])),
        'dur_h': float(meta['dur_h'].to_numpy()[idx].mean()),
        'series_share': float(meta['is_series'].to_numpy()[idx].mean()),
        'genre_diversity': genre_diversity(meta['genres'].to_numpy()[idx]),
    }
    if ref_top is not None:
        inter = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(top, ref_top)]
        out['jaccard_vs_sft'] = float(np.mean(inter))
    return out


def genre_diversity(genres, max_users=3000):
    """Mean pairwise Jaccard distance of genre sets inside a list."""
    vals = []
    for row in genres[:max_users]:
        sets = [set(g.split(', ')) - {''} for g in row]
        d = [1 - len(a & b) / max(1, len(a | b)) for i, a in enumerate(sets) for b in sets[i + 1:]]
        vals.append(np.mean(d) if d else 0)
    return float(np.mean(vals))
