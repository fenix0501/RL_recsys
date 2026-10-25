"""Stage 1: SFT next-item GPT-2 and the Popular baseline."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from common import N_SPECIAL, PAD, UNK, append_jsonl, log, seed_all
from metrics import relevance, tokens_to_items
from models import Policy, gpt_config, recommend, save_policy
from sequences import chunks, pad


def popular_top(data, split, k=10, days=7):
    """Most watched vocab items in the last `days` before the cutoff, minus seen items."""
    cutoff = split.cutoff
    recent = (data['day'] < cutoff) & (data['day'] >= cutoff - days) & (data['tok'] >= N_SPECIAL)
    order = np.argsort(-np.bincount(data['tok'][recent], minlength=N_SPECIAL + len(data['vocab_items'])),
                       kind='stable')
    order = order[order >= N_SPECIAL][:k + 200]
    top = []
    for u, e in zip(split.users, split.end):
        seen = set(data['tok'][data['ptr'][u]:e].tolist())
        top.append([t for t in order if t not in seen][:k])
    return np.array(top)


def evaluate_policy(model, data, split, cfg, device, k=10):
    top = recommend(model, data, split.users, split.end, cfg['max_len'], device, k=k)
    return relevance(tokens_to_items(data, top), split.targets, k).mean().to_dict(), top


def train_sft(data, val, cfg, device, out_dir, smoke=False):
    g = cfg['gpt']
    seed_all(cfg['seed'])
    windows = chunks(data, cfg['max_len'])
    vocab = N_SPECIAL + len(data['vocab_items'])
    model = Policy(gpt_config(vocab, g, cfg['max_len'])).to(device)
    log(f'SFT: {len(windows)} windows, {sum(p.numel() for p in model.parameters())/1e6:.2f}M params, device {device}')
    opt = torch.optim.AdamW(model.parameters(), lr=g['lr'], weight_decay=g['weight_decay'])
    epochs = 2 if smoke else g['max_epochs']
    steps_total = epochs * (len(windows) // g['batch_size'] + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=g['lr'], total_steps=steps_total, pct_start=0.05)
    rng = np.random.default_rng(cfg['seed'])
    best, bad = -1.0, 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(windows))
        losses = []
        for s in range(0, len(order), g['batch_size']):
            batch = [windows[i][0] for i in order[s:s + g['batch_size']]]
            x = torch.from_numpy(pad(batch)).to(device)
            y = x[:, 1:].clone()
            y[(y == PAD) | (y == UNK)] = -100
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        metrics, _ = evaluate_policy(model, data, val, cfg, device)
        row = {'epoch': epoch, 'train_loss': float(np.mean(losses)), **{f'val_{k}': v for k, v in metrics.items()}}
        append_jsonl(out_dir / 'sft_log.jsonl', row)
        log('SFT', row)
        if metrics['ndcg10'] > best:
            best, bad = metrics['ndcg10'], 0
            save_policy(model, out_dir / 'sft.pt', epoch=epoch, val=metrics)
        else:
            bad += 1
            if bad >= g['patience']:
                break
    return best
