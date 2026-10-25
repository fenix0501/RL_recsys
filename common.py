"""Paths, device, seeding and small IO helpers shared by all task-2 stages."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

# torch is imported lazily: the RM-B worker process (LightGBM) must never load it,
# because two OpenMP runtimes in one process crash on macOS.

TASK2 = Path(__file__).resolve().parent

PAD, BOS, UNK = 0, 1, 2
N_SPECIAL = 3


def pick_device(name='auto'):
    import torch
    if name != 'auto':
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + '\n')
    tmp.replace(path)


def append_jsonl(path, row):
    with open(path, 'a') as f:
        f.write(json.dumps(row, default=float) + '\n')


def atomic_save(obj, path):
    import torch
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)
