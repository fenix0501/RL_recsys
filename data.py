"""Pinned KION download and the calendar weeks shared by every stage."""
from __future__ import annotations

import hashlib
import subprocess
import zipfile

import pandas as pd

from common import TASK2, log

# Full three-table archive; the separate interactions.csv.zip in the same repository is truncated.
COMMIT = 'f69775be31fa5779907cf0a92ddedb70037fb5ae'
SHA256 = '50fc5b7db78cbda4c3d0ee5bb5bea4c416cff41e8af58aab0b661d0d3e9dfa9b'
URL = f'https://raw.githubusercontent.com/irsafilo/KION_DATASET/{COMMIT}/data_original.zip'
RAW = TASK2 / 'data/raw/data_original.zip'
N_ROWS = 5_476_251


def download():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        log('downloading', URL)
        part = RAW.with_suffix('.zip.part')
        subprocess.run(['curl', '-fLsS', '--retry', '3', '--connect-timeout', '15',
                        '--max-time', '600', URL, '-o', str(part)], check=True)
        part.replace(RAW)
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f'KION archive checksum {digest} does not match the pinned input')
    return RAW


def load_raw():
    download()
    cache = TASK2 / f'data/cache/interactions_{SHA256[:12]}.parquet'
    if cache.exists():
        events = pd.read_parquet(cache)
    else:
        with zipfile.ZipFile(RAW) as archive, archive.open('data_original/interactions.csv') as stream:
            events = pd.read_csv(stream)
        events['last_watch_dt'] = pd.to_datetime(events['last_watch_dt'], format='%Y-%m-%d')
        events['user_id'] = events['user_id'].astype('int32')
        events['item_id'] = events['item_id'].astype('int32')
        cache.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(cache, index=False)
    if len(events) != N_ROWS:
        raise ValueError(f'Incomplete KION input: {len(events)} rows, expected {N_ROWS:,}')
    with zipfile.ZipFile(RAW) as archive:
        with archive.open('data_original/users.csv') as stream:
            users = pd.read_csv(stream)
        with archive.open('data_original/items.csv') as stream:
            items = pd.read_csv(stream)
    return events, users, items


def split_dates(events):
    # A complete calendar week is Monday 00:00 through next Monday 00:00.
    observed_end = events['last_watch_dt'].max().normalize() + pd.Timedelta(days=1)
    end = observed_end - pd.Timedelta(days=observed_end.weekday())
    return {name: (end - pd.Timedelta(days=offset), end - pd.Timedelta(days=offset - 7))
            for name, offset in [('train', 21), ('validation', 14), ('test', 7)]}


if __name__ == '__main__':
    print(download(), SHA256)
