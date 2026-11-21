"""Finish or reconcile the immutable bank handoff without regenerating any visits."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

NAMES = {'BANKS.private.json','PARENT_UPDATES.private.jsonl','target.npy','robodojo.npy','libero.npy',
         'robodojo_noise_cursors.npy','libero_noise_cursors.npy','rates.npy','rng.npy'}


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected regular file '+path.name)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
        os.fsync(f.fileno())
    if h.hexdigest()!=expected:
        raise ValueError('Bank package changed at '+path.name)


def finalize(dest, checksum):
    with dest.with_name(dest.name+'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        incoming=dest.with_name(dest.name+'.incoming')
        if incoming.exists() and dest.exists():
            raise ValueError('Conflicting bank directories')
        tree=dest if dest.exists() else incoming
        if tree.is_symlink() or not tree.is_dir():
            raise ValueError('Missing bank directory')
        verify(tree/'SHA256SUMS',checksum)
        entries={line[66:]:line[:64] for line in (tree/'SHA256SUMS').read_text().splitlines()}
        if set(entries)!=NAMES or {p.name for p in tree.iterdir()}!=NAMES|{'SHA256SUMS'}:
            raise ValueError('Bank membership differs')
        for name,sha in entries.items():
            verify(tree/name,sha)
        sync_dir(tree)
        if tree==incoming:
            incoming.rename(dest)
        sync_dir(dest.parent)
        return entries['BANKS.private.json']


if __name__=='__main__':
    print(finalize(Path(sys.argv[1]),sys.argv[2]))
