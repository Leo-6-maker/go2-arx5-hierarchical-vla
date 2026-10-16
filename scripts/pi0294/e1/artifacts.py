"""Atomic completed numeric payloads shared by CPU and neural workers."""
import os
from pathlib import Path
import numpy as np


def atomic_npz(path,**arrays):
    path=Path(path);temp=path.with_name(path.name+'.pending.'+str(os.getpid()))
    with temp.open('xb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    temp.chmod(0o600);os.replace(temp,path)
