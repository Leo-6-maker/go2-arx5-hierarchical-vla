from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(jsonable(value),sort_keys=True,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def manifest(folder):
    root=Path(folder)
    return [{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':file_hash(p)}
            for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ['FILE_MANIFEST.json']]
