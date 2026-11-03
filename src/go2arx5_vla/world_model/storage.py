"""Private local transition streams; only load trusted project artifacts."""
import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path


def jsonable(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(jsonable(value), stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def dump_private(path, value):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
            pickle.dump(value, stream, protocol=5)
        raw.flush()
        os.fsync(raw.fileno())


def load_private(path):
    with gzip.open(path, 'rb') as stream:
        return pickle.load(stream)
