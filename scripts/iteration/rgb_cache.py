"""Bounded verified RGB decode cache for a future, immutable data adapter.

No model/processor imports. Every hit rechecks file identity; every miss rechecks
SHA and format. Cache admission still requires full snapshot auditing and an
equivalence test. It is deliberately not wired into the PI0292 NativeDataset.
"""
from collections import OrderedDict
import hashlib
import io
import os
from pathlib import Path
import stat


def file_identity(info):
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


class VerifiedRGBLRU:
    def __init__(self, root, max_bytes):
        if type(max_bytes) is not int or max_bytes<=0:raise ValueError('BOUNDED_CACHE_REQUIRED')
        self.root=Path(root).resolve(strict=True);self.max_bytes=max_bytes
        self.rows=OrderedDict();self.bytes=0;self.hits=0;self.misses=0

    def get(self, relative, expected_sha256):
        path=(self.root/relative).resolve(strict=True)
        if not path.is_relative_to(self.root):raise ValueError('RGB_PATH_ESCAPE')
        with path.open('rb') as handle:
            info=os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):raise ValueError('REGULAR_RGB_REQUIRED')
            identity=file_identity(info);key=(str(path),expected_sha256)
            if key in self.rows:
                previous,value=self.rows[key]
                if previous!=identity:raise ValueError('CACHED_RGB_IDENTITY_CHANGED')
                if file_identity(path.stat())!=identity:raise ValueError('RGB_CHANGED_DURING_READ')
                self.hits+=1;self.rows.move_to_end(key)
                return value.copy()
            raw=handle.read()
            if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('RGB_HASH_CHANGED')
            if file_identity(os.fstat(handle.fileno()))!=identity or file_identity(path.stat())!=identity:
                raise ValueError('RGB_CHANGED_DURING_READ')
        import numpy as np
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as image:
            if image.mode!='RGB' or image.size!=(256,256):raise ValueError('RGB_FORMAT_CHANGED')
            value=np.asarray(image).copy()
        self.misses+=1
        if value.nbytes<=self.max_bytes:
            while self.rows and self.bytes+value.nbytes>self.max_bytes:
                _,(_,removed)=self.rows.popitem(last=False);self.bytes-=removed.nbytes
            value.flags.writeable=False
            self.rows[key]=(identity,value);self.bytes+=value.nbytes
        return value.copy()
