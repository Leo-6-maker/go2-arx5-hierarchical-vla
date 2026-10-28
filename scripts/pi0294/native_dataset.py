"""Qualified PI0292 native adapter with bounded immutable-identity CPU caches."""
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.pi0292.train_adapter import NativeDataset as OriginalDataset
from scripts.pi0292.train_adapter import ProcessedCollator, official_import, sha
from scripts.iteration.rgb_cache import file_identity


class ByteLRU:
    def __init__(self, limit):
        self.limit,self.bytes,self.values = int(limit),0,OrderedDict()
        if self.limit <= 0:
            raise ValueError('POSITIVE_CACHE_BOUND_REQUIRED')

    def get(self,key):
        value = self.values.pop(key,None)
        if value is not None:
            self.values[key] = value
            return value[0]

    def put(self,key,value,size):
        if size > self.limit:
            return
        old = self.values.pop(key,None)
        if old is not None:
            self.bytes -= old[1]
        while self.bytes+size > self.limit:
            _,old = self.values.popitem(last=False)
            self.bytes -= old[1]
        self.values[key] = value,int(size)
        self.bytes += int(size)


class NativeDataset(OriginalDataset):
    def __init__(self,*args,numeric_cache_bytes=128*1024**2,rgb_cache_bytes=256*1024**2,**kwargs):
        super().__init__(*args,**kwargs)
        self.numeric_bound,self.rgb_bound = numeric_cache_bytes,rgb_cache_bytes
        self.numeric_cache,self.rgb_cache = ByteLRU(numeric_cache_bytes),ByteLRU(rgb_cache_bytes)

    def __getstate__(self):
        state = super().__getstate__()
        state.update(numeric_cache=ByteLRU(self.numeric_bound),rgb_cache=ByteLRU(self.rgb_bound))
        return state

    def raw(self,index):
        row,anchor = self.rows[index]
        path = self.root/row['file']
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('NUMERIC_PATH_ESCAPE')
        identity = file_identity(path.stat())
        key = (self.manifest_sha,self.statistics_sha,row['sha256'],str(path.resolve()))
        cached = self.numeric_cache.get(key)
        if cached is not None:
            previous,packet=cached
            if previous!=identity:raise ValueError('CACHED_NUMERIC_IDENTITY_CHANGED')
        else:
            if sha(path) != row['sha256']:
                raise ValueError('NUMERIC_EXPORT_CHANGED')
            with np.load(path,allow_pickle=False) as saved:
                packet = {k:saved[k].copy() for k in saved.files}
            if file_identity(path.stat())!=identity:raise ValueError('NUMERIC_CHANGED_DURING_READ')
            self.numeric_cache.put(key,(identity,packet),sum(v.nbytes for v in packet.values()))
        if not packet['valid_window'][anchor] or packet['label_representation'].item() != 'ABSOLUTE_COMMITTED_EEF10':
            raise ValueError('INVALID_ABSOLUTE_SUPERVISION')
        rawroot = Path(self.raw_bindings.get(row['id'],row['raw_root'])).resolve()
        images = []
        for camera in range(2):
            frame = (rawroot/packet['rgb_paths'][anchor,camera]).resolve()
            if not frame.is_relative_to(rawroot):
                raise ValueError('RGB_PATH_ESCAPE')
            identity = file_identity(frame.stat())
            key = (self.manifest_sha,self.statistics_sha,'RGB256_DETERMINISTIC_V1',str(frame),
                        str(packet['rgb_sha256'][anchor,camera]))
            cached = self.rgb_cache.get(key)
            if cached is not None:
                previous,image=cached
                if previous!=identity:raise ValueError('CACHED_RGB_IDENTITY_CHANGED')
            else:
                if sha(frame) != packet['rgb_sha256'][anchor,camera]:
                    raise ValueError('RGB_CHANGED')
                with Image.open(frame) as opened:
                    if opened.mode != 'RGB' or opened.size != (256,256):
                        raise ValueError('INVALID_RGB_FORMAT')
                    image = np.asarray(opened).copy()
                if file_identity(frame.stat())!=identity:raise ValueError('RGB_CHANGED_DURING_READ')
                self.rgb_cache.put(key,(identity,image),image.nbytes)
            images.append(image.copy())
        return packet['actual_state10'][anchor].copy(),packet['committed_absolute10'][anchor:anchor+40].copy(),images,packet['instruction'].item()

    def identity(self,index):
        value = super().identity(index)
        row,_ = self.rows[index]
        value.update(task=row.get('task','T1'),family=row.get('family','F0'),origin=row.get('origin','legacy'))
        return value
