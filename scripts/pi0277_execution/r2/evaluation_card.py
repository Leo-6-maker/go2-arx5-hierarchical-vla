"""Freeze registered evaluation inputs/noise without importing a model or simulator."""
import hashlib,json,os
from pathlib import Path
import numpy as np
COMMON=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0277');ROOT=COMMON/'R2'
OLD=COMMON.with_name('PI0276')
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0277/R2')
OLD_RAW=RAW.parent.with_name('PI0276')
RELEASE='1a20200ff066ef5a003650110cd20106f9a4bf64'

def read(p):return json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
    with tmp.open('w') as f:json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)

def freeze(p,value):
    if Path(p).exists():assert read(p)==value,p
    else:write(p,value)

def packet(row):
    folder=Path(row['episode']);index=row['anchor']
    with np.load(folder/'observations.npz',allow_pickle=False) as z:
        value=dict(body_rgb=z['base_camera_96'][index],wrist_rgb=z['hand_camera_96'][index],state_v1=z['state_v1'][index])
    assert value['state_v1'].shape==(10,)
    assert all(value[k].shape==(96,96,3) and value[k].dtype==np.uint8 for k in ('body_rgb','wrist_rgb'))
    return value
