"""Typed offline target snapshot container. This does not export or restore a live controller."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np

REQUIRED = {
    'simulator':('integration',),
    'wbc':('observation_history','previous_action','controller_hidden'),
    'mpc':('solver','policy','warmstart'),
    'receiver':('installed_reference','pending_reference','progress','clock_map'),
    'gripper':('actuator_state','calibration'),
    'rng':('generator_state',),
    'monitor':('task_history','safety_history'),
}


def array_hash(array):
    a=np.ascontiguousarray(array)
    return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def validate(blocks,identity,clock):
    if identity.get('backend')!='GO2_ARX5_MUJOCO_OCS2' or not identity.get('frozen_files_sha256'):
        raise ValueError('Explicit target identity is mandatory')
    if any(not isinstance(h,str)or not re.fullmatch('[0-9a-f]{64}',h)for h in identity['frozen_files_sha256'].values()):
        raise ValueError('Full frozen file SHA256 identity required')
    if (type(clock.get('native_tick'))is not int or clock['native_tick']<0
            or clock.get('physics_dt_s')!=.002 or not np.isfinite(clock.get('time_s',np.nan))):
        raise ValueError('Native 2ms clock is mandatory')
    if abs(clock['time_s']-clock['native_tick']*.002)>1e-8:
        raise ValueError('Absolute simulator tick/time differs; keep MPC clock offsets separately')
    if set(blocks)!=set(REQUIRED):raise ValueError('qpos/qvel alone is not a complete control snapshot')
    arrays={}
    for block,fields in REQUIRED.items():
        if set(blocks[block])!=set(fields):raise ValueError('Missing control state:'+block)
        for field in fields:
            value=np.asarray(blocks[block][field])
            if value.dtype.kind not in 'biuf' or value.ndim>8 or not value.size or not np.isfinite(value).all():
                raise ValueError('Finite typed numeric payload required; no Python object serialization')
            arrays[block+'.'+field]=np.ascontiguousarray(value)
    return arrays


def save_snapshot(path,blocks,identity,clock):
    arrays=validate(blocks,identity,clock)
    meta=dict(schema='WM02_OFFLINE_CONTROL_CONTAINER_V1',identity=identity,clock=clock,
        native_exporter_bound=False,live_restore_performed=False,
        fields={k:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=array_hash(v),role='RESTORE_ONLY')for k,v in arrays.items()})
    arrays['__metadata_utf8']=np.frombuffer(json.dumps(meta,sort_keys=True,allow_nan=False).encode(),dtype=np.uint8)
    with Path(path).open('xb')as stream:np.savez_compressed(stream,**arrays)
    return meta


def load_snapshot(path,expected_identity):
    with np.load(path,allow_pickle=False)as saved:
        meta=json.loads(saved['__metadata_utf8'].tobytes().decode())
        if meta.get('schema')!='WM02_OFFLINE_CONTROL_CONTAINER_V1'or meta['identity']!=expected_identity:
            raise ValueError('Snapshot source/controller identity differs')
        if meta.get('native_exporter_bound')is not False or meta.get('live_restore_performed')is not False:
            raise ValueError('Offline container cannot claim a native exporter or restore')
        if set(saved.files)!=set(meta['fields'])|{'__metadata_utf8'}:raise ValueError('Container fields differ')
        blocks={name:{}for name in REQUIRED}
        for key,desc in meta['fields'].items():
            if not re.fullmatch(r'[a-z_]+\.[a-z_]+',key):raise ValueError('Invalid field')
            a=saved[key].copy()
            if desc.get('role')!='RESTORE_ONLY'or a.dtype.str!=desc['dtype']or list(a.shape)!=desc['shape']or array_hash(a)!=desc['sha256']:
                raise ValueError('Snapshot bytes/shape/dtype differ')
            block,field=key.split('.');blocks[block][field]=a
    validate(blocks,meta['identity'],meta['clock']);return blocks,meta
