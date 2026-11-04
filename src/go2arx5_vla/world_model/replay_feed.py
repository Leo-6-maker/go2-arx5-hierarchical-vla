"""Private replay-file bridge only. No sockets, ROS, control publisher or live-target claim."""
from pathlib import Path
import json,os,time
import numpy as np
from .storage import jsonable


def publish_replay(rows,config,path,stop,*,rate=1.):
    path=Path(path)
    if path.exists():raise FileExistsError('New private replay feed required')
    t0=rows[0]['time_s'];start=time.monotonic()
    for sequence,row in enumerate(rows,1):
        delay=start+(row['time_s']-t0)/rate-time.monotonic()
        if delay>0 and stop.wait(delay):break
        if stop.is_set():break
        message=dict(schema='WM02_READONLY_REPLAY_FEED_V1',label='TARGET_FAILED_REPLAY',source_kind='RECORDED_REPLAY_NOT_LIVE_TARGET',source_sha=config['original_native_source_sha'],scene_sha=config['scene']['sha256'],journal_sha=config['journal']['sha256'],emission_monotonic_s=time.monotonic(),recorded_monotonic_s=None,physics_t=row['time_s'],step=row['step'],sequence=sequence,qpos=row['qpos'],qvel=row['qvel'],ctrl=row['ctrl'],contacts=row['contacts'])
        tmp=path.with_suffix('.pending')
        with tmp.open('w')as f:json.dump(jsonable(message),f,allow_nan=False,separators=(',',':'));f.flush()
        os.replace(tmp,path)


def read_snapshot(path,config,last_sequence=0):
    path=Path(path)
    if not path.exists():return None
    raw=path.read_bytes()
    if len(raw)>2*1024**2:raise ValueError('Replay feed frame too large')
    v=json.loads(raw)
    if(v['schema']!='WM02_READONLY_REPLAY_FEED_V1'or v['label']!='TARGET_FAILED_REPLAY'or v['source_kind']!='RECORDED_REPLAY_NOT_LIVE_TARGET'or v['source_sha']!=config['original_native_source_sha']or v['scene_sha']!=config['scene']['sha256']or v['journal_sha']!=config['journal']['sha256']):raise PermissionError('Exact private recorded replay identity required; live target is not admitted')
    if v['sequence']<=last_sequence:return None
    q=np.array(v['qpos']);dq=np.array(v['qvel']);ctrl=np.array(v['ctrl'])
    if q.shape!=(34,)or dq.shape!=(32,)or ctrl.shape!=(20,)or not np.isfinite(np.r_[q,dq,ctrl,v['physics_t'],v['emission_monotonic_s']]).all()or abs(v['physics_t']-v['step']*.002)>1e-8:raise ValueError('Replay feed ABI/time invalid')
    return dict(step=v['step'],time_s=v['physics_t'],qpos=q,qvel=dq,ctrl=ctrl,contacts=v['contacts'],sequence=v['sequence'],emission_monotonic_s=v['emission_monotonic_s'])
