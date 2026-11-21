"""Package only individually admitted TRAIN episodes on the four-episode or six-hour clock."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from hopper_parent_custody import PLAN, digest, sync_dir, write_once


def read(path):
    return json.loads(path.read_bytes())


def read_pinned(pin):
    path=Path(pin['path'])
    if digest(path)!=pin['sha256']:
        raise ValueError('Producer evidence changed: '+path.name)
    return read(path)


def object_file(source, expected, depot, shared=None):
    obj=depot/expected
    if obj.is_file():
        if obj.stat().st_size!=source.stat().st_size or obj.stat().st_mode & 0o222:
            raise ValueError('Snapshot object changed')
        return obj
    if shared is not None:
        old=Path(shared)/expected
        if old.is_file() and not old.is_symlink() and not old.stat().st_mode & 0o222:
            if old.stat().st_size!=source.stat().st_size or digest(old)!=expected:
                raise ValueError('Original snapshot object differs')
            os.link(old,obj)
            sync_dir(depot)
            return obj
    with tempfile.NamedTemporaryFile(dir=depot,delete=False) as out:
        temp=Path(out.name)
        with source.open('rb') as f:
            shutil.copyfileobj(f,out,1024*1024)
        out.flush()
        os.fsync(out.fileno())
    if digest(temp)!=expected:
        raise ValueError('Snapshot copy differs from original capture: '+source.name)
    temp.chmod(0o400)
    temp.rename(obj)
    sync_dir(depot)
    return obj


def publish(root, producer, rows, cfg, *, origins=None):
    n=len(rows)
    final=root/'snapshots'/f'TRAIN-{n:03d}'
    if final.exists():
        raise FileExistsError('Reconcile already published snapshot')
    stage=Path(tempfile.mkdtemp(prefix=final.name+'.incoming.',dir=final.parent))
    episodes=[]
    cases=[]
    seen_results=set()
    seen_payloads=set()
    for row in rows:
        ref=row['episode']
        ep=read_pinned(ref)
        eid=ep['id']
        folder=origins[eid]['root'] if origins is not None else producer
        location=folder/'admission/TRAIN'
        epdir=location/eid
        if (epdir.resolve().parent!=location.resolve() or Path(ref['path'])!=epdir/'EPISODE.private.json'
                or eid!=row['condition'] or ep['split']!='TRAIN' or ep['individual_pass'] is not True
                or not row['task_and_mobile_success'] or row['hard_STOP'] or row['technical_failure']
                or row['requests']!=1 or ep['native_result_sha256']!=row['result']['sha256']
                or ep['windows']!=ref['windows'] or ref['confirmed_P10_atoms']!=10*ep['windows']):
            raise ValueError('Invalid TRAIN admission: '+eid)
        native=read_pinned(row['result'])
        if native['physical_requests']!=1 or native['actual_native_integrations']!=row['ticks']:
            raise ValueError('Native denominator changed: '+eid)
        if origins is not None:
            metadata=origins[eid]
            case=folder/'cases'/eid
            if (Path(row['result']['path'])!=case/'RESULT.private.json'
                    or native['source_sha']!=metadata['source'] or native['condition']['collection_episode']!=eid
                    or read(case/'EXECUTION_SPEC.json')['source_sha']!=metadata['source']):
                raise ValueError('Producer differs from the actual native case: '+eid)
        members=[ep['labels'],*ep['frame_shards']]
        payload=tuple(m['sha256'] for m in members)
        if ep['native_result_sha256'] in seen_results or payload in seen_payloads:
            raise ValueError('Duplicate capture in cumulative snapshot')
        seen_results.add(ep['native_result_sha256'])
        seen_payloads.add(payload)
        (stage/eid).mkdir()
        for m in members:
            rel=Path(m['file'])
            origin=location/rel
            if rel.parent!=Path(eid) or origin.is_symlink() or origin.resolve().parent!=epdir.resolve():
                raise ValueError('Capture member outside admitted episode')
            os.link(object_file(origin,m['sha256'],root/'objects',cfg.get('shared_objects')),stage/rel)
        sync_dir(stage/eid)
        episodes.append(ep)
        cases.append(dict(id=eid,instruction=ep['task'],windows=ep['windows'],confirmed_P10_atoms=ref['confirmed_P10_atoms'],
                   scene_sha256=ep['scene_sha256'],reset=ep['reset'],native_result_sha256=ep['native_result_sha256'],
                   episode_sha256=ref['sha256'],native_ticks=row['ticks'],admitted=True))
        if origins is not None:
            cases[-1].update(producer_spec=metadata['spec'],producer_source=metadata['source'])
    total=sum(ep['windows'] for ep in episodes)
    manifest=dict(schema='SLA_TARGET_ADAPTATION_NEW_EXPERTS_V1',plan=PLAN,source=cfg['source_revision'],
                  representation='DENSE_REFERENCE_G7_V6',state='committed',split='TRAIN',episodes=episodes,
                  statistics_sha256=cfg['C20_statistics_sha256'],windows=total,live_SCALE_training_data_changed=False,
                  phase_input_to_student=False,teacher_memory_input=False,unconsumed_suffix_is_executed=False)
    write_once(stage/'MANIFEST.private.json',manifest)
    msha=digest(stage/'MANIFEST.private.json')
    proof=dict(cases=cases,admitted_episodes=n,admitted_windows=total,dataset_sha256=msha)
    if origins is None:
        proof['producer_spec']=cfg['producer_spec_sha256']
    else:
        from hopper_snapshot_producers import SCHEMA, ORIGINAL, check_admission_provenance
        proof.update(schema=SCHEMA,original_campaign_spec=ORIGINAL)
        check_admission_provenance(proof)
    write_once(stage/'ACTUAL.private.json',proof)
    write_once(stage/'SEALED.private.json',dict(plan=PLAN,split='TRAIN',count=n,windows=total,
               manifest_sha256=msha,admission_sha256=digest(stage/'ACTUAL.private.json'),
               epoch=time.time(),episode_sha256=[r['episode']['sha256'] for r in rows]))
    sync_dir(stage)
    stage.rename(final)
    sync_dir(final.parent)


def run(root):
    cfg=read(root/'MOUNT.private.json')
    producer=Path(cfg['producer'])
    if (cfg['plan']!=PLAN or os.getuid()!=cfg['uid'] or digest(Path(__file__))!=cfg['worker_sha256']
            or digest(producer/'EXECUTION_SPEC.json')!=cfg['producer_spec_sha256']):
        raise ValueError('Snapshot owner, source or producer identity differs')
    scheduled=cfg['T0_epoch']+21600
    previous=[]
    existing=sorted((root/'snapshots').glob('TRAIN-???'))
    if existing:
        last=read(existing[-1]/'SEALED.private.json')
        previous=last['episode_sha256']
        scheduled=last['epoch']+21600
    while True:
        now=time.time()
        stop=(root/'STOP.json').exists() or (producer/'RESULT.private.json').is_file() or now>=cfg['deadline_epoch']
        src=producer/'RECORDS.private.json'
        rows=[r for r in (read(src) if src.exists() else []) if r.get('admitted') and r['split']=='TRAIN']
        pins=[r['episode']['sha256'] for r in rows]
        if pins[:len(previous)]!=previous:
            raise ValueError('Previously sealed TRAIN membership changed')
        pending=len(rows)-len(previous)
        while pending>=4 or (pending and (now>=scheduled or stop)):
            count=len(previous)+min(pending,4)
            publish(root,producer,rows[:count],cfg)
            previous=pins[:count]
            pending=len(rows)-count
            scheduled=time.time()+21600
        pulse=dict(pid=os.getpid(),epoch=now,admitted=len(rows),published=len(previous),
                   next_six_hour_epoch=scheduled,terminal=stop)
        # Heartbeat and terminal have separate lifetimes.
        tmp=root/'heartbeat.pending'
        tmp.write_text(json.dumps(pulse))
        os.replace(tmp,root/'HEARTBEAT.private.json')
        if stop:
            write_once(root/'RESULT.private.json',pulse)
            return
        time.sleep(min(30,max(.1,cfg['deadline_epoch']-time.time())))


if __name__=='__main__':
    os.umask(0o077)
    root=Path.cwd()
    with (root/'OWNER.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(root)
        except Exception as error:
            write_once(root/'FAILURE.private.json',dict(reason=str(error),epoch=time.time()))
            raise
