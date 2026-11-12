"""Finite native-camera GPU child of the original registry/Journal/neural owner."""
from __future__ import annotations
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from contextlib import closing


def run(path):
    path=Path(path);root=path.parent;s=json.loads(path.read_bytes());source=Path(s['source_root'])
    sys.path[:0]=[str(source/'src'),str(source)]
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model.native_arx5_training_profile_v1 import SCOPE
    from go2arx5_vla.world_model.native_arx5_training_jobs_v1 import run as science
    if (os.getuid()!=2009 or os.uname().nodename!='corelab-4080s' or s['science_role']!=SCOPE
        or s['scope']!=SCOPE or s['physics']is not False or s['WBC_changes']is not False
        or s['owner']!='EXISTING_LOCAL_PC' or str(root)!=s['registered_root']):
        raise PermissionError('EXACT_NEW_REGISTERED_CORE_GPU_CHILD')
    owner=Path(s['owner_root']);spec_sha=digest(path);started=time.monotonic();last_check=0.
    def guard():
        nonlocal last_check
        now=time.time()
        if now+60>=s['deadline_epoch']or(root/'STOP.json').exists()or(owner/'STOP.json').exists():
            raise RuntimeError('FINITE_DEADLINE_OR_ORIGINAL_STOP')
        if time.monotonic()-last_check<1:return
        lease=json.loads((root/'control/LEASE.json').read_bytes())
        if lease['status']!='ACTIVE'or lease['spec_sha256']!=spec_sha or digest(path)!=spec_sha:
            raise PermissionError('SAME_OWN_ACTIVE_SPEC_LEASE')
        last_check=time.monotonic()
    with closing(OwnerLock(root/'OWNER.lock')),closing(OwnerLock(s['neural_owner_lock'])):
        guard()
        reg=json.loads(Path(s['registry']).read_bytes())
        if not any(e.get('path')==str(root)and e.get('scope')==SCOPE
                   and e.get('source_sha')==s['source_sha']for e in reg['entries']):
            raise PermissionError('ORIGINAL_REGISTRY_HAS_OWN_NEW_GPU_CHILD')
        for name,pin in s['source_files'].items():
            p=source/name
            if not p.resolve().is_relative_to(source.resolve())or digest(p)!=pin:
                raise PermissionError('IMMUTABLE_SOURCE_CHANGED:'+name)
        journal=Journal(owner/'control/journal',owner.name);key=root.name.upper()+'-SCIENCE'
        failure=None;code=1;actual=None
        try:
            actual=journal.once(key,dict(spec_sha256=spec_sha),lambda:science(s,root,guard))
            code=0
        except BaseException:
            failure=traceback.format_exc();write(root/'FAILURE.private.json',dict(error=failure))
        usage=resource.getrusage(resource.RUSAGE_SELF)
        result=dict(exit_code=code,failure=failure,science=actual,wall_s=time.monotonic()-started,
                    CPU_s=usage.ru_utime+usage.ru_stime,max_RSS_KiB=usage.ru_maxrss,source_sha=s['source_sha'])
        write(root/'RESULT.private.json',result)
        write(root/'WORKER_EXIT.private.json',dict(exit_code=code,failure=failure))
        lease=json.loads((root/'control/LEASE.json').read_bytes())
        write(root/'control/LEASE.json',dict(lease,status='CLOSED',worker_exit_code=code,
                    result_sha256=digest(root/'RESULT.private.json'),closed_epoch=time.time()))
        print(json.dumps(dict(exit_code=code,wall_s=result['wall_s'],failure=failure)),flush=True)
        return code


if __name__=='__main__':
    raise SystemExit(run(sys.argv[1]))
