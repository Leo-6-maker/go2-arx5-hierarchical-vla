"""One persistent serial dispatcher; reconnect inspects it instead of restarting."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback

from .durable import OwnerLock, read, write
from .resource_guard import Budget


def main(root, python, phase, resume_audit=False):
    root = Path(root)
    lock = OwnerLock(root/'DISPATCH_OWNER.lock')
    state_path = root/'evidence'/('DISPATCH_'+phase+('_resume_audit' if resume_audit else '')+'.json')
    if state_path.exists():
        raise RuntimeError('EXISTING_DISPATCH_REQUIRES_STATUS_INSPECTION_NOT_RESTART')
    assert read(root/'RUNNER_RELIABILITY_TEST.json')['status']=='PASS'
    assert read(root/'WORKSPACE_HANDSHAKE.json')['status']=='PASS'
    assert read(root/'SCIENCE_SOURCE_EQUIVALENCE.json')['status']=='PASS'
    if phase=='supplement':
        assert read(root/'ENV_CANARY_GATE.json')['status']=='PASS'
    plan = read(root/'ATTEMPT_PLAN.json')
    order = plan['order'][:2] if phase=='canary' else plan['order'][2:]
    authority = read(root/'ADOPTION.json')
    source = read(root/'CODE_SEAL.json')['source_sha']
    resume_ids=set()
    if resume_audit:
        prior=read(root/'evidence'/('DISPATCH_'+phase+'.json'))
        assert prior['status']=='TERMINAL_STOP' and prior['reason']=='RAW_AUDIT_ENGINEERING_FAILURE'
        assert not Path('/proc',str(prior['pid'])).exists(), 'Prior owner handle must be absent'
        ledger=read(root/'RESOURCE_LEDGER.json')
        resume_ids={r['id'] for r in ledger['starts'] if r['id'] in order}
        assert all(r['status']=='VALID_COMPLETE' for r in ledger['starts'] if r['id'] in resume_ids)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'code')+':'+str(root/'code/src'),
        CUDA_VISIBLE_DEVICES='',MUJOCO_GL='egl',LIBGL_ALWAYS_SOFTWARE='1',
        __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json',
        LP_NUM_THREADS='4',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env['PYTHONPATH']+=':'+str(root/'code/scripts/pi0291_arx5_data')
    env.pop('MUJOCO_EGL_DEVICE_ID',None)
    state = dict(pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],
        status='RUNNING',phase=phase,order=order,current=None,records=[],began=time.time())
    write(state_path,state)
    def child(label, command, cap):
        with (root/'logs'/f'{label}.stdout').open('xb') as out, (root/'logs'/f'{label}.stderr').open('xb') as err:
            proc = subprocess.Popen(command,cwd=root/'code',env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
            state['child_pid']=proc.pid
            write(state_path,state)
            try:
                return proc.wait(timeout=min(cap,max(.01,authority['new_work_cutoff_epoch']-time.time())))
            except subprocess.TimeoutExpired:
                # Only the child launched by this owner is signalled.
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise RuntimeError('OWN_CHILD_TIME_BUDGET_EXHAUSTED')
            finally:
                state['child_pid']=None
                write(state_path,state)
    try:
        for identifier in order:
            if time.time()+1200 >= authority['new_work_cutoff_epoch']:
                raise RuntimeError('INSUFFICIENT_EPISODE_AND_AUDIT_TIME')
            state.update(current=identifier,updated=time.time())
            write(state_path,state)
            rc=0 if identifier in resume_ids else child(identifier+'_physics',[python,'-m','scripts.pi0292.run_episode','--root',str(root),'--id',identifier,'--source-sha',source],920)
            result=read(root/'episodes'/identifier/'RESULT.json')
            if rc or result['status']!='VALID_COMPLETE':
                state['records'].append(dict(id=identifier,physics_exit=rc,result=result,admitted=False))
                raise RuntimeError('PHYSICAL_OR_OWNER_FAILURE_PRESERVED')
            audit_path=root/'evidence'/('AUDIT_BOUND_'+identifier+'.json')
            meter=Budget(root)
            try:
                rc=child(identifier+'_audit_bound',[python,str(Path(__file__).with_name('audit_data.py')),
                    '--root',str(root),'--ids',identifier,'--physics-code-root',str(root/'code'),'--output',str(audit_path)],300)
            finally:meter.finish_cpu('AUDIT_'+identifier)
            if rc:raise RuntimeError('RAW_AUDIT_ENGINEERING_FAILURE')
            audit=read(audit_path)['episodes'][0]
            admitted=(audit['status']=='PASS_FULL_RAW_AUDIT' and audit['independent_score']['persistent_success']
                      and audit['camera_visibility']['initial_and_grasp_carry_release_visible'])
            rc=child(identifier+'_video',[python,'-m','scripts.pi0292.video','--root',str(root),'--id',identifier],300)
            if rc:raise RuntimeError('VIDEO_ENGINEERING_FAILURE')
            state['records'].append(dict(id=identifier,physics_exit=0,audit_status=audit['status'],
                admitted=admitted,score=audit['independent_score'],video_exit=rc))
            write(state_path,state)
            if not audit['independent_score']['safety_valid']:
                raise RuntimeError('REAL_SAFETY_VIOLATION_STOP')
            if phase=='canary' and not admitted:
                raise RuntimeError('ENVIRONMENT_CANARY_QUALITY_GATE_FAIL')
        state['status']='TERMINAL_SUCCESS'
        if phase=='canary':
            write(root/'ENV_CANARY_GATE.json',dict(status='PASS',records=state['records'],excluded_from_TRAIN_DEV=True))
    except Exception as exc:
        state.update(status='TERMINAL_STOP',reason=str(exc),traceback=traceback.format_exc())
    finally:
        state.update(current=None,updated=time.time())
        write(state_path,state)
        lock.close()
    return 0 if state['status']=='TERMINAL_SUCCESS' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--python',required=True)
    p.add_argument('--phase',choices=['canary','supplement'],required=True);p.add_argument('--resume-audit',action='store_true');a=p.parse_args()
    raise SystemExit(main(a.root,a.python,a.phase,a.resume_audit))
