"""Finite single-A40 capacity/resume/FIT4 chain; PBS remains the scheduler.

This adapter consumes an already submitted, registered allocation. It never
submits another job, runs physics, or contacts a live Core worker.
"""
import json, os, subprocess, time
from pathlib import Path
from contextlib import closing

from . import target_gr00t_profile_v1 as profile
from .storage import digest


def pbs_script(root, source, python, wall_s):
    import shlex
    profile.need(type(wall_s) is int and 3600 <= wall_s <= 6*3600, 'FINITE_A40_SLICE')
    h, rem = divmod(wall_s, 3600); m, sec = divmod(rem, 60)
    q = shlex.quote
    return '\n'.join([
        '#!/bin/bash -l', '#PBS -N wm02_target_gr00t', '#PBS -P CFP04-CF-007',
        '#PBS -q auto', '#PBS -l select=1:ncpus=8:mem=64GB:ngpus=1:gpu_model=A40',
        f'#PBS -l walltime={h:02d}:{m:02d}:{sec:02d}', '#PBS -m n', '#PBS -r n',
        '#PBS -j oe', '#PBS -o '+str(Path(root)/'pbs.log'), 'set -eu',
        'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        'export PYTHONPATH='+q(str(Path(source)/'src')+':'+str(source)),
        'exec '+q(str(python))+' -B -m scripts.wm02.target_gr00t_a40_allocation_v1 --config '+
        q(str(Path(root)/'ALLOCATION_CONFIG.json')), ''])


def child_environment(source, root):
    """Inherit the device mask already proved to expose one PBS-assigned A40."""
    return dict(os.environ,PYTHONPATH=str(Path(source)/'src')+':'+str(source),
        TMPDIR=str(Path(root)/'tmp'),XDG_CACHE_HOME=str(Path(root)/'cache'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')


def fit4_gate(root, *, original_before=None):
    """Predeclared 4mm-task checks on matched actual free generations."""
    import numpy as np
    root = Path(root)
    before = json.loads((Path(original_before) if original_before else root/'before/RESULT.json').read_bytes())['records']
    after = json.loads((root/'after/RESULT.json').read_bytes())['records']
    profile.need(len(before) == len(after) == 4, 'FOUR_REAL_MATCHED_TRAIN_DIAGNOSTICS')
    rows = []
    for a, b in zip(before, after):
        profile.need(a['episode'] == b['episode'] and a['seed'] == b['seed']
                     and a['noise'] == b['noise'], 'MATCHED_PRE_POST_INPUT_NOISE')
        with np.load(root/'after'/(b['episode']+'.npz'), allow_pickle=False) as z:
            pred, expert, state = z['physical_H40'], z['expert_policy_H40'], z['actual_state10']
        direction = expert[-1,:3]-state[:3]
        norm = np.linalg.norm(direction)
        directional_m = float(np.dot(pred[-1,:3]-state[:3], direction)/norm) if norm else -1.
        def err(r, horizon, name):
            value = r['errors'][horizon][name]
            return value['mean'] if isinstance(value, dict) else value
        values = {k:err(r,h,n) for k,r,h,n in (
            ('before_XYZ',a,'H40','XYZ_euclidean_m'),('after_XYZ',b,'H40','XYZ_euclidean_m'),
            ('prefix8_XYZ',b,'prefix8','XYZ_euclidean_m'),('before_SO3',a,'H40','SO3_rad'),
            ('after_SO3',b,'H40','SO3_rad'),('before_grip',a,'H40','gripper_MAE'),
            ('after_grip',b,'H40','gripper_MAE'))}
        finite = all(v is not None and np.isfinite(v) for v in values.values())
        rows.append(dict(episode=b['episode'],command=b['command'],**values,
            correct_direction=directional_m > 0, directional_endpoint_m=directional_m,
            passes=bool(finite and directional_m > 0 and values['after_XYZ'] <= .001
                and values['prefix8_XYZ'] <= .001 and values['after_SO3'] <= .01
                and values['after_grip'] <= .02 and values['after_SO3'] <= values['before_SO3']+.001
                and values['after_grip'] <= values['before_grip']+.001)))
    initial = float(np.mean([r['before_XYZ'] for r in rows]))
    final = float(np.mean([r['after_XYZ'] for r in rows]))
    return dict(schema='WM02_TARGET_GR00T_FIT4_GATE_V1',rows=rows,
        passes=all(r['passes'] for r in rows) and final <= .9*initial,
        mean_XYZ_before_m=initial,mean_XYZ_after_m=final,
        continuation500_allowed=final < initial and all(r['after_SO3'] is not None
            and r['after_SO3'] <= r['before_SO3']+.001 and r['after_grip'] <= r['before_grip']+.001 for r in rows),
        thresholds=dict(position_m=.001,rotation_rad=.01,gripper=.02,
            allowed_rotation_degradation_rad=.001,allowed_gripper_degradation=.001,
            required_relative_mean_position_improvement=.1),
        original_source_B0=False,comparison='ZERO_STEP_PARENT_AND_CANDIDATE_WITH_SAME_NEW_TARGET_PROCESSOR',
        native_task_success='NOT_MEASURED_BY_OPEN_LOOP_GATE')


def compare_next_update(first, resumed):
    """Actual uninterrupted step2 versus independently reconstructed cp1->2."""
    import torch, numpy as np, hashlib
    def identity(v):
        if torch.is_tensor(v):
            a=v.detach().cpu().contiguous()
            return (str(a.dtype),tuple(a.shape),hashlib.sha256(a.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
        if isinstance(v,np.ndarray):return(str(v.dtype),v.shape,hashlib.sha256(v.tobytes()).hexdigest())
        if isinstance(v,dict):return {k:identity(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [identity(x) for x in v]
        return v
    checks={}
    for name in ('target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth'):
        a=torch.load(Path(first)/name,map_location='cpu',weights_only=False)
        b=torch.load(Path(resumed)/name,map_location='cpu',weights_only=False)
        checks[name]=identity(a)==identity(b)
    for name in ('SAMPLER_STATE.json','PARAMETER_CONTRACT.json','FROZEN_PARENT_TENSORS.json'):
        checks[name]=json.loads((Path(first)/name).read_bytes())==json.loads((Path(resumed)/name).read_bytes())
    profile.need(all(checks.values()), 'ACTUAL_RESUMED_NEXT_UPDATE_NOT_EQUAL')
    return dict(status='PASS_ACTUAL_UNINTERRUPTED_VERSUS_RESUMED_NEXT_UPDATE',checks=checks,
        next_update_parameters_optimizer_scheduler_RNG_sampler_exact=True,
        qualification_weights_used_for_fit4=False)


def with_allocation(c, operation):
    import signal
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot, once
    from scripts.pi0294.multitask.shared_vanda import job, seconds
    from scripts.pi0294.multitask.shared_site import validate_vanda, project_quota
    root=Path(c['root']); need=profile.need; start=time.monotonic();stop_requested=[]
    def stop(sig,frame):
        stop_requested.append(sig)
        write(root/'STOP_REQUESTED.json',dict(signals=stop_requested,epoch=time.time(),
            action='FORWARD_TO_OWNED_CHILD_AND_SAVE_AT_OPTIMIZER_BOUNDARY'))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    need(os.getuid()==22182 and os.environ.get('PBS_JOBID'), 'REAL_ALLOCATED_OWNER')
    jid,j=job(); need(j['Job_Name']=='wm02_target_gr00t', 'PINNED_TARGET_ALLOCATION_NAME')
    lease=json.loads((root/'control/LEASE.json').read_bytes())
    need(lease['status']=='PBS_SUBMITTED' and lease['job_id']==jid
         and lease['config_sha256']==digest(root/'ALLOCATION_CONFIG.json'), 'PINNED_NEW_PBS_SCOPE')
    now=time.time(); quota_path=Path('/home/svu/vanda/quotas/CFP04-CF-007.proj_scratch.txt')
    quota=project_quota(quota_path.read_text(),user=os.environ['USER'],report_epoch=quota_path.stat().st_mtime,observed_epoch=now)
    deadline=min(c['absolute_deadline_epoch'],now+seconds(j['Resource_List']['walltime'])-
                 seconds(j.get('resources_used',{}).get('walltime','00:00:00'))-120)
    context=dict(schema='SHARED_RAW_VANDA_A40_EXECUTION_V1',job_id=jid,owner_uid=os.getuid(),
        owner_user=os.environ['USER'],science_run_id=c['generation_id'],submission_epoch=c['submission_epoch'],
        submission_nonce=c['submission_nonce'],authorization_mode='SAME_PBS_ALLOCATION_AFTER_FRESH_SUBMISSION',
        project_receipt=c['project_receipt'],quota_receipt=quota,predicted_new_peak_bytes=c['predicted_new_peak_bytes'],
        allocation_deadline_epoch=deadline)
    validate_vanda(context,j,now=now,uid=os.getuid(),user=os.environ['USER'],hostname=os.uname().nodename,job_id=jid)
    need(now+2400 < deadline,'TIME_FOR_MODEL_QUALIFICATION_AND_CLOSEOUT')
    for name,pin in c['source_files'].items():need(digest(Path(c['source_root'])/name)==pin,'ALLOCATION_SOURCE:'+name)
    for name,pin in c['dataset_files'].items():need(digest(Path(c['bindings']['dataset'])/name)==pin,'ALLOCATION_DATA:'+name)
    journal=Journal(root/'control/journal',root.name)
    need(not (root/'ALLOCATION_STARTED.json').exists(),'UNKNOWN_ALLOCATION_NEVER_RESTART')
    write(root/'ALLOCATION_STARTED.json',dict(job_id=jid,PID=os.getpid(),epoch=now,deadline_epoch=deadline,context=context))
    # Short independent device probe exits before model construction.
    q=subprocess.run([c['python'],'-B','-c',"import json,torch; p=torch.cuda.get_device_properties(0);print(json.dumps(dict(count=torch.cuda.device_count(),name=p.name,uuid=str(p.uuid),total_memory=p.total_memory)))"],
                     text=True,capture_output=True,check=True,timeout=90)
    gpu=json.loads(q.stdout);need(gpu['count']==1 and gpu['name']=='NVIDIA A40','PBS_VISIBLE_SINGLE_A40')
    write(root/'ACTUAL_GPU.json',gpu)
    children=[]
    def child(name,mode,variant,endpoint,*,resume=None,start_step=0,generation=None,microbatch=1):
        need(not stop_requested and time.time()+2400 < deadline,'ALLOCATION_MODEL_START_AND_CLOSEOUT')
        r=root/name
        def register():
            r.mkdir();(r/'control').mkdir();(r/'tmp').mkdir();(r/'cache').mkdir()
            alias=r/'model_inputs/nvidia/Cosmos-Reason2-2B';alias.parent.mkdir(parents=True)
            alias.symlink_to(c['bindings']['cosmos'],target_is_directory=True)
            recipe=profile.recipe(c['dataset_sha256'],c['statistics_sha256'],mode=mode,variant=variant,endpoint=endpoint,microbatch=microbatch)
            s=dict(c['training_common'],registered_root=str(r),source_root=c['source_root'],source_sha=c['source_sha'],
                source_files=c['source_files'],bindings=c['bindings'],parent=c['parent'],
                profile=profile.PROFILE,kind=profile.KIND,execution_site='VANDA_A40',
                recipe=recipe,allocation_request_id=c['generation_id'],
                generation_id=generation or (c['generation_id']+'-'+name),start_step=start_step,resume=resume,
                GPU_uuid=gpu['uuid'],GR00T_updates=endpoint-start_step,WM_updates=0,Actor_updates=0,TASK_ADAPTER_updates=0,
                physics=False,WBC_changes=False,parent_manifest_sha256=profile.PARENT,role='TARGET-GR00T-'+name,
                registry=c['registry'],submission_epoch=c['submission_epoch'],submission_nonce=c['submission_nonce'],
                project_receipt=c['project_receipt'],predicted_new_peak_bytes=c['predicted_new_peak_bytes'],
                storage_roots=c['storage_roots'],deadline_epoch=deadline,closeout_reserve_s=1200)
            profile.validate(s);write(r/'EXECUTION_SPEC.json',s)
            with closing(OwnerLock(c['ledger_lock'])):
                reg,pin=snapshot(c['registry']);need(not any(e.get('path')==str(r)for e in reg['entries']),'NEW_CHILD_REGISTRY')
                reg['entries'].append(dict(path=str(r),role=s['role'],scope=profile.PROFILE,local_lease=str(r/'control/LEASE.json'),
                    owner='EXISTING_LOCAL_PC',source_sha=c['source_sha'],executor_node=os.uname().nodename))
                once(Path(c['registry']),reg,root.name+'-'+name+'-REGISTER',expected_before_sha256=pin)
            write(r/'control/LEASE.json',dict(status='ACTIVE',job_id=jid,profile=profile.PROFILE,
                spec_sha256=digest(r/'EXECUTION_SPEC.json'),deadline_epoch=deadline,GR00T_updates=endpoint-start_step))
            return dict(spec_sha256=digest(r/'EXECUTION_SPEC.json'))
        registered=journal.once(name+'-REGISTER',dict(mode=mode,variant=variant,endpoint=endpoint,resume=resume),register)
        def execute():
            env=child_environment(c['source_root'],r)
            t=time.monotonic()
            with (r/'stdout.log').open('xb')as out,(r/'stderr.log').open('xb')as err:
                p=subprocess.Popen([c['python'],'-B','-m','scripts.wm02.train_target_gr00t_v1',
                    '--spec',str(r/'EXECUTION_SPEC.json')],cwd=c['source_root'],env=env,stdout=out,stderr=err,
                    start_new_session=True)
                write(r/'ACTUAL_STARTED.json',dict(PID=p.pid,job_id=jid,epoch=time.time(),
                    proc_start=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
                sent=[]
                while p.poll() is None:
                    for sig in stop_requested[len(sent):]:
                        if p.poll()is None:os.killpg(p.pid,sig)
                        sent.append(sig)
                    if time.time() > deadline+30:
                        if p.poll()is None:os.killpg(p.pid,signal.SIGTERM)
                        try:p.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            if p.poll()is None:os.killpg(p.pid,signal.SIGKILL)
                            p.wait(timeout=20)
                            raise RuntimeError('UNKNOWN_WORKER_WITHOUT_SAFE_CLOSEOUT_PRESERVE_PROGRESS')
                    else:
                        try:p.wait(timeout=5)
                        except subprocess.TimeoutExpired:pass
            processes=subprocess.check_output(['ps','-eo','pid,pgid'],text=True,timeout=10)
            need(not any(len(v:=line.split())==2 and v[1]==str(p.pid)
                         for line in processes.splitlines()[1:]),'NO_REMAINING_OWNED_MODEL_PROCESS_GROUP')
            actual=json.loads((r/'RESULT.json').read_bytes());closed=json.loads((r/'control/LEASE.json').read_bytes())
            need(closed['status']=='CLOSED' and closed['worker_exit_code']==p.returncode,'ACTUAL_CHILD_TERMINAL')
            return dict(name=name,root=str(r),exit_code=p.returncode,result_sha256=digest(r/'RESULT.json'),
                status=actual['status'],GR00T_updates=actual.get('GR00T_updates',0),
                global_step=actual.get('global_step'),requested_endpoint=endpoint,
                endpoint_complete=actual.get('global_step')==endpoint,
                PBS_GPU_wall_s_upper=time.monotonic()-t,checkpoint=actual.get('checkpoint'),
                inference_package=actual.get('inference_package'),actual_model_loads=actual.get('actual_model_loads',0))
        row=journal.once(name+'-EXECUTE',registered,execute);children.append(row)
        write(root/'CHILDREN_ACTUAL.json',dict(children=children));return row
    return operation(root,need,deadline,gpu,child,children,start,stop_requested)


def run(c):
    from scripts.pi0292.durable import write
    if c.get('allocation_mode')=='FORMAL_SLICE':
        from .target_gr00t_formal_v1 import execute
        return with_allocation(c,lambda *args:execute(c,*args))
    def capacity_fit4(root,need,deadline,gpu,child,children,start,stop_requested):
        selected=None
        for variant in profile.VARIANTS:
            name='capacity-'+variant.lower().replace('_','-')
            cap=child(name,'CAPACITY',variant,2)
            if cap['exit_code']:
                error=(Path(cap['root'])/'stderr.log').read_text()
                need('out of memory' in error.lower() or 'ORIGINAL_TWO_GIB_MEMORY_MARGIN' in error,
                     'NON_CAPACITY_ENGINEERING_FAILURE_REQUIRES_REPAIR_NOT_VARIANT_SEARCH')
                continue
            if not cap['endpoint_complete']:
                return dict(status='PARTIAL_CAPACITY_CHECKPOINT_SEALED',children=children,
                    actual_GR00T_updates=sum(r['GR00T_updates']for r in children),endpoint_complete=False)
            selected=variant;first=Path(cap['checkpoint']);cp1=first.parent/'checkpoint-1'
            s=json.loads((Path(cap['root'])/'EXECUTION_SPEC.json').read_bytes())
            resumed=child(name+'-restore-next','CAPACITY',variant,2,start_step=1,
                resume=dict(path=str(cp1),sha256=digest(cp1/'CHECKPOINT_COMPLETE.json')),generation=s['generation_id'])
            need(resumed['exit_code']==0 and resumed['endpoint_complete'],'INDEPENDENT_CAPACITY_RESUME_FAILED')
            proof=compare_next_update(first,resumed['checkpoint']);write(root/'NEXT_UPDATE_RESTORE_ACTUAL.json',proof)
            break
        need(selected is not None,'AWAITING_SCALE_REVIEW_ALL_SINGLE_A40_SCOPES_FAILED')
        timings=[json.loads(line)for line in (Path(cap['root'])/'OPTIMIZER_STEPS.jsonl').read_text().splitlines()]
        need([r['step']for r in timings]==[1,2]and all(r['optimizer_update_wall_s']>0 for r in timings),
             'TWO_MEASURED_REAL_UPDATES_FOR_FIT4_TIME_ADMISSION')
        forecast=200*1.25*max(r['optimizer_update_wall_s']for r in timings)+cap['PBS_GPU_wall_s_upper']+1200
        time_admission=dict(predicted_fit4_200_wall_s=forecast,remaining_allocation_s=deadline-time.time(),
            measured_update_wall_s=[r['optimizer_update_wall_s']for r in timings],
            measured_full_load_save_probe_restore_overhead_upper_s=cap['PBS_GPU_wall_s_upper'],
            closeout_reserve_s=1200,passes=time.time()+forecast<deadline)
        write(root/'FIT4_TIME_ADMISSION.json',time_admission)
        if not time_admission['passes']:
            return dict(status='CAPACITY_QUALIFIED_WAIT_FRESH_FIT4_ALLOCATION',children=children,
                actual_GR00T_updates=sum(r['GR00T_updates']for r in children),time_admission=time_admission,
                capacity_and_next_update_already_complete_do_not_repeat=True)
        fit=child('fit4-200','FIT4',selected,200)
        need(fit['exit_code']==0,'FIT4_ENGINEERING_FAILURE')
        if not fit['endpoint_complete']:
            return dict(status='PARTIAL_FIT4_CHECKPOINT_SEALED',children=children,
                actual_GR00T_updates=sum(r['GR00T_updates']for r in children),endpoint_complete=False)
        gate=fit4_gate(fit['root']);write(root/'FIT4_GATE_200.json',gate)
        # A 500 continuation is a separately registered same-generation slice.
        if not gate['passes'] and gate['continuation500_allowed']:
            timings=[json.loads(line)for line in (Path(fit['root'])/'OPTIMIZER_STEPS.jsonl').read_text().splitlines()]
            remaining_prediction=300*1.25*max(r['optimizer_update_wall_s']for r in timings)+cap['PBS_GPU_wall_s_upper']+1200
            if time.time()+remaining_prediction>=deadline or stop_requested:
                return dict(status='FIT4_200_PROGRESS_WAIT_FRESH500_ALLOCATION',children=children,
                    actual_GR00T_updates=sum(r['GR00T_updates']for r in children),fit4_gate=gate,
                    predicted_300_update_and_restore_s=remaining_prediction)
            s=json.loads((Path(fit['root'])/'EXECUTION_SPEC.json').read_bytes());cp=Path(fit['checkpoint'])
            fit=child('fit4-500','FIT4',selected,500,start_step=200,
                resume=dict(path=str(cp),sha256=digest(cp/'CHECKPOINT_COMPLETE.json')),generation=s['generation_id'])
            need(fit['exit_code']==0,'FIT4_CONTINUATION_ENGINEERING_FAILURE')
            if not fit['endpoint_complete']:
                return dict(status='PARTIAL_FIT4_CONTINUATION_SEALED',children=children,
                    actual_GR00T_updates=sum(r['GR00T_updates']for r in children),endpoint_complete=False)
            gate=fit4_gate(fit['root'],original_before=root/'fit4-200/before/RESULT.json')
            write(root/'FIT4_GATE_500.json',gate)
        return dict(status='FIT4_CHAIN_SEALED',children=children,selected_variant=selected,
            actual_GR00T_updates=sum(r['GR00T_updates']for r in children),wall_s=time.monotonic()-start,
            fit4_learnability_passed=gate['passes'],fit4_gate=gate,
            formal_training='NOT_STARTED_REQUIRES_SEALED_DIVERSE_TRAIN_DEV_AND_FIT4_DECISION',
            native_physics=0,WM_updates=0,Actor_updates=0,baseline_changed=False)
    return with_allocation(c,capacity_fit4)
