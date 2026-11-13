"""Finite N20 closeout -> original raw TRAIN8 -> fixed new DEV48.

Only the existing registered native/group executor performs physics. Waiting
holds no GPU owner, loads no model and never restarts the candidate or a row.
"""
import json
import os
from pathlib import Path
import sys
import time

from scripts.pi0292.durable import write
from scripts.wm02.sla_queue_native_v1 import unit_state,wait_capacity,never_dispatched,CAPACITY_REASON
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.stable_task_scene_v1 import pinned
from go2arx5_vla.world_model import sla_dense_native_v5 as dense,sla_dev48_v5 as dev
from go2arx5_vla.world_model import task_learning_grasp_v1 as task

SCHEMA='SLA_FINITE_N20_RAW3K_TRAIN8_THEN_NEW_DEV48_FOLLOWUP_V5'


def descriptor(path):
    path=Path(path);return dict(path=str(path),sha256=digest(path))


def all_entry_rejections(rows):
    return len(rows)==8 and all(r['technical_valid']and r['policy_proposal_rejected']
        and r['actual_native_integrations']==task.B for r in rows)


def verify_saved_state(path):
    """Full saved-file readback once; no parent/teacher/optimizer is loaded."""
    import numpy as np
    p=Path(path);m=json.loads(p.read_bytes());total=0
    task.need(m['schema']=='SLA_COMPLETE_PEFT_STATE_V1'and m['step']==3000
        and m['cursor']==96000 and len(m['arrays'])==87,'N20_FULL87_COMPLETE3K_STATE')
    for row in m['arrays']:
        q=p.parent/row['file']
        task.need(q.resolve().is_relative_to(p.parent.resolve())and digest(q)==row['sha256'],
                  'N20_ALL_SAVED_ARRAY_BYTES')
        v=np.load(q,allow_pickle=False)
        task.need(list(v.shape)==row['shape']and str(v.dtype)==row['dtype']and np.isfinite(v).all(),
                  'N20_ALL_SAVED_ARRAY_SHAPE_DTYPE_FINITE')
        total+=q.stat().st_size
    return dict(complete_sha256=digest(p),arrays_read=87,saved_array_file_bytes=total,
        model_loads=0,optimizer_updates=0,physical_requests=0)


def group_terminal(root,panel,end):
    root=Path(root);lease=json.loads((root/'control/LEASE.json').read_bytes())
    result=json.loads((root/'RESULT.private.json').read_bytes())
    inp=json.loads((root/'INPUT.private.json').read_bytes())
    rows=json.loads((root/'OUTCOMES.private.json').read_bytes())
    task.need(unit_state(root)['MainPID']=='0'and lease['status']=='CLOSED'
        and lease['input_sha256']==digest(root/'INPUT.private.json')
        and lease['worker_exit_code']==result['exit_code'] and result['unresolved_pair']is None
        and inp['panel']==panel['path'] and inp['panel_sha256']==panel['sha256']
        and inp['assignment_slice'][1]==end and result['cumulative_completed']==len(rows)==end,
        'N20_FOLLOWUP_ACTUAL_CLOSED_GROUP_NO_UNKNOWN_REPLAY')
    return result['exit_code']==0 and all(r['technical_valid']and not r['hard_safety_stop']for r in rows),rows


def run(spec):
    from go2arx5_vla.world_model.sla_native_v1 import closed
    from go2arx5_vla.world_model.task_learning_student_registration_v2 import register_group,dispatch_group
    from scripts.wm02.admit_dense_sla_native_v5 import build
    from scripts.pi0294.e1.journal import Journal
    root=Path(spec['registered_root']);parent=Path(spec['candidate_spec']['path']).parent
    campaign=Path(spec['campaign_root']);start=time.time()
    task.need(spec['schema']==SCHEMA and os.getuid()==2009 and os.uname().nodename=='corelab-4080s'
        and spec['source_root']==str(Path(__file__).resolve().parents[2])
        and spec['poll_s']==60 and spec['method']==dense.NAMED_METHOD,
        'FINITE_EXISTING_CORE_N20_ENDPOINT_FOLLOWUP')
    source=pinned(dict(path=str(Path(spec['source_root'])/'SOURCE_STAGED.private.json'),
        sha256=spec['source_seal_sha256']))
    task.need(source['source_sha']==spec['source_sha']and source['files']==spec['source_files'],
              'N20_FOLLOWUP_IMMUTABLE_SOURCE_SEAL')
    candidate=pinned(spec['candidate_spec'])
    task.need(candidate['source_sha']=='cdc6ec75400ec1d2b508b4349166cda4a395da4c',
              'FOLLOW_ONLY_THE_EXISTING_FRESH_N20_OWNER')
    def live():
        task.need(time.time()<spec['deadline_epoch']and not(root/'STOP.json').exists()
            and not(campaign/'STOP.json').exists(),'FINITE_N20_FOLLOWUP_OR_OPERATOR_STOP')
    while True:
        live();state=unit_state(parent);lease=json.loads((parent/'control/LEASE.json').read_bytes())
        task.need(state['LoadState']=='loaded','N20_CANDIDATE_UNIT_UNKNOWN_NO_RESTART')
        if state['MainPID']=='0'and lease['status']=='CLOSED':break
        write(root/'PROGRESS.private.json',dict(state='WAIT_EXISTING_N20_ACTUAL_CLOSEOUT',epoch=time.time(),
            new_model_loads=0,optimizer_updates=0,physical_requests=0,GPU_owner_held=False))
        time.sleep(60)
    owner=descriptor(parent/'RESULT.private.json');result=closed(owner)
    task.need(result['schema']=='SLA_CORE_FRESH_N20_FM3K_RGB_ACTUAL_V5'
        and result['complete3K']and result['fresh_RGB16_bit_exact']and result['step']==3000
        and result['target_visits']==96000 and result['source_visits']==0,
        'ACTUAL_COMPLETE3K_FRESH_RGB_NOT_INTERMEDIATE500_OR_PREFIX')
    trained=json.loads((parent/'candidate3000/RESULT.private.json').read_bytes())
    complete=Path(trained['last_checkpoint']['path'])/'COMPLETE.json'
    readback_start=time.time();readback=verify_saved_state(complete)
    write(root/'SAVED_ARRAYS_READBACK.private.json',dict(readback,wall_s=time.time()-readback_start))
    request=dict(method=dense.NAMED_METHOD,candidate_owner=owner['path'],
        training_result=str(parent/'candidate3000/RESULT.private.json'),complete_state=str(complete),
        parameter_inventory=str(parent/'candidate3000/work/PARAMETERS.private.json'),ordinary_RGB_owner=owner['path'],
        ordinary_RGB_reconstruction=str(parent/'fresh_RGB/REPLAY_RESULT.private.json'),
        ordinary_RGB_spec=str(parent/'fresh_RGB/EXECUTION_SPEC.json'))
    admission=build(request);write(root/'POLICY_ADMISSION.private.json',admission)
    policy=descriptor(root/'POLICY_ADMISSION.private.json')
    base=pinned(spec['base_panel'])['registration'];static=pinned(spec['dev_static_spec'])
    dev_actual=closed(descriptor(Path(static['registered_root'])/'RESULT.private.json'))
    common=dict(source_root=spec['source_root'],source_sha=spec['source_sha'],source_files=source['files'],
        source_seal=descriptor(Path(spec['source_root'])/'SOURCE_STAGED.private.json'),
        native_base=base['native_base'],gpu_base=base['gpu_base'],Native_body_qualification=policy,
        memory_execution='SLA_ORDINARY_RGB_JAX_SINGLE_CORE_V1',memory_execution_reference=admission['ordinary_RGB_reconstruction'],
        capacity_wait_before_pair_v3=dict(version='SLA_UNSTARTED_PAIR_RESOURCE_WAIT_V3',poll_s=60))
    train_request=dict(common,root=spec['train_panel_root'],scenes=base['scenes'],scene_producers=base['scene_producers'],
        scene_allowlist=base['scene_allowlist'],scene_allowlist_sha256=base['scene_allowlist_sha256'],
        preregistration=descriptor(Path(spec['source_root'])/dense.NAMED_DECLARATION))
    development_request=dict(common,root=spec['dev_panel_root'],population=dev.POPULATION,
        scenes=dev_actual['scenes'],scene_producers={l:spec['dev_static_spec']for l in dev_actual['scenes']},
        scene_allowlist=str(Path(static['registered_root'])/'SCENE_ALLOWLIST.private.json'),
        scene_allowlist_sha256=digest(Path(static['registered_root'])/'SCENE_ALLOWLIST.private.json'),
        scene_disjointness=dev_actual['scene_disjointness'],preregistration=descriptor(Path(spec['source_root'])/dev.DECLARATION))
    for kind,request,slices in (('TRAIN8',train_request,[(0,8)]),('DEV48',development_request,[(i,i+8)for i in range(0,48,8)])):
        live();task.need(not Path(request['root']).exists(),'FOLLOWUP_PANEL_EXISTS_RECONCILE_NEVER_REPLAY')
        panel=dense.register_panel(request)['panel'];previous=None
        for begin,end in slices:
            live();remaining=int(spec['deadline_epoch']-time.time()-120)
            task.need(remaining>3900,'N20_FOLLOWUP_NO_LATE_NEW_GROUP')
            group=Path(request['root'])/(Path(request['root']).name+f'-group-{begin:03d}-{end:03d}')
            gr=dict(root=str(group),panel=panel,assignment_slice=[begin,end],population_predecessor=previous,
                pilot1_binding=spec['pilot1_binding'],fusion_binding=spec['fusion_binding'],max_wall_s=min(3*3600,remaining))
            registered=register_group(gr);inp=pinned(registered['input'])
            journal=Journal(campaign/'control/journal',campaign.name)
            def dispatch():
                try:return dispatch_group(registered['input']['path'])
                except (ValueError,BlockingIOError)as error:
                    if (not isinstance(error,BlockingIOError)and str(error)!=CAPACITY_REASON
                            or not never_dispatched(group,journal)):
                        raise RuntimeError('UNKNOWN_OR_EFFECTFUL_DISPATCH_NEVER_REPLAY')from error
                    raise
            ready=wait_capacity(dispatch,min(spec['deadline_epoch'],inp['deadline_epoch']-3900),
                lambda value:write(root/'PROGRESS.private.json',dict(value,population=kind)),
                stopped=lambda:(root/'STOP.json').exists()or(campaign/'STOP.json').exists())
            if not ready:return dict(status='FINITE_PRE_DISPATCH_CAPACITY_WAIT_EXPIRED',exit_code=75,population=kind)
            write(root/'PROGRESS.private.json',dict(state='ACTUAL_NATIVE_GROUP_DISPATCHED',population=kind,
                group=str(group),begin=begin,end=end,epoch=time.time()))
            while unit_state(group)['MainPID']!='0':
                task.need(time.time()<inp['deadline_epoch']+180,'N20_GROUP_TERMINAL_UNKNOWN_NO_RESTART')
                time.sleep(60)
            good,rows=group_terminal(group,panel,end)
            write(root/f'{kind}_{end:02d}_TERMINAL.private.json',dict(result=descriptor(group/'RESULT.private.json'),
                good=good,conditions=end,task_successes=sum(r['task_success']for r in rows)))
            if not good:return dict(status='ACTUAL_TECHNICAL_OR_SAFETY_STOP_RETAINED',exit_code=1,population=kind,conditions=end)
            previous=descriptor(group/'RESULT.private.json')
            if kind=='TRAIN8'and all_entry_rejections(rows):
                return dict(status='ALL_RAW_TRAIN8_ENTRY_REJECTIONS_REPAIR_BEFORE_DEV48',exit_code=0,
                    TRAIN8_completed=8,DEV48_completed=0,DEV48_unstarted=48)
    return dict(status='COMPLETE_FIXED_TRAIN8_AND_NEW_DEV48_NOT_GUARANTEED_SUCCESS',exit_code=0,
        TRAIN8_completed=8,DEV48_completed=48,wall_s=time.time()-start)


if __name__=='__main__':
    s=json.loads(Path(sys.argv[1]).read_bytes())
    write(Path(s['registered_root'])/'RESULT.private.json',run(s))
