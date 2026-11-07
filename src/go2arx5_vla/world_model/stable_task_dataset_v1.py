"""Rebuild supervised examples from sealed real new-layout native episodes.

Labels are consumed commands, never future achieved states. Failed/partial
episodes remain in the manifest without fabricated full-horizon labels.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image

from scripts.pi0292.relay_protocol import unseal
from .storage import digest,write_json
from . import stable_task_adapter_v1 as adapter
from . import stable_task_data_v1 as data

SCHEMA = 'WM02_REAL_STABLE_TASK_SUPERVISION_V1'
NATIVE = 'f30450e669236724a63c125cc88df86bb28aad9136ee644b1c8328e088b0adea'


def need(ok,reason):
    if not ok:raise ValueError('STABLE_TASK_DATASET_' + reason)


def load(path,pin=None,*,maximum=384*1024**2):
    path=Path(path)
    need(path.is_absolute() and path.resolve()==path and not path.is_symlink()
         and path.stat().st_size<=maximum,'CANONICAL_BOUNDED_INPUT')
    actual=digest(path);need(pin is None or actual==pin,'PINNED_INPUT')
    return json.loads(path.read_bytes()),dict(path=str(path),sha256=actual)


def consumed_label(qualification):
    suffix=qualification.get('generated_segment')or{};rows=suffix.get('windows')or[]
    need(qualification.get('capability_reference_all40_consumed')is True
         and suffix.get('all_40_knots_consumed')is True and suffix.get('raw_knot_consumed')==[True]*40
         and len(rows)==suffix.get('complete_suffix_atoms')==176,'FULL_REAL_REFERENCE_CONSUMPTION')
    refs=[]
    for i,row in enumerate(rows):
        ref=np.asarray(row['actual_installed_consumed_EEF10_BASE7'],dtype=float)
        need(row['native_start_step']==data.B+25*i and row['native_end_step']==data.B+25*(i+1)
             and row['actual_ticks']==25 and row['solver_policies_with_exact_installed_target']
             and row['gripper']['command_consumed']is True
             and row['gripper']['submitted']['sequence']==(data.B-2503)//25+i
             and ref.shape==(17,)and np.isfinite(ref).all(),'REAL_SOLVER_GRIPPER_COMMAND_JOIN')
        refs.append(ref)
    value=refs[-1][:3]-refs[0][:3]
    need(np.linalg.norm(value)<=adapter.STEP_M+1e-10,'BOUNDED_ACTUAL_CONSUMED_COMMAND_LABEL')
    return value


def actual_features(root,spec,spec_pin,qualification):
    identity=qualification['student_observation'];path=Path(identity['path'])
    need(path.is_relative_to(root/'native-01/PAUSED_OBSERVATIONS'),'ACTUAL_OWNED_CAPTURE')
    record,descriptor=load(path,identity['sha256'],maximum=16*1024**2);body=unseal(record)
    need(body['native_step']==identity['native_step']==data.B
         and body['world']['execution_spec_sha256']==spec_pin
         and body['world']['scene_sha256']==digest(spec['model_path'])
         and body['world']['native_output']==str(root/'native-01'),'SAME_REAL_WORLD_SPEC_AND_DECISION')
    for name,pin in body['files'].items():
        member=path.parent/name
        need(member.resolve().is_relative_to(path.parent)and digest(member)==pin,'EXACT_RETAINED_OBSERVATION_MEMBER')
    shadow=unseal(body['original_shadow_ABI'])
    need(shadow['native_step']==data.B
         and shadow['target_observation']['target_runtime_sha256']==spec_pin
         and shadow['target_observation']['native_state_sha256']==body['files']['STATE.private.bin'],
         'REAL_TARGET_STATE_NOT_REPLACED_SOURCE_STATE')
    images={}
    for slot in ('body_rgb','wrist_rgb'):
        p=path.parent/'images'/f'000000-{slot}.png'
        with Image.open(p)as picture:
            need(picture.mode=='RGB'and picture.size==(256,256),'ACTUAL_RGB_NO_MODE_REPLACEMENT')
            images[slot]=np.asarray(picture).copy()
    x=adapter.features(**images,actual_target_eef10=shadow['target_observation']['actual_state_eef10'],
                       task_command=spec['task_command'])
    return x,descriptor


def read_episode(root,row,panel_pin):
    root=Path(root)
    spec,spec_descriptor=load(root/'EXECUTION_SPEC.json',maximum=16*1024**2)
    need(spec.get('phase')==data.PHASE and spec.get('kind')==data.KIND
         and spec.get('data_role')==row['split']and spec.get('data_parent')==row['parent']
         and spec.get('task_command')==row['command']
         and spec.get('stable_task_panel_sha256')==panel_pin,'NEW_PHYSICAL_ROLE_NO_DIAGNOSTIC_RELABEL')
    data.validate(spec)
    lease,lease_descriptor=load(root/'control/LEASE.json',maximum=1024**2)
    worker,worker_descriptor=load(root/'WORKER_EXIT.private.json',maximum=1024**2)
    need(lease.get('status')=='CLOSED'and lease.get('spec_sha256')==spec_descriptor['sha256']
         and lease.get('worker_exit_code')==worker.get('exit_code')==0
         and worker.get('failure')is None,'SEALED_CLOSED0_REAL_EPISODE')
    result,result_descriptor=load(root/'native-01/RESULT.json')
    trace=result.get('trace_audit')or{};writer=trace.get('writer')or{}
    need(result.get('status')=='NATIVE_STAGE_PASSED'and result.get('phase')==data.PHASE
         and result.get('failure')is None and result.get('actual_target_integrations')==data.E
         and result.get('native_binary_sha256')==NATIVE
         and result.get('original_joint_or_warning_samples')==0
         and result.get('native_outer_guard_rejection_count')==0 and not result.get('physics_stops')
         and result.get('trace_audit_error')is None
         and trace.get('status')=='RECORDED_TRACE_CHECKS_PASS'and trace.get('native_step_records')==data.E
         and writer.get('dropped')==0 and writer.get('events')==writer.get('written'),
         'ACTUAL_ORIGINAL_SAFE_COMPLETE_NATIVE_TRACE')
    q=result['reference_qualification']
    need(q['data_role']==row['split']and q['data_parent']==row['parent']
         and q['stable_task_data']==spec['stable_task_data']and q.get('raw_reference_is_GR00T')is False,
         'REAL_NEW_EXPERT_PRODUCER_AND_ROLE')
    label=consumed_label(q)
    proposal,proposal_descriptor=load(root/'native-01/STABLE_TASK_EXPERT_PROPOSAL.private.json',
        q['executed_reference_proposal_sha256'],maximum=16*1024**2)
    raw=np.asarray(proposal['original_physical_H40'],float)
    need(raw.shape==(40,10)and np.allclose(label,raw[-1,:3]-raw[0,:3],rtol=0,atol=1e-10),
         'LABEL_IS_CONSUMED_COMMAND_NOT_FUTURE_ACHIEVED_POSE')
    x,observation_descriptor=actual_features(root,spec,spec_descriptor['sha256'],q)
    return dict(run=root.name,root=str(root),parent_id=row['parent'],data_role=row['split'],
        task_command=row['command'],features=x.tolist(),reference_delta_m=label.tolist(),
        provenance='REAL_SUBMITTED_AND_SOLVER_GRIPPER_CONSUMED_COMMANDS',
        actual_task_success=q['task_condition_success'],
        files=dict(spec=spec_descriptor,lease=lease_descriptor,worker=worker_descriptor,
            native_result=result_descriptor,proposal=proposal_descriptor,observation=observation_descriptor))


def build(panel_path,runs_root,output):
    panel,panel_descriptor=load(panel_path,maximum=16*1024**2)
    need(panel.get('schema')=='WM02_STABLE_TASK_DATA_PANEL_V1','FROZEN_NEW_DATA_PANEL')
    episodes=[];excluded=[];missing=[]
    for row in panel['assignments']:
        root=Path(runs_root)/row['native_run']
        if not(root/'WORKER_EXIT.private.json').exists():
            missing.append(dict(run=root.name,status='MISSING_OR_RUNNING_NOT_USED'));continue
        try:episodes.append(read_episode(root,row,panel_descriptor['sha256']))
        except Exception as error:
            excluded.append(dict(run=root.name,root=str(root),data_role=row['split'],parent_id=row['parent'],
                status='FAILED_OR_PARTIAL_PRESERVED_WITHOUT_FULL_HORIZON_LABEL',reason=type(error).__name__+': '+str(error)))
    value=dict(schema=SCHEMA,status='COMPLETE_ALL_EXPECTED_EPISODES_TERMINAL'if not missing else'PARTIAL',
        panel=panel_descriptor,episodes=episodes,excluded=excluded,missing=missing,
        expected_episode_count=len(panel['assignments']),actual_eligible_episodes=len(episodes),
        actual_failed_or_unlabelled_episodes=len(excluded),new_physics=0,new_optimizer_updates=0,
        features_definition=adapter.SCHEMA,old_DIAGNOSTIC_data_used=False)
    write_json(output,value);return value


def load_for_training(path,pin):
    value,descriptor=load(path,pin,maximum=32*1024**2)
    need(value.get('schema')==SCHEMA and value.get('status')=='COMPLETE_ALL_EXPECTED_EPISODES_TERMINAL'
         and value.get('features_definition')==adapter.SCHEMA,'REAL_FINISHED_DATASET_NOT_SYNTHETIC')
    panel,_=load(value['panel']['path'],value['panel']['sha256'],maximum=16*1024**2)
    rows={r['native_run']:r for r in panel['assignments']};actual=[]
    for saved in value['episodes']:
        rebuilt=read_episode(saved['root'],rows[saved['run']],value['panel']['sha256'])
        need(rebuilt['files']==saved['files']and rebuilt['data_role']==saved['data_role']
             and rebuilt['parent_id']==saved['parent_id']
             and np.allclose(rebuilt['features'],saved['features'],rtol=0,atol=1e-6)
             and np.allclose(rebuilt['reference_delta_m'],saved['reference_delta_m'],rtol=0,atol=1e-10),
             'REBUILT_REAL_INPUT_AND_CONSUMED_LABEL_IDENTITIES')
        actual.append(rebuilt)
    train,dev=adapter.split_rows(actual)
    x=np.asarray([r['features']for r in actual],np.float32)
    y=np.asarray([r['reference_delta_m']for r in actual],np.float32)
    return x,y,actual,train,dev,descriptor
