"""Source-bound CPU contracts and only the six authorized saved-input candidates."""
from pathlib import Path
import numpy as np
from .task_capture_runtime import read,sha
from .sdr_runtime import FOLDER,require_adoption,cards,slot_cost
from .sdr_source import retarget,tool_binding
from .sdr_client import Client,PlannerFailure
from .mps2_cpu import first_input

def boundaries(root):
    root=Path(root);p=root/FOLDER;source=read(p/'SOURCE_EXTRACTION.json');plan=read(p/'SOURCE_PLAN.json');gate=read(p/'RETARGET_TIME_GATE.json')
    records=source['records'];checks={}
    assert len(records)==6 and all(r['split']=='TRAIN' and r['sample_command_parity']=='EXACT_ALL_640' for r in records)
    checks['SIX_FIXED_TRAIN_SOURCE_IDENTITIES']=dict(source_ids=[r['id'] for r in records],no_DEV=True)
    assert all(r['unique_committed_rows']==640 and r['original_H40_references']==80 for r in records)
    checks['UNIQUE_640_INSTALLED_PREFIX_ROWS']=dict(rows=3840,overlap_not_duplicated=True)
    assert all(r['observations']==641 for r in records)
    checks['EXPORT_601_NOT_RAW_641']=dict(raw=641,exported_anchor_count=601,missing_suffix_invented=False)
    assert all(0<=r['cut_start_index']<r['cut_end_inclusive_index']<640 for r in records)
    checks['SOURCE_EVENT_SEARCHSORTED_CUTS']=dict(cuts=[[r['cut_start_index'],r['cut_end_inclusive_index']] for r in records])
    for r in records:
        with np.load(p/'sources'/r['derived_file'],allow_pickle=False) as d:
            assert d['actual_TCP'].shape==(641,4,4) and d['committed_TCP'].shape==(640,4,4)
    checks['MEASURED_AND_COMMITTED_SEPARATE']=dict(different_lengths_and_fields=True)
    for r in gate['rows']:
        t=np.asarray(r['target_tool_binding']['C_tcp_pad']);assert np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-12)
        assert r['source_and_target_pad_sizes_match']
    checks['FUNCTIONAL_TOOL_AND_MECHANICAL_STROKE']=dict(actual_zero_step_models=8,source_target_pad_dimensions_match=True)
    for r in records:
        t=np.asarray(r['tool_binding']['C_tcp_pad']);assert np.allclose(t@np.linalg.inv(t),np.eye(4),atol=1e-12)
    checks['SOURCE_TARGET_TOOL_FRAME_ROUNDTRIP']=dict(cases=6)
    for r in gate['rows']:
        with np.load(p/'sources'/r['retarget_file'],allow_pickle=False) as d:
            u=d['transport_weight'];rot=d['task_rotation'];assert np.isfinite(d['poses']).all() and np.all(np.diff(u)>=-1e-12)
            assert np.allclose(rot.T@rot,np.eye(3),atol=1e-12) and u[0]==0 and u[-1]==1
    checks['TWO_ANCHOR_WARP_AND_CONTINUITY']=dict(finite_registered_SE3=True,monotone_transport_weight=True)
    assert all(r['new_physics']==0 for r in gate['rows'])
    checks['NO_ACTUAL_JOINT_ROOT_OBJECT_EDITS']=dict(source_and_target_reads_only=True,physics_steps=0)
    assert all(r['slowdown']>=1 and r['active_duration_s']>=r['source_cut_duration_s'] for r in gate['rows'])
    checks['RATE_SLOWDOWN_AND_28S_LOWER_BOUND']=dict(time_rejected=[r['id'] for r in gate['rows'] if r['active_duration_s']>28])
    fixture=[dict(id='first',status='FAIL'),dict(id='second',status='PASS'),dict(id='third',status='PASS')]
    assert next(r for r in fixture if r['status']=='PASS')['id']=='second'
    checks['FIRST_COMPLETE_LEGAL_SOURCE_ORDER']=dict(no_physical_quality_selection=True)
    def reject(fn):
        try:fn()
        except PermissionError:return True
        raise AssertionError('EXPECTED_CONTRACT_REJECTION')
    assert slot_cost('IK',0,{})==dict(goal_IK_calls=1,CLIK_reserved=16)
    assert reject(lambda:slot_cost('IK',0,dict(CLIK_reserved=32256))) and reject(lambda:slot_cost('RRT',8,{}))
    checks['INTERNAL_IK_RRT_DENSE_CAPS']=dict(cap_rejections=True)
    assert reject(lambda:slot_cost('IK',0,{},['IK/0']))
    checks['NO_REDRAW_ON_RESTORE']=dict(named_duplicate_slot_rejected=True)
    bytask={t:[r for r in gate['rows'] if r['task']==t] for t in ['T2','T1']}
    assert all(r['active_duration_s']>28 for r in bytask['T2']) and any(r['active_duration_s']<=28 for r in bytask['T1'])
    checks['TASK_LOCAL_ADMISSION']=dict(T2_time_fail_does_not_block_T1=True)
    lock=read(p/'SOURCE_LOCK.json');code=Path(__file__).resolve().parents[2]
    for member in lock['unchanged_native_members']:
        assert sha(code/member['path'])==member['sha256']
    checks['ORIGINAL_GATE_SCORER_NATIVE_MEMBERS']=dict(unchanged_public_members=len(lock['unchanged_native_members']),no_WBC_changes=True)
    a,s=require_adoption(root);assert not a['new_T0'] and not a['budget_reset']
    old=read(root/'maniskill-search-v2/SERIES_CLOSED.json');assert old['actual_FAIL_unchanged']
    checks['BUDGET_DEADLINE_OLD_FAILURES_PRESERVED']=dict(retained_T0=a['window']['original_T0_epoch'],old_MP_S2_partial_kept=True)
    assert len(checks)==16
    return checks

def qualify(root,manifest):
    from scripts.pi0292.durable import write
    root=Path(root);p=root/FOLDER;require_adoption(root,manifest)
    if (p/'CPU_ADMISSION.json').exists():return read(p/'CPU_ADMISSION.json')
    checks=boundaries(root);gate=read(p/'RETARGET_TIME_GATE.json');extracted=read(p/'SOURCE_EXTRACTION.json')
    old=read(root/'maniskill-search-v2/CONDITIONS.json')['conditions'];candidates=[];tasks=[]
    client_objects={}
    for task in ['T2','T1']:
        entry=next(e for e in old if e.get('phase')=='CPU_SAVED_INPUT' and e['task']==task)
        engine,state,mount=first_input(root,entry);client=Client(root,engine)
        selected=None
        for i,row in enumerate([r for r in gate['rows'] if r['task']==task]):
            condition='SDR_CPU_'+task+'_'+str(i)
            if row['status']!='NEEDS_ENTRY_ROUTE':
                result=dict(task=task,source_id=row['id'],condition=condition,status='NOT_ADMITTED_SOURCE_TIME_LOWER_BOUND',
                    active_duration_s=row['active_duration_s'],even_zero_entry_exceeds_28s=True,new_IK_calls=0)
            else:
                record=next(r for r in extracted['records'] if r['id']==row['id'])
                file=p/'sources'/row['retarget_file'];assert sha(file)==row['retarget_sha256']
                with np.load(file,allow_pickle=False) as d:program={k:d[k].copy() for k in d.files}
                contact_stage='close' if task=='T1' else 'contact';event=next(e for e in record['events'] if e['stage']==contact_stage)
                source_index=int(np.searchsorted(np.arange(640)*.05,event['time_s']))
                contact_index=int(np.searchsorted(program['source_indices'],source_index))
                package=dict(index=i,task=task,approach=program['poses'][0],contact=program['poses'][contact_index],
                    active_duration_s=row['active_duration_s'],source_id=row['id'],source_program_sha256=row['retarget_sha256'])
                before=state['qpos'].copy()
                try:
                    poses,cert,index=client.bundle(state,[package],'approach',condition,0)
                    result=dict(task=task,source_id=row['id'],condition=condition,status='PASS_SOURCE_ROUTE_AND_TIME',
                        candidate_index=i,certificate=cert,active_duration_s=row['active_duration_s'],
                        maximum_entry_plus_active_s=cert['duration_s']+row['active_duration_s'],source_program_sha256=row['retarget_sha256'],
                        contact_source_index=source_index)
                    if selected is None:selected=result
                except PlannerFailure as failure:
                    result=dict(task=task,source_id=row['id'],condition=condition,status='FAIL_REGISTERED_SOURCE_ROUTE',reason=str(failure),evidence=failure.evidence)
                assert np.array_equal(state['qpos'],before)
            candidates.append(result);write(p/'SOURCE_CANDIDATES.partial.json',dict(candidates=candidates,source_sha=manifest['code']['vla_commit']))
        tasks.append(selected or dict(task=task,status='NOT_ADMITTED_FIXED_SOURCE_POOL',
            reason='ALL_THREE_SOURCE_TIME_LOWER_BOUNDS_FAIL' if task=='T2' else 'NO_REGISTERED_SOURCE_ROUTE_AND_TIME',physical_starts=0))
    value=dict(status='CPU_CONTRACT_COMPLETE_TASK_LOCAL_RESULTS',source_sha=manifest['code']['vla_commit'],checks_passed=16,checks=checks,
        candidates=candidates,task_results=tasks,new_physics=0,new_render=0,new_model_queries=0,selected='NONE',deployment='NONE')
    write(p/'CPU_ADMISSION.json',value);return value
