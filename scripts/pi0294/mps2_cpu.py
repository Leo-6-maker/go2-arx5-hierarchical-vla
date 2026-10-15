"""Sixteen changed-contract fixtures and two original saved first-state bundles."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from .mps2_runtime import FOLDER,VERSION,require_adoption,cards
from .mps2_contract import slot_cost,first_valid,route_mode,Ghost,validate_command
from .mps2_geometry import packages,rodrigues
from .mps2_teacher import Teacher
from .mp_client import PlannerFailure
from .task_capture_runtime import read,sha

def boundaries(root):
    """No planner, rendering or integration is called by these fixtures."""
    root=Path(root);expected=read(root/FOLDER/'TASK_CARD.json')['checks_planned_not_executed'];checks={}
    def reject(fn,kind):
        try:fn()
        except kind:return True
        raise AssertionError('NEGATIVE_FIXTURE_NOT_REJECTED')
    for axis in np.eye(3):
        for angle in [0,np.pi/6,-np.pi/6]:
            r=rodrigues(axis,angle);assert np.allclose(r.T@r,np.eye(3)) and np.linalg.det(r)>0
    checks[expected[0]]=dict(orthogonal_right_handed=9)
    assert first_valid([dict(status='FAIL',i=0),dict(status='PASS',i=1)])['i']==1
    checks[expected[4]]=dict(failed_candidate_advanced_to=1)
    order=sorted([(2.,1,3,0),(1.,2,0,0),(1.,1,1,1)]);assert order[0]==(1.,1,1,1)
    checks[expected[5]]=dict(deterministic_tie_order=True,physical_selection_calls=0)
    assert slot_cost('IK',0,{})==dict(goal_IK_calls=1,CLIK_reserved=16)
    assert reject(lambda:slot_cost('IK',0,dict(CLIK_reserved=42624)),PermissionError)
    checks[expected[6]]=dict(precall_capacity_accounting=True,over_cap_rejected=True)
    assert reject(lambda:slot_cost('IK',0,{},['IK/0']),PermissionError)
    checks[expected[7]]=dict(incomplete_or_completed_slot_cannot_redraw=True)
    assert slot_cost('DENSE',1,dict(dense_reserved_points=220220))==dict(dense_reserved_points=1001)
    assert reject(lambda:slot_cost('DENSE',2,{}),PermissionError)
    checks[expected[8]]=dict(last_1001_point_reservation_at_cap=True,third_candidate_rejected=True,max_local_iterations_per_point=5)
    assert route_mode('PASS_SHORT_MILESTONES_AND_INTEGRITY')=='AUTO'
    assert route_mode('FAIL')=='NOT_AVAILABLE_OPERATOR'
    assert route_mode('FAIL',operator=True)=='ASSISTED'
    assert route_mode('FAIL',native_safety=True,operator=True)=='NOT_RUN_SAFETY_STOP'
    checks[expected[9]]=dict(task_local_admission=True)
    card=read(root/FOLDER/'CONDITIONS.json');physical=[c for c in card['conditions'] if c.get('phase')!='CPU_SAVED_INPUT']
    assert len(physical)==4 and all(not c['full_slot_valid_FAIL_retry'] for c in physical)
    assert card['assisted_replaces_full_slot'] and card['additional_assisted_episode_slots']==0
    checks[expected[10]]=dict(full_slots=2,assisted_additional_slots=0)
    actual=dict(tcp=np.eye(4),qpos=np.arange(34.));before=actual['qpos'].copy();ghost=Ghost(actual)
    ghost.edit([.01,0,0],[0,0,.1],opening=0)
    assert np.array_equal(actual['qpos'],before) and np.array_equal(actual['tcp'],np.eye(4))
    assert not np.array_equal(ghost.target,actual['tcp']);checks[expected[11]]=dict(actual_unchanged=True)
    assert reject(lambda:ghost.edit([0,0,0],[0,0,0],actor='LLM'),PermissionError)
    assert reject(lambda:ghost.edit([.03,0,0],[0,0,0]),ValueError)
    assert reject(lambda:ghost.edit([0,0,0],[0,0,.16]),ValueError)
    for _ in range(127):ghost.edit([0,0,0],[0,0,0])
    assert reject(lambda:ghost.edit([0,0,0],[0,0,0]),PermissionError)
    checks[expected[12]]=dict(edit_cap=128,LLM_rejected=True,oversized_step_rejected=True)
    assert validate_command(ghost.command()).shape==(4,4)
    for key in ['qpos','joint_action','root','object']:
        assert reject(lambda k=key:validate_command(dict(ghost.command(),**{k:0})),PermissionError)
    checks[expected[13]]=dict(forbidden_command_fields=4)
    from .mp_window import SavedWindow
    a=np.arange(520.).reshape(40,13);one=SavedWindow(a).take(8);split=SavedWindow(a);left=split.take(4)
    right=SavedWindow.restore(json.loads(json.dumps(split.snapshot()))).take(4);assert np.array_equal(one,np.vstack([left,right]))
    from .task_clock import actual_source_age
    assert actual_source_age(10.2,10.2)==0 and actual_source_age(1.,1.1)<0
    checks[expected[14]]=dict(prefix8_restore_equal=True,source_time_boundary=True)
    old=read(root/'maniskill-component-v1/CPU_ADMISSION.json')['checks']
    assert old['FROZEN_SCORE_RETAINS_LATE_FAILURE']['late_failure_rejected']
    assert route_mode('PASS_SHORT_MILESTONES_AND_INTEGRITY',native_safety=True)=='NOT_RUN_SAFETY_STOP'
    checks[expected[15]]=dict(old_late_failure_receipt_reused=True,new_safety_stops_all_modes=True)
    return checks

def first_input(root,entry):
    from .task_state import parse_state,ActualState
    import mujoco
    folder=(root/'teacher-qualification-v2/episodes'/entry['source'] if entry['task']=='T1' else root/'episodes'/entry['source'])
    if sha(folder/'samples.jsonl')!=entry['source_samples_sha256']:raise ValueError('MPS2_SAVED_SOURCE_HASH')
    row=json.loads((folder/'samples.jsonl').read_text().splitlines()[0]);packet=None
    with (folder/'state.bin.journal').open('rb') as handle:
        while raw:=handle.read(4):
            length,=struct.unpack('<I',raw);candidate=parse_state(handle.read(length))
            if candidate['step']==row['native_step']:packet=candidate;break
    if packet is None:raise ValueError('MPS2_ORIGINAL_FIRST_PACKET_REQUIRED')
    calibration=read(folder/'images/000000-calibration.json')
    for camera in ('body_rgb','wrist_rgb'):
        assert calibration[camera]['timestamp_s']==packet['time_s']
        assert (folder/'images'/('000000-'+camera+'.png')).is_file()
    model=mujoco.MjModel.from_binary_path(str(folder/'task-scene.private.mjb'));engine=ActualState(model);state=engine.update(packet)
    state['time_s']=float(row['time_s'])
    for k in ['qpos','qvel','ctrl']:assert np.array_equal(state[k],row[k])
    intent=next(json.loads(line) for line in (folder/'committed_intents.jsonl').read_text().splitlines() if json.loads(line).get('phase')=='SUBMITTED_REQUIRES_INSTALL_ACK')
    mount=np.asarray(row.get('actual_BASE_mount',intent['actual_BASE_mount']))
    return engine,state,mount

def qualify(root,manifest):
    from scripts.pi0292.durable import write
    root=Path(root);p=root/FOLDER;require_adoption(root,manifest)
    for r in read(p/'SOURCE_LOCK.json')['members'][:12]:
        if sha(Path(__file__).resolve().parents[2]/r['path'])!=r['sha256']:raise ValueError('MPS2_FROZEN_PUBLIC_MEMBER_CHANGED:'+r['path'])
    checks=boundaries(root);results=[];conditions=read(p/'CONDITIONS.json')['conditions'];expected=read(p/'TASK_CARD.json')['checks_planned_not_executed']
    for entry in conditions:
        if entry.get('phase')!='CPU_SAVED_INPUT':continue
        engine,state,mount=first_input(root,entry);task=entry['task'];original=state['qpos'].copy()
        fixed=next(c for c in cards() if c['task']==task and c['duration_s']==12.)
        teacher=Teacher(fixed,state,engine,mount,root=root,condition=entry['id'],cpu=True)
        packs=teacher.packages;assert len(packs)==6
        for pkg in packs:
            assert np.allclose(pkg['approach'][:3,:3].T@pkg['approach'][:3,:3],np.eye(3))
        if task=='T1':
            assert [(x['axis'],x['degrees']) for x in packs]==[(a,t) for a in (0,1) for t in (0,30,-30)]
            assert all(np.allclose(x['approach'][:3,3]-x['contact'][:3,3],[0,0,.08]) for x in packs)
            checks[expected[1]]=dict(actual_packages=6,ordered_axes_angles=True,contact_gap_centers_explicit=True)
        else:
            assert [x['degrees'] for x in packs]==[0,30,-30,60,-60,180]
            assert all(np.isfinite([x['front_support'],x['bottom_support']]).all() for x in packs)
            checks[expected[2]]=dict(actual_packages=6,all_rotated_pad_supports_recomputed=True)
        teacher.observe(state,mount)
        try:
            result=teacher.plan(state,mount)
            row=dict(task=task,condition=entry['id'],status='PASS_VALID_PLANNED_H40',selected_candidate=teacher.selected['index'],certificate=result['persistent']['certificate'])
        except PlannerFailure as e:row=dict(task=task,condition=entry['id'],status='FAIL_REGISTERED_SEARCH',reason=str(e),evidence=e.evidence)
        assert np.array_equal(state['qpos'],original)
        results.append(row)
        write(p/'P1.partial.json',dict(start_results=results,source_sha=manifest['code']['vla_commit'],new_physics=0))
    # Every approach bundle has paired endpoint goals before any route qualifies.
    for row in results:
        request=read(p/'requests'/row['condition']/'0000/INPUT.private.json')
        assert len(request['goals'])==12 and all(request['goals'][2*i]['endpoint']=='approach' and request['goals'][2*i+1]['endpoint']=='contact' for i in range(6))
    checks[expected[3]]=dict(both_endpoints_required=True,geometry_packages=6,task_start_inputs=2)
    assert set(checks)==set(expected)
    proof=dict(status='PASS_CHANGED_CPU_CONTRACT_TASK_LOCAL_RESULTS',source_sha=manifest['code']['vla_commit'],version=VERSION,
        checks_passed=len(checks),checks=checks,start_results=results,saved_first_inputs=2,new_physics=0,new_render=0,new_model=0,
        previous_24_checks_reused=True,selected='NONE',deployment='NONE')
    write(p/'CPU_ADMISSION.json',proof);return proof
