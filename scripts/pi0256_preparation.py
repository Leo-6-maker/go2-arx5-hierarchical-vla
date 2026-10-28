"""CPU audit and frozen preparation contract; no simulator/model imports."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from go2arx5_vla.trajectory.so3 import quat_to_matrix,matrix_to_quat,geodesic_angle
from pi0255_motion3 import sha,write

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0256')
OLD=ROOT.parent/'PI0255'
AUTH='PI0256_PREPARATION_REPAIR_TO_MULTITASK_PILOT_GO_V1'
CONTROL=ROOT.parent/'PI0252/pre_full_sim/controls/A_attempt2'


def entry_reference(template,position,quaternion):
    """Rigidly transport the evidenced C0 curve from the actual entry pose once."""
    rotation=quat_to_matrix(quaternion)@quat_to_matrix(template['quaternion'][0]).T
    times=np.arange(101,dtype=float)*.05
    old_times=template['timestamp']-template['timestamp'][0]
    xyz=np.stack([np.interp(times,old_times,template['position'][:,i]) for i in range(3)],axis=-1)
    xyz=(xyz-template['position'][0])@rotation.T+position
    # The existing C0 orientation is constant. Preserve its rigid transported pose.
    assert np.max(geodesic_angle(quat_to_matrix(template['quaternion']),quat_to_matrix(template['quaternion'][0])))<1e-7
    return dict(timestamp=times,position=xyz,quaternion=np.tile(quaternion,(len(times),1)))


def qualified(row):
    numbers=[row[k] for k in ('speed_mps','angular_speed_radps','position_error_m','rotation_error_rad')]
    return all(np.isfinite(numbers)) and row['fresh'] and row['clean'] and not row['joint_limit_violation'] and (
        numbers[0]<=.02 and numbers[1]<=.20 and numbers[2]<=.035 and numbers[3]<=.10)


def advance(state,row,start,stable):
    assert state=='PREPARE'
    elapsed=row['time_s']-start
    assert elapsed>=0
    if not row['clean']:return 'TERMINAL',None
    stable=(row['time_s'] if stable is None else stable) if qualified(row) else None
    if stable is not None and row['time_s']-stable>=.5-1e-8 and elapsed<=5.+1e-8:return 'SETTLED',stable
    if elapsed>=5.-1e-8:return 'TERMINAL',stable
    return state,stable


def unchanged(before,after):
    """Reference synchronization must not advance or mutate physical state."""
    assert before==after,'reference synchronization mutated physics/history'


def failure_pose(post,event):
    """Only a pre-autoreset event can describe an autoreset failure pose."""
    return event['achieved_pos'] if event is not None else post['TCP_world']


def audit():
    timelines=[]
    for i in range(6):
        p=OLD/'episodes'/('pi0255-family-%03d-M1-c0'%i)/'attempt-00'
        d=json.loads((p/'result.json').read_text());ticks=[json.loads(x) for x in (p/'raw_ticks.jsonl').read_text().splitlines()]
        assert sha(p/'raw_ticks.jsonl')==d['raw_ticks_sha256']
        handoffs=[json.loads(x) for x in (p/'handoffs.jsonl').read_text().splitlines()]
        last=ticks[-1];event=last['termination_event'];clean=[x for x in ticks if x['physical_clean']]
        errors=[np.linalg.norm(np.asarray(x['achieved_world'])-x['desired_world']) for x in clean]
        angles=[float(geodesic_angle(quat_to_matrix(x['achieved_quaternion']),quat_to_matrix(x['desired_quaternion']))) for x in clean]
        timelines.append(dict(family=i,episode=d['episode']['episode_id'],reset_seed=d['episode']['reset_seed'],
            implicit_reset_step=True,first_physics_s=json.loads((p/'physics_started.json').read_text())['sim_time_before'],
            reset_finished_s=d['initial_reset']['actual_sim_time_s'],construction_reference=d['reference_load_events'][0],
            first_bind=d['reference_load_events'][1],first_actor_outputs=ticks[0]['WBC_policy'],
            actual_actor_input_history='UNOBSERVED_IN_ORIGINAL_BYTES',first_post_actor_tick_s=ticks[0]['actual_sim_time_s'],
            reference_loads=d['reference_load_events'],clean_prefix_ticks=len(clean),
            clean_prefix_XYZ_rms_m=float(np.sqrt(np.mean(np.square(errors)))),clean_prefix_SO3_rms_rad=float(np.sqrt(np.mean(np.square(angles)))),
            outcome=d['failure'],end_s=d['end_actual_sim_time_s'],terminal_gate=handoffs[-1]['gate'],
            pre_autoreset_event=event,post_step_TCP=last['achieved_world'],
            post_step_TCP_is_clean_failure_pose=False if event else True,
            raw_sha256=d['raw_ticks_sha256'],result_sha256=sha(p/'result.json'),
            complete_handoff_arrays=sum('reference_file' in x for x in handoffs),handoffs=len(handoffs)))
    write(ROOT/'evidence/legacy_six_family_timelines.json',dict(schema='PI0256_SOURCE_AND_RAW_TIMELINES_V1',families=timelines,new_physics=0))
    control=json.loads((CONTROL/'control.json').read_text());assert control['status']=='PASS' and control['rows']==97
    with np.load(CONTROL/'actual_expert_reference.npz',allow_pickle=False) as d:
        template={k:d[k].copy() for k in ('timestamp','position','quaternion')}
    np.savez_compressed(ROOT/'evidence/evidenced_C0_template.npz',**template)
    registry=json.loads((OLD/'design/task_registry_v1.json').read_text());old=registry['families'][:6];cases=[]
    for group in ('A','B','C'):
        for i,f in enumerate(old):
            family=json.loads(json.dumps(f))
            if group=='C':
                family['family_id']='pi0256-prep-confirm-%02d'%i;family['reset_seed']=2560600+i
                family['center_world']=(np.asarray(family['center_world'])+[.007,.009,0.]).tolist()
                family['factors']['x_shift']+=.007;family['factors']['y_shift']+=.009
            cases.append(dict(case_id='pi0256-prep-%s%02d'%(group,i),group=group,family=family,
                              seed=family['reset_seed'],release='B_ALL6_PASS' if group=='C' else 'CPU_AND_FREEZE'))
    contract=dict(schema='PI0256_PREPARATION_V2_CONTRACT_V1',authority=AUTH,authority_commit='fcfd37bf2ced86ae3c3ce9ee43c2a0b9ec810450',
        state_machine=['RESET','PREPARE','SETTLED','TASK','TERMINAL'],preparation_elapsed_deadline_s=5.,stable_interval_s=.5,
        thresholds=dict(speed_mps=.02,angular_speed_radps=.20,position_error_m=.035,rotation_error_rad=.10),
        reference=dict(source=str(CONTROL/'actual_expert_reference.npz'),source_sha256=sha(CONTROL/'actual_expert_reference.npz'),
            evidence_sha256=sha(CONTROL/'control.json'),template_sha256=sha(ROOT/'evidence/evidenced_C0_template.npz'),
            procedure='one rigid transport of evidenced complete C0 curve to reset-time actual root/DOF FK TCP; measured orientation; terminal hold to5s',
            native_prepare_points=101,native_prepare_period_s=.05,policy_horizon_unchanged=40,
            hold_scope='terminal part of nondegenerate continuous curve; not new idle action or minimum-arc workaround',
            physically_proved='original C0 finite execution only; repaired package/new stability predicates require B/C evidence'),
        package=dict(anchor='one immutable measured-reset FK translation; world-aligned orientation',
            reference_refresh='call existing _update_ee_task_space_error after project-owned reference load; no history clear/extra observation read',
            native_controller_weights_gains_dynamics_reset_unchanged=True),
        joint_predicate='original native telemetry joint_limit_violation; soft/hard margins also reported, never silently relaxed',
        preparation_native_reference='continuous native expert asset; no change to future H40 policy ABI',
        cases=cases,prep_start_cap=24,global_native_start_cap=384,native_step_cap=480000,
        new_model_queries=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    write(ROOT/'design/PREPARATION_V2_CONTRACT.json',contract)
    print(json.dumps(dict(status='P0_TIMELINES_AND_PROSPECTIVE_PACKAGE',families=len(timelines),cases=len(cases))),flush=True)


def check():
    row=dict(time_s=10.,fresh=True,clean=True,joint_limit_violation=False,speed_mps=.01,angular_speed_radps=.1,position_error_m=.01,rotation_error_rad=.02)
    assert qualified(row)
    for field,value in [('fresh',False),('speed_mps',np.nan),('angular_speed_radps',.21),('rotation_error_rad',.11),('joint_limit_violation',True)]:
        assert not qualified(dict(row,**{field:value}))
    assert advance('PREPARE',dict(row,time_s=10.6),10.,10.)[0]=='SETTLED'
    assert advance('PREPARE',dict(row,time_s=15.1),10.,None)[0]=='TERMINAL'
    assert advance('PREPARE',dict(row,time_s=10.6,clean=False),10.,10.)[0]=='TERMINAL'
    with np.load(ROOT/'evidence/evidenced_C0_template.npz',allow_pickle=False) as d:template={k:d[k].copy() for k in d.files}
    q=np.asarray([0.,0.,np.sin(.3),np.cos(.3)]);p=np.asarray([.2,-.1,.6]);before={k:v.copy() for k,v in template.items()}
    reference=entry_reference(template,p,q)
    assert np.array_equal(reference['position'][0],p) and np.array_equal(reference['quaternion'][0],q)
    assert all(np.array_equal(before[k],template[k]) for k in before)
    assert np.array_equal(reference['position'][39:],np.tile(reference['position'][39],(62,1)))
    assert np.linalg.norm(reference['position'][-1]-p)>.1
    anchor=np.array([.4,.2,.1]);native=reference['position']-anchor
    assert np.allclose(native+anchor,reference['position']) and not np.allclose(native+2*anchor,reference['position'])
    assert not np.allclose(quat_to_matrix(q),quat_to_matrix(np.roll(q,1)))
    unchanged({'root':'a','dof':'b','clock':10.,'history':'c'},{'root':'a','dof':'b','clock':10.,'history':'c'})
    try:unchanged({'clock':10.},{'clock':10.02})
    except AssertionError:pass
    else:raise AssertionError('physics mutation accepted')
    assert failure_pose({'TCP_world':[9.,9.,9.]},{'achieved_pos':[1.,2.,3.]})==[1.,2.,3.]
    original=json.loads((ROOT/'evidence/runtime_readback.json').read_text());assert all(x['matches'] for x in original['files'])
    # Test the actual native error-refresh source on CPU without importing IsaacGym.
    import ast,types,torch
    f=next(x for x in original['files'] if x['name']=='wbc_env');tree=ast.parse(Path(f['path']).read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_update_ee_task_space_error')
    scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),f['path'],'exec'),scope)
    target=torch.tensor([[.2,.3,.4]]);actual=torch.tensor([[.1,.1,.1]])
    dummy=types.SimpleNamespace(_arm_target_pos_world=lambda:target,_arm_target_quat_world=lambda:torch.tensor([[0.,0.,0.,1.]]),
        end_effector_state=torch.cat((actual,torch.tensor([[0.,0.,0.,1.]])),dim=1),ee_pos_err=torch.zeros(1,3),ee_rot_err_axis_angle=torch.zeros(1,3),
        _goal_reaching_enabled=lambda:False,cfg=types.SimpleNamespace(use_rot6d=False),arm_target_quat_body=torch.tensor([[0.,0.,0.,1.]]),
        arm_target_pos_body=torch.zeros(1,3),commands_arm_obs=torch.zeros(1,6))
    scope.update(torch=torch,quat_error_axis_angle=lambda a,b:torch.zeros(a.shape[0],3))
    assert not torch.equal(dummy.ee_pos_err,target-actual)
    scope['_update_ee_task_space_error'](dummy);assert torch.equal(dummy.ee_pos_err,target-actual)
    write(ROOT/'design/preparation_cpu_receipt.json',dict(status='PASS_PREPARATION_CPU_BOUNDARIES',source_sha256=sha(__file__),
        actual_native_error_refresh_tested=True,tests=['stale_error_cache','elapsed_clock_offset','nonfinite','angular_without_translation',
        'wrong_quaternion','double_anchor','reference_immutable','physics_mutation','post_reset_not_clean_failure'],new_physics=0,new_GR00T=0))
    print('PASS_PREPARATION_CPU_BOUNDARIES',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');p.add_argument('--cpu-check',action='store_true');a=p.parse_args()
    if a.audit:audit()
    elif a.cpu_check:check()
    else:p.print_help()
