"""Pure array/schema admission for a future source-domain episode."""
import ast,hashlib,json
from pathlib import Path
ALLOWED=frozenset(('body_rgb','wrist_rgb','state_v1','language'))
def policy_packet(value):
    import numpy as np
    if frozenset(value)!=ALLOWED:raise ValueError('Privileged/missing policy input')
    for key in ('body_rgb','wrist_rgb'):
        if value[key].shape!=(96,96,3) or value[key].dtype!=np.uint8:raise ValueError('RGB ABI')
    if value['state_v1'].shape!=(10,) or not np.isfinite(value['state_v1']).all() or not isinstance(value['language'],str):raise ValueError('State/language ABI')
    return value

def validate(episode):
    import numpy as np
    required={'scene_id','parent_group','split','initial_state_sha256','observation_episode_id','command_episode_id','control_time_s','frame_time_s','state_v1','body_rgb','wrist_rgb','commanded_eef_gripper','installed_tcp','achieved_tcp','instruction','evidence'}
    if set(episode)!=required:raise ValueError('Episode schema keys')
    if episode['observation_episode_id']!=episode['command_episode_id']:raise ValueError('Stale images paired with another executor')
    evidence=episode['evidence'];required_evidence={'source_commit','target_WBC_calls','contacts','native_predicates','analysis_phase','camera_identity','opening_command_and_measurement_separate'}
    if not isinstance(evidence,dict) or not required_evidence<=set(evidence):raise ValueError('Source/physical evidence missing')
    if evidence['source_commit']!='62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3' or evidence['target_WBC_calls']!=0 or evidence['opening_command_and_measurement_separate'] is not True:raise ValueError('Source/gripper/WBC contract mismatch')
    times=np.asarray(episode['control_time_s']);frames=np.asarray(episode['frame_time_s']);command=np.asarray(episode['commanded_eef_gripper']);n=len(times)
    if n<40 or command.shape!=(n,10) or episode['state_v1'].shape!=(n,10):raise ValueError('No genuine H40 or command ABI')
    if not np.array_equal(times,frames) or not np.allclose(np.diff(times),.05,rtol=0,atol=1e-8):raise ValueError('Fresh synchronized20Hz frames required')
    if not np.isfinite(command).all() or np.min(command[:,9])<0 or np.max(command[:,9])>1:raise ValueError('Source command fraction invalid')
    for key in ('installed_tcp','achieved_tcp'):
        if episode[key].shape!=(n,4,4) or not np.isfinite(episode[key]).all():raise ValueError('Separate actual/installed TCP evidence required')
    for key in ('body_rgb','wrist_rgb'):
        if episode[key].shape!=(n,96,96,3) or episode[key].dtype!=np.uint8:raise ValueError('Episode RGB ABI')
    if len(episode['initial_state_sha256'])!=64 or any(c not in '0123456789abcdef' for c in episode['initial_state_sha256']):raise ValueError('Initial-state identity missing')
    if not np.isfinite(episode['state_v1']).all() or not np.isfinite(times).all():raise ValueError('Nonfinite state/time')
    if episode['split'] not in ('DEV','TRAIN','VAL','TEST') or not isinstance(episode['instruction'],str) or not episode['instruction']:raise ValueError('Split/instruction missing')
    for value in (command[:,3:9],episode['state_v1'][:,3:9]):
        rows=value.reshape(n,2,3)
        if not np.allclose(np.sum(rows*rows,axis=-1),1,atol=1e-5) or not np.allclose(np.sum(rows[:,0]*rows[:,1],axis=-1),0,atol=1e-5):raise ValueError('Invalid source row6d')
    for key in ('installed_tcp','achieved_tcp'):
        t=episode[key];r=t[:,:3,:3]
        if not np.allclose(t[:,3],np.array([0,0,0,1]),atol=1e-6) or not np.allclose(r.transpose(0,2,1)@r,np.eye(3),atol=1e-5) or not np.allclose(np.linalg.det(r),1,atol=1e-5):raise ValueError('Invalid actual TCP transform')
    for i in (0,n-1):policy_packet(dict(body_rgb=episode['body_rgb'][i],wrist_rgb=episode['wrist_rgb'][i],state_v1=episode['state_v1'][i],language=episode['instruction']))
    return dict(status='CPU_EPISODE_SCHEMA_PASS',genuine_H40_windows=n-39,excluded_suffix_frames=39,teacher_fields_evidence_only=True,source_is_not_target_WBC=True)

def qualified_bridge(source_worker,rotation_source):
    """Reuse only the checked pure functions; never import the native worker."""
    import numpy as np,torch
    from scipy.spatial.transform import Rotation
    scope=dict(np=np,torch=torch,Rotation=Rotation)
    for path,names in [(rotation_source,{'_axis_angle_rotation','euler_angles_to_matrix'}),(source_worker,{'array','native_action','opening'})]:
        nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes}!=names:raise ValueError('Qualified pure function inventory changed')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),scope)
    return scope

DATA_SCHEMA=dict(schema='POST_PI0276_COMMAND_OBSERVATION_EVIDENCE_V1',policy_allowlist=sorted(ALLOWED),state_fields=['world_TCP_XYZ','world_TCP_row6d','measured_pad_opening_fraction'],
    action_fields=['future_commanded_world_EEF_XYZ','future_commanded_world_EEF_row6d','gripper_command_fraction'],action_shape=[40,10],dt_s=.05,
    required_evidence=['parent_group','initial_state_sha256','fresh_RGB_actual_state_source','raw_reference','installed_TCP','achieved_TCP','contacts','native_predicates','analysis_phase','failures','full_H40_gate_before_and_after_codec'],
    privileged_fields_never_policy=['object_goal','teacher_phase','source_time','reward','success','contact_derived_grasp','scene_id'],
    split_before_windows=True,TRAIN_only_statistics=True,no_fabricated_future=True,statistics_strategy='Matched visual-only future comparison shares fixed statistics; new geometry/tasks require prospective TRAIN-only identity',
    target_WBC_filter=False,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')

if __name__=='__main__':print(json.dumps(DATA_SCHEMA,indent=2))
