"""Read-only audit of SHA-pinned project S records; export numeric, non-pickle windows."""
from pathlib import Path
from collections import Counter
import gzip
import hashlib
import json
import pickle
import numpy as np
from .contracts import validate_atom
from .transitions import validate_record
from .storage import digest,write_json

GROUPS=(('joint_position',6),('joint_velocity',6),('tcp_position',3),('tcp_rotation',9),
        ('tool_velocity',6),('gripper_fraction',1),('gripper_speed',1),
        ('object_position',3),('object_rotation',9),('object_velocity',6))
SLICES={};_offset=0
for _name,_size in GROUPS:SLICES[_name]=slice(_offset,_offset+_size);_offset+=_size
STATE_DIM=_offset
SCHEMA=dict(version='WM02_S_NATIVE_STATE_V1',groups=GROUPS,state_dim=STATE_DIM,
    domain='S_FIXED_BASE_ARX5_NO_BASE_OR_LEG_PADDING',
    context=['goal_xyz','table_top_m','cube_edge_m'],
    events=['left_object_contact','right_object_contact','native_grasp','illegal_contact'],
    clock=['absolute_time_s_div32','installed_reference_elapsed_s'],
    reference='actual installed H40 world row-rot6d mechanical gripper fraction',
    future_controller_inputs=[],Teacher_phase_input=False)


class NumericProjectUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        # Legacy data is our own SHA-pinned artifact. Still refuse arbitrary code/classes.
        allowed={('numpy','dtype'),('numpy','ndarray')}
        allowed|={(m,n)for m in ('numpy.core.multiarray','numpy._core.multiarray')for n in ('_reconstruct','scalar')}
        allowed|={(m,'_frombuffer')for m in ('numpy.core.numeric','numpy._core.numeric')}
        if (module,name)not in allowed:raise ValueError('Non-numeric legacy pickle global:'+module+'.'+name)
        return super().find_class(module,name)


def load_rows(entry):
    if digest(entry['path'])!=entry['sha256']:raise ValueError('Original stream SHA differs')
    with gzip.open(entry['path'],'rb')as stream:value=NumericProjectUnpickler(stream).load()
    if not isinstance(value,dict)or not isinstance(value.get('rows'),list):raise ValueError('Unknown original record container')
    return value['rows'],value.get('events',[])


def tensor_state(s):
    tcp=np.asarray(s['tcp']);obj=np.asarray(s['object_pose'])
    values=(s['arm_q'],s['arm_qd'],tcp[:3,3],tcp[:3,:3].reshape(-1),
        np.r_[s['tool_linear_velocity'],s['tool_angular_velocity']],
        [s['gripper_fraction']],[s['gripper_speed']],obj[:3,3],obj[:3,:3].reshape(-1),
        np.r_[s['object_linear_velocity'],s['object_angular_velocity']])
    out=np.concatenate(values).astype(np.float64)
    if out.shape!=(STATE_DIM,)or not np.isfinite(out).all():raise ValueError('Missing/nonfinite measured S fields; never pad T fields')
    for r in (tcp[:3,:3],obj[:3,:3]):
        if not np.allclose(r.T@r,np.eye(3),atol=1e-7)or abs(np.linalg.det(r)-1)>1e-7:raise ValueError('Non-SO3 measured state')
    return out


def fingerprint(row):
    h=hashlib.sha256()
    for x in (row['native_before']['integration'],row['native_after']['integration'],row['reference_compiled'],
              row['reference_installed'],[row['t0'],row['t1']]):
        a=np.ascontiguousarray(x,dtype='<f8');h.update(str(a.shape).encode());h.update(a.tobytes())
    h.update(row['context']['model_sha256'].encode());return h.hexdigest()


def aligned(row):
    validate_atom(row);validate_record(row,native_dt=.002)
    if row.get('domain')!='S_FIXED_BASE_DIAGNOSTIC'or row['split']!='TRAIN':raise ValueError('Only original S TRAIN audit is authorized here')
    tensor_state(row['state_before']);tensor_state(row['state_after'])
    ref=np.asarray(row['reference_compiled']);point=row['native_progress']['point']
    if ref.shape!=(40,10)or not np.isfinite(ref).all()or point not in range(8):raise ValueError('Missing full installed H40/progress')
    if not np.array_equal(ref,row['installed_H40'])or not np.array_equal(ref[point],row['reference_installed']):raise ValueError('Compiled-installed-consumed mismatch')
    if row['native_progress']['forecast_index']!=(point+1)*25:raise ValueError('Consumed progress mismatch')
    # Validate physical raw labels against commands, never achieved successor pose.
    from scipy.spatial.transform import Rotation
    before=row['state_before']['tcp']
    if point:
        prev=ref[point-1];rows=prev[3:9].reshape(2,3)
        r=np.stack((rows[0],rows[1],np.cross(*rows)));pos=prev[:3]
    else:r=before[:3,:3];pos=before[:3,3]
    cur=ref[point];rr=cur[3:9].reshape(2,3);rot=np.stack((rr[0],rr[1],np.cross(*rr)))
    command=np.r_[cur[:3]-pos,Rotation.from_matrix(rot@r.T).as_rotvec(),cur[-1]]
    if not np.allclose(command,row['action_raw'],atol=1e-8,rtol=0):raise ValueError('Raw physical action is not the installed command delta')
    return row['physics_ticks']==25 and row['valid_state_target']


def continuous(a,b):
    keys=('episode_id','parent_id','scene_id','split','executor_id','run_id')
    return (not(a['terminal']or a['truncated'])and all(a[k]==b[k]for k in keys)
        and abs(a['t1']-b['t0'])<1e-8
        and all(np.array_equal(a['state_after'][k],b['state_before'][k])for k in ('qpos','qvel','tcp','object_pose')))


def build(manifest_path,output,*,prediction_atoms=8):
    if prediction_atoms not in (8,16):raise ValueError('Bounded 8/16 prediction diagnostics only')
    window_length=8+prediction_atoms
    output=Path(output);output.mkdir(exist_ok=False)
    manifest=json.loads(Path(manifest_path).read_text())
    if manifest.get('schema')!='WM02_TRUSTED_SOURCE_REPLAY_MANIFEST_V1'or manifest.get('trusted_project_pickle')is not True:raise ValueError('Explicit SHA-pinned trusted-project input required')
    entries=manifest['files'];loaded={};rejections=0
    for e in entries:
        if e['path']in loaded:raise ValueError('Duplicate manifest file')
        rows,events=load_rows(e);loaded[e['path']]=rows;rejections+=len(events)
        for a,b in zip(rows,rows[1:]):
            if a['episode_id']==b['episode_id']and a['run_id']==b['run_id']and(a['terminal']or a['truncated']):
                raise ValueError('Original stream contains a successor after terminal/timeout without a new episode')
    all_rows=[r for rows in loaded.values()for r in rows]
    if len(all_rows)!=manifest['expected_raw_transitions']:raise ValueError('Raw transition denominator differs')
    valid={id(r):aligned(r)for r in all_rows}
    # Union provenance aliases: same model/physical layout OR same declared parent/scene.
    group_parent={}
    def find(k):
        group_parent.setdefault(k,k)
        if group_parent[k]!=k:group_parent[k]=find(group_parent[k])
        return group_parent[k]
    def union(a,b):
        a,b=find(a),find(b);group_parent[max(a,b)]=min(a,b)
    for r in all_rows:
        m='model:'+r['context']['model_sha256'];union(m,'parent:'+r['parent_id']);union(m,'scene:'+r['scene_id'])
    canonical={m:find('model:'+m)for m in {r['context']['model_sha256']for r in all_rows}}
    group_names=sorted(set(canonical.values()));holdout=group_names[-1]if len(group_names)>1 else None
    unique={fingerprint(r):r for r in all_rows};windows={};streams=[];discontinuities=0
    for e in entries:
        rows=loaded[e['path']]
        if e.get('prefix'):
            if e['prefix']not in loaded:raise ValueError('Prefix must also be pinned in manifest')
            rows=loaded[e['prefix']]+rows
        streams.append(dict(kind=e['kind'],rows=len(rows),physical_group=canonical[rows[0]['context']['model_sha256']]if rows else None))
        segment=[]
        for row in rows:
            if segment and not continuous(segment[-1],row):segment=[];discontinuities+=1
            if not valid[id(row)]:segment=[];continue
            segment.append(row)
            if len(segment)>=window_length:
                chunk=segment[-window_length:];key=hashlib.sha256(''.join(fingerprint(r)for r in chunk).encode()).hexdigest()
                windows.setdefault(key,chunk)
    arrays={k:[]for k in ('state','reference','context','events','done','timeout','clock','valid','group','window_id','record_ids','raw_action','installed_command','tracking_residual')}
    from scipy.spatial.transform import Rotation
    for key,chunk in sorted(windows.items()):
        first=chunk[0];c=first['context'];states=[first['state_before']]+[r['state_after']for r in chunk]
        arrays['state'].append(np.stack([tensor_state(s)for s in states]))
        arrays['reference'].append(np.stack([r['reference_compiled']for r in chunk]))
        arrays['raw_action'].append(np.stack([r['action_raw']for r in chunk]));arrays['installed_command'].append(np.stack([r['reference_installed']for r in chunk]))
        arrays['context'].append([*c['goal_xyz'],c['table_top_m'],c['cube_edge_m']])
        arrays['events'].append([[s[k]for k in SCHEMA['events']]for s in states[1:]])
        arrays['done'].append([r['terminal']for r in chunk]);arrays['timeout'].append([r['truncated']for r in chunk])
        arrays['clock'].append([[r['t0']/32,r['native_progress']['point']*.05]for r in chunk]);arrays['valid'].append([True]*window_length)
        arrays['group'].append(canonical[c['model_sha256']]);arrays['window_id'].append(key);arrays['record_ids'].append([fingerprint(r)for r in chunk])
        residual=[]
        for r in chunk:
            tcp=r['state_before']['tcp'];ref=r['reference_installed'];rr=ref[3:9].reshape(2,3);rot=np.stack((rr[0],rr[1],np.cross(*rr)))
            residual.append(np.r_[ref[:3]-tcp[:3,3],Rotation.from_matrix(rot@tcp[:3,:3].T).as_rotvec()])
        arrays['tracking_residual'].append(residual)
    for k in arrays:arrays[k]=np.asarray(arrays[k])
    arrays['group']=np.asarray(arrays['group'],dtype=str)
    arrays['partition']=np.where(arrays['group']==holdout,'TRAIN_ORIGIN_HOLDOUT','FIT')
    arrays.update(schema_json=np.array(json.dumps(SCHEMA)),purpose=np.array('REAL_S_NATIVE_DIAGNOSTIC_REPLAY'),manifest_sha256=np.array(digest(manifest_path)))
    with (output/'REPLAY.npz').open('xb')as f:np.savez_compressed(f,**arrays)
    def events(rows):
        c=Counter()
        for r in rows:
            a,b=r['state_before'],r['state_after']
            c['contact_onset']+=int(not(a['left_object_contact']or a['right_object_contact'])and(b['left_object_contact']or b['right_object_contact']))
            c['grasp_onset']+=int(not a['native_grasp']and b['native_grasp']);c['grasp_sustained_transitions']+=int(a['native_grasp']and b['native_grasp'])
            c['release_onset']+=int(a['native_grasp']and not b['native_grasp']);c['terminal_failure']+=int(r['terminal']);c['timeout']+=int(r['truncated'])
            c['gripper_opening_increase']+=int(b['gripper_fraction']>a['gripper_fraction']+1e-4)
        return dict(c)
    fit=arrays['partition']=='FIT';hold=~fit
    fit_ids=set(arrays['record_ids'][fit].flatten());hold_ids=set(arrays['record_ids'][hold].flatten())
    if fit_ids&hold_ids:raise ValueError('Transition leakage across physical scene groups')
    report=dict(schema='WM02_REAL_REPLAY_AUDIT_V1',raw_record_count=len(all_rows),valid_full_50ms_records=sum(valid.values()),
        unique_observed_state_command_time_pairs=len(unique),duplicate_observed_pairs=len(all_rows)-len(unique),
        original_episode_ids=len({r['episode_id']for r in all_rows}),original_parent_ids=len({r['parent_id']for r in all_rows}),
        independent_physical_scene_groups=len(group_names),groups=group_names,holdout_group=holdout,
        split_policy='Original TRAIN preserved; diagnostic holdout by connected physical scene/parent/model, not formal DEV/TEST',
        burn_in_atoms=8,prediction_atoms=prediction_atoms,window_transitions=window_length,
        continuous_windows=len(windows),fit_windows=int(fit.sum()),holdout_windows=int(hold.sum()),
        discontinuities_not_joined=discontinuities,restore_branch_files=sum(e['kind']=='restore_branch'for e in entries),
        rejection_events=rejections,raw_events=events(all_rows),unique_pair_events=events(unique.values()),
        reward_train_missing_records=sum(r.get('reward_train')is None for r in all_rows),
        full_H40_nonholding_tail_records=sum(not np.array_equal(np.asarray(r['reference_compiled'])[8:],np.repeat(np.asarray(r['reference_compiled'])[7:8],32,axis=0))for r in all_rows),
        compiled_installed_consumed_raw_alignment='PASS_ALL_RECORDS',state_schema=SCHEMA,
        all_original_files_unchanged=all(digest(e['path'])==e['sha256']for e in entries),
        manifest_sha256=digest(manifest_path),numeric_replay_sha256=digest(output/'REPLAY.npz'),
        actual_new_native_transitions=0,formal_complete_task_successes=0,TEST_read=False,streams=streams)
    write_json(output/'AUDIT.json',report);return report
