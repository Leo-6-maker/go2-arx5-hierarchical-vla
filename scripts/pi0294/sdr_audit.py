"""Independently recompute time bounds and native FK of saved solutions; no IK."""
import collections
from pathlib import Path
import numpy as np
from .task_capture_runtime import read,sha
from .sdr_runtime import FOLDER
from .mps2_cpu import first_input

def angle(a,b):
    relative=np.swapaxes(a,-1,-2)@b
    skew=np.stack([relative[...,2,1]-relative[...,1,2],relative[...,0,2]-relative[...,2,0],relative[...,1,0]-relative[...,0,1]],axis=-1)/2
    return np.arctan2(np.linalg.norm(skew,axis=-1),np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1))

def run(root,manifest):
    import mujoco
    from scripts.pi0292.durable import write
    from .mp_client import contact_permitted
    root=Path(root);p=root/FOLDER;output=p/'INDEPENDENT_CPU_AUDIT.json'
    if output.exists():return read(output)
    ledger_before=sha(p/'SEARCH_LEDGER.json');gate=read(p/'RETARGET_TIME_GATE.json');source=read(p/'SOURCE_EXTRACTION.json');cpu=read(p/'CPU_ADMISSION.json')
    times=[];solutions=[];all_calls=0;request_rows=[]
    for row in gate['rows']:
        record=next(r for r in source['records'] if r['id']==row['id'])
        source_file=p/'sources'/record['derived_file'];assert sha(source_file)==record['derived_sha256']
        target_file=p/'sources'/row['retarget_file'];assert sha(target_file)==row['retarget_sha256']
        with np.load(source_file,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
        with np.load(target_file,allow_pickle=False) as f:saved={k:f[k].copy() for k in f.files}
        tool_src=np.asarray(record['tool_binding']['C_tcp_pad']);tool_dst=np.asarray(row['target_tool_binding']['C_tcp_pad'])
        old=read(root/'maniskill-search-v2/CONDITIONS.json')['conditions']
        entry=next(e for e in old if e.get('phase')=='CPU_SAVED_INPUT' and e['task']==row['task'])
        engine,state,_=first_input(root,entry)
        a=np.asarray(arrays['actual_object'])[0,:3,3];b=np.asarray(state['object_pose'])[:3,3]
        g0=np.r_[record['source_card']['goal_xy'],a[2]];card=next(c for c in __import__('scripts.pi0294.sdr_runtime',fromlist=['cards']).cards() if c['task']==row['task']);g1=np.r_[card['goal_xy'],b[2]]
        heading0=np.arctan2(*(g0-a)[[1,0]]);heading1=np.arctan2(*(g1-b)[[1,0]])
        co,si=np.cos(heading1-heading0),np.sin(heading1-heading0);rotation=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
        pads=arrays['committed_TCP']@tool_src;t=arrays['source_time']
        event_start=next(e['time_s'] for e in record['events'] if e['stage']==('carry' if row['task']=='T1' else 'push'))
        event_end=next(e['time_s'] for e in record['events'] if e['previous']==('carry' if row['task']=='T1' else 'push'))
        i0,i1=np.searchsorted(t,[event_start,event_end]);increments=np.linalg.norm(np.diff(pads[i0:i1+1,:3,3],axis=0),axis=1)
        progress=np.r_[0.,np.cumsum(increments)];progress/=progress[-1];weight=np.r_[np.zeros(i0),progress,np.ones(639-i1)]
        expected=pads.copy();expected[:,:3,3]=b+(pads[:,:3,3]-a)@rotation.T+weight[:,None]*(g1-b-rotation@(g0-a))
        expected[:,:3,:3]=rotation@pads[:,:3,:3];expected=expected@np.linalg.inv(tool_dst)
        first,last=record['cut_start_index'],record['cut_end_inclusive_index'];expected=expected[first:last+1]
        assert np.allclose(saved['poses'],expected,atol=1e-12,rtol=0)
        linear=np.linalg.norm(np.diff(expected[:,:3,3],axis=0),axis=1)/.05;angular=angle(expected[:-1,:3,:3],expected[1:,:3,:3])/.05
        factor=max(1.,linear.max()/.06,angular.max()/.3);duration=(last-first)*.05*factor
        assert abs(factor-row['slowdown'])<1e-7 and abs(duration-row['active_duration_s'])<1e-6
        peak=int(np.argmax(linear));global_indices=[first+peak,first+peak+1]
        source_linear=np.linalg.norm(np.diff(arrays['committed_TCP'][first:last+1,:3,3],axis=0),axis=1)/.05
        times.append(dict(id=row['id'],task=row['task'],status=row['status'],independent_min_uniform_slowdown=float(factor),
            independent_zero_entry_duration_s=float(duration),duration_cap_s=28,raw_max_linear_speed_m_s=float(linear.max()),
            raw_max_angular_speed_rad_s=float(angular.max()),source_before_retarget_max_linear_m_s=float(source_linear.max()),
            peak_source_indices=global_indices,peak_is_reference_boundary=global_indices[1]%8==0,
            source_vs_target_functional_tool_matrix_max_abs=float(np.max(np.abs(tool_src-tool_dst))),
            maximum_saved_pose_reconstruction_abs=float(np.max(np.abs(saved['poses']-expected)))))
    binding=read(root/'maniskill-component-v1/planning-assets/ASSET_BINDING.private.json')
    entry=next(e for e in read(root/'maniskill-search-v2/CONDITIONS.json')['conditions'] if e.get('phase')=='CPU_SAVED_INPUT' and e['task']=='T1')
    engine,state,_=first_input(root,entry);m=engine.model;d=mujoco.MjData(m);site=engine.tcp
    for condition in ['SDR_CPU_T1_'+str(i) for i in range(3)]:
        folder=p/'requests'/condition/'0000';request=read(folder/'INPUT.private.json');response=read(folder/'IK_RESULT.private.json')
        assert response['input_sha256']==sha(folder/'INPUT.private.json') and len(request['goals'])==2
        initial=np.asarray(response['initial_fk_world']);assert np.linalg.norm(initial[:3,3]-state['tcp'][:3,3])<=1e-5 and angle(initial[:3,:3],state['tcp'][:3,:3])<=1e-4
        for goal_index,goal in enumerate(response['goals']):
            file=folder/f'IK-{goal_index:02d}.private.json';result=read(file);target=np.asarray(request['goals'][goal_index]['pose']);rows=[]
            assert result['input_sha256']==sha(folder/'INPUT.private.json')
            for k,candidate in enumerate(result['internal_IK']):
                all_calls+=1;q=np.asarray(candidate['solution']);normalized=q.copy();violations=[]
                for index,joint in enumerate(binding['native_joint_ids']):
                    low,high=m.jnt_range[joint]
                    if m.jnt_type[joint]==mujoco.mjtJoint.mjJNT_HINGE:normalized[index]+=2*np.pi*np.round(((low+high)/2-normalized[index])/(2*np.pi))
                    excess=max(low-normalized[index],normalized[index]-high,0.)
                    if excess>0:violations.append(dict(joint=binding['joint_names'][index],violation=float(excess),candidate=float(normalized[index]),lower=float(low),upper=float(high)))
                d.qpos[:]=state['qpos'];d.qvel[:]=state['qvel'];d.ctrl[:]=state['ctrl'];d.qpos[binding['joint_qpos_addresses']]=normalized;mujoco.mj_forward(m,d)
                ep=float(np.linalg.norm(d.site_xpos[site]-target[:3,3]));er=float(angle(d.site_xmat[site].reshape(3,3),target[:3,:3]))
                collisions=[]
                for c in d.contact:
                    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in [c.geom1,c.geom2]]
                    if not contact_permitted(names,c.dist,request['goals'][goal_index]['stage']):collisions.append(dict(names=names,distance_m=float(c.dist)))
                witness=bool(not violations and not collisions and ep<=1e-3 and er<=1e-3)
                rows.append(dict(index=k,numerically_converged=candidate['success'],native_FK_position_error_m=ep,native_FK_rotation_error_rad=er,
                    equivalent_angle_native_limits_valid=not violations,violations=violations,original_rule_collisions=collisions,native_legal_goal_witness=witness))
            solutions.append(dict(condition=condition,endpoint=request['goals'][goal_index]['endpoint'],library_status=result['status'],
                candidate_count=len(rows),numerically_converged=sum(r['numerically_converged'] for r in rows),
                native_limit_valid=sum(r['equivalent_angle_native_limits_valid'] for r in rows),native_legal_witnesses=sum(r['native_legal_goal_witness'] for r in rows),
                violations_by_joint=dict(collections.Counter(v['joint'] for r in rows for v in r['violations'])),
                candidates=rows,request_sha256=sha(folder/'INPUT.private.json'),response_sha256=sha(file)))
        request_rows.append(dict(condition=condition,pairs=len(response['pairs']),request_sha256=sha(folder/'INPUT.private.json'),response_sha256=sha(folder/'IK_RESULT.private.json')))
    assert ledger_before==sha(p/'SEARCH_LEDGER.json')
    unresolved=any(row['native_legal_witnesses'] for row in solutions)
    value=dict(status='INTEGRITY_EXCEPTION_REQUIRES_RECONCILIATION' if unresolved else 'PASS_INDEPENDENT_BOUNDED_SOURCE_ADMISSION_FAIL_RECONCILIATION',
        auditor_source_sha=manifest['code']['vla_commit'],original_CPU_source_sha=cpu['source_sha'],time_cases=times,native_goal_cases=solutions,
        saved_requests=request_rows,actual_saved_CLIK_initializations=all_calls,search_ledger_sha256=ledger_before,search_ledger_unchanged=True,
        new_global_IK=0,new_RRT=0,new_physics=0,new_render=0,new_neural_queries=0,unresolved_native_legal_witness=unresolved,
        global_IK_infeasibility_proven=False,WBC_task_failure_proven=False,selected='NONE',deployment='NONE')
    write(output,value);return value
