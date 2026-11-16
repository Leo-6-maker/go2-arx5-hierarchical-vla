"""All saved C20 native queries and the independent contact event: zero effects."""
import json,sys,time,struct,hashlib,traceback
from pathlib import Path
from collections import Counter
import numpy as np
from scripts.pi0292.durable import write
from scripts.pi0294.task_state import parse_state
from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import read,sha
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles,pad32
from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
from go2arx5_vla.trajectory.so3 import quat_to_matrix

def pose7(p):
    v=np.eye(4);v[:3,3]=p[:3];v[:3,:3]=quat_to_matrix(np.asarray(p[3:7]));return v

def packets(path):
    with Path(path).open('rb')as h:
        while b:=h.read(4):yield parse_state(h.read(struct.unpack('<I',b)[0]))

def collision(case,spec,result):
    import mujoco
    from scripts.pi0294.task_contacts import allowed
    from go2arx5_vla.world_model.native_replay import forbid_integration
    model=mujoco.MjModel.from_xml_path(spec['model_path']);data=mujoco.MjData(model);last=[]
    for row in packets(case/'native-01/state.bin.journal'):
        if row['step']>=result['actual_native_integrations']-1250:last.append(row)
    rows=[]
    with forbid_integration(mujoco)as calls:
        for packet in last:
            data.qpos[:]=packet['qpos'];data.qvel[:]=packet['qvel'];data.ctrl[:]=packet['ctrl'];data.time=packet['time_s'];mujoco.mj_forward(model,data)
            bad=[]
            for c in packet['contacts']:
                names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c[g])for g in ('geom0','geom1')]
                if not allowed(*names):bad.append(dict(pair=names,distance_m=c['distance_m'],normal_force_N=float(c['wrench'][0]),point=c['point'].tolist(),contact_time_s=packet['contact_time_s']))
            clearance={}
            table=model.geom('task_table').id;top=float(data.geom_xpos[table,2]+model.geom_size[table,2])
            for name in ('task_left_pad','task_right_pad'):
                g=model.geom(name).id;corners=np.asarray([[x,y,z]for x in (-1,1)for y in (-1,1)for z in (-1,1)])*model.geom_size[g];world=corners@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
                clearance[name]=dict(min_box_corner_z_minus_table_top_m=float(world[:,2].min()-top),geom_type=int(model.geom_type[g]),table_top_m=top)
            tcp=model.site('x5_ee').id;base=model.body('x5_base_link').id
            if bad or packet['step']%25==3 or packet is last[-1]:rows.append(dict(step=packet['step'],state_time_s=packet['time_s'],contact_time_s=packet['contact_time_s'],illegal_saved_contacts=bad,pad_clearance=clearance,TCP=data.site_xpos[tcp].tolist(),BASE=data.xpos[base].tolist(),arm_qpos=packet['qpos'].tolist(),ctrl=packet['ctrl'].tolist()))
    return dict(original_stop=result['physics_stops'],last_2p5s=rows,first_illegal=next((r for r in rows if r['illegal_saved_contacts']),None),integrations=0,controller_restoration_claimed=False,passive_geometry_only=True,forbidden_integration_calls=len(calls),geometry_source=dict(path=spec['model_path'],sha256=sha(spec['model_path'])))

def run(s):
    root=Path(s['registered_root']);old=Path(s['C20_root']);stats=read(old/'dataset/STATISTICS.json');quant=Quantiles(stats['actions']['q01'],stats['actions']['q99']);all_rows=[];episodes=[];collisions=[]
    for panel in ('TRAIN8','DEV16'):
        group=old/panel/'native';batch=read(group/'RESULT.private.json')
        for outcome in batch['outcomes']:
            case=group/'runs'/outcome['root'];spec=read(case/'EXECUTION_SPEC.json');result=read(case/'RESULT.private.json');assert sha(case/'RESULT.private.json')==outcome['result_sha256']
            native=case/'native-01';commands=[json.loads(l)for l in(native/'COMMANDS.private.jsonl').read_text().splitlines()];boundaries={c['native_step']:c for c in commands if c['phase']=='mobile-accepted-boundary-050'}
            samples=[json.loads(l)for l in(native/'MOBILE_SAMPLES.private.jsonl').read_text().splitlines()];sample_by_step={r['native_end_step']:r for r in samples};episode=[]
            import mujoco
            model=mujoco.MjModel.from_xml_path(spec['model_path']);query_ticks={int(read(q/'INPUT_META.private.json')['native_step'])for q in(native/'queries').iterdir()};contact_packets={r['step']:r for r in packets(native/'state.bin.journal')if r['step']in query_ticks}
            initial_rotation=np.asarray(sample_by_step[5703]['actual_TCP'])[:3,:3]
            for query in sorted((native/'queries').iterdir()):
                meta=read(query/'INPUT_META.private.json');response=read(query/'RESPONSE.private.json');assert sha(query/'INPUT.private.npz')==meta['input_sha256'];assert sha(response['action'])==response['action_sha256']
                with np.load(query/'INPUT.private.npz',allow_pickle=False)as z:state=z['state_eef10'];frame=z['T_policy_from_mount'];image_shapes={k:list(z[k].shape)for k in ('body_rgb','wrist_rgb')}
                with np.load(response['action'],allow_pickle=False)as z:raw=z['physical_H40'];normalized=z['normalized']
                decoded=GripLogitCodec(frame).decode(state,pad32(quant.unnormalize(normalized[0,:,:7])));decode_error=float(np.max(np.abs(decoded-raw)))
                if decode_error>1e-12:raise ValueError('SAVED_NORMALIZER_CODEC_AND_SINGLE_SIGMOID_DISAGREE')
                step=meta['native_step'];witness=boundaries[step];reference=pose7(witness['accepted_pose7']);diagnostic=analyze(raw,state,np.asarray(spec['policy_from_world']),reference)
                sample=sample_by_step[step];objects=np.asarray(sample['object_pose']);tcp=np.asarray(sample['actual_TCP']);grasp_center=objects[:3,3]+np.array([0,-.0017,.012+(.008 if spec['mobile_condition']['family']=='F2'else 0)])+initial_rotation@np.array([-.012,0.,0.])
                contacts=[dict(pair=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c[g])for g in ('geom0','geom1')],distance_m=c['distance_m'],normal_force_N=float(c['wrench'][0]))for c in contact_packets[step]['contacts']]
                row=dict(panel=panel,case=case.name,query=int(query.name),input=dict(path=str(query/'INPUT.private.npz'),sha256=meta['input_sha256']),output=dict(path=response['action'],sha256=response['action_sha256']),native_step=step,normalized_shape=list(normalized.shape),codec_decode_error=decode_error,sigmoid_count=1,image_shapes=image_shapes,diagnostic=diagnostic,query_context=dict(actual_TCP=tcp.tolist(),object_pose=objects.tolist(),distance_TCP_to_V3_geometric_grasp_goal_m=float(np.linalg.norm(tcp[:3,3]-grasp_center)),V3_geometric_grasp_goal_audit_only=grasp_center.tolist(),same_full_control_state_expert_plan_paired=False,actual_measured_gripper_fraction=float(state[9]),actual_saved_contacts=contacts,actual_gripper_command=sample['gripper_command'],native_grasp=sample['native_grasp'],BASE=sample['actual_mount'],accepted_reference=reference.tolist()),actual_consumed_ticks=max(0,min(250,result['actual_native_integrations']-step)),terminal_native_tick=result['actual_native_integrations'])
                episode.append(row);all_rows.append(row)
            if len(episode)!=result['model_generations']:raise ValueError('ALL_TRUE_QUERY_DENOMINATOR')
            rejection=(result.get('detail')or{}).get('valid_policy_rejection');bad=[r for r in episode if not r['diagnostic']['original_full_H40_pass']]
            if (len(bad)!=bool(rejection)or rejection and bad[0]['query']!=rejection['query']):raise ValueError('REPRODUCE_ORIGINAL_NATIVE_GATE_REJECTION')
            episodes.append(dict(case=case.name,panel=panel,queries=len(episode),H40_pass=len(episode)-len(bad),first_rejection=bad[0]if bad else None,native_ticks=result['actual_native_integrations'],unsafe=result['unsafe'],task_success=(result.get('detail')or{}).get('task_success',False),ever_grasp=bool(result['metrics']['grasped_atoms'])))
            if result['unsafe']:collisions.append(dict(case=case.name,**collision(case,spec,result),last_query=episode[-1]))
            write(root/'PROGRESS.private.json',dict(epoch=time.time(),completed_episodes=len(episodes),audited_queries=len(all_rows),new_model_generations=0,new_optimizer_updates=0,physical_requests=0))
    bad=[r for r in all_rows if not r['diagnostic']['original_full_H40_pass']];assert len(episodes)==13 and len(bad)==12 and len(collisions)==1
    write(root/'ALL_QUERIES.private.json',all_rows);write(root/'COLLISION.private.json',collisions)
    result=dict(schema='MFREE_C20_ALL_QUERY_ZERO_UPDATE_AUDIT_V2',epoch=time.time(),episodes=episodes,actual_original_queries=len(all_rows),original_H40_pass=len(all_rows)-len(bad),original_H40_rejected=len(bad),first_channels=dict(Counter(r['diagnostic']['first_temporal_exceedance']['channel']for r in bad)),first_regions=dict(Counter(r['diagnostic']['first_temporal_exceedance']['region']for r in bad)),max_rejection_ratio=max(r['diagnostic']['first_temporal_exceedance']['peak_ratio']for r in bad),new_model_generations=0,new_optimizer_updates=0,new_training_visits=0,new_physical_requests=0,new_integrations=0,all_query_records=dict(path=str(root/'ALL_QUERIES.private.json'),sha256=sha(root/'ALL_QUERIES.private.json')),collision=dict(path=str(root/'COLLISION.private.json'),sha256=sha(root/'COLLISION.private.json')),old_RP8_passed=False)
    write(root/'RESULT.private.json',result);return result

if __name__=='__main__':
 s=read(sys.argv[sys.argv.index('--audit')+1]);root=Path(s['registered_root'])
 try:run(s);code=0
 except BaseException:write(root/'RESULT.private.json',dict(exit_code=1,failure=traceback.format_exc(),new_optimizer_updates=0,new_physical_requests=0));code=1
 from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
 close(s,code);raise SystemExit(code)
