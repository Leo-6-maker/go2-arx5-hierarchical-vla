"""Two named PD4 label views from the same new complete expert episodes.

Each view fits its own TRAIN action statistics after the physical transform.
This CPU audit is not a native replay of the newly fitted statistics or a
training admission. The captured native transport used frozen N20 statistics.
"""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from scripts.pi0291_arx5_data.native_dataset import matrix_from10,pose10
from scripts.pi0292.durable import write
from scripts.pi0294.e1.artifacts import atomic_npz
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles,pad32,roundtrip_error,TOLERANCES
from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec,EPSILON
from go2arx5_vla.sla_adapter.action_contract_v61 import consumed_mask,identity
from go2arx5_vla.world_model.sla_dense_policy_v5 import compile_proposal
from go2arx5_vla.world_model.task_learning_grasp_v1 import FiftyMsProgram,WORKING_OPEN_M,WORKING_CLOSED_M,MECHANICAL_TRAVEL_M
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix,rotation_matrix_to_rot6d_rows

VIEWS=('PD4_EXECUTED_REF','PD4_FINAL_PLAN')
CORE=('schema','query_id','native_step','final_H40','installed_reference_H40','planner_sha256',
      'adapter_sha256','normalizer_sha256','command_period_ticks','physics_dt_s')


def reference_labels(state_native,reference_before_query,installed,nominal_grip):
    """Invert the original transport with explicit R0,not installed[0]."""
    state,reference,installed=np.asarray(state_native),np.asarray(reference_before_query),np.asarray(installed)
    raw=installed.copy();raw[:,:3]=state[:3,3]+installed[:,:3]-reference[:3,3]
    raw[:,3:9]=rotation_matrix_to_rot6d_rows(state[:3,:3]@reference[:3,:3].T@rot6d_rows_to_matrix(installed[:,3:9]))
    raw[:,9]=nominal_grip
    return raw


def run(producer,output):
    producer,output=Path(producer),Path(output);started=time.time();cpu=time.process_time()
    group=json.loads((producer/'RESULT.private.json').read_bytes());lease=json.loads((producer/'control/LEASE.json').read_bytes())
    if(output.exists()or lease['status']!='CLOSED' or lease['worker_exit_code']!=0
            or(group['requested'],group['terminal'],group['full'])!=(4,4,4)
            or group['old_RP8_passed']is not False):raise ValueError('FOUR_ACTUAL_COMPLETE_NEW_EPISODES_AND_NEW_OUTPUT')
    output.mkdir();banks={v:[]for v in VIEWS};states=[];source_rows=[];all_states=[]
    for row in sorted(group['outcomes'],key=lambda r:r['layout']):
        root=producer/'runs'/row['native_run'];spec_path=root/'EXECUTION_SPEC.json';spec=json.loads(spec_path.read_bytes())
        plan_path=root/'native-01/FINAL_PLAN_CAPTURE_ACTUAL.private.json';capture=json.loads(plan_path.read_bytes());plans=capture['plans']
        smp=root/'data/sla-pt-shards/MANIFEST.private.json';sm=json.loads(smp.read_bytes())
        if(capture['writer']['committed']is not True or(capture['queries'],capture['supervised_points'],capture['consumed_points'])!=(61,2440,640)
                or not capture['expert_task_success_observed'] or sm['training_dataset_admitted']is not False
                or sm['source_proof']['spec']['sha256']!=sha(spec_path)):
            raise ValueError('COMPLETE_FINAL_PLANS_NATIVE_MASKS_AND_RAW_SHARD_CUSTODY')
        labels=root/'data/sla-pt-shards'/sm['label_shard']['file']
        if sha(labels)!=sm['label_shard']['sha256']:raise ValueError('ACTUAL_CONSUMED_SHARD_BYTES')
        with np.load(labels,allow_pickle=False)as z:old={k:z[k].copy()for k in z.files}
        if not np.array_equal(old['command_native_step'],5703+np.arange(640)*25):raise ValueError('ORIGINAL_20HZ_COMMAND_CLOCK')
        frame=np.asarray(spec['task_phase_source_from_target']);mount=sm['source_proof']['mount']['T_policy_from_mount'];codec=GripLogitCodec(mount)
        source_frames=[];source_steps=[]
        for shard in sm['frame_shards']:
            fp=root/'data/sla-pt-shards'/shard['file']
            if sha(fp)!=shard['sha256']:raise ValueError('ORIGINAL_LOSSLESS_FRAME_SHARD_BYTES')
            with np.load(fp,allow_pickle=False)as z:source_frames.extend(z['state_eef10']);source_steps.extend(z['native_step'])
        if not np.array_equal(source_steps,5703+np.arange(321)*50):raise ValueError('ORIGINAL_10HZ_CAMERA_CLOCK')
        state10=np.asarray(source_frames)[np.arange(61)*5];state7=old['state7'];all_states.extend(state7)
        command_ids=[];query_ids=[]
        for a,p in enumerate(plans):
            if(p['origin_query_index']!=a or p['native_step']!=5703+250*a
                    or identity({k:p[k]for k in CORE})!=p['plan_sha256']
                    or p['normalizer_sha256']is not None or p['supervision_mask']!=[True]*40
                    or consumed_mask(p,p['events'])!=p['consumed_mask']
                    or p['consumed_mask']!=([True]*10+[False]*30 if a<60 else[True]*40)
                    or p['transport_proof']['expert_segment_generator_applied']is not False):
                raise ValueError('IMMUTABLE_ORIGIN_PLAN_AND_NO_FUTURE_QUERY_BACKFILL')
            count=10 if a<60 else 40;command_ids.extend(p['command_ids'][:count]);query_ids.extend([p['query_id']]*count)
        if len(command_ids)!=640:raise ValueError('ALL640_ACTUAL_COMMAND_IDENTITIES')
        phases=[]
        for a in range(61):
            path=root/'native-01'/('TASK_LEARNING_DECISION_'+str(a).zfill(3)+'.private.json')
            phases.append(json.loads(path.read_bytes())['stage']if a<60 else'HOLD')
        records={v:[]for v in VIEWS}
        for a,p in enumerate(plans):
            actual=np.linalg.inv(frame)@matrix_from10(state10[a]);reference=np.asarray(p['reference_before_query']);base=np.asarray(p['base_pose_hold'])
            if not np.allclose(p['transport_query_state_eef10'],state10[a],rtol=0,atol=1e-9)or not np.allclose(p['transport_policy_from_mount'],mount,rtol=0,atol=1e-9):
                raise ValueError('SAME_CAPTURED_CURRENT_STATE_AND_EPISODE_MOUNT')
            ix=slice(a*10,a*10+40);installed=old['command_installed_eef10'][ix]
            ref=reference_labels(actual,reference,installed,old['command_nominal_eef10'][ix,9])
            for view,raw in zip(VIEWS,(ref,np.asarray(p['final_H40'])),strict=True):
                program=FiftyMsProgram(raw,actual,reference,base)
                expected=installed if view==VIEWS[0]else np.asarray(p['installed_reference_H40'])
                if not np.allclose(program.reference_path,expected,rtol=0,atol=1e-12):raise ValueError('EXACT_CURRENT_REFERENCE_TRANSPORT_IDENTITY')
                target=pose10(frame@matrix_from10(raw),raw[:,9]);s7,latent=codec.encode(state10[a],target)
                if not np.allclose(s7,state7[a],rtol=0,atol=1e-12):raise ValueError('UNCHANGED_STUDENT_STATE')
                records[view].append(dict(target=target,latent=latent,reference_before=reference,base=base,
                    ids=command_ids[ix]if view==VIEWS[0]else p['command_ids'],
                    queries=query_ids[ix]if view==VIEWS[0]else[p['query_id']]*40,
                    consumed=[True]*40 if view==VIEWS[0]else p['consumed_mask']))
        common=dict(id=row['layout']+'-r14',layout=row['layout'],source_run=root.name,split='TRAIN',task='PICK_HOLD',
            mount=mount,source_spec_sha256=sha(spec_path),source_native_result_sha256=sha(root/'native-01/RESULT.json'),
            raw_shard_manifest_sha256=sha(smp),final_plan_summary_sha256=sha(plan_path),source_root=str(root))
        for view in VIEWS:
            banks[view].append(dict(common,state_eef10=state10,state7=state7,phase=phases,records=records[view],frame_shards=sm['frame_shards']))
    result=dict(schema='SLA_PD4_TWO_VIEW_STATISTICS_AND_FP32_AUDIT_V6_1',views={},models=0,updates=0,physics_integrations=0,
        old_RP8_passed=False,new_statistics_native_replay=False,new_statistics_training_admitted=False)
    for view,episodes in banks.items():
        dest=output/view;dest.mkdir();actions=np.asarray([r['latent']for e in episodes for r in e['records']]).reshape(-1,7)
        state_values=np.asarray(all_states);stat=dict(state=Quantiles.fit(state_values).summary(state_values),actions=Quantiles.fit(actions).summary(actions),
            split='TRAIN',fit_unique_windows=244,supervised_action_points=9760,consumed_mask_filters_statistics=False,
            fit_input='UNNORMALIZED_RELATIVE_XYZ_SO3_AND_LOGIT_GRIP',resampling_multiplicity=False)
        write(dest/'STATISTICS.private.json',stat);q=Quantiles(stat['actions']['q01'],stat['actions']['q99']);descriptors=[];maximum=dict(xyz_m=0.,so3_rad=0.,mechanical_grip_m=0.);count=0
        for e in episodes:
            folder=dest/e['id'];folder.mkdir();root=Path(e['source_root']);spec=json.loads((root/'EXECUTION_SPEC.json').read_bytes());frame=np.asarray(spec['task_phase_source_from_target']);codec=GripLogitCodec(e['mount'])
            for shard in e['frame_shards']:os.link(root/'data/sla-pt-shards'/shard['file'],folder/shard['file'])
            r=e['records'];atomic_npz(folder/'TRAIN_WINDOWS.npz',state_eef10=e['state_eef10'],state7=e['state7'],latent7=np.stack([x['latent']for x in r]),
                nominal_policy_H40=np.stack([x['target']for x in r]),native_step=5703+np.arange(61)*250,frame_index=np.arange(61)*5,phase=np.asarray(e['phase']),
                supervision_mask=np.ones((61,40),bool),consumed_mask=np.asarray([x['consumed']for x in r]),
                origin_query_ids=np.asarray([x['queries']for x in r]),origin_command_ids=np.asarray([x['ids']for x in r]))
            audit_folder=folder/'FP32_FILES';audit_folder.mkdir()
            for a,x in enumerate(r):
                packed=pad32(q.normalize(x['latent'])).astype(np.float32)[None];container=audit_folder/(str(a).zfill(2)+'-MODEL.npz');atomic_npz(container,normalized=packed)
                with np.load(container,allow_pickle=False)as z:actual=z['normalized'].copy()
                decoded=codec.decode(e['state_eef10'][a],pad32(q.unnormalize(actual[0,:,:7])));action=audit_folder/(str(a).zfill(2)+'-ACTION.npz');atomic_npz(action,normalized=actual,physical_H40=decoded)
                compiled,_=compile_proposal(action,sha(action),source_from_target=frame,query_state10=e['state_eef10'][a],
                    actual_TCP=np.linalg.inv(frame)@matrix_from10(e['state_eef10'][a]),consumed_reference_TCP=x['reference_before'],base_hold=x['base'])
                desired=FiftyMsProgram(pose10(np.linalg.inv(frame)@matrix_from10(x['target']),x['target'][:,9]),
                    np.linalg.inv(frame)@matrix_from10(e['state_eef10'][a]),x['reference_before'],x['base'])
                err=roundtrip_error(desired.reference_path,compiled.reference_path);width=err['gripper']*MECHANICAL_TRAVEL_M
                if(err['xyz_m']>TOLERANCES['xyz_m']or err['so3_rad']>TOLERANCES['so3_rad']or width>EPSILON*(WORKING_OPEN_M-WORKING_CLOSED_M)+1e-12):raise ValueError('NEW_VIEW_FROZEN_STATISTICS_FP32_RAW_COMPILER')
                maximum['xyz_m']=max(maximum['xyz_m'],err['xyz_m']);maximum['so3_rad']=max(maximum['so3_rad'],err['so3_rad']);maximum['mechanical_grip_m']=max(maximum['mechanical_grip_m'],width);count+=1
            descriptors.append({k:v for k,v in e.items()if k not in('state_eef10','state7','phase','records','frame_shards')}|dict(
                labels=dict(file=e['id']+'/TRAIN_WINDOWS.npz',sha256=sha(folder/'TRAIN_WINDOWS.npz')),
                frame_shards=[dict(d,file=e['id']+'/'+d['file'])for d in e['frame_shards']]))
        manifest=dict(schema='SLA_PROSPECTIVE_PD4_DUAL_VIEW_DATASET_V6_1',dataset_id=view,episodes=descriptors,windows=count,
            statistics=dict(file='STATISTICS.private.json',sha256=sha(dest/'STATISTICS.private.json')),producer_result_sha256=sha(producer/'RESULT.private.json'),
            training_dataset_admitted=False,PD20_complete=False,old_RP8_passed=False)
        if count!=244:raise ValueError('ALL244_WINDOWS_NO_DROPPING')
        write(dest/'MANIFEST.private.json',manifest);result['views'][view]=dict(windows=count,manifest_sha256=sha(dest/'MANIFEST.private.json'),statistics_sha256=manifest['statistics']['sha256'],FP32_max_reference_error=maximum)
    result.update(passes=True,wall_s=time.time()-started,CPU_s=time.process_time()-cpu);write(output/'RESULT.private.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--producer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.producer,a.output)))
