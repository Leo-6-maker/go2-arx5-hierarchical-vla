"""Derive all1220 command labels;reuse immutable RGB without camera requalification."""
import argparse,json,os,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from scripts.pi0291_arx5_data.native_dataset import matrix_from10,pose10
from scripts.pi0292.durable import write
from scripts.pi0294.e1.artifacts import atomic_npz
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles,pad32,roundtrip_error
from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec,EPSILON,VERSION as CODEC
from go2arx5_vla.sla_adapter.reference_labels_v6 import derive,VERSION
from go2arx5_vla.sla_adapter import reference_dataset_v6 as data
from go2arx5_vla.world_model.task_learning_grasp_v1 import FiftyMsProgram,WORKING_CLOSED_M,WORKING_OPEN_M,MECHANICAL_TRAVEL_M


def envelope(raw,state):
    desired=raw.copy();desired[:,9]=(WORKING_CLOSED_M+(WORKING_OPEN_M-WORKING_CLOSED_M)*raw[:,9])/MECHANICAL_TRAVEL_M
    anchor=matrix_from10(state)
    FiftyMsProgram.validate_envelope(SimpleNamespace(desired_path=desired,decision_TCP_target=anchor,
        reference_path=desired,initial_reference_TCP_target=anchor))


def run(parent,output):
    parent,output=Path(parent),Path(output);start=time.time();cpu=time.process_time()
    p=parent/'MANIFEST.private.json'
    if sha(p)!=data.ORIGINAL_MANIFEST_SHA or output.exists():raise ValueError('NEW_OUTPUT_AND_EXACT_SEALED_N20_PARENT_REQUIRED')
    m=json.loads(p.read_bytes());output.mkdir();write(output/'STATE.private.json',dict(state='preparing',models=0,updates=0,physics=0))
    (output/'ORIGINAL_N20_MANIFEST.private.json').write_bytes(p.read_bytes())
    episodes=[];states=[];actions=[];reports=[];identity=0.;old_pass=0;fp32=0;maximum=dict(xyz_m=0.,so3_rad=0.,gripper=0.)
    try:
        for e in m['episodes']:
            spec_path=Path(e['mount']['source_spec']['path']);source=spec_path.parent;spec=json.loads(spec_path.read_bytes())
            smp=source/'data/sla-pt-shards/MANIFEST.private.json';sm=json.loads(smp.read_bytes());lp=smp.parent/sm['label_shard']['file']
            if (sha(spec_path)!=e['source_spec_sha256'] or sha(smp)!=e['original_recording_manifest_sha256']
                    or sha(lp)!=sm['label_shard']['sha256'] or sm['source_command_count']!=640
                    or sm['source_proof']['result']['sha256']!=e['source_native_result_sha256']
                    or sm['source_proof']['spec']['sha256']!=e['source_spec_sha256']):
                raise ValueError('ORIGINAL_CONSUMPTION_PROVEN_COMMAND_AND_FRAME_CUSTODY')
            oldpath=parent/e['labels']['file']
            if sha(oldpath)!=e['labels']['sha256']:raise ValueError('ORIGINAL_LABEL_BYTES_CHANGED')
            with np.load(oldpath,allow_pickle=False)as z:old={k:z[k].copy()for k in z.files}
            with np.load(lp,allow_pickle=False)as z:steps=z['command_native_step'].copy();nom=z['command_nominal_eef10'].copy();installed=z['command_installed_eef10'].copy()
            if not np.array_equal(steps,5703+np.arange(640)*25):raise ValueError('ALL640_ACTUAL_COMMAND_CLOCK_POINTS')
            frame=np.asarray(spec['task_phase_source_from_target']);nom=pose10(frame@matrix_from10(nom),nom[:,9]);installed=pose10(frame@matrix_from10(installed),installed[:,9])
            codec=GripLogitCodec(e['mount']['T_policy_from_mount']);new=[];latent=[];errors=[]
            for a in range(61):
                ix=slice(a*10,a*10+40);state=old['state_eef10'][a];before=old['nominal_policy_H40'][a]
                if not np.allclose(before,nom[ix],rtol=0,atol=1e-12):raise ValueError('EVERY_OLD_NOMINAL_JOINS_ITS_ACTUAL_ISSUING_COMMAND')
                try:envelope(before,state);old_pass+=1
                except ValueError:pass
                value,proof=derive(state,before,installed[ix],anchor_native_step=int(old['native_step'][a]),command_native_steps=steps[ix])
                state7,label=codec.encode(state,value)
                if not np.allclose(state7,old['state7'][a],rtol=0,atol=1e-12):raise ValueError('ORIGINAL_MEASURED_STATE_INPUT_MUST_NOT_CHANGE')
                identity=max(identity,proof['reconstructed_reference_max_error']);new.append(value);latent.append(label);states.append(old['state7'][a]);actions.extend(label)
            folder=output/e['id'];folder.mkdir()
            for desc in e['frame_shards']:
                before=parent/desc['file'];after=output/desc['file'];os.link(before,after)
                if (before.stat().st_dev,before.stat().st_ino)!=(after.stat().st_dev,after.stat().st_ino):raise ValueError('SAME_IMMUTABLE_RGB_INODE_REQUIRED')
            os.link(oldpath,folder/'ORIGINAL_ISSUING_QUERY_LABELS.private.npz')
            atomic_npz(folder/'TRAIN_WINDOWS.npz',**dict(old,latent7=np.stack(latent),nominal_policy_H40=np.stack(new)))
            episodes.append(dict(e,labels=dict(file=e['id']+'/TRAIN_WINDOWS.npz',sha256=sha(folder/'TRAIN_WINDOWS.npz')),
                original_issuing_query_labels=dict(file=e['id']+'/ORIGINAL_ISSUING_QUERY_LABELS.private.npz',sha256=e['labels']['sha256']),
                original_consumed_label_shard_sha256=sm['label_shard']['sha256']))
            reports.append(dict(episode=e['id'],windows=61,source_command_shard_sha256=sm['label_shard']['sha256']))
        states,actions=np.asarray(states),np.asarray(actions)
        original_stats_path=parent/m['statistics']['file']
        if sha(original_stats_path)!=m['statistics']['sha256']:raise ValueError('ORIGINAL_TRAIN_ONLY_STATE_NORMALIZER_BYTES')
        original_stats=json.loads(original_stats_path.read_bytes());fitted=Quantiles.fit(states)
        if not all(np.allclose(getattr(fitted,k),original_stats['state'][k],rtol=0,atol=1e-12)for k in('q01','q99')):
            raise ValueError('CURRENT_STATE_POPULATION_AND_NORMALIZER_MUST_STAY_IDENTICAL')
        stats=dict(state=original_stats['state'],actions=Quantiles.fit(actions).summary(actions),
            fit_unique_windows=1220,fitted_before_oversampling=True,split='TRAIN',codec_version=CODEC,
            epsilon=EPSILON,state_gripper_transformed=False,label_semantics=VERSION)
        write(output/'STATISTICS.private.json',stats);q=Quantiles(stats['actions']['q01'],stats['actions']['q99'])
        for e in episodes:
            with np.load(output/e['labels']['file'],allow_pickle=False)as z:labels=z['latent7'].copy();state=z['state_eef10'].copy();nominal=z['nominal_policy_H40'].copy()
            codec=GripLogitCodec(e['mount']['T_policy_from_mount'])
            for label,s,target in zip(labels,state,nominal,strict=True):
                packed=pad32(q.normalize(label)).astype(np.float32);decoded=codec.decode(s,pad32(q.unnormalize(packed[:,:7])))
                error=roundtrip_error(target,decoded)
                if error['xyz_m']>1e-6 or error['so3_rad']>1e-5 or error['gripper']>EPSILON+1e-6:raise ValueError('REFERENCE_LABEL_FP32_CODEC_ROUNDTRIP')
                for k,v in error.items():maximum[k]=max(maximum[k],v)
                envelope(decoded,s);fp32+=1
        if len(states)!=1220 or fp32!=1220:raise ValueError('NO_DROPPED_OR_SELECTED_WINDOWS')
        audit=dict(schema=data.AUDIT_SCHEMA,passes=True,read_windows=1220,episodes=20,
            original_manifest_sha256=data.ORIGINAL_MANIFEST_SHA,original_full_H40_pass=old_pass,
            original_full_H40_reject=1220-old_pass,full_H40_envelope_pass=1220,FP32_decoded_envelope_pass=fp32,
            reconstructed_installed_reference_max_error=identity,FP32_roundtrip_max_error=maximum,
            label_semantics=VERSION,original_observations_and_states_unchanged=True,all40_target_knots_actually_consumed=True,
            RGB_requalified=False,frame_storage='ORIGINAL_IMMUTABLE_HARDLINKS',model_loads=0,
            optimizer_updates=0,physics_integrations=0,DEV_TEST_reads=0,episodes_provenance=reports,
            wall_s=time.time()-start,CPU_s=time.process_time()-cpu)
        write(output/'REFERENCE_CONVERSION.private.json',audit)
        result=dict(m,schema=data.SCHEMA,dataset_id=data.DATASET_ID,scope=data.SCOPE,episodes=episodes,
            label_semantics=VERSION,original_observations_and_states_unchanged=True,all40_target_knots_actually_consumed=True,
            native_reference_transport_changed=False,original_target_manifest=dict(file='ORIGINAL_N20_MANIFEST.private.json',sha256=data.ORIGINAL_MANIFEST_SHA),
            reference_conversion=dict(file='REFERENCE_CONVERSION.private.json',sha256=sha(output/'REFERENCE_CONVERSION.private.json')),
            statistics=dict(file='STATISTICS.private.json',sha256=sha(output/'STATISTICS.private.json')))
        data.manifest_contract(result);write(output/'MANIFEST.private.json',result)
        write(output/'STATE.private.json',dict(state='committed',windows=1220,models=0,updates=0,physics=0))
        return dict(audit,manifest_sha256=sha(output/'MANIFEST.private.json'),statistics_sha256=result['statistics']['sha256'])
    except BaseException as e:
        write(output/'STATE.private.json',dict(state='incomplete',failure=repr(e),original_dataset_modified=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.parent,a.output)))
