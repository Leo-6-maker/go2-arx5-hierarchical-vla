"""Actual FP32 files -> existing decoder -> unmodified raw student compiler.

No model,MuJoCo integration or native certificate is run here. RP8 must still
reconstruct actual control history and consume its registered P10 conditions.
"""
import argparse,collections,json,time
from pathlib import Path
import numpy as np
from scripts.pi0294.e1.artifacts import atomic_npz
from scripts.pi0291_arx5_data.native_dataset import matrix_from10
from go2arx5_vla.sla_adapter.action_contract_v61 import DATA_SHA,STATS_SHA,selection
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles,pad32,roundtrip_error,TOLERANCES
from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec,EPSILON
from go2arx5_vla.world_model.sla_dense_policy_v5 import compile_proposal,VERSION as CONSUMER
from go2arx5_vla.world_model.task_learning_grasp_v1 import MECHANICAL_TRAVEL_M,WORKING_OPEN_M,WORKING_CLOSED_M


def run(dataset,output):
    dataset,output=Path(dataset),Path(output);start=time.time();cpu=time.process_time()
    if sha(dataset/'MANIFEST.private.json')!=DATA_SHA or output.exists():raise ValueError('EXACT_V6_DATA_AND_FRESH_AUDIT_OUTPUT')
    m=json.loads((dataset/'MANIFEST.private.json').read_bytes());stats_path=dataset/m['statistics']['file']
    if sha(stats_path)!=STATS_SHA:raise ValueError('FROZEN_FULL_N20_STATISTICS_NOT_MICROFIT_STATISTICS')
    stats=json.loads(stats_path.read_bytes());q=Quantiles(stats['actions']['q01'],stats['actions']['q99']);chosen=selection(m)
    output.mkdir();(output/'S16_RP8.private.json').write_text(json.dumps(chosen,indent=2)+'\n')
    phases=collections.Counter();reports=[];maximum=dict(xyz_m=0.,so3_rad=0.,mechanical_grip_m=0.);count=0
    for e in m['episodes']:
        root=Path(e['mount']['source_spec']['path']).parent;spec_path=root/'EXECUTION_SPEC.json'
        if sha(spec_path)!=e['source_spec_sha256']:raise ValueError('ORIGINAL_SPEC_CHANGED')
        spec=json.loads(spec_path.read_bytes());frame=np.asarray(spec['task_phase_source_from_target']);labels=dataset/e['labels']['file']
        if sha(labels)!=e['labels']['sha256']:raise ValueError('V6_WINDOW_LABEL_BYTES_CHANGED')
        smp=root/'data/sla-pt-shards/MANIFEST.private.json';sm=json.loads(smp.read_bytes());lp=smp.parent/sm['label_shard']['file']
        if sha(smp)!=e['original_recording_manifest_sha256']or sha(lp)!=sm['label_shard']['sha256']:
            raise ValueError('ORIGINAL_CONSUMED_COMMAND_SHARD_CHANGED')
        # Use the actual recorded BASE7 target,never an invented fixed-base zero.
        cycle_path=root/'native-01/STABLE_TASK_CYCLES/00/RESULT.private.json'
        cycle=json.loads(cycle_path.read_bytes());base=np.asarray(cycle['windows'][0]['actual_installed_consumed_EEF10_BASE7'])[10:]
        if base.shape!=(7,)or not np.isfinite(base).all():raise ValueError('ACTUAL_ORIGINAL_BASE_HOLD_REQUIRED')
        with np.load(lp,allow_pickle=False)as z:installed=z['command_installed_eef10'].copy()
        with np.load(labels,allow_pickle=False)as z:states=z['state_eef10'].copy();latents=z['latent7'].copy()
        codec=GripLogitCodec(e['mount']['T_policy_from_mount']);folder=output/e['id'];folder.mkdir()
        for a,(state,label)in enumerate(zip(states,latents,strict=True)):
            normalized=pad32(q.normalize(label)).astype(np.float32)[None]
            fp=folder/(str(a).zfill(2)+'-model-container.npz');atomic_npz(fp,normalized=normalized)
            with np.load(fp,allow_pickle=False)as z:packed=z['normalized'].copy()
            if packed.dtype!=np.float32 or not np.array_equal(packed,normalized):raise ValueError('REAL_SERIALIZED_FP32_CONTAINER')
            physical=codec.decode(state,pad32(q.unnormalize(packed[0,:,:7])))
            action=folder/(str(a).zfill(2)+'-ACTION.npz');atomic_npz(action,normalized=packed,physical_H40=physical)
            actual=np.linalg.inv(frame)@matrix_from10(state);expected=installed[a*10:a*10+40]
            program,proof=compile_proposal(action,sha(action),source_from_target=frame,query_state10=state,
                actual_TCP=actual,consumed_reference_TCP=matrix_from10(expected[0]),base_hold=base)
            err=roundtrip_error(expected,program.reference_path);width=err['gripper']*MECHANICAL_TRAVEL_M
            if(err['xyz_m']>TOLERANCES['xyz_m']or err['so3_rad']>TOLERANCES['so3_rad']
                    or width>EPSILON*(WORKING_OPEN_M-WORKING_CLOSED_M)+1e-12
                    or not np.array_equal(program.base_pose_hold,base)
                    or proof['command_clips']!=0 or proof['endpoint_basis_applied']):
                raise ValueError('FULL_STUDENT_FILE_CODEC_FRAME_REFERENCE_IDENTITY')
            maximum['xyz_m']=max(maximum['xyz_m'],err['xyz_m']);maximum['so3_rad']=max(maximum['so3_rad'],err['so3_rad']);maximum['mechanical_grip_m']=max(maximum['mechanical_grip_m'],width)
            phases[e['expert_stage_audit_only'][a]]+=1;count+=1
        reports.append(dict(episode=e['id'],windows=len(states),original_BASE7_source_sha256=sha(cycle_path)))
    if count!=1220:raise ValueError('ALL1220_WINDOWS_NO_SELECTION_OR_DROPPING')
    result=dict(schema='SLA_ALL1220_FP32_STUDENT_PACKET_COMPILER_ACTUAL_V6_1',passes=True,
        data_manifest_sha256=DATA_SHA,statistics_sha256=STATS_SHA,consumer=CONSUMER,
        windows=count,phases=dict(phases),maximum_installed_reference_error=maximum,episodes=reports,
        actual_file_serialization=True,ordinary_student_compiler=True,actual_recorded_BASE7=True,
        native_certifier_run=False,native_control_history_restored=False,RP8_complete=False,
        model_loads=0,optimizer_updates=0,physical_requests=0,physics_integrations=0,
        wall_s=time.time()-start,CPU_s=time.process_time()-cpu)
    (output/'RESULT.private.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.dataset,a.output)))
