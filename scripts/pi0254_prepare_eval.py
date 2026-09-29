"""Seal actual CPU codec, inference conditions and evaluation noise before queries."""
from pathlib import Path
import hashlib
import json
import sys

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT=BASE/'PI0254'
sys.path[:0]=[str(ROOT/'execution'),str(BASE/'PI0253/runtime')]
import pi0253_full_screen_v2 as inherited


def noise_seed(source, family):
    key=f'PI0254_EVAL_{family}|{25410 if family=="B0" else 25411}|source={source}'
    return key,int.from_bytes(hashlib.sha256(key.encode('ascii')).digest()[:8],'big')&((1<<63)-1)


def main():
    import numpy as np
    import torch
    import pi0242_train_infer_bridge as bridge
    import pi0246_paired_screen as paired
    import pi0243_staged_learning_to_sim as stage43
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat,rot6d_rows_to_matrix,geodesic_angle
    old,_,_,_,_,ctx=inherited.context()
    processor,embodiment=ctx['processor'],ctx['embodiment']
    manifest=inherited.read(ROOT/'source640/manifest.json')
    assert manifest['status']=='PASS_ORIGINAL_FILES_LABEL_STATE_TIME_BINDING' and len(manifest['files'])==640
    roles={name:inherited.read(ROOT/f'design/{name}.json')['rows'] for name in ('O64','P64','E64')}
    eval_sources={r['source_episode_id'] for name in ('P64','E64') for r in roles[name]}
    assert len(eval_sources)==128
    records=[];noises=[];gate=TrajectorySafetyGateV1()
    for number,row in enumerate(manifest['files'],1):
        source=row['source_episode_id'];path=ROOT/'source640'/row['path']
        assert inherited.sha(path)==row['sha256']
        with np.load(path,allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
        state,expert=arrays['state'],arrays['expert']
        encoded,error=ctx['p1'].encode_expert(processor,embodiment,expert,state)
        assert encoded.shape==(40,10) and np.isfinite(encoded).all() and error<=1e-5
        raw=np.zeros((1,40,132),np.float32);raw[0,:,:10]=encoded
        decoded=bridge._decode_physical(processor,embodiment,raw,state)[0]
        xyz=float(np.linalg.norm(decoded[:,:3]-expert[:,:3],axis=1).max())
        angle=float(geodesic_angle(rot6d_rows_to_matrix(decoded[:,3:9]),rot6d_rows_to_matrix(expert[:,3:9])).max())
        assert xyz<=1e-5 and angle<=1e-5
        achieved_quat=matrix_to_quat(rot6d_rows_to_matrix(state[None,3:9]))[0]
        expert_gate=gate.evaluate(selector._build_chunk(expert[None],chunk_id=f'pi0254-expert-{row["episode_index"]}'),
                                 now=0.,achieved_position=state[:3],achieved_orientation=achieved_quat)
        record=dict(source_episode_id=source,episode_index=row['episode_index'],source_sha256=row['sha256'],
                    codec_max_abs=error,codec_max_xyz_m=xyz,codec_max_so3_rad=angle,expert_gate=expert_gate.as_dict())
        if source in eval_sources:
            parent=ROOT/'conditions'/f'source-{row["episode_index"]:06d}'
            parent.mkdir(parents=True,exist_ok=True)
            mask=np.zeros((40,132),np.float64);mask[:,:10]=1
            source_path=parent/'tensors.npz'
            if not source_path.exists():ctx['p1'].durable_npz(source_path,**arrays,action_mask=mask)
            else:
                with np.load(source_path,allow_pickle=False) as previous:
                    assert all(np.array_equal(previous[k],v) for k,v in {**arrays,'action_mask':mask}.items())
            policy,identity,*_=paired.policy_for(dict(resolved_tensor_path=str(source_path)),processor,embodiment,bridge,stage43)
            condition_path=parent/'condition.pt'
            if not condition_path.exists():ctx['p1'].durable_torch(condition_path,policy)
            else:assert bridge._canonical_hash(bridge._leaf_inventory(torch.load(condition_path,weights_only=False)))==identity
            record.update(condition_identity=identity,condition_sha256=inherited.sha(condition_path),source_tensor_sha256=inherited.sha(source_path))
            inherited.write(parent/'record.json',record)
            for family in ('B0','B1'):
                key,seed=noise_seed(source,family)
                value=torch.randn((1,40,132),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)
                bf16=value.to(torch.bfloat16)
                packet=ROOT/'noise'/f'{row["episode_index"]:06d}_{family}.npz';packet.parent.mkdir(exist_ok=True)
                f32=value.numpy();bits=bf16.view(torch.uint16).numpy()
                if not packet.exists():ctx['p1'].durable_npz(packet,float32=f32,bfloat16_bits=bits)
                else:
                    with np.load(packet,allow_pickle=False) as previous:
                        assert np.array_equal(previous['float32'],f32) and np.array_equal(previous['bfloat16_bits'],bits)
                noises.append(dict(source_episode_id=source,noise_name=family,key=key,seed=seed,path=str(packet),
                                   file_sha256=inherited.sha(packet),float32_sha256=hashlib.sha256(f32.tobytes()).hexdigest(),
                                   bfloat16_sha256=hashlib.sha256(bits.tobytes()).hexdigest(),shape=list(value.shape),
                                   generator_device='cpu',generator_dtype='torch.float32',injected_dtype='torch.bfloat16'))
        records.append(record)
        if number%64==0:print(json.dumps(dict(status='CPU_CODEC_AND_GATE',count=number)),flush=True)
    inherited.write(ROOT/'cpu_codec_conditions.json',dict(schema='PI0254_CODEC_CONDITIONS_V1',status='PASS_CPU_RAW_CODEC_CONDITIONS',
                        source_manifest_sha256=inherited.sha(ROOT/'source640/manifest.json'),rows=records,
                        expert_gate_filter=False,new_model_queries=0,new_optimizer_updates=0,
                        torch_version=torch.__version__,processor_config_sha256=inherited.sha(inherited.INPUTS/'processor_base/processor_config.json')))
    inherited.write(ROOT/'noise/manifest.json',dict(schema='PI0254_ACTUAL_EVAL_NOISE_V1',rows=noises))
    print(json.dumps(dict(status='PASS_PREQUERY_INPUT_AND_NOISE_MATERIALIZATION',codec_windows=len(records),
                         condition_windows=len(eval_sources),actual_noise_tensors=len(noises),new_model_queries=0)),flush=True)


if __name__=='__main__':main()
