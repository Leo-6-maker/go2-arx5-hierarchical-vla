"""Fixed TRAIN48 x noise2, same C20 model/statistics, zero optimization."""
import hashlib,json,os,sys,time,traceback
from pathlib import Path
from collections import Counter
import numpy as np
from scripts.pi0292.durable import OwnerLock,write
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import read,sha
from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze
from go2arx5_vla.sla_adapter.codec_v1 import pad32,roundtrip_error
from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
from scripts.sla.export_mfree_c20_v1 import _reference
PHASES=('PREGRASP','DESCEND','CLOSE','LIFT','HOLD')

def select_windows(dataset):
    chosen=[]
    for family in ('F1','F2'):
        episodes=[e for e in dataset.manifest['episodes']if e['family']==family]
        for phase in PHASES:
            for position in (0,3,6,9):
                ep=episodes[position];indices=np.flatnonzero(np.asarray(ep['expert_stage_audit_only'])==phase);anchor=int(indices[len(indices)//2]);index=next(i for i,(e,a)in enumerate(dataset.rows)if e['id']==ep['id']and a==anchor)
                chosen.append(dict(index=index,kind='PHASE',family=family,phase=phase,episode=ep['id'],anchor=anchor))
    for family in ('F1','F2'):
        episodes=[e for e in dataset.manifest['episodes']if e['family']==family]
        for phase in PHASES[1:]:
            for ep in episodes:
                anchor=int(np.flatnonzero(np.asarray(ep['expert_stage_audit_only'])==phase)[0]);index=next(i for i,(e,a)in enumerate(dataset.rows)if e['id']==ep['id']and a==anchor)
                if index not in {r['index']for r in chosen}:break
            chosen.append(dict(index=index,kind='STAGE_SEAM_OR_GRIP_EVENT',family=family,phase=phase,episode=ep['id'],anchor=anchor))
    if len(chosen)!=48 or len({r['index']for r in chosen})!=48:raise ValueError('FIXED48_DISTINCT_TRAIN_WINDOWS')
    return chosen

def compare_inputs(dataset,provider,index):
    from go2arx5_vla.sla_adapter.dataset_v1 import images224
    from go2arx5_vla.sla_adapter.model_v1 import observation
    from scripts.sla.train_v1 import exact_trees
    state,_,images,text=dataset.raw(index);ep,_=dataset.rows[index];codec=GripLogitCodec(dataset.frames[ep['id']]['T_policy_from_mount']);dummy=np.broadcast_to(state,(40,10)).copy();dummy[:,9]=.5;s,_=codec.encode(state,dummy)
    provider_item=images224(images,provider.resize);provider_item.update(state=pad32(provider.statistics['state'].normalize(s)).astype(np.float32),actions=np.zeros((40,32),np.float32),prompt=text)
    training_item=dataset[index];a,_=observation(training_item,provider.tokenizer);b,_=observation(provider_item,provider.tokenizer);result=exact_trees(a,b)
    if not result['bit_exact']:raise ValueError('TRAIN_EVAL_AND_ORDINARY_RGB_PREPROCESS_DISAGREE')
    return dict(index=index,**result,checked=['image_RGB_resize_scaling','image_mask','pixel_padding_mask','measured_state7_and_padding25','prompt_tokens_and_state_tokens','dropout_disabled'],loss_target_actions_not_model_input=True)

def parameter_changes(old,package):
    initial=read(old/'candidate1000/work/step0000/COMPLETE.json');end=read(old/'candidate1000/work/step1000/COMPLETE.json');a={r['key']:r for r in initial['arrays']};b={r['key']:r for r in end['arrays']};rows=[]
    for r in read(package/'MANIFEST.private.json')['adapter_arrays']:
        key=r['key'];start=old/'candidate1000/work/step0000'/a[key]['file'];finish=old/'candidate1000/work/step1000'/b[key]['file'];changed=a[key]['sha256']!=b[key]['sha256']
        row=dict(key=key,shape=r['shape'],changed=changed,initial_sha256=a[key]['sha256'],final_sha256=b[key]['sha256'],action_block=any(t in key.lower()for t in ('action','time_mlp','state_proj')))
        if row['action_block']:
            x=np.load(start,mmap_mode='r').reshape(-1);y=np.load(finish,mmap_mode='r').reshape(-1);square=0.;peak=0.
            for offset in range(0,len(x),262144):
                diff=y[offset:offset+262144].astype(np.float64)-x[offset:offset+262144].astype(np.float64);square+=float(diff@diff);peak=max(peak,float(abs(diff).max()))
            row.update(delta_L2=square**.5,maximum_delta=peak)
        rows.append(row)
    logs=[json.loads(x)for x in(old/'candidate1000/work/updates.private.jsonl').read_text().splitlines()]
    return dict(active_parameter_arrays=rows,loss_first= {k:logs[0][k]for k in ('mean_FM32_loss','active7','padding25','grad_norm_before_clip')},loss_final={k:logs[-1][k]for k in ('mean_FM32_loss','active7','padding25','grad_norm_before_clip')},total_FM_not_physical_legality=True,new_optimizer_updates=0)

def run(s):
    import jax
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter.mfree_training_v1 import MobileDataset
    from go2arx5_vla.sla_adapter.provider_mfree_v1 import Provider
    root=Path(s['registered_root']);old=Path(s['C20_root']);train=read(old/'candidate1000/RESULT.private.json');package=Path(train['package']['path']);cache=root/'cache';jax.config.update('jax_compilation_cache_dir',str(cache))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=s['GPU_uuid']or os.environ.get('XLA_FLAGS')!=read(s['numeric_rule']['path'])['XLA_FLAGS']:raise PermissionError('FIXED_SLA_NUMERIC_ENVIRONMENT')
    dataset=MobileDataset(old/'dataset',sha(old/'dataset/MANIFEST.private.json'),resize_with_pad=resize_with_pad,max_cache_bytes=512*1024**2)
    windows=select_windows(dataset);write(root/'WINDOW48_PLAN.private.json',dict(schema='C20_TRAIN48_NOISE2_FIXED_DIAGNOSTIC_V2',windows=windows,noises=[0,1],manifest_sha256=dataset.manifest_sha,statistics_sha256=dataset.manifest['statistics']['sha256'],max_new_generations=96,new_updates=0,phase_is_selection_audit_only=True))
    # Reuse actual old TRAIN evaluation forwards and the independent RGB output
    # hashes, then audit both entry tensors separately. No Adam is loaded.
    saved=read(old/'candidate1000/ENDPOINT_RGB.private.json');fresh=read(old/'freshRGB/REPLAY_RESULT.private.json');fresh_map={(r['index'],r['noise']):r for r in fresh['records']};cached={}
    for r in saved:
        path=old/'candidate1000/work/step1000-TRAIN'/r['file'];f=fresh_map[(r['index'],r['noise'])]
        if any(r[k]!=f[k]for k in ('normalized_sha256','physical_sha256')):raise ValueError('SAVED_REAL_TRAIN_FORWARD_AND_RGB_OUTPUT_DIFFER')
        with np.load(path,allow_pickle=False)as z:
            if hashlib.sha256(z['normalized'].tobytes()).hexdigest()!=r['normalized_sha256']or hashlib.sha256(z['physical_H40'].tobytes()).hexdigest()!=r['physical_sha256']:raise ValueError('CACHED_TRAIN_FORWARD_BYTES_CHANGED')
        cached[(r['index'],r['noise'])]=path
    write(root/'PROGRESS.private.json',dict(phase='LOAD_PINNED_C20_PROVIDER',epoch=time.time(),new_generations=0,new_optimizer_updates=0))
    provider=Provider(package,manifest_sha256=train['package']['sha256'],parent_params=s['raw_parent'],parent_manifest_path=s['parent_manifest']['path'])
    consistency=[dict(noise=r['noise'],forward_outputs_bit_exact=True,output_source='REUSED_REAL_TRAIN_EVAL_AND_FRESH_RGB_FORWARDS',**compare_inputs(dataset,provider,r['index']))for r in saved]
    write(root/'ENTRY16_INPUT_AND_DECODE.private.json',dict(records=consistency,all_bit_exact=True,previous_forward_outputs_reused=True,new_generations=0,optimizer_loaded=False))
    source_rows={};records=[];generations=0
    for w in windows:
        index=w['index'];ep,anchor=dataset.rows[index];state,expert,images,text=dataset.raw(index);frame=dataset.frames[ep['id']]['T_policy_from_mount'];codec=GripLogitCodec(frame)
        if ep['id']not in source_rows:
            m=read(ep['original_live_capture']['path']);source_rows[ep['id']]=read(m['labels']['path'])
        label=source_rows[ep['id']][anchor];reference=_reference(label);spec=read(Path(ep['native_result']['path']).parent/'EXECUTION_SPEC.json');policy_from_world=np.asarray(spec['policy_from_world'])
        _,latent=codec.encode(state,expert);fp32=codec.decode(state,pad32(dataset.statistics['actions'].unnormalize(dataset.statistics['actions'].normalize(latent).astype(np.float32))));label_check=analyze(fp32,state,policy_from_world,reference)
        if not label_check['original_full_H40_pass']:raise ValueError('EXPERT_FP32_SAME_CHAIN_GATE_FAILURE')
        for noise in (0,1):
            if (index,noise)in cached:
                with np.load(cached[(index,noise)],allow_pickle=False)as z:value=dict(normalized=z['normalized'].copy(),physical_H40=z['physical_H40'].copy())
                origin='EXACT_OLD_FORWARD_CACHE'
            else:
                if generations>=96:raise PermissionError('FINITE96_DIAGNOSTIC_GENERATION_BUDGET')
                value=provider.predict(state,images,text,frame,noise_id=noise);generations+=1;origin='NEW_PINNED_C20_ORDINARY_RGB'
            target=root/'data'/f"{index:04d}-{noise}.npz"
            with target.open('xb')as handle:np.savez(handle,**value,expert_policy_H40=expert,state_eef10=state)
            prediction=value['physical_H40'];diag=analyze(prediction,state,policy_from_world,reference)
            records.append(dict(**w,noise=noise,origin=origin,prediction=dict(path=str(target),sha256=sha(target)),expert_same_chain_pass=label_check['original_full_H40_pass'],expert_envelope=label_check,prediction_envelope=diag,errors=dict(path_XYZ_RMS_m=float(np.sqrt(np.mean(np.sum((prediction[:,:3]-expert[:,:3])**2,axis=1)))),endpoint_XYZ_m=float(np.linalg.norm(prediction[-1,:3]-expert[-1,:3])),gripper_MAE=float(np.abs(prediction[:,9]-expert[:,9]).mean()),predicted_first_close_k=next((int(k)for k,g in enumerate(prediction[:,9])if g<.5),None),expert_first_close_k=next((int(k)for k,g in enumerate(expert[:,9])if g<.5),None))))
            write(root/'PROGRESS.private.json',dict(phase='TRAIN48_NOISE2_DIAGNOSTIC',epoch=time.time(),records=len(records),new_generations=generations,new_optimizer_updates=0,new_physical_requests=0))
    write(root/'TRAIN48_PREDICTIONS.private.json',records);changes=parameter_changes(old,package);write(root/'ACTION_PARAMETER_AND_LOSS_AUDIT.private.json',changes)
    step500=read(old/'candidate1000/work/step500-TRAIN/DETAILED.private.json')
    bad=[r for r in records if not r['prediction_envelope']['original_full_H40_pass']]
    result=dict(schema='C20_FIXED48_ZERO_UPDATE_DIAGNOSTIC_V2',epoch=time.time(),unique_TRAIN_windows=48,predictions=len(records),new_model_generations=generations,cached_predictions=len(records)-generations,expert_label_passes=48,expert_state_generated_H40_pass=len(records)-len(bad),expert_state_generated_H40_fail=len(bad),failure_channels=dict(Counter(r['prediction_envelope']['first_temporal_exceedance']['channel']for r in bad)),failure_regions=dict(Counter(r['prediction_envelope']['first_temporal_exceedance']['region']for r in bad)),entry16_bit_exact=True,entry16_existing_forward_outputs_reused=True,step500_original_TRAIN_diagnostic=dict(predictions=len(step500),EEF_full_H40_pass=sum(r['EEF_full_H40_pass']for r in step500)),step1000_original_TRAIN_diagnostic=train['diagnostics'],new_optimizer_updates=0,new_training_visits=0,new_physical_requests=0,new_integrations=0,neural_training_choice='NOT_AUTOMATICALLY_REGISTERED_REQUIRES_MECHANISM_DECISION',F1_physical_isolation_preserved=True)
    write(root/'RESULT.private.json',result);return result

if __name__=='__main__':
 s=read(sys.argv[sys.argv.index('--diagnostic')+1]);root=Path(s['registered_root']);lock=OwnerLock(s['neural_owner_lock'])
 try:run(s);code=0
 except BaseException:write(root/'RESULT.private.json',dict(exit_code=1,failure=traceback.format_exc(),new_optimizer_updates=0,new_physical_requests=0));code=1
 finally:lock.close()
 from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
 close(s,code);raise SystemExit(code)
