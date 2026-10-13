"""Fixed two-noise native offline evaluation; model execution stays in own PBS."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from .durable import OwnerLock,read,write
from .train_adapter import NativeDataset,ProcessedCollator,official_import,sha
from .eval_contract import noise_seed,gate_action,error_metrics,summarize_records
from .neural_train import verify_allocation,verify_inputs,COSMOS_REVISION


def run(args):
    root=Path(args.root);lock=OwnerLock(root/'NEURAL_OWNER.lock')
    allocation=verify_allocation(root,'offline');bindings=read(args.bindings)
    verify_inputs(root,bindings,'offline');official_import(bindings['source'])
    probe=read(root/'evidence/OPEN_LOOP_PROBE_SEAL.json')
    if probe['status']!='SEALED_BEFORE_MAIN':raise PermissionError('EXPERT_POSITIVE_PROBE_GATE_REQUIRED')
    for name,digest in probe['evaluation_script_hashes'].items():
        if sha(Path(__file__).parent/name)!=digest:raise ValueError('EVALUATION_SCRIPT_CHANGED_AFTER_SEAL')
    if probe['statistics_sha256']!=sha(Path(bindings['dataset'])/'statistics.json'):raise ValueError('PROBE_STATS_CHANGED')
    output=root/'offline'/args.profile;output.mkdir(parents=True,exist_ok=False)
    import torch
    from transformers import AutoModel,set_seed
    from gr00t.data.embodiment_tags import EmbodimentTag
    import gr00t.model.gr00t_n1d7.gr00t_n1d7
    from .trainer_adapter import tensor_identity
    torch.set_num_threads(4)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    path=Path(bindings['base']) if args.profile=='B0' else root/'training/main'/('checkpoint-'+args.profile[1:])
    if args.profile!='B0':
        checkpoint=read(path/'CHECKPOINT_COMPLETE.json')
        if checkpoint['global_step']!=int(args.profile[1:]):raise ValueError('FIXED_CHECKPOINT_STEP_REQUIRED')
        for name,digest in checkpoint['required_files'].items():
            if sha(path/name)!=digest:raise ValueError('CHECKPOINT_CHANGED')
    weights=sorted(path.glob('*.safetensors'))
    if not weights:raise ValueError('COMPLETE_PROFILE_WEIGHTS_REQUIRED')
    profile_identity={p.name:sha(p) for p in weights}
    ledger=read(root/'HOPPER_RESOURCE_LEDGER.json');ledger['model_loads']+=1;write(root/'HOPPER_RESOURCE_LEDGER.json',ledger)
    set_seed(29200)
    model,loading=AutoModel.from_pretrained(path,local_files_only=True,trust_remote_code=True,output_loading_info=True,
        transformers_loading_kwargs=dict(local_files_only=True,trust_remote_code=True,cache_dir=bindings['cosmos_hub'],revision=COSMOS_REVISION),
        tune_llm=False,tune_visual=False,tune_projector=True,tune_diffusion_model=True,tune_vlln=True,
        state_dropout_prob=0.,backbone_trainable_params_fp32=True,load_bf16=False)
    missing=loading.get('missing_keys',[])
    if any('mask_token' not in k for k in missing) or loading.get('unexpected_keys') or loading.get('mismatched_keys'):
        raise ValueError('OFFLINE_PROFILE_LOADING_MISMATCH')
    if any('mask_token' in k for k in missing) and model.action_head.mask_token is not None:
        with torch.no_grad():model.action_head.mask_token.copy_(.02*torch.randn_like(model.action_head.mask_token))
    model.to('cuda').eval()
    if model.action_head.num_inference_timesteps!=4:raise ValueError('NATIVE_FOUR_STEPS_REQUIRED')
    raw_bindings=read(bindings['raw_bindings'])
    datasets={split:NativeDataset(bindings['dataset'],bindings['source'],bindings['cosmos'],split,raw_bindings=raw_bindings) for split in ('TRAIN','DEV')}
    indices={split:{(row['id'],anchor):i for i,(row,anchor) in enumerate(dataset.rows)} for split,dataset in datasets.items()}
    state=dict(status='RUNNING',profile=args.profile,pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],
        allocation=allocation,started=time.time(),queries=0,planned=probe['per_profile_queries'],records=[])
    write(output/'STATE.json',state)
    captured=[]
    def initial_noise(module,inputs):
        if not captured:captured.append(inputs[0].detach().clone())
    hook=model.action_head.action_encoder.register_forward_pre_hook(initial_noise)
    try:
        for window in probe['probes']:
            dataset=datasets[window['split']];index=indices[window['split']][window['id'],window['anchor']]
            actual,expert,images,text=dataset.raw(index)
            item=dataset[index];batch=ProcessedCollator(dataset.processor)([item])
            del batch['inputs']['action'];del batch['inputs']['action_mask']
            if any(name in batch['inputs'] for name in ('object_pose','goal','teacher_stage','action','labels')):
                raise ValueError('PRIVILEGED_OR_FUTURE_INPUT')
            batch={k:{n:v.to('cuda') if torch.is_tensor(v) else v for n,v in values.items()} for k,values in batch.items()}
            for bank in probe['noise_banks']:
                if time.time()>=min(read(root/'ADOPTION.json')['new_work_cutoff_epoch'],allocation['allocation_end_epoch']-60):
                    raise TimeoutError('OFFLINE_GENERATION_CUTOFF')
                identifier=f"{window['id']}-{window['anchor']:03d}-{bank}"
                record={**window,'noise_bank':bank,'profile':args.profile,'query_id':identifier}
                ledger=read(root/'HOPPER_RESOURCE_LEDGER.json')
                if ledger.get('offline_queries',0)>=1152 or ledger['generation_queries']>=2624 or state['queries']>=384:
                    raise PermissionError('ACTION_GENERATION_BUDGET')
                ledger['offline_queries']=ledger.get('offline_queries',0)+1;ledger['generation_queries']+=1
                write(root/'HOPPER_RESOURCE_LEDGER.json',ledger)
                state['queries']+=1;state['pending']=identifier;write(output/'STATE.json',state)
                seed=noise_seed(bank,window['id'],window['anchor'])
                if str(seed)!=window['noise_seeds'][str(bank)]:raise ValueError('PAIRED_NOISE_SEED_CHANGED')
                set_seed(seed);captured.clear()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    predicted=model.get_action(**batch)['action_pred']
                if len(captured)!=1:raise ValueError('NATIVE_INITIAL_NOISE_NOT_OBSERVED')
                normalized=predicted.detach().float().cpu().numpy()
                record.update(noise=tensor_identity(captured[0]),normalized_outside_unit_components=int(np.count_nonzero(np.abs(normalized[...,:10])>1)),
                    native_inverse_clip=True,extra_correction=False)
                try:
                    decoded=dataset.processor.decode_action(normalized,EmbodimentTag.NEW_EMBODIMENT,
                        {'eef_9d':actual[None,None,:9],'gripper_position':actual[None,None,9:]})
                    physical=np.concatenate([decoded[k] for k in ('eef_9d','gripper_position')],axis=-1)[0]
                    record.update(gate=gate_action(physical,actual,window['anchor']/20,identifier),errors=error_metrics(physical,expert,actual))
                except (ValueError,AssertionError,FloatingPointError) as error:
                    physical=np.full((40,10),np.nan)
                    record.update(gate=dict(decision='REJECT',issues=[dict(code='NATIVE_DECODE_FAILURE',severity='REJECT')]),
                        decode_error=type(error).__name__+': '+str(error),errors=error_metrics(physical,expert,actual))
                np.savez_compressed(output/(identifier+'.npz'),normalized_action=normalized,decoded_action=physical,
                    expert=expert,actual_state=actual,noise_bits=captured[0].contiguous().view(torch.uint8).cpu().numpy())
                record['array_sha256']=sha(output/(identifier+'.npz'))
                write(output/(identifier+'.json'),record)
                state['records'].append(record);state.pop('pending',None);write(output/'STATE.json',state)
        if state['queries']!=probe['per_profile_queries']:raise ValueError('INCOMPLETE_FIXED_PROFILE')
        summary=summarize_records(state['records'])
        write(output/'SUMMARY.json',dict(status='OFFLINE_PROFILE_EVALUATED_NOT_TASK_SUCCESS',profile=args.profile,
            profile_weights=profile_identity,loading=loading,probe_sha256=sha(root/'evidence/OPEN_LOOP_PROBE_SEAL.json'),
            queries=state['queries'],summary=summary,rejected_errors_retained=True,selected='NONE',deployment='NONE'))
        state['status']='OFFLINE_PROFILE_EVALUATED_NOT_TASK_SUCCESS'
    except Exception as error:
        state.update(status='FAILED_PRESERVED_NO_RESAMPLING',error_type=type(error).__name__,error=str(error));raise
    finally:
        hook.remove();usage=resource.getrusage(resource.RUSAGE_SELF)
        state.update(finished=time.time(),CPU_s=usage.ru_utime+usage.ru_stime);write(output/'STATE.json',state);lock.close()
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--bindings',required=True)
    parser.add_argument('--profile',choices=['B0','S2048','S4096'],required=True)
    result=run(parser.parse_args());print(json.dumps(dict(status=result['status'],queries=result['queries'])))
