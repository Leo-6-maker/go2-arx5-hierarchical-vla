"""Independent TRAIN100 frozen-feature adaptation and numerical attribution.

The original cross-site gate and FIT4 worker are deliberately not entry points.
All model calls here are offline; this module grants no physical execution.
"""
import copy
import json
import time
from pathlib import Path

from . import task_learning_features_v1 as features
from .storage import digest

SCOPE = 'TASK_LEARNING_CORE_LOCAL_TRAIN100_LAST2_V1'
MEMORY = 'TRAIN100_SFT2000_FROZEN_FEATURE_LAST2_V1'


def setup(s, root, *, ordinary=False):
    from scripts.pi0294.numerics import prepare_environment, configure_torch, MODE
    from scripts.pi0294.native_dataset import official_import
    prepare_environment(MODE); official_import(s['bindings']['source'])
    import gr00t.model.gr00t_n1d7.gr00t_n1d7  # register the pinned architecture
    import torch
    from transformers import set_seed
    from .target_gr00t_training_v1 import _load_original_parent, action_contract, gpu_identity
    from .target_gr00t_delivery_v1 import install
    from .target_gr00t_core_v1 import trim_unused_logits
    from .task_learning_native_dataset_v1 import TaskLearningNativeDataset
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from scripts.pi0294.local_backbone import local_model_name
    from scripts.pi0294.neural import COSMOS_REVISION
    configure_torch(torch, MODE); torch.set_num_threads(4); set_seed(s['seed'])
    gpu_identity(torch, dict(s, execution_site='CORELAB_4080S'))
    (root/'model_inputs/nvidia').mkdir(parents=True)
    (root/'model_inputs/nvidia/Cosmos-Reason2-2B').symlink_to(s['bindings']['cosmos'], target_is_directory=True)
    model = _load_original_parent(s, root, 'MODEL_LOAD')
    policy_arm=s.get('policy_arm','B1')
    features.need(policy_arm in ('B0','B1'),'DECLARED_LOCAL_POLICY_ARM')
    if policy_arm=='B1':install(model,s['package']['path'],s['package']['sha256'],s['parent']['files'])
    else:model.requires_grad_(False);model.eval()
    contract = None if ordinary else action_contract(model,'INTERFACES_LAST2_DIT',profile_name='TASK_LEARNING_GR00T_V1')
    if contract is not None:
        features.need(contract.describe()['trainable_numel']==77847684,'EXACT_LAST2_SCOPE')
    trim_unused_logits(model); model.eval()
    data=TaskLearningNativeDataset(s['bindings']['dataset'],s['bindings']['source'],s['bindings']['cosmos'],
        for_optimization=False,optimization_site='CORELAB_4080S')
    processor_root=Path(s['package']['path'])if policy_arm=='B1'else Path(s['parent']['path'])
    data.processor=Gr00tN1d7Processor.from_pretrained(processor_root/'processor',
        model_name=local_model_name(root,s['bindings']),local_files_only=True,
        transformers_loading_kwargs=dict(local_files_only=True,trust_remote_code=True,
            cache_dir=s['bindings']['cosmos_hub'],revision=COSMOS_REVISION))
    data.processor.eval()
    features.need(data.manifest_sha==s['dataset_sha256'] and data.statistics_sha==s['statistics_sha256']
                  and len(data)==1450,'SAME_UNIQUE_TRAIN100_DATA_AND_STATISTICS')
    return model,contract,data


def saved_features(manifest, query):
    """Recover the recorded output tensor dtype, including BF16 stored as FP32."""
    import torch
    import numpy as np
    path=Path(manifest['path'])
    features.need(digest(path)==manifest['sha256'],'PINNED_PAIRED_FEATURE_MANIFEST')
    value=json.loads(path.read_bytes())
    row=next(r for r in value['records'] if r['query_index']==query and r['module']=='backbone'
             and r['stage']=='output' and r['call']==0)
    file=path.parent/row['file']; features.need(digest(file)==row['sha256'],'PAIRED_FEATURE_MEMBER')
    result={}
    with np.load(file,allow_pickle=False) as packet:
        for item in row['structure']:
            features.need(len(item['path'])==1 and 'key'in item,'KNOWN_BACKBONE_FEATURE_PACKET')
            tensor=torch.from_numpy(packet[item['key']].copy()).to(getattr(torch,item['dtype'].split('.')[-1]))
            features.need(list(tensor.shape)==item['shape'],'ORIGINAL_FEATURE_SHAPE')
            result[item['path'][0]]=tensor
    return result


def feature_exchange(s, model, data, root, guard):
    import torch
    import numpy as np
    from scripts.pi0292.durable import write
    from scripts.pi0294.native_train import tensor_identity, tree_identity
    from gr00t.data.embodiment_tags import EmbodimentTag
    output=root/'feature-exchange'; output.mkdir()
    records=json.loads(Path(s['Core40_result']['path']).read_bytes())['records'][:4]
    indexes={(e['id'],a):i for i,(e,a) in enumerate(data.rows)}
    model.backbone.to('cpu'); model.action_head.to('cuda'); results=[]
    for q,record in enumerate(records):
        guard(); index=indexes[record['episode'],record['anchor']]
        inputs=data.processor.collator([data[index]['processed']])['inputs']
        state,_,_,_=data.raw(index); outputs={}; noises={}
        for origin in ('Core','H100'):
            row=dict(inputs=features.tensor_tree(inputs,'cpu'),features=saved_features(s['paired_features'][origin],q))
            for repeat in range(2):
                noise=[]
                def capture(module,args):
                    if not noise: noise.append(tensor_identity(args[0]))
                hook=model.action_head.action_encoder.register_forward_pre_hook(capture)
                try:
                    with torch.random.fork_rng(devices=[0]),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        torch.manual_seed(record['seed'])
                        value=model.action_head.get_action(features.feature_packet(row),
                            features.action_packet(model,row,model.dtype,generation=True))['action_pred']
                    normalized=value.detach().float().cpu().numpy()
                finally: hook.remove()
                decoded=data.processor.decode_action(normalized,EmbodimentTag.NEW_EMBODIMENT,
                    {'eef_9d':state[None,None,:9],'gripper_position':state[None,None,9:]})
                physical=np.concatenate([decoded[k]for k in('eef_9d','gripper_position')],axis=-1)[0]
                file=output/f'q{q:02d}-{origin}-r{repeat}.npz';np.savez(file,normalized=normalized,physical_H40=physical)
                outputs[origin,repeat]=(normalized,physical);noises[origin,repeat]=noise[0]
        with np.load(Path(s['Core40_samples'])/record['sample_file'],allow_pickle=False) as baseline:
            baseline_exact=all(np.array_equal(outputs['Core',0][i],baseline[k])for i,k in enumerate(('normalized','physical_H40')))
        features.need(len(set(json.dumps(x,sort_keys=True)for x in noises.values()))==1
                      and noises['Core',0]==record['noise'],'EXACT_MATCHED_GENERATION_NOISE')
        results.append(dict(query_index=q,episode=record['episode'],anchor=record['anchor'],seed=record['seed'],
            input_identity=tree_identity(inputs),same_state_masks_generation_steps=True,
            Core_baseline_bit_exact=baseline_exact,
            within_origin_repeat_bit_exact={o:all(np.array_equal(outputs[o,0][i],outputs[o,1][i])for i in range(2))for o in ('Core','H100')},
            swapped_feature_normalized_max_abs=float(np.max(np.abs(outputs['Core',0][0].astype(float)-outputs['H100',0][0].astype(float)))),
            swapped_feature_physical_component_max_abs=np.max(np.abs(outputs['Core',0][1].astype(float)-outputs['H100',0][1].astype(float)),axis=0).tolist()))
    result=dict(schema='TASK_LEARNING_CORE_SAME_HEAD_FEATURE_EXCHANGE_V1',records=results,
        original_windows=4,feature_sources=2,repetitions=2,actual_action_head_generations=16,
        backbone_generations=0,optimizer_updates=0,physical_integrations=0,
        all_Core_baselines_bit_exact=all(r['Core_baseline_bit_exact']for r in results),
        same_action_head_weights=True,Processor_fixed=True,precision='ORIGINAL_BF16_AUTOCAST_FP32_PARAMETERS',
        denoising_steps=4,physical_execution_authorized=False,original_cross_hardware_reject_replaced=False)
    write(output/'RESULT.private.json',result);return result


def prepare_cache(s, model, contract, data, root, guard):
    import torch
    from scripts.pi0292.durable import write
    from scripts.pi0294.native_train import tree_identity
    output=root/'TRAIN100-features';output.mkdir(); rows=[]
    indexes={(r['episode'],r['anchor']):r for r in json.loads(Path(s['Core40_result']['path']).read_bytes())['records']}
    model.to('cuda').eval(); proofs=[]
    for index,(episode,anchor) in enumerate(data.rows):
        guard();inputs=data.processor.collator([data[index]['processed']])['inputs']
        row=dict(identity=data.identity(index),inputs=features.tensor_tree(inputs,'cpu'),input_identity=tree_identity(inputs))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            backbone_input,_=model.prepare_input(copy.deepcopy(inputs))
            row['features']=features.tensor_tree(model.backbone(backbone_input),'cpu')
        probe=indexes.get((episode['id'],anchor))
        if probe is not None:
            with torch.random.fork_rng(devices=[0]),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                torch.manual_seed(probe['seed']);original=model.get_action(features.inference_inputs(inputs))['action_pred'].detach().cpu()
                torch.manual_seed(probe['seed']);cached=model.action_head.get_action(features.feature_packet(row),features.action_packet(model,row,model.dtype,generation=True))['action_pred'].detach().cpu()
            features.need(torch.equal(original,cached),'SFT2000_TRAIN100_FULL_H40_CACHE_EXACT')
            proofs.append(dict(index=index,episode=episode['id'],anchor=anchor,full_H40_exact=True))
        file=output/f'row-{index:04d}.pt';features.atomic_torch(file,row)
        rows.append(dict(index=index,episode=episode['id'],anchor=anchor,phase_audit_only=episode['expert_stage_audit_only'][anchor],file=file.name,sha256=digest(file)))
        write(output/'PROGRESS.private.json',dict(prepared=len(rows),requested=1450,original_generations=len(proofs),cached_generations=len(proofs)))
    features.need(len(rows)==1450 and len(proofs)==40,'COMPLETE_UNIQUE_TRAIN100_CACHE')
    result=dict(schema=MEMORY,dataset_sha256=s['dataset_sha256'],statistics_sha256=s['statistics_sha256'],
        initial_package=s['package'],rows=rows,parameter_contract=contract.describe(),
        cache_generation_proofs=proofs,backbone_feature_calls=1450,full_original_generations=40,cached_generations=40,
        optimizer_updates=0,old_FIT4_identity_reused=False,formal_FULL_ACTION_SIDE_continuation=False)
    write(output/'CACHE_MANIFEST.private.json',result);model.backbone.to('cpu');torch.cuda.empty_cache()
    return result


def sampling_tables(cache, manifest, events):
    """Keep the original3200 exposure entries, changing only25% event-arm draws."""
    lookup={(r['episode'],r['anchor']):r['index']for r in cache['rows']}
    original=[lookup[e['id'],a]for e in manifest['episodes']if e['split']=='TRAIN'
              for a in e['optimization_anchors']]
    event=[lookup[r['episode'],r['anchor']]for r in events['windows']if r['actual_contact_onset']]
    features.need(len(lookup)==1450 and len(original)==3200 and set(original)==set(range(1450))
                  and bool(event) and len(event)==len(set(event)),'COMPLETE_UNCHANGED_BASE_AND_REAL_EVENT_POPULATIONS')
    return original,event


def sample_at(cursor, arm, original, event, seed):
    features.need(arm in ('ORIGINAL','CONTACT_EVENTS'),'DECLARED_SAMPLING_ARM')
    if arm=='CONTACT_EVENTS' and cursor%4==3:
        return event[features.sample_index(cursor//4,len(event),seed+1000000)]
    offset=cursor if arm=='ORIGINAL' else cursor-cursor//4
    return original[features.sample_index(offset,len(original),seed)]


def save_state(path, model, contract, optimizer, scheduler, step, recipe, cache_pin, frozen):
    from scripts.pi0292.durable import write
    from scripts.pi0294.native_train import tree_identity,rng_identity
    path=Path(path);path.mkdir()
    features.atomic_torch(path/'target_parameters.pt',{n:model.get_parameter(n).detach().cpu()for n in contract.trainable_names})
    features.atomic_torch(path/'optimizer.pt',optimizer.state_dict())
    features.atomic_torch(path/'scheduler.pt',scheduler.state_dict())
    features.atomic_torch(path/'rng.pt',features.rng_state())
    write(path/'STATE_IDENTITY.json',dict(parameters=features.state_identity(model,contract.trainable_names),
        optimizer=tree_identity(optimizer.state_dict()),scheduler=scheduler.state_dict(),RNG=rng_identity(),cursor=step*32))
    write(path/'FROZEN_STARTING_TENSORS.json',frozen)
    write(path/'CHECKPOINT_COMPLETE.json',dict(schema=MEMORY,adaptation_step=step,parent_global_step=2000,
        cursor=step*32,recipe=recipe,feature_cache=cache_pin,parameter_contract=contract.describe(),
        files={p.name:digest(p)for p in path.iterdir()if p.is_file()},scaler='NOT_APPLICABLE_BF16',
        formal_FULL_ACTION_SIDE_continuation=False,old_FIT4_identity_reused=False))


def restore_state(path, model, contract, optimizer, scheduler, recipe, cache_pin, frozen):
    import torch
    from scripts.pi0294.native_train import tree_identity,rng_identity
    path=Path(path);saved=json.loads((path/'CHECKPOINT_COMPLETE.json').read_bytes())
    features.need(saved['schema']==MEMORY and saved['recipe']==recipe and saved['feature_cache']==cache_pin
                  and saved['parameter_contract']==contract.describe(),'SAME_NEW_ADAPTATION_IDENTITY')
    features.need(all(digest(path/n)==pin for n,pin in saved['files'].items()),'COMPLETE_CHECKPOINT_MEMBERS')
    features.need(json.loads((path/'FROZEN_STARTING_TENSORS.json').read_bytes())==frozen,'SAME_FROZEN_SFT2000_WEIGHTS')
    values=torch.load(path/'target_parameters.pt',map_location='cpu',weights_only=True)
    features.need(set(values)==set(contract.trainable_names),'ALL_LAST2_PARAMETERS')
    with torch.no_grad():
        for name,value in values.items():
            target=model.get_parameter(name)
            features.need(value.shape==target.shape and value.dtype==target.dtype,'UNCHANGED_PARAMETER_LAYOUT')
            target.copy_(value)
    optimizer.load_state_dict(torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False))
    scheduler.load_state_dict(torch.load(path/'scheduler.pt',map_location='cpu',weights_only=False))
    features.restore_rng(torch.load(path/'rng.pt',map_location='cpu',weights_only=False))
    actual=dict(parameters=features.state_identity(model,contract.trainable_names),optimizer=tree_identity(optimizer.state_dict()),
        scheduler=scheduler.state_dict(),RNG=rng_identity(),cursor=saved['cursor'])
    features.need(actual==json.loads((path/'STATE_IDENTITY.json').read_bytes()),'ACTUAL_FRESH_PROCESS_FULL_STATE_EXACT')
    return saved['adaptation_step']


def train(s, model, contract, data, root, guard):
    import torch
    from collections import OrderedDict
    from transformers import Trainer,set_seed
    from scripts.pi0292.durable import write
    from scripts.pi0294.native_train import NativeNoiseAudit
    from scripts.pi0294.generalization.minimal_adapter import frozen_fingerprint
    from scripts.pi0294.e1.trainer import learning_rate_multiplier
    from scripts.pi0294.multitask.activation_offload import SavedActivationOffload,context
    from .target_gr00t_core_v1 import activation_policy
    cp=Path(s['feature_cache']['path']);features.need(digest(cp)==s['feature_cache']['sha256'],'PINNED_NEW_TRAIN100_CACHE')
    cache=json.loads(cp.read_bytes());events_path=Path(s['contact_events']['path'])
    features.need(digest(events_path)==s['contact_events']['sha256'],'PINNED_ACTUAL_CONTACT_AUDIT')
    events=json.loads(events_path.read_bytes())
    features.need(cache['dataset_sha256']==events['dataset']['sha256']==s['dataset_sha256']
                  and cache['initial_package']==s['package'],'SAME_TRAIN100_AND_SFT2000_START')
    original,event=sampling_tables(cache,data.manifest,events)
    model.backbone.to('cpu');model.action_head.to('cuda');model.eval();frozen=frozen_fingerprint(model)
    decay=set(Trainer.get_decay_parameter_names(None,model));recipe=s['recipe']
    groups=[dict(params=[model.get_parameter(n)for n in contract.trainable_names if(n in decay)==use],
                 weight_decay=recipe['weight_decay']if use else 0.)for use in(True,False)]
    optimizer=torch.optim.AdamW(groups,lr=recipe['learning_rate'],betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:learning_rate_multiplier(step,8000,200))
    set_seed(recipe['seed']);step=restore_state(s['resume'],model,contract,optimizer,scheduler,recipe,s['feature_cache'],frozen)if s.get('resume')else 0
    start=step;audit=NativeNoiseAudit(model);storage=SavedActivationOffload(model,policy=activation_policy());lru=OrderedDict()
    write(root/'RESTORE_ACTUAL.private.json',dict(restored=s.get('resume')is not None,adaptation_step=step,all_states_exact=s.get('resume')is not None))
    def row_at(index):
        if index in lru:row=lru.pop(index);lru[index]=row;return row
        pin=cache['rows'][index];file=cp.parent/pin['file'];features.need(digest(file)==pin['sha256'],'UNCHANGED_FROZEN_FEATURE_ROW')
        row=torch.load(file,map_location='cpu',weights_only=False);lru[index]=row
        if len(lru)>8:lru.popitem(last=False)
        return row
    try:
        while step<s['endpoint']:
            guard();started=time.monotonic();optimizer.zero_grad(set_to_none=True);losses=[]
            for offset in range(32):
                cursor=step*32+offset;index=sample_at(cursor,s['arm'],original,event,recipe['seed']);row=row_at(index)
                with context(storage),torch.autocast('cuda',dtype=torch.bfloat16):
                    value=model.action_head(features.feature_packet(row),features.action_packet(model,row,model.dtype))
                    loss=features.check_loss(value);(loss/32).backward()
                features.need(audit.verified and not audit.active,'ORIGINAL_NOISE_AND_MASKED_LOSS_AUDIT')
                losses.append(float(loss.detach()))
                with(root/'SAMPLE_TRACE.jsonl').open('a')as stream:
                    stream.write(json.dumps(dict(cursor=cursor,index=index,identity=row['identity'],native_noise=audit.record))+'\n')
                del loss,value
            features.need(all(p.grad is None or bool(torch.isfinite(p.grad).all())for p in contract.parameters()),'FINITE_LAST2_GRADIENTS')
            torch.nn.utils.clip_grad_norm_(contract.parameters(),1.)
            used_lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step();step+=1
            progress=dict(adaptation_step=step,parent_global_step=2000,cursor=step*32,actual_optimizer_calls=step-start,
                nonzero_LR=used_lr>0,learning_rate_used=used_lr,native_loss=sum(losses)/32,wall_s=time.monotonic()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            with(root/'OPTIMIZER_STEPS.jsonl').open('a')as stream:stream.write(json.dumps(progress)+'\n')
            write(root/'PROGRESS.private.json',progress)
            if step==s['endpoint']:save_state(root/f'checkpoint-{step}',model,contract,optimizer,scheduler,step,recipe,s['feature_cache'],frozen)
        features.need(frozen_fingerprint(model)==frozen,'FROZEN_SFT2000_TENSORS_UNCHANGED')
        if step==500:
            measure(s,model,data,root/'after-500',guard,cache_manifest=cache)
            if s.get('restore_compare'):
                compare_saved_generations(s['restore_compare'],root/'after-500',root/'RESTORED_PREDICTIONS_ACTUAL.private.json')
        return dict(actual_optimizer_calls=step-start,adaptation_step=step,cursor=step*32,
            checkpoint=str(Path(s['resume'])if step==start and s.get('resume')else root/f'checkpoint-{step}'),
            arm=s['arm'],distinct_training_windows=1450,
            formal_FULL_ACTION_SIDE_continuation=False)
    finally:audit.close()


def measure(s,model,data,output,guard,*,cache_manifest=None):
    import torch
    import numpy as np
    from gr00t.data.embodiment_tags import EmbodimentTag
    from scripts.pi0292.durable import write
    from scripts.pi0292.eval_contract import error_metrics
    from scripts.pi0294.native_train import rng_identity
    from .task_learning_local_analysis_v1 import geometry
    output=Path(output);output.mkdir();state=features.rng_state();before=rng_identity();rows=[]
    records=json.loads(Path(s['Core40_result']['path']).read_bytes())['records']
    lookup={(e['id'],a):i for i,(e,a)in enumerate(data.rows)}
    cached={(r['episode'],r['anchor']):r for r in cache_manifest['rows']}if cache_manifest else None
    try:
        for record in records:
            guard();index=lookup[record['episode'],record['anchor']];actual,expert,_,_=data.raw(index)
            with torch.random.fork_rng(devices=[0]),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                torch.manual_seed(record['seed'])
                if cached is None:
                    inputs=data.processor.collator([data[index]['processed']])['inputs']
                    prediction=model.get_action(features.inference_inputs(inputs))['action_pred']
                else:
                    pin=cached[record['episode'],record['anchor']];file=Path(s['feature_cache']['path']).parent/pin['file']
                    features.need(digest(file)==pin['sha256'],'SEALED_EVALUATION_FEATURE_ROW')
                    row=torch.load(file,map_location='cpu',weights_only=False)
                    prediction=model.action_head.get_action(features.feature_packet(row),features.action_packet(model,row,model.dtype,generation=True))['action_pred']
                normalized=prediction.detach().float().cpu().numpy()
            decoded=data.processor.decode_action(normalized,EmbodimentTag.NEW_EMBODIMENT,
                {'eef_9d':actual[None,None,:9],'gripper_position':actual[None,None,9:]})
            physical=np.concatenate([decoded[k]for k in('eef_9d','gripper_position')],axis=-1)[0]
            file=output/record['sample_file'];np.savez(file,normalized=normalized,physical_H40=physical,
                expert_policy_H40=expert,actual_state10=actual,noise_seed=np.asarray(record['seed'],dtype=np.int64))
            rows.append(dict(episode=record['episode'],anchor=record['anchor'],seed=record['seed'],
                phase_audit_only=record['phase_audit_only'],sample_file=file.name,sha256=digest(file),
                errors=error_metrics(physical,expert,actual),envelope=geometry(physical,actual)))
    finally:features.restore_rng(state)
    features.need(before==rng_identity()and len(rows)==40,'COMPLETE_PHASE_PROBES_WITHOUT_TRAIN_RNG_CHANGE')
    result=dict(schema='TASK_LEARNING_CORE_LOCAL_LAST2_PHASE_PROBES_V1',records=rows,
        actual_queries=40,original_full_inference=cache_manifest is None,parent_global_step=2000,
        adaptation_step=s.get('adaptation_step',500),policy_arm=s.get('policy_arm','B1'),
        native_task_success='NOT_MEASURED_BY_OPEN_LOOP',new_optimizer_updates=0)
    write(output/'RESULT.private.json',result);return result


def compare_saved_generations(first,second,output):
    import numpy as np
    from scripts.pi0292.durable import write
    first,second=Path(first),Path(second)
    def records(folder):
        path=folder/'RESULT.private.json'
        return json.loads(path.read_bytes())['records']
    one,two=records(first),records(second);rows=[]
    features.need(len(one)==len(two)==40,'ALL_FORTY_SAME_CARD_REPLAY_PREDICTIONS')
    for a,b in zip(one,two):
        features.need(all(a[k]==b[k]for k in('episode','anchor','seed','sample_file')),'MATCHED_SAME_CARD_QUERY_IDENTITIES')
        x,y=first/a['sample_file'],second/b['sample_file']
        features.need(digest(x)==a['sha256']and digest(y)==b['sha256'],'ALL_REPLAY_MEMBERS_PINNED')
        with np.load(x,allow_pickle=False)as aa,np.load(y,allow_pickle=False)as bb:
            exact={k:(aa[k].shape==bb[k].shape and aa[k].dtype==bb[k].dtype and aa[k].tobytes()==bb[k].tobytes())
                   for k in('normalized','physical_H40','actual_state10','expert_policy_H40','noise_seed')}
        rows.append(dict(episode=a['episode'],anchor=a['anchor'],exact=exact))
    result=dict(schema='TASK_LEARNING_CORE_LOCAL_SAME_CARD_REPLAY_V1',requested=40,matched=40,records=rows,
                all_bit_exact=all(all(r['exact'].values())for r in rows),cross_hardware_qualified=False)
    write(output,result);features.need(result['all_bit_exact'],'SAME_CARD_ALL_PREDICTIONS_BIT_EXACT')
    return result


def local_replay(s,model,data,root,guard):
    """Dedicated ordinary local replay, independent of the rejected cross-site gate."""
    import torch
    from scripts.pi0292.durable import write
    model.requires_grad_(False);model.to('cuda').eval()
    measure(dict(s,adaptation_step=0),model,data,root/'replay-1',guard)
    if s['policy_arm']=='B0':
        measure(dict(s,adaptation_step=0),model,data,root/'replay-2',guard)
        compare=compare_saved_generations(root/'replay-1',root/'replay-2',root/'SAME_CARD_ACTUAL.private.json')
    else:
        # The sealed ordinary Core40 result is the first same-card realization.
        # Its prediction members live in its sealed-TRAIN40 child directory.
        reference=root/'sealed-reference';reference.mkdir()
        import os
        source=json.loads(Path(s['Core40_result']['path']).read_bytes())
        for row in source['records']:
            member=Path(s['Core40_samples'])/row['sample_file'];features.need(digest(member)==row['sha256'],'SEALED_ORDINARY_REFERENCE')
            os.link(member,reference/row['sample_file'])
        write(reference/'RESULT.private.json',source)
        compare=compare_saved_generations(reference,root/'replay-1',root/'SAME_CARD_ACTUAL.private.json')
    result=dict(schema='TASK_LEARNING_CORE_LOCAL_ORDINARY_REPLAY_ACTUAL_V1',policy_arm=s['policy_arm'],
        same_card_reproducible=compare['all_bit_exact'],actual_queries=80 if s['policy_arm']=='B0'else 40,
        weights=s['parent']if s['policy_arm']=='B0'else s['package'],precision='ORIGINAL_BF16_AUTOCAST_FP32_PARAMETERS',
        GPU_uuid=s['GPU_uuid'],denoising_steps=4,original_cross_hardware_REJECT_preserved=True,
        mechanical_safety_or_consumption_audit_changed=False,clipping=False,Teacher_fallback=False,
        native_panel_admission_issued=False,optimizer_updates=0,native_integrations=0,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write(root/'LOCAL_REPLAY_ACTUAL.private.json',result);return result


def rebuild(s, model, data, root, guard):
    """Fresh ordinary full model: SFT2000 plus the new internal last2 delta."""
    import torch
    import numpy as np
    import shutil
    from scripts.pi0292.durable import write
    checkpoint=Path(s['trained_checkpoint']['path'])
    features.need(digest(checkpoint/'CHECKPOINT_COMPLETE.json')==s['trained_checkpoint']['sha256'],'SEALED_NEW500_CHECKPOINT')
    saved=json.loads((checkpoint/'CHECKPOINT_COMPLETE.json').read_bytes())
    features.need(saved['schema']==MEMORY and saved['adaptation_step']==500 and saved['parent_global_step']==2000
                  and saved['recipe']==s['recipe'] and saved['feature_cache']==s['feature_cache']
                  and all(digest(checkpoint/n)==pin for n,pin in saved['files'].items()),'ACTUAL_COMPLETE500_STATE')
    values=torch.load(checkpoint/'target_parameters.pt',map_location='cpu',weights_only=True)
    features.install_ordinary_diagnostic_delta(model,values,saved['parameter_contract'])
    del values
    model.to('cuda').eval();actual=measure(s,model,data,root/'original-full-inference',guard)
    expected=json.loads((Path(s['trained_probes'])/'RESULT.private.json').read_bytes())
    comparisons=[]
    for a,b in zip(expected['records'],actual['records']):
        features.need(all(a[k]==b[k]for k in('episode','anchor','seed','sample_file')),'SAME_PHASE_RECONSTRUCTION_QUERIES')
        left,right=Path(s['trained_probes'])/a['sample_file'],root/'original-full-inference'/b['sample_file']
        features.need(digest(left)==a['sha256'] and digest(right)==b['sha256'],'BOTH_ACTUAL_GENERATION_MEMBERS')
        with np.load(left,allow_pickle=False)as x,np.load(right,allow_pickle=False)as y:
            exact={k:np.array_equal(x[k],y[k])for k in('normalized','physical_H40')}
            delta={k:float(np.max(np.abs(x[k].astype(float)-y[k].astype(float))))for k in exact}
        comparisons.append(dict(episode=a['episode'],anchor=a['anchor'],exact=exact,max_abs=delta))
    features.need(len(comparisons)==40,'ALL_FORTY_RECONSTRUCTION_QUERIES')
    result=dict(schema='TASK_LEARNING_CORE_LOCAL_LAST2_ORDINARY_RECONSTRUCTION_V1',
        actual_full_original_generations=40,matched_cached_generations=40,records=comparisons,
        all_H40_and_physical_bit_exact=all(all(r['exact'].values())for r in comparisons),
        parent_package=s['package'],trained_checkpoint=s['trained_checkpoint'],optimizer_updates=0,native_integrations=0,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        cross_hardware_qualified=False,original_DEV_or_TEST_results=False)
    write(root/'RECONSTRUCTION_ACTUAL.private.json',result)
    features.need(result['all_H40_and_physical_bit_exact'],'FRESH_ORDINARY500_RECONSTRUCTION_EXACT')
    package=root/'inference-package';package.mkdir()
    shutil.copy2(checkpoint/'target_parameters.pt',package/'target_parameters.pt')
    shutil.copytree(Path(s['package']['path'])/'processor',package/'processor')
    write(package/'PARAMETER_CONTRACT.json',saved['parameter_contract'])
    write(package/'PACKAGE_MANIFEST.json',dict(schema='TASK_LEARNING_CORE_LOCAL_LAST2_PACKAGE_V1',
        parent_package=s['package'],parent_global_step=2000,adaptation_step=500,recipe=s['recipe'],
        dataset_sha256=s['dataset_sha256'],statistics_sha256=s['statistics_sha256'],
        reconstruction=dict(path=str(root/'RECONSTRUCTION_ACTUAL.private.json'),sha256=digest(root/'RECONSTRUCTION_ACTUAL.private.json')),
        files={p.relative_to(package).as_posix():digest(p)for p in package.rglob('*')if p.is_file()},
        standalone_without_parent_package=False,cross_hardware_qualification=False,old_FIT4_identity_reused=False))
    return dict(result,package=dict(path=str(package),sha256=digest(package/'PACKAGE_MANIFEST.json')))
