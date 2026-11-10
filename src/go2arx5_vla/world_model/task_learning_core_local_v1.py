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


def setup(s, root):
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
    install(model,s['package']['path'],s['package']['sha256'],s['parent']['files'])
    contract = action_contract(model,'INTERFACES_LAST2_DIT',profile_name='TASK_LEARNING_GR00T_V1')
    features.need(contract.describe()['trainable_numel']==77847684,'EXACT_LAST2_SCOPE')
    trim_unused_logits(model); model.eval()
    data=TaskLearningNativeDataset(s['bindings']['dataset'],s['bindings']['source'],s['bindings']['cosmos'],
        for_optimization=False,optimization_site='CORELAB_4080S')
    data.processor=Gr00tN1d7Processor.from_pretrained(Path(s['package']['path'])/'processor',
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
