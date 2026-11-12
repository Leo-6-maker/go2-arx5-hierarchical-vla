"""Rebuild the original RGB N1.7 inference graph from an own complete endpoint.

No feature bank, optimizer update, clipping, Teacher, or physical execution.
The new package binds the official parent, learned whitelist and saved Processor.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil

from . import native_arx5_training_runtime_v1 as rt
from . import native_arx5_training_profile_v1 as profile
from . import task_learning_features_v1 as features
from .storage import digest


def read_checkpoint(pin,contract,s):
    path=Path(pin['path']);rt.require(digest(path/'COMPLETE.json')==pin['sha256'],'ENDPOINT_COMPLETE_PIN')
    value=json.loads((path/'COMPLETE.json').read_bytes())
    rt.require(value['schema']==rt.CHECKPOINT_SCHEMA and value['parent_files_identity']==profile.PARENT_ID
        and value['parameter_contract']==contract.describe() and value['Processor_saved'] is True
        and value['recipe']==profile.recipe(s['recipe']['stage'],s['dataset_sha256'],s['statistics_sha256'])
        and value['global_step']in(4,250,500)
        and (value['global_step']!=4 or s['qualification_only']is True)
        and digest(path/'statistics.json')==s['statistics_sha256']
        and all(digest(path/n)==sha for n,sha in value['files'].items()), 'COMPLETE_OWN_QUAL4_OR_FIXED_250_500')
    return path,value


def install(values,model,contract):
    import torch
    rt.require(set(values)==set(contract.trainable_names),'EXACT_ENDPOINT_WHITELIST')
    with torch.no_grad():
        for n,value in values.items():
            p=model.get_parameter(n)
            rt.require(value.shape==p.shape and value.dtype==p.dtype,'ENDPOINT_TENSOR_SHAPE_PRECISION')
            p.copy_(value)


def load_processor(path,s,root):
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from scripts.pi0294.local_backbone import local_model_name
    from scripts.pi0294.neural import COSMOS_REVISION
    processor=Gr00tN1d7Processor.from_pretrained(path,model_name=local_model_name(root,s['bindings']),
        local_files_only=True,transformers_loading_kwargs=dict(local_files_only=True,trust_remote_code=True,
            cache_dir=s['bindings']['cosmos_hub'],revision=COSMOS_REVISION))
    processor.eval()
    rt.require(processor.embodiment_id_mapping['new_embodiment']==10,'SAME_TARGET_PROCESSOR_SLOT')
    return processor


def probes(model,data,root,guard):
    import numpy as np
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from scripts.pi0292.durable import write
    from scripts.pi0294.native_train import tree_identity
    from .task_learning_local_analysis_v1 import geometry
    root.mkdir();records=[]
    for layout_index,layout in enumerate(('TRAIN01','TRAIN07','TRAIN13','TRAIN25')):
        for anchor in(0,9):
            index=next(i for i,(ep,a)in enumerate(data.rows)
                if ep['parent']=='TASK_LEARNING_UNDER_HAND_'+layout and a==anchor)
            guard();state,_,_,_=data.raw(index)
            inputs=data.processor.collator([data[index]['processed']])['inputs']
            seed=2026111200+100*layout_index+anchor;file=root/f'{layout}-{anchor:02d}.npz'
            write(file.with_suffix('.intent.json'),dict(actual_model_invocation_requested=1,index=index,seed=seed))
            with torch.random.fork_rng(devices=[0]),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                torch.manual_seed(seed)
                normalized=model.get_action(features.inference_inputs(inputs))['action_pred'].detach().float().cpu().numpy()
            decoded=data.processor.decode_action(normalized,EmbodimentTag.NEW_EMBODIMENT,
                {'eef_9d':state[None,None,:9],'gripper_position':state[None,None,9:]})
            physical=np.concatenate([decoded[k]for k in('eef_9d','gripper_position')],axis=-1)[0]
            rt.require(normalized.shape==(1,40,132)and physical.shape==(40,10)
                and np.isfinite(normalized).all()and np.isfinite(physical).all(),'FINITE_COMPLETE_NORMALIZED_AND_PHYSICAL_H40')
            np.savez(file,normalized=normalized,physical_H40=physical,actual_state10=state)
            records.append(dict(layout=layout,anchor=anchor,index=index,seed=seed,file=file.name,sha256=digest(file),
                input_identity=tree_identity(inputs),raw_envelope=geometry(physical,state)))
    write(root/'PROBES_ACTUAL.private.json',dict(actual_ordinary_RGB_generations=8,records=records))
    return records


def run(s,model,contract,data,frozen,root,guard):
    import numpy as np
    import torch
    from scripts.pi0292.durable import write
    from scripts.pi0294.native_train import rng_identity
    from scripts.pi0294.generalization.minimal_adapter import consume_for_official_inference,TargetCategoryLinear
    cp,saved=read_checkpoint(s['checkpoint'],contract,s)
    rt.require(json.loads((cp/'FROZEN.json').read_bytes())==frozen,'SAME_FROZEN_OFFICIAL_ENDPOINT_PARENT')
    install(torch.load(cp/'trainable.pt',map_location='cpu',weights_only=True),model,contract)
    guard();state=features.rng_state();before_rng=rng_identity()
    try:
        package=root/'inference-package';package.mkdir()
        shutil.copyfile(cp/'trainable.pt',package/'trainable.pt')
        shutil.copyfile(cp/'statistics.json',package/'statistics.json')
        shutil.copytree(cp/'processor',package/'processor')
        data.processor=load_processor(cp/'processor',s,root)
        reference=probes(model,data,root/'restored-wrapped-RGB',guard)
        contract._training_session_used=True
        model=consume_for_official_inference(contract)
        model.to('cuda').eval();data.processor=load_processor(package/'processor',s,root)
        ordinary=probes(model,data,root/'ordinary-RGB',guard);equal=True
        for a,b in zip(reference,ordinary):
            rt.require(a['index']==b['index']and a['seed']==b['seed']and a['input_identity']==b['input_identity'],
                       'SAME_REAL_RGB_STATE_LANGUAGE_AND_PACKAGED_PROCESSOR')
            with np.load(root/'restored-wrapped-RGB'/a['file'])as x,np.load(root/'ordinary-RGB'/b['file'])as y:
                equal=equal and all(np.array_equal(x[k],y[k])for k in('normalized','physical_H40','actual_state10'))
        pure=not any(isinstance(m,TargetCategoryLinear)for m in model.modules())
        manifest=dict(schema='NATIVE_ARX5_OFFICIAL_ENDPOINT_INFERENCE_PACKAGE_V1',
            parent_files_identity=profile.PARENT_ID,revision=profile.REVISION,checkpoint=s['checkpoint'],
            global_step=saved['global_step'],recipe=saved['recipe'],parameter_contract=saved['parameter_contract'],
            statistics_sha256=s['statistics_sha256'],
            files={p.relative_to(package).as_posix():digest(p)for p in package.rglob('*')if p.is_file()},
            ordinary_original_RGB_reconstruction_exact=equal and pure,Core_local_only=True,
            original_cross_hardware_REJECT_preserved=True,physical_execution_authorized=False)
        write(package/'MANIFEST.private.json',manifest)
        result=dict(schema='NATIVE_ARX5_ORDINARY_RGB_ENDPOINT_RECONSTRUCTION_ACTUAL_V1',passes=equal and pure,
            global_step=saved['global_step'],actual_ordinary_RGB_generations=16,actual_optimizer_updates=0,
            normalized_and_physical_H40_bit_exact=equal,original_graph_without_wrappers=pure,
            actual_Processor_roundtrip=True,training_feature_cache_used=False,checkpoint=s['checkpoint'],
            package=dict(path=str(package),sha256=digest(package/'MANIFEST.private.json')),
            max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),
            physics_integrations=0,records=ordinary,qualification_only=s['qualification_only'])
    finally:
        features.restore_rng(state)
    result['evaluation_RNG_restored']=rng_identity()==before_rng
    write(root/'RECONSTRUCTION_ACTUAL.private.json',result)
    rt.require(result['passes']and result['evaluation_RNG_restored'],'ORDINARY_RGB_PROCESSOR_H40_RECONSTRUCTION')
    return result
