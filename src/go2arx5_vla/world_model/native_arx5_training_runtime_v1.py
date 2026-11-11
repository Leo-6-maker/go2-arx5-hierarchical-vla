"""New official-parent transfer runtime, called by the existing finite owner.

No launcher or allocation authority. Cache only the validated frozen-backbone
boundary. All action-head computation, gradients and Adam moments stay on CUDA.
Historical SFT/FIT4 identities and schedulers are deliberately not used.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time

from . import native_arx5_training_profile_v1 as profile
from . import task_learning_features_v1 as features
from .storage import digest

CACHE_SCHEMA = 'NATIVE_ARX5_OFFICIAL_BACKBONE_CACHE_V1'
CHECKPOINT_SCHEMA = 'NATIVE_ARX5_OFFICIAL_LAST2_FULL_RESUME_V1'


def require(value, reason):
    if not value:
        raise ValueError('NATIVE_ARX5_' + reason)


def checked_parent(checkpoint, audit):
    """Reuse the actual seven-file stream audit only while native stats match."""
    from scripts.pi0294.multitask.raw_parent_delivery import FILES,stat_identity
    require(audit.get('schema') == 'NATIVE_ARX5_OFFICIAL_PARENT_SEVEN_FILE_IDENTITY_V1'
        and audit.get('passes') is True and audit.get('revision') == profile.REVISION
        and audit.get('parent_files_identity') == profile.PARENT_ID
        and audit.get('full_stream_bytes') == sum(size for _,size in FILES.values())
        and set(audit.get('files',{})) == set(FILES), 'ACTUAL_OFFICIAL_PARENT_AUDIT')
    root = Path(checkpoint)
    require(str(root) == audit['official_parent'] and root.resolve(strict=True) == root,
            'SAME_AUDITED_NATIVE_PARENT_PATH')
    for name,(expected,size) in FILES.items():
        row = audit['files'][name]
        require(row['sha256'] == expected and row['stat']['bytes'] == size
                and stat_identity(root/name) == row['stat'], 'UNCHANGED_OFFICIAL_PARENT_'+name)


def load_official(s, root, guard):
    """Strict new parent identity; caller already owns GPU and finite lease."""
    guard()
    audit_path = Path(s['official_parent_audit']['path'])
    require(digest(audit_path) == s['official_parent_audit']['sha256'], 'PINNED_PARENT_AUDIT')
    checked_parent(s['official_parent'], json.loads(audit_path.read_bytes()))
    from scripts.pi0294.native_dataset import official_import
    official_import(s['bindings']['source'])
    from scripts.pi0294.generalization.minimal_model import loader_kwargs,validate_loading
    from scripts.pi0294.local_backbone import local_model_name
    from scripts.pi0294.neural import COSMOS_REVISION
    from scripts.pi0292.durable import write
    from transformers import AutoModel
    import gr00t.model.gr00t_n1d7.gr00t_n1d7
    write(Path(root)/'OFFICIAL_PARENT_LOAD_INTENT.private.json', dict(
        parent_files_identity=profile.PARENT_ID, revision=profile.REVISION, epoch=time.time()))
    model, loading = AutoModel.from_pretrained(s['official_parent'], **loader_kwargs(
        local_model_name(root,s['bindings']),s['bindings']['cosmos_hub'],COSMOS_REVISION))
    write(Path(root)/'OFFICIAL_PARENT_CONSTRUCTED_ACTUAL.private.json', dict(
        actual_model_loads=1, loading=loading, parent_files_identity=profile.PARENT_ID))
    validate_loading(model,loading)
    contract = profile.parameter_contract(model)
    contract.validate()
    from scripts.pi0294.native_train import parameter_inventory
    from scripts.pi0294.generalization.minimal_adapter import frozen_fingerprint
    # Actual new-parent inventory, including hashes, before any candidate update.
    write(Path(root)/'PARAMETER_INVENTORY.private.json', parameter_inventory(model))
    write(Path(root)/'PARAMETER_CONTRACT.private.json', contract.describe())
    frozen = frozen_fingerprint(model)
    write(Path(root)/'FROZEN_START.private.json',frozen)
    from .target_gr00t_core_v1 import trim_unused_logits
    trim_unused_logits(model)  # same previously qualified unused-vocabulary option
    model.to('cuda').eval()
    guard()
    return model,contract,frozen


def build_target_data(s,*,for_optimization=False):
    from scripts.pi0292.train_adapter import build_processor
    from .native_arx5_dataset_v1 import NativeCameraTransferDataset
    root = Path(s['native_camera_dataset'])
    require(digest(root/'DATASET_MANIFEST.json') == s['dataset_sha256']
        and digest(root/'statistics.json') == s['statistics_sha256'], 'NEW_TARGET_DATA_AND_STATS')
    processor = build_processor(s['bindings']['source'],s['bindings']['cosmos'],
                                json.loads((root/'statistics.json').read_bytes()))
    return NativeCameraTransferDataset(root,s['camera_audit'],processor=processor,
        for_optimization=for_optimization,training_admission=s.get('training_admission'))


def cache_backbone(model, data, output, guard):
    """Versioned real-image cache; no DiT-prefix output is stored."""
    import torch
    from scripts.pi0292.durable import write
    from scripts.pi0294.native_train import tree_identity
    from .task_learning_gpu_rows_v3 import compact_row,row_bytes,FEATURE_KEYS
    output = Path(output); output.mkdir()
    require(not any(p.requires_grad for p in model.backbone.parameters()), 'FROZEN_BACKBONE_ONLY')
    rows=[];started=time.monotonic();total=0
    for index in range(len(data)):
        guard()
        inputs=data.processor.collator([data[index]['processed']])['inputs']
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            backbone_input,_=model.prepare_input(copy.deepcopy(inputs))
            encoded=model.backbone(backbone_input)
        require(set(encoded)==set(FEATURE_KEYS), 'ONLY_VERIFIED_BACKBONE_CACHE_BOUNDARY')
        row=compact_row(dict(identity=data.identity(index),inputs=inputs,features=encoded),device='cpu')
        file=output/f'row-{index:05d}.pt';features.atomic_torch(file,row);size=row_bytes(row);total+=size
        rows.append(dict(index=index,identity=data.identity(index),file=file.name,sha256=digest(file),
                         full_input_identity=tree_identity(inputs),tensor_bytes=size))
        if index%25==0 or index+1==len(data):
            write(output/'PROGRESS.private.json',dict(actual_backbone_calls=index+1,rows=index+1,
                tensor_bytes=total,wall_s=time.monotonic()-started,optimizer_updates=0))
    manifest=dict(schema=CACHE_SCHEMA,parent_files_identity=profile.PARENT_ID,revision=profile.REVISION,
        dataset_sha256=data.manifest_sha,statistics_sha256=data.statistics_sha,
        row_count=len(rows),rows=rows,input_dtype=str(model.dtype),tensor_bytes=total,
        cached_keys=list(FEATURE_KEYS),DiT_prefix_cached=False,training_admitted=False,
        actual_backbone_calls=len(rows),optimizer_updates=0,wall_s=time.monotonic()-started)
    write(output/'MANIFEST.private.json',manifest)
    return dict(path=str(output/'MANIFEST.private.json'),sha256=digest(output/'MANIFEST.private.json'))


class ResidentRows:
    """Variable population; validated once, then zero steady-state feature copies."""
    def __init__(self,pin,*,dataset_sha256,statistics_sha256,maximum_bytes,guard):
        import torch
        from .task_learning_gpu_rows_v3 import compact_row,row_bytes
        path=Path(pin['path']);require(digest(path)==pin['sha256'],'CACHE_MANIFEST')
        manifest=json.loads(path.read_bytes())
        require(manifest['schema']==CACHE_SCHEMA and manifest['parent_files_identity']==profile.PARENT_ID
            and manifest['dataset_sha256']==dataset_sha256 and manifest['statistics_sha256']==statistics_sha256
            and len(manifest['rows'])==manifest['row_count'] and 0<manifest['row_count']<=3200
            and manifest['tensor_bytes']<=maximum_bytes and manifest['DiT_prefix_cached'] is False,
            'INDEPENDENT_NEW_PARENT_AND_IMAGE_FEATURES')
        self.rows=[];self.bytes=0;self.pin=pin;self.manifest=manifest;started=time.monotonic()
        for index,entry in enumerate(manifest['rows']):
            guard();file=path.parent/entry['file']
            require(file.resolve().is_relative_to(path.parent.resolve()) and digest(file)==entry['sha256'], 'CACHE_ROW')
            row=torch.load(file,map_location='cpu',weights_only=False)
            require(entry['index']==index and row['identity']==entry['identity']
                and row['identity']['split']=='TRAIN' and row_bytes(row)==entry['tensor_bytes'], 'TRAIN_ROW_ORDER')
            self.rows.append(compact_row(row,device='cuda'));self.bytes+=row_bytes(self.rows[-1])
        require(self.bytes==manifest['tensor_bytes'],'COMPLETE_GPU_BANK')
        self.preload_wall_s=time.monotonic()-started

    def __call__(self,index):
        return self.rows[index]

    def describe(self):
        return dict(schema='NATIVE_ARX5_VARIABLE_POPULATION_CUDA_BANK_V1',rows=len(self.rows),
            tensor_bytes=self.bytes,preload_wall_s=self.preload_wall_s,device='cuda',
            steady_state_CPU_to_GPU_feature_action_copies=0,DiT_prefix_cached=False)


def optimizer_and_scheduler(model,contract,recipe):
    import torch
    from transformers import Trainer
    decay=set(Trainer.get_decay_parameter_names(None,model))
    groups=[dict(params=[model.get_parameter(n)for n in contract.trainable_names if(n in decay)==use],
                 weight_decay=recipe['weight_decay']if use else 0.)for use in(True,False)]
    optimizer=torch.optim.AdamW(groups,lr=recipe['learning_rate'],betas=tuple(recipe['betas']),eps=recipe['eps'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,profile.lr_multiplier)
    return optimizer,scheduler


def update(model,contract,optimizer,scheduler,visits,step,row_at,*,full_data=None,audit,guard,qualifying=False,trace=None):
    """Original 32-microbatch masked FM update; no CPU activation context."""
    import torch
    from scripts.pi0294.native_train import tree_identity,rng_identity
    from .task_learning_core_local_v1 import training_tensor_residency
    require(0<=step<500 and len(visits)==16000, 'FINITE_MATCHED_CURSOR')
    require(optimizer.param_groups[0]['lr']==profile.learning_rates()[step], 'ACTUAL_USED_LR')
    contract.validate();guard();optimizer.zero_grad(set_to_none=True);losses=[];records=[];started=time.monotonic()
    for offset in range(32):
        guard();cursor=step*32+offset;index=int(visits[cursor]);row=row_at(index)
        if trace is not None:trace(dict(cursor=cursor,index=index,status='MODEL_FORWARD_ATTEMPT'))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if full_data is None:
                value=model.action_head(features.feature_packet(row),features.action_packet(model,row,model.dtype))
            else:
                inputs=full_data.processor.collator([full_data[index]['processed']])['inputs']
                value=model(inputs=copy.deepcopy(inputs))
            loss=features.check_loss(value)
            (loss/32).backward()
        require(audit.verified and not audit.active, 'ORIGINAL_FM_NOISE_AUDIT')
        losses.append(float(loss.detach()))
        if trace is not None:trace(dict(cursor=cursor,index=index,status='BACKWARD_COMPLETE'))
        if qualifying:
            records.append(dict(cursor=cursor,index=index,identity=row['identity'],noise=copy.deepcopy(audit.record)))
        del value,loss
    require(all(model.get_parameter(n).grad is not None for n in contract.trainable_names),
            'ALL_WHITELIST_GRADIENTS_PRESENT')
    gradient_identity=tree_identity({n:model.get_parameter(n).grad for n in contract.trainable_names})if qualifying else None
    norm=float(torch.nn.utils.clip_grad_norm_(contract.parameters(),1.,error_if_nonfinite=True))
    used_lr=optimizer.param_groups[0]['lr']
    if trace is not None:trace(dict(global_step=step+1,status='OPTIMIZER_ATTEMPT'))
    optimizer.step();scheduler.step()
    if trace is not None:trace(dict(global_step=step+1,status='OPTIMIZER_COMPLETE'))
    residency=training_tensor_residency(model,contract,optimizer)
    result=dict(global_step=step+1,cursor=(step+1)*32,actual_optimizer_calls=1,actual_head_forward_calls=32,
        actual_full_model_training_calls=32 if full_data is not None else 0,
        used_lr=used_lr,next_lr=optimizer.param_groups[0]['lr'],mean_native_FM_loss=sum(losses)/32,
        grad_norm_before_clip=norm,wall_s=time.monotonic()-started,residency=residency,
        max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())
    if qualifying:
        result['equivalence']=dict(gradients=gradient_identity,parameters=features.state_identity(model,contract.trainable_names),
            optimizer=tree_identity(optimizer.state_dict()),scheduler=scheduler.state_dict(),RNG=rng_identity(),
            per_sample_loss_hex=[v.hex()for v in losses],samples=records,cursor=(step+1)*32)
    return result


def save_checkpoint(path,model,contract,optimizer,scheduler,step,recipe,cache_pin,frozen,visits_sha256,
                    *,processor=None,statistics_file=None):
    import shutil
    from scripts.pi0292.durable import write
    from scripts.pi0294.native_train import tree_identity,rng_identity
    require(type(step)is int and 0<=step<=500,'CHECKPOINT_CURSOR')
    path=Path(path);path.mkdir()
    if processor is not None:
        require(statistics_file is not None and digest(statistics_file)==recipe['statistics_sha256'],
                'CHECKPOINT_ORIGINAL_STATISTICS')
        state=features.rng_state()
        try:
            processor.save_pretrained(path/'processor')
            shutil.copyfile(statistics_file,path/'statistics.json')
        finally:
            features.restore_rng(state)
    features.atomic_torch(path/'trainable.pt',{n:model.get_parameter(n).detach().cpu()for n in contract.trainable_names})
    features.atomic_torch(path/'optimizer.pt',optimizer.state_dict())
    features.atomic_torch(path/'scheduler.pt',scheduler.state_dict())
    features.atomic_torch(path/'rng.pt',features.rng_state())
    write(path/'STATE_IDENTITY.json',dict(parameters=features.state_identity(model,contract.trainable_names),
        optimizer=tree_identity(optimizer.state_dict()),scheduler=scheduler.state_dict(),RNG=rng_identity(),cursor=step*32))
    write(path/'FROZEN.json',frozen)
    write(path/'COMPLETE.json',dict(schema=CHECKPOINT_SCHEMA,global_step=step,cursor=32*step,recipe=recipe,
        parent_files_identity=profile.PARENT_ID,parameter_contract=contract.describe(),feature_cache=cache_pin,
        visits_sha256=visits_sha256,scaler='NOT_APPLICABLE_BF16',
        Processor_saved=processor is not None,
        files={p.relative_to(path).as_posix():digest(p)for p in path.rglob('*')if p.is_file()}))


def restore_checkpoint(path,model,contract,optimizer,scheduler,recipe,cache_pin,frozen,visits_sha256):
    import torch
    from scripts.pi0294.native_train import tree_identity,rng_identity
    path=Path(path);saved=json.loads((path/'COMPLETE.json').read_bytes())
    require(saved['schema']==CHECKPOINT_SCHEMA and saved['recipe']==recipe
        and saved['parent_files_identity']==profile.PARENT_ID and saved['feature_cache']==cache_pin
        and saved['parameter_contract']==contract.describe() and saved['visits_sha256']==visits_sha256,
        'SAME_NEW_STAGE_COMPLETE_STATE')
    require(all(digest(path/n)==sha for n,sha in saved['files'].items())
        and json.loads((path/'FROZEN.json').read_bytes())==frozen,'CHECKPOINT_FILES_AND_FROZEN_PARENT')
    values=torch.load(path/'trainable.pt',map_location='cpu',weights_only=True)
    require(set(values)==set(contract.trainable_names),'EXACT_SAVED_WHITELIST')
    with torch.no_grad():
        for n,value in values.items():
            target=model.get_parameter(n)
            require(target.shape==value.shape and target.dtype==value.dtype,'SAVED_TENSOR_LAYOUT')
            target.copy_(value)
    optimizer.load_state_dict(torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False))
    scheduler.load_state_dict(torch.load(path/'scheduler.pt',map_location='cpu',weights_only=False))
    features.restore_rng(torch.load(path/'rng.pt',map_location='cpu',weights_only=False))
    actual=dict(parameters=features.state_identity(model,contract.trainable_names),optimizer=tree_identity(optimizer.state_dict()),
        scheduler=scheduler.state_dict(),RNG=rng_identity(),cursor=saved['cursor'])
    require(actual==json.loads((path/'STATE_IDENTITY.json').read_bytes()) and saved['cursor']==32*saved['global_step'],
            'ACTUAL_RESTORED_PARAMETERS_ADAM_SCHEDULE_RNG_CURSOR')
    return saved['global_step']
