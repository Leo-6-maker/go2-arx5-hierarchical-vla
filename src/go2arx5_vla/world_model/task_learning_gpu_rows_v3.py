"""Optional immutable GPU rows for the already qualified frozen-feature seam.

Ordinary model inference still consumes real images through the original path.
This optimization is only for offline head training; it does not remove any
head input, action label, mask or native noise/consumption audit.
"""
import json
import time
from pathlib import Path

from . import task_learning_features_v1 as features
from .storage import digest

ACTION_KEYS=('state','embodiment_id','action','action_mask')
FEATURE_KEYS=('backbone_features','backbone_attention_mask','image_mask')
SCHEMA='TASK_LEARNING_GPU_FROZEN_FEATURE_ROW_BANK_V3'


def compact_row(row,*,device):
    """Copy only tensors actually read by the pinned action-head forward."""
    import torch
    features.need(all(k in row['inputs']for k in ACTION_KEYS)and set(row['features'])==set(FEATURE_KEYS),
                  'EXACT_ORIGINAL_HEAD_AND_FEATURE_KEYS')
    result=dict(identity=dict(row['identity']),
        inputs={k:features.tensor_tree(row['inputs'][k],device)for k in ACTION_KEYS},
        features=features.tensor_tree(row['features'],device))
    features.need(all(torch.is_tensor(v)and v.device.type==torch.device(device).type
        and not v.requires_grad for group in ('inputs','features')for v in result[group].values()),
        'IMMUTABLE_HEAD_ROW_TENSORS_ON_DECLARED_DEVICE')
    return result


def row_bytes(row):
    return sum(t.numel()*t.element_size()for key in ('inputs','features')for t in row[key].values())


class ResidentRows:
    def __init__(self,pin,*,guard,maximum_bytes):
        import torch
        p=Path(pin['path']);features.need(digest(p)==pin['sha256'],'UNCHANGED_ORIGINAL_FEATURE_MANIFEST')
        manifest=json.loads(p.read_bytes());features.need(len(manifest['rows'])==1450,'EXACT_TRAIN1450_FEATURE_BANK')
        self.rows=[];self.bytes=0;started=time.monotonic()
        for index,entry in enumerate(manifest['rows']):
            guard();file=p.parent/entry['file'];features.need(digest(file)==entry['sha256'],'ADMIT_EACH_IMMUTABLE_ROW_ONCE')
            row=torch.load(file,map_location='cpu',weights_only=False)
            features.need(entry['index']==index and row['identity']['split']=='TRAIN'
                and row['identity']['episode_id']==entry['episode'] and row['identity']['anchor']==entry['anchor'],
                'ORIGINAL_ORDER_AND_TRAIN_ONLY_ROW_BANK')
            size=sum(v.numel()*v.element_size()for k in FEATURE_KEYS for v in [row['features'][k]])+sum(
                row['inputs'][k].numel()*row['inputs'][k].element_size()for k in ACTION_KEYS)
            features.need(self.bytes+size<=maximum_bytes,'PREDECLARED_GPU_ROW_BANK_CAPACITY')
            resident=compact_row(row,device='cuda');self.rows.append(resident);self.bytes+=row_bytes(resident)
        self.preload_wall_s=time.monotonic()-started

    def __call__(self,index):return self.rows[index]

    def describe(self):
        return dict(schema=SCHEMA,TRAIN_rows=len(self.rows),resident_tensor_bytes=self.bytes,
            preload_wall_s=self.preload_wall_s,device='cuda',retained_action_keys=list(ACTION_KEYS),
            original_images_reused_in_ordinary_model=True,original_cache_modified=False,
            steady_state_CPU_to_GPU_feature_or_action_transfers=0)


def qualification(spec,model,contract,data,root,guard):
    """One matched update per route from the same C checkpoint; no candidate step."""
    import torch
    from scripts.pi0292.durable import write
    from . import task_learning_core_local_v1 as base
    from . import task_learning_execution_loss_v3 as loss
    cache=json.loads(Path(spec['feature_cache']['path']).read_bytes())
    states,experts,identities=loss.physical_inputs(spec,cache)
    decoder=loss.PhysicalDecoder.from_processor(data.processor)
    saved=json.loads((Path(spec['resume'])/'CHECKPOINT_COMPLETE.json').read_bytes());results={};bank=None
    for route in ('ORIGINAL_ROW_READS','CUDA_TRAIN1450_BANK'):
        guard();output=root/route;output.mkdir();model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        if route=='CUDA_TRAIN1450_BANK':bank=ResidentRows(spec['feature_cache'],guard=guard,maximum_bytes=3*1024**3)
        torch.cuda.reset_peak_memory_stats()
        objective=loss.ExecutionLoss(model.action_head,decoder,states,experts,identities,
            coefficient=spec['recipe']['execution_loss']['coefficient'])
        diagnostic=dict(spec,qualification_only=True,operation='QUALIFY_TRAINING_ROW_BANK_V3',endpoint=saved['adaptation_step']+1)
        try:actual=base.train(diagnostic,model,contract,data,output,guard,
            loss_transform=objective,row_loader=bank)
        finally:objective.close()
        results[route]=dict(actual=actual,progress=json.loads((output/'PROGRESS.private.json').read_bytes()),
            update=json.loads((output/'UPDATE_IDENTITY.private.json').read_bytes()),
            samples=[json.loads(x)for x in(output/'SAMPLE_TRACE.jsonl').read_text().splitlines()])
    left,right=[results[k]for k in ('ORIGINAL_ROW_READS','CUDA_TRAIN1450_BANK')]
    exact=left['update']==right['update']and left['samples']==right['samples']
    result=dict(schema=SCHEMA,status='PASS_MATCHED_COMPLETE_UPDATE'if exact else'REJECT_MATCHED_UPDATE_DIFFERENCE',
        source_sha=spec['source_sha'],checkpoint=spec['resume'],exact_loss_gradients_parameters_optimizer_scheduler_RNG_noise_and_auxiliary=exact,
        actual_non_candidate_optimizer_calls=2,candidate_optimizer_calls=0,matched_windows=32,
        bank=bank.describe(),executions=results,physics_integrations=0,original_DEV_TEST_queries=0,
        speedup_not_claimed_from_single_update=True)
    write(root/'QUALIFICATION_ACTUAL.private.json',result)
    features.need(exact,'ACTUAL_GPU_ROW_BANK_COMPLETE_UPDATE_EQUIVALENCE')
    return result
