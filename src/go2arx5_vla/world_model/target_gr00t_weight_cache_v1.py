"""Actual same-model loss/gradient proof before disabling AMP weight storage.

Autocast stays BF16. Each ordinary backbone and action forward still executes;
only PyTorch's temporary cast-weight cache is disabled, with no feature cache.
"""
import copy
from pathlib import Path
from .target_gr00t_profile_v1 import need

MEMORY='CORE_FROZEN_BACKBONE_SWAP_NO_WEIGHT_CACHE_V5'


def qualification_inputs(data,root):
    # Keep the original positioned-sampler contract and give the non-optimizer
    # comparison its own trace, separate from the real Trainer's sample cursor.
    probe=copy.copy(data)
    probe.trace_root=Path(root)/'weight-cache-qualification-sample-trace'
    probe.trace_root.mkdir(parents=True)
    return probe.processor.collator([probe[(0,0)]['processed']])['inputs']


def enforce(model):
    import torch
    state=dict(schema='NATIVE_BF16_FORWARD_WITHOUT_CAST_WEIGHT_CACHE_V1',forward_calls=0,
        weight_cache_enabled=False,autocast_dtype='torch.bfloat16')
    def check(module,args):
        need(torch.is_autocast_enabled('cuda')and torch.get_autocast_dtype('cuda')==torch.bfloat16
            and not torch.is_autocast_cache_enabled(),
            'ACTUAL_NATIVE_BF16_FORWARD_WITH_DISABLED_WEIGHT_CACHE')
        state['forward_calls']+=1
    model.action_head.register_forward_pre_hook(check)
    return state


def qualify(model,contract,data,root,guard):
    import torch
    from transformers import set_seed
    from scripts.pi0292.durable import write
    from scripts.pi0294.native_train import NativeNoiseAudit,rng_identity
    from scripts.pi0294.multitask.activation_offload import SavedActivationOffload,context
    from .task_learning_features_v1 import rng_state,restore_rng,gradients,check_loss
    from .target_gr00t_core_v1 import activation_policy
    need(len(data)==4 and all(not p.requires_grad for p in model.backbone.parameters()),
        'NATIVE_CACHE_POLICY_QUALIFICATION_ON_FROZEN_FOUR_WINDOW_DATA_ONLY')
    inputs=qualification_inputs(data,root)
    saved_rng=rng_state();training=model.training
    audit=NativeNoiseAudit(model)
    records=[]
    try:
        model.train()
        for enabled,maximum in[(True,2*1024**3),(False,4*1024**3)]:
            guard();set_seed(2026110901);model.zero_grad(set_to_none=True)
            storage=SavedActivationOffload(model,policy=activation_policy(maximum))
            with context(storage),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=enabled):
                output=model(inputs=copy.deepcopy(inputs));loss=check_loss(output)
                (loss/32).backward()
            row=dict(loss=loss.detach().cpu().clone(),gradients=gradients(model,contract.trainable_names),
                noise=copy.deepcopy(audit.record),RNG=rng_identity())
            need(len(row['gradients'])==len(contract.trainable_names)
                and any(bool(torch.count_nonzero(t))for t in row['gradients'].values()),
                'ALL_ORIGINAL_ACTION_GRADIENTS_ACTUALLY_COMPUTED')
            records.append(row);del output,loss
            model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        a,b=records
        need(torch.equal(a['loss'],b['loss'])and set(a['gradients'])==set(b['gradients'])
            and all(torch.equal(a['gradients'][n],b['gradients'][n])for n in a['gradients']),
            'WEIGHT_CACHE_DISABLED_ORIGINAL_LOSS_AND_EVERY_GRADIENT_EXACT')
        need(a['noise']==b['noise']and a['RNG']==b['RNG'],
            'WEIGHT_CACHE_DISABLED_NATIVE_NOISE_AND_RNG_EXACT')
        proof=dict(status='PASS_REAL_WEIGHT_CACHE_DISABLED_LOSS_GRADIENT_NOISE_RNG_EXACT',
            native_loss=float(a['loss']),gradient_tensors=len(a['gradients']),
            gradient_exact=True,noise_and_RNG_exact=True,forward_autocast_dtype='torch.bfloat16',
            parameter_dtypes_changed=False,backbone_feature_cache=False,
            reference_weight_cache=True,candidate_weight_cache=False,
            reference_activation_budget_bytes=2*1024**3,candidate_activation_budget_bytes=4*1024**3,
            optimizer_calls=0,physics_integrations=0)
        write(root/'WEIGHT_CACHE_EQUIVALENCE_ACTUAL.json',proof)
        return proof
    finally:
        audit.close();model.zero_grad(set_to_none=True);model.train(training)
        restore_rng(saved_rng);torch.cuda.empty_cache()
