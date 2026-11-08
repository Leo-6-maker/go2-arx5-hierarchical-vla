"""Explicit frozen-feature Core diagnostic; original GR00T action head and loss.

This finite FIT4 route is not a deployed policy or full-model Trainer claim.
Only immutable, already admitted observations may use its cache. Qualification
requires exact original-path generations, native loss and action gradients.
"""
import copy
import json
import os
import random
from pathlib import Path

from .storage import digest

SCOPE = 'GR00T_TASK_LEARNING_CORE_HOPPER_V1'
MEMORY = 'FROZEN_BACKBONE_FEATURES_CORE_DIAGNOSTIC_V1'


def need(ok, reason):
    if not ok:
        raise ValueError('TASK_LEARNING_FEATURES_' + reason)


def sample_index(cursor, size, seed):
    import numpy as np
    cycle, offset = divmod(cursor, size)
    return int(np.random.Generator(np.random.PCG64(seed + cycle)).permutation(size)[offset])


def tensor_tree(value, device=None):
    import torch
    if torch.is_tensor(value):
        return value.detach().to(device or value.device).clone()
    if hasattr(value, 'items'):
        return {k: tensor_tree(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tensor_tree(v, device) for v in value)
    return copy.deepcopy(value)


def feature_packet(row):
    from transformers.feature_extraction_utils import BatchFeature
    return BatchFeature(data=tensor_tree(row['features'], 'cuda'))


def action_packet(model, row, input_dtype):
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    values = model.action_head.prepare_input(copy.deepcopy(row['inputs']))
    return BatchFeature(data={k: v.to('cuda', dtype=input_dtype if torch.is_floating_point(v) else v.dtype)
                              for k, v in values.items()})


def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(value):
    import numpy as np
    import torch
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['cpu'])
    torch.cuda.set_rng_state_all(value['cuda'])


def atomic_torch(path, value):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + '.part')
    need(not path.exists() and not temporary.exists(), 'FRESH_TENSOR_ARTIFACT')
    torch.save(value, temporary)
    os.replace(temporary, path)


def state_identity(model, names):
    from scripts.pi0294.native_train import tensor_identity
    return {n: tensor_identity(model.get_parameter(n)) for n in names}


def gradients(model, names):
    return {n: model.get_parameter(n).grad.detach().cpu().clone()
            for n in names if model.get_parameter(n).grad is not None}


def check_loss(output):
    import torch
    mask, masked, loss = output['action_mask'], output['action_loss'], output['loss']
    need(mask.shape[1:] == (40, 132) and bool((mask[..., :10] == 1).all())
         and bool((mask[..., 10:] == 0).all()) and bool((masked[..., 10:] == 0).all()), 'ORIGINAL_ACTION_MASK')
    need(torch.equal(loss, masked.sum() / (mask.sum() + 1e-6)) and bool(torch.isfinite(loss)), 'ORIGINAL_NATIVE_LOSS')
    return loss


def qualify_cache(model, contract, data, root, guard):
    import torch
    from transformers import set_seed
    from scripts.pi0292.durable import write
    from scripts.pi0294.native_train import NativeNoiseAudit, tree_identity, rng_identity
    from scripts.pi0294.multitask.activation_offload import SavedActivationOffload, context
    from .target_gr00t_core_v1 import activation_policy
    root = Path(root)
    need(len(data) == 4 and all(not p.requires_grad for p in model.backbone.parameters()), 'FROZEN_FIT4_ONLY')
    input_dtype = model.dtype
    rows, generation_proofs = [], []
    model.to('cuda').eval()
    for index in range(len(data)):
        guard()
        inputs = data.processor.collator([data[index]['processed']])['inputs']
        row = dict(identity=data.identity(index), inputs=tensor_tree(inputs, 'cpu'),
                   input_identity=tree_identity(inputs))
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            backbone_input, _ = model.prepare_input(copy.deepcopy(inputs))
            row['features'] = tensor_tree(model.backbone(backbone_input), 'cpu')
        seed = 2026110900 + index
        with torch.random.fork_rng(devices=[0]), torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            torch.manual_seed(seed)
            original = model.get_action(copy.deepcopy(inputs))['action_pred'].detach().cpu()
            torch.manual_seed(seed)
            cached = model.action_head.get_action(feature_packet(row), action_packet(model, row, input_dtype))['action_pred'].detach().cpu()
        need(torch.equal(original, cached), 'FULL_H40_ORIGINAL_AND_CACHE_EXACT')
        row['original_normalized_H40'] = original
        generation_proofs.append(dict(index=index, seed=seed, exact=True, output=tree_identity(original)))
        rows.append(row)
    audit = NativeNoiseAudit(model)
    storage = SavedActivationOffload(model, policy=activation_policy())
    try:
        set_seed(2026110901)
        original_rng = rng_state()
        model.zero_grad(set_to_none=True)
        with context(storage), torch.autocast('cuda', dtype=torch.bfloat16):
            original_output = model(inputs=copy.deepcopy(rows[0]['inputs']))
            original_loss = check_loss(original_output)
            (original_loss / 32).backward()
        reference_loss = original_loss.detach().cpu().clone()
        reference_gradients = gradients(model, contract.trainable_names)
        reference_noise = copy.deepcopy(audit.record)
        reference_rng_after = rng_identity()
        need(reference_gradients and any(bool(torch.count_nonzero(v)) for v in reference_gradients.values()), 'REAL_REFERENCE_GRADIENT')
        del original_output, original_loss
        model.zero_grad(set_to_none=True)
        # Keep the original frozen weights on CPU; do not cast or replace them.
        model.backbone.to('cpu')
        torch.cuda.empty_cache()
        restore_rng(original_rng)
        with context(storage), torch.autocast('cuda', dtype=torch.bfloat16):
            cached_output = model.action_head(feature_packet(rows[0]), action_packet(model, rows[0], input_dtype))
            cached_loss = check_loss(cached_output)
            (cached_loss / 32).backward()
        candidate_gradients = gradients(model, contract.trainable_names)
        need(torch.equal(reference_loss, cached_loss.detach().cpu()), 'CACHED_NATIVE_LOSS_EXACT')
        need(set(reference_gradients) == set(candidate_gradients) and all(
            torch.equal(reference_gradients[n], candidate_gradients[n]) for n in reference_gradients), 'CACHED_ACTION_GRADIENTS_EXACT')
        need(reference_noise == audit.record and reference_rng_after == rng_identity(), 'CACHED_NATIVE_NOISE_AND_RNG_EXACT')
        # Inference with a live training audit must remain outside that audit.
        stream_before = rng_identity()
        with torch.random.fork_rng(devices=[0]), torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            model.action_head.get_action(feature_packet(rows[0]), action_packet(model, rows[0], input_dtype))
        need(not audit.active and audit.pending is None and rng_identity() == stream_before, 'TRAIN_GENERATE_AUDIT_SCOPE')
        proof = dict(status='PASS_REAL_GR00T_FROZEN_FEATURE_OUTPUT_LOSS_GRADIENT_EQUIVALENCE',
                     generations=generation_proofs, gradient_tensors=len(reference_gradients),
                     original_loss=float(reference_loss), gradient_exact=True, noise_and_RNG_exact=True,
                     training_generation_hook_scoped=True, precision_changed=False,
                     original_input_dtype=str(input_dtype), dataset_sha256=data.manifest_sha,
                     statistics_sha256=data.statistics_sha,
                     reference_memory_route='CORE_UNUSED_LOGITS1_CPU_ACTIVATIONS_2G_V2')
        atomic_torch(root / 'feature_cache.pt', dict(rows=rows, proof=proof, input_dtype=input_dtype))
        write(root / 'FEATURE_CACHE_QUALIFICATION.json', dict(proof, cache_sha256=digest(root / 'feature_cache.pt')))
        return proof
    finally:
        audit.close()
        model.zero_grad(set_to_none=True)


def save_checkpoint(path, model, contract, optimizer, scheduler, step, recipe, cache_sha, frozen):
    from scripts.pi0292.durable import write
    from scripts.pi0294.native_train import tree_identity, rng_identity
    path = Path(path)
    path.mkdir()
    atomic_torch(path / 'target_parameters.pt', {n: model.get_parameter(n).detach().cpu() for n in contract.trainable_names})
    atomic_torch(path / 'optimizer.pt', optimizer.state_dict())
    atomic_torch(path / 'scheduler.pt', scheduler.state_dict())
    atomic_torch(path / 'rng.pt', rng_state())
    write(path / 'STATE_IDENTITY.json', dict(parameters=state_identity(model, contract.trainable_names),
        optimizer=tree_identity(optimizer.state_dict()), scheduler=scheduler.state_dict(), RNG=rng_identity(), cursor=step * 32))
    write(path / 'FROZEN_PARENT_TENSORS.json', frozen)
    write(path / 'CHECKPOINT_COMPLETE.json', dict(schema=MEMORY, step=step, cursor=step * 32, recipe=recipe,
        feature_cache_sha256=cache_sha, parameter_contract=contract.describe(),
        files={p.name: digest(p) for p in path.iterdir() if p.is_file()}, scaler='NOT_APPLICABLE_BF16'))


def load_checkpoint(path, model, contract, optimizer, scheduler, recipe, cache_sha, frozen):
    import torch
    path = Path(path)
    saved = json.loads((path / 'CHECKPOINT_COMPLETE.json').read_text())
    need(saved['schema'] == MEMORY and saved['recipe'] == recipe and saved['feature_cache_sha256'] == cache_sha,
         'SAME_CACHE_DATA_MODEL_AND_SCHEDULE')
    need(saved['parameter_contract'] == contract.describe(), 'SAME_PARAMETER_CONTRACT')
    need(json.loads((path / 'FROZEN_PARENT_TENSORS.json').read_text()) == frozen, 'SAME_FROZEN_PARENT')
    need(all(digest(path / n) == h for n, h in saved['files'].items()), 'ALL_CHECKPOINT_MEMBERS')
    values = torch.load(path / 'target_parameters.pt', map_location='cpu', weights_only=True)
    need(set(values) == set(contract.trainable_names), 'COMPLETE_TRAINABLE_PARAMETERS')
    with torch.no_grad():
        for name, value in values.items():
            target = model.get_parameter(name)
            need(target.dtype == value.dtype and target.shape == value.shape, 'PRESERVED_PARAMETER_DTYPE')
            target.copy_(value)
    optimizer.load_state_dict(torch.load(path / 'optimizer.pt', map_location='cpu', weights_only=False))
    scheduler.load_state_dict(torch.load(path / 'scheduler.pt', map_location='cpu', weights_only=False))
    restore_rng(torch.load(path / 'rng.pt', map_location='cpu', weights_only=False))
    from scripts.pi0294.native_train import tree_identity, rng_identity
    actual = dict(parameters=state_identity(model, contract.trainable_names), optimizer=tree_identity(optimizer.state_dict()),
                  scheduler=scheduler.state_dict(), RNG=rng_identity(), cursor=saved['step'] * 32)
    need(actual == json.loads((path / 'STATE_IDENTITY.json').read_text()), 'ACTUAL_ALL_STATE_RESTORED')
    return saved['step']


def train_cached(model, contract, root, cache, recipe, endpoint, guard, resume=None):
    import torch
    from transformers import Trainer, set_seed
    from scripts.pi0292.durable import write
    from scripts.pi0294.generalization.minimal_adapter import frozen_fingerprint
    from scripts.pi0294.native_train import NativeNoiseAudit, tree_identity
    from scripts.pi0294.e1.trainer import learning_rate_multiplier
    from scripts.pi0294.multitask.activation_offload import SavedActivationOffload, context
    from .target_gr00t_core_v1 import activation_policy
    import time
    root = Path(root)
    packet = torch.load(cache, map_location='cpu', weights_only=False)
    need(packet['proof']['gradient_exact'] and len(packet['rows']) == 4, 'REAL_QUALIFIED_FIXED_FEATURE_CACHE')
    cache_sha = digest(cache)
    rows, input_dtype = packet['rows'], packet['input_dtype']
    need(input_dtype == model.dtype and packet['proof']['dataset_sha256'] == recipe['dataset_sha256']
         and packet['proof']['statistics_sha256'] == recipe['statistics_sha256'], 'SAME_INPUT_DTYPE_DATA_STATISTICS')
    model.backbone.to('cpu')
    model.action_head.to('cuda')
    model.eval()
    frozen = frozen_fingerprint(model)
    # Follow the pinned Trainer's decay grouping, with the same AdamW defaults.
    decay = set(Trainer.get_decay_parameter_names(None, model))
    groups = [dict(params=[model.get_parameter(n) for n in contract.trainable_names if (n in decay) == use_decay],
                   weight_decay=recipe['weight_decay'] if use_decay else 0.) for use_decay in (True, False)]
    optimizer = torch.optim.AdamW(groups, lr=recipe['learning_rate'], betas=(.9, .999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_multiplier(step, 500, 8))
    set_seed(recipe['seed'])
    step = load_checkpoint(resume, model, contract, optimizer, scheduler, recipe, cache_sha, frozen) if resume else 0
    start_step = step
    before = state_identity(model, contract.trainable_names)
    audit = NativeNoiseAudit(model)
    storage = SavedActivationOffload(model, policy=activation_policy())
    try:
        while step < endpoint:
            guard()
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for offset in range(32):
                cursor = step * 32 + offset
                index = sample_index(cursor, len(rows), recipe['seed'])
                row = rows[index]
                with context(storage), torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model.action_head(feature_packet(row), action_packet(model, row, input_dtype))
                    loss = check_loss(output)
                    (loss / 32).backward()
                need(audit.verified and not audit.active, 'AUDITED_REAL_ACTION_BACKWARD')
                losses.append(float(loss.detach()))
                with (root / 'SAMPLE_TRACE.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(cursor=cursor, index=index, identity=row['identity'],
                        input_identity=row['input_identity'], native_noise=audit.record)) + '\n')
                del loss, output
            need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in contract.parameters()), 'FINITE_GRADIENTS')
            torch.nn.utils.clip_grad_norm_(contract.parameters(), 1.)
            used_lr = optimizer.param_groups[0]['lr']
            optimizer.step()
            scheduler.step()
            step += 1
            current = state_identity(model, contract.trainable_names)
            progress = dict(step=step, cursor=step * 32, optimizer_calls=step - start_step,
                learning_rate_used=used_lr, next_learning_rate=optimizer.param_groups[0]['lr'],
                native_loss=sum(losses) / 32, action_parameters_changed=current != before,
                wall_s=time.monotonic() - started, epoch=time.time(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
            with (root / 'OPTIMIZER_STEPS.jsonl').open('a') as stream:
                stream.write(json.dumps(progress) + '\n');stream.flush();os.fsync(stream.fileno())
            write(root / 'PROGRESS.json', progress)
            if step <= 4 or step == endpoint or step % 250 == 0:
                need(frozen_fingerprint(model) == frozen, 'FROZEN_MODEL_UNCHANGED')
                save_checkpoint(root / ('checkpoint-' + str(step)), model, contract, optimizer, scheduler,
                                step, recipe, cache_sha, frozen)
        need(state_identity(model, contract.trainable_names) != before, 'NONZERO_LEARNED_PARAMETER_CHANGE')
        # Exercise train -> generation -> train-scope cleanup without moving the frozen VLM.
        from scripts.pi0294.native_train import rng_identity
        before_probe = rng_identity()
        with torch.random.fork_rng(devices=[0]), torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            generated = model.action_head.get_action(feature_packet(rows[0]), action_packet(model, rows[0], input_dtype))
        need(not audit.active and audit.pending is None and rng_identity() == before_probe, 'POST_TRAIN_GENERATION_SCOPE_AND_RNG')
        write(root / 'RESULT.json', dict(status='REAL_GR00T_CACHED_ACTION_DIAGNOSTIC_COMPLETE',
            global_step=step, cursor=step * 32, optimizer_calls=step - start_step,
            nonzero_parameters_changed=True, frozen_parameters_unchanged=True,
            native_generation=tree_identity(generated), training_generation_audit_passed=True,
            memory_route=MEMORY, formal_model=False, native_task_conditions=0))
        return root / ('checkpoint-' + str(step))
    finally:
        audit.close()


def compare_four(continuous, resumed, split):
    def identity(path):return json.loads((Path(path) / 'STATE_IDENTITY.json').read_text())
    left, right = identity(continuous), identity(resumed)
    need(left == right and left['cursor'] == 128, 'FOUR_CONTINUOUS_EQUALS_FRESH_TWO_PLUS_TWO_ALL_STATE')
    need(identity(split)['cursor'] == 64, 'SAVED_TWO_STEP_BOUNDARY')
    return dict(status='PASS_REAL_FOUR_VERSUS_TWO_PLUS_TWO_EXACT', model=True, optimizer=True,
                scheduler=True, RNG=True, sampler_cursor=128, precision_relaxed=False)
