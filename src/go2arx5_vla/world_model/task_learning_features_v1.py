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


def inference_inputs(inputs):
    """Free generation has no training action/RTC prefix or expert label."""
    values = copy.deepcopy(inputs)
    values.pop('action', None)
    values.pop('action_mask', None)
    return values


def action_packet(model, row, input_dtype, *, generation=False):
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    values = model.action_head.prepare_input(inference_inputs(row['inputs']) if generation else copy.deepcopy(row['inputs']))
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
            original = model.get_action(inference_inputs(inputs))['action_pred'].detach().cpu()
            torch.manual_seed(seed)
            cached = model.action_head.get_action(feature_packet(row), action_packet(model, row, input_dtype, generation=True))['action_pred'].detach().cpu()
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
            model.action_head.get_action(feature_packet(rows[0]), action_packet(model, rows[0], input_dtype, generation=True))
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


def measure_cached(model, packet, data, folder, step, guard):
    """Matched original decode on every FIT4 row; preserve the training RNG."""
    import numpy as np
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from scripts.pi0292.durable import write
    from scripts.pi0292.eval_contract import error_metrics
    from scripts.pi0294.e1.artifacts import atomic_npz
    from scripts.pi0294.native_train import tensor_identity, tree_identity, rng_identity
    folder = Path(folder)
    folder.mkdir()
    need(len(data) == len(packet['rows']) == 4, 'EVERY_FIT4_GENERATION_DENOMINATOR')
    before, records = rng_state(), []
    before_identity = rng_identity()
    try:
        for index, row in enumerate(packet['rows']):
            guard()
            need(data.identity(index) == row['identity'], 'MATCHED_CACHE_OBSERVATION_IDENTITY')
            inputs = data.processor.collator([data[index]['processed']])['inputs']
            need(tree_identity(inputs) == row['input_identity'], 'MATCHED_ORIGINAL_PROCESSOR_INPUT')
            state, expert, _, _ = data.raw(index)
            noise = []
            def capture(module, args):
                if not noise:noise.append(tensor_identity(args[0]))
            hook = model.action_head.action_encoder.register_forward_pre_hook(capture)
            seed = 2026110800 + index
            try:
                torch.manual_seed(seed)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    value = model.action_head.get_action(feature_packet(row),
                        action_packet(model, row, packet['input_dtype'], generation=True))['action_pred']
                    normalized = value.detach().float().cpu().numpy()
            finally:
                hook.remove()
            decoded = data.processor.decode_action(normalized, EmbodimentTag.NEW_EMBODIMENT,
                {'eef_9d':state[None,None,:9], 'gripper_position':state[None,None,9:]})
            physical = np.concatenate([decoded[k] for k in ('eef_9d', 'gripper_position')], axis=-1)[0]
            need(physical.shape == (40,10) and np.isfinite(physical).all() and len(noise) == 1,
                 'FULL_FINITE_ORIGINAL_SINGLE_DECODE')
            episode = data.rows[index][0]
            file = folder / (episode['id'] + '.npz')
            atomic_npz(file, physical_H40=physical, normalized=normalized, expert_policy_H40=expert,
                       actual_state10=state, noise_seed=np.asarray(seed, dtype=np.int64))
            records.append(dict(episode=episode['id'], command=episode['command'], seed=seed, noise=noise[0],
                model_global_step=step, errors=error_metrics(physical, expert, state), sha256=digest(file),
                Processor_decode_calls=1, original_processed_input_identity=row['input_identity']))
    finally:
        restore_rng(before)
    need(rng_identity() == before_identity, 'FIT4_PROBES_PRESERVE_ALL_TRAINING_RNG')
    write(folder / 'RESULT.json', dict(records=records, actual_VLA_queries=len(records),
        optimizer_stream_RNG_preserved=True, cache_equivalence_qualified=True,
        native_task_success='NOT_MEASURED_BY_OPEN_LOOP_DIAGNOSTIC'))
    return records


def train_cached(model, contract, root, cache, recipe, endpoint, guard, resume=None, evaluation_data=None):
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
        if evaluation_data is not None:
            measure_cached(model, packet, evaluation_data, root / 'before', step, guard)
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
            if evaluation_data is not None and step in (50, 100, endpoint):
                measure_cached(model, packet, evaluation_data,
                               root / ('after' if step == endpoint else 'probe-' + str(step)), step, guard)
        need(state_identity(model, contract.trainable_names) != before, 'NONZERO_LEARNED_PARAMETER_CHANGE')
        # Exercise train -> generation -> train-scope cleanup without moving the frozen VLM.
        from scripts.pi0294.native_train import rng_identity
        before_probe = rng_identity()
        with torch.random.fork_rng(devices=[0]), torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            generated = model.action_head.get_action(feature_packet(rows[0]), action_packet(model, rows[0], input_dtype, generation=True))
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


def reconstruct_original(model, contract, data, cache, checkpoint, recipe, output, trained_probes, guard):
    """Fresh parent plus learned delta, original full backbone/free generation.

    The cache is used only for the matched equality check. It is never used by
    the reconstructed full model. This does not claim a deployed task policy.
    """
    import torch
    import numpy as np
    from gr00t.data.embodiment_tags import EmbodimentTag
    from scripts.pi0292.durable import write
    from scripts.pi0294.generalization.minimal_adapter import frozen_fingerprint
    from scripts.pi0294.native_train import tree_identity,NativeNoiseAudit
    from scripts.pi0294.e1.artifacts import atomic_npz
    from .target_gr00t_core_v1 import trim_unused_logits
    checkpoint,output=Path(checkpoint),Path(output)
    saved=json.loads((checkpoint/'CHECKPOINT_COMPLETE.json').read_text())
    need(saved['schema']==MEMORY and saved['step']==200 and saved['cursor']==6400
         and saved['recipe']==recipe and saved['feature_cache_sha256']==digest(cache),
         'ACTUAL_COMPLETED_FIT200_CACHE_AND_RECIPE')
    need(all(digest(checkpoint/name)==pin for name,pin in saved['files'].items()), 'COMPLETE_FIT200_MEMBERS')
    need(saved['parameter_contract']==contract.describe()
        and json.loads((checkpoint/'FROZEN_PARENT_TENSORS.json').read_text())==frozen_fingerprint(model),
        'FRESH_ORIGINAL_FROZEN_PARENT_AND_ARCHITECTURE')
    before=state_identity(model,contract.trainable_names)
    values=torch.load(checkpoint/'target_parameters.pt',map_location='cpu',weights_only=True)
    need(set(values)==set(contract.trainable_names),'ALL_FIT200_TRAINABLE_PARAMETERS')
    with torch.no_grad():
        for name,value in values.items():
            target=model.get_parameter(name)
            need(target.dtype==value.dtype and target.shape==value.shape and bool(torch.isfinite(value).all()),
                 'FINITE_ORIGINAL_DTYPE_LEARNED_PARAMETER')
            target.copy_(value)
    actual=state_identity(model,contract.trainable_names)
    need(actual!=before and actual==json.loads((checkpoint/'STATE_IDENTITY.json').read_text())['parameters'],
         'LEARNED_FIT200_DELTA_INSTALLED_IN_FRESH_MODEL_EXACTLY')
    packet=torch.load(cache,map_location='cpu',weights_only=False)
    need(len(packet['rows'])==len(data)==4,'EXACT_FOUR_REAL_DIAGNOSTIC_INPUTS')
    trim_unused_logits(model)  # The previously qualified official unused-logit option.
    model.to('cuda').eval()
    audit=NativeNoiseAudit(model)
    records=[]
    try:
        for index,row in enumerate(packet['rows']):
            guard()
            inputs=data.processor.collator([data[index]['processed']])['inputs']
            need(tree_identity(inputs)==row['input_identity'],'SAME_RECONSTRUCTED_PROCESSOR_INPUTS')
            seed=2026110800+index
            with torch.random.fork_rng(devices=[0]),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                torch.manual_seed(seed)
                original=model.get_action(inference_inputs(inputs))['action_pred'].detach().cpu()
                torch.manual_seed(seed)
                cached=model.action_head.get_action(feature_packet(row),
                    action_packet(model,row,packet['input_dtype'],generation=True))['action_pred'].detach().cpu()
            need(torch.equal(original,cached),'LEARNED_FULL_ORIGINAL_MODEL_EQUALS_CACHED_HEAD_EXACT')
            state,expert,_,_=data.raw(index);episode=data.rows[index][0]['id']
            normalized=original.float().numpy()
            decoded=data.processor.decode_action(normalized,EmbodimentTag.NEW_EMBODIMENT,
                {'eef_9d':state[None,None,:9],'gripper_position':state[None,None,9:]})
            physical=np.concatenate([decoded[k]for k in ('eef_9d','gripper_position')],axis=-1)[0]
            with np.load(Path(trained_probes)/(episode+'.npz'),allow_pickle=False)as trained:
                need(np.array_equal(normalized,trained['normalized'])
                    and np.array_equal(physical,trained['physical_H40']), 'FRESH_MODEL_EQUALS_ACTUAL_FIT200_GENERATION')
            path=output/(episode+'.npz')
            atomic_npz(path,normalized=normalized,physical_H40=physical,expert_policy_H40=expert,
                actual_state10=state,noise_seed=np.asarray(seed,dtype=np.int64))
            records.append(dict(episode=episode,seed=seed,normalized_H40_exact=True,
                physical_H40_exact=True,output_sha256=digest(path),Processor_decode_calls=1))
        need(not audit.active and audit.pending is None,'FRESH_ORIGINAL_INFERENCE_OUTSIDE_TRAINING_AUDIT')
        proof=dict(status='PASS_FRESH_ORIGINAL_GR00T_PARENT_PLUS_LEARNED_FIT200_DELTA',records=records,
            source_checkpoint_sha256=digest(checkpoint/'CHECKPOINT_COMPLETE.json'),
            full_original_backbone_generations=4,cache_comparison_generations=4,
            all132_H40_outputs_bit_exact=True,physical_outputs_bit_exact=True,precision_relaxed=False,
            optimizer_updates=0,physics_integrations=0,native_task_conditions=0,
            standalone_policy=False,formal_model=False,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(output/'RESULT.json',proof)
        return proof
    finally:audit.close()
