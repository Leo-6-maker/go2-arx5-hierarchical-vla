"""Finite small-GPU measurements and checkpoint-specific inference admission.

The normal worker retains its 24GiB entry rule. Only an identified ten-update
capacity run may measure the same model, images, batch and full AdamW restore
on a smaller device. Training is never admitted by an inference certificate.
"""
GIB = 1024**3
REQUEST = dict(schema='N17_SMALL_GPU_NATIVE10_CAPACITY_V1', optimizer_updates=10,
    microbatch=1, accumulation=32, effective_batch=32, minimum_headroom_bytes=GIB,
    training_admission=False, automatic_retry=False)
INFERENCE_REQUEST = dict(schema='N17_SMALL_GPU_SAVED_TRAIN32_INFERENCE_V1',
    optimizer_updates=0, physics_integrations=0, generation_queries=96,
    minimum_headroom_bytes=GIB, training_admission=False, automatic_retry=False)


def runtime_inference_capacity(torch, previous, now):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    minimum = min(previous['minimum_free_bytes'], free)
    return dict(last_sample_monotonic=now, measurements=previous['measurements']+1,
        minimum_free_bytes=minimum, total_bytes=total,
        maximum_allocated_bytes=torch.cuda.max_memory_allocated(),
        maximum_reserved_bytes=torch.cuda.max_memory_reserved(), capacity_pass=minimum>=GIB)


def qualified_dev_entry(manifest, admission):
    """Use a closed native96 certificate for this exact checkpoint and GPU."""
    import hashlib
    import json
    from pathlib import Path
    from .book_phase_diagnostic import KIND
    expected = {'RUN_MANIFEST.json', 'RESULT.json', 'GPU_INFERENCE_CAPACITY.json', 'control/LEASE.json'}
    limits = dict(model_loads=1, optimizer_updates=0, physics_integrations=185000,
        generation_queries=800, RGB_pairs=6420, source_constructors=10)
    if (admission.get('schema') != 'N17_NATIVE96_CERTIFIED_DEV10_INFERENCE_V1'
            or set(admission.get('files', {})) != expected
            or manifest.get('kind') != 'V7_BOOK_FROZEN_DEV10'
            or any(manifest.get('limits', {}).get(k) != v for k, v in limits.items())
            or manifest.get('GPU_capacity_qualification') is not None
            or manifest.get('GPU_inference_qualification') is not None):
        raise PermissionError('EXACT_CERTIFIED_DEV10_NO_TRAINING')
    root = Path(admission['qualification_root'])
    if not root.is_absolute() or root.is_symlink():
        raise PermissionError('NATIVE96_CERTIFICATE_ROOT')
    values = {}
    for name, pin in admission['files'].items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise PermissionError('NATIVE96_CERTIFICATE_PATH')
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != pin:
            raise PermissionError('NATIVE96_CERTIFICATE_CHANGED:'+name)
        values[name] = json.loads(body)
    q, result, capacity, lease = (values[n] for n in
        ('RUN_MANIFEST.json', 'RESULT.json', 'GPU_INFERENCE_CAPACITY.json', 'control/LEASE.json'))
    if (q.get('kind') != KIND or q.get('GPU_inference_qualification') != INFERENCE_REQUEST
            or q.get('GPU_inference_admission') is not None
            or q.get('GPU_uuid') != manifest.get('GPU_uuid')
            or not manifest.get('delta_checkpoint_manifest_sha256')
            or q.get('delta_checkpoint_manifest_sha256') != manifest['delta_checkpoint_manifest_sha256']
            or result.get('evaluated_checkpoint_manifest_sha256') != manifest['delta_checkpoint_manifest_sha256']
            or result.get('status') != 'COMPLETE_MATCHED_TRAIN32_PHASE_DIAGNOSTIC' or result.get('error') is not None
            or any(result.get(k) != v for k, v in dict(model_load_attempts=1, generation_attempts=96,
                completed_generations=96, optimizer_updates=0, physics_integrations=0, formal_DEV_conditions=0).items())
            or lease.get('status') != 'CLOSED'
            or lease.get('manifest_sha256') != admission['files']['RUN_MANIFEST.json']
            or lease.get('result_sha256') != admission['files']['RESULT.json']
            or capacity.get('capacity_pass') is not True or capacity.get('completed_generations') != 96
            or capacity.get('minimum_free_bytes', 0) < GIB
            or len(capacity.get('measurements', [])) != 97):
        raise PermissionError('COMPLETE_CLOSED_SAME_GPU_CHECKPOINT_NATIVE96_REQUIRED')
    entry_requirement(q)
    return 13*GIB


def entry_requirement(manifest):
    admission = manifest.get('GPU_inference_admission')
    if admission is not None:
        return qualified_dev_entry(manifest, admission)
    request = manifest.get('GPU_capacity_qualification')
    inference = manifest.get('GPU_inference_qualification')
    if inference is not None:
        from .book_phase_diagnostic import KIND
        limits=manifest.get('limits',{})
        if (request is not None or inference!=INFERENCE_REQUEST or manifest.get('kind')!=KIND
                or any(limits.get(k)!=v for k,v in dict(model_loads=1,optimizer_updates=0,
                    physics_integrations=0,generation_queries=96,RGB_pairs=0,source_constructors=0).items())):
            raise PermissionError('EXACT_SAVED_TRAIN32_INFERENCE_NO_TRAINING_OR_PHYSICS')
        return 13*GIB
    if request is None:
        return 24*GIB
    recipe = manifest.get('recipe', {})
    if (request != REQUEST or manifest.get('kind') != 'V7_BOOK_M32_FINITE_LEARNING'
            or recipe.get('max_steps') != 10
            or recipe.get('capacity_qualification_only') is not True
            or recipe.get('limited_DiT_blocks') != 2
            or recipe.get('TRAIN_visits') != 320):
        raise PermissionError('EXACT_TEN_UPDATE_NATIVE_CAPACITY_ONLY')
    return 13*GIB
