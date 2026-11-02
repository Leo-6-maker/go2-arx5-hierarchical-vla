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
TRAINING_REQUEST = dict(schema='M32_LOSSLESS_CPU_NATIVE10_TRAINING_CAPACITY_V1',
    optimizer_updates=10, microbatch=1, accumulation=32, effective_batch=32,
    minimum_headroom_bytes=2*GIB, automatic_retry=False)


def qualified_training_entry(manifest):
    """Admit only the explicit memory path and a same-device full native10."""
    import hashlib,json
    from pathlib import Path
    from .activation_offload import POLICY
    from .book_m32_learning import is_plan,KIND
    from .book_bc_dagger import SCHEMA as BC_SCHEMA
    from .book_phase_boundary import PLAN_SCHEMA as PB_SCHEMA
    raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
    pin=hashlib.sha256(raw(POLICY)).hexdigest()
    p=manifest.get('recipe',{});limits=manifest.get('limits',{})
    request=manifest.get('GPU_training_qualification');admission=manifest.get('GPU_training_admission')
    if (manifest.get('kind')!=KIND or not is_plan(p) or p.get('schema')not in(BC_SCHEMA,PB_SCHEMA)
            or manifest.get('memory_execution_sha256')!=pin
            or manifest.get('sealed_inputs',{}).get('M32_MEMORY_EXECUTION.json')!=pin
            or manifest.get('training_GPU_entry_MiB')not in(15360,24576)
            or any(manifest.get(k)is not None for k in('GPU_capacity_qualification','GPU_inference_qualification','GPU_inference_admission'))
            or any(limits.get(k)!=v for k,v in dict(model_loads=1,optimizer_updates=p['max_steps'],physics_integrations=0,
                generation_queries=p['generation_queries'],RGB_pairs=0,source_constructors=0).items())):
        raise PermissionError('EXACT_SEALED_M32_LOSSLESS_TRAINING_ONLY')
    required=manifest['training_GPU_entry_MiB']*1024**2
    if request is not None:
        if request!=TRAINING_REQUEST or admission is not None or p['max_steps']!=10 or p['capacity_qualification_only']is not True:
            raise PermissionError('EXACT_LOSSLESS_NATIVE10_NOT_FORMAL_TRAINING')
        return required
    files={'RUN_MANIFEST.json','RESULT.json','M32_CAPACITY_RESULT.json','ACTUAL_DELTA_CHECKPOINT_RESTORE.json',
           'ACTUAL_ACTIVATION_OFFLOAD.json','control/LEASE.json'}
    if (p['max_steps']!=4096 or p['capacity_qualification_only']is not False or not isinstance(admission,dict)
            or admission.get('schema')!='M32_SAME_GPU_LOSSLESS_NATIVE10_CERTIFIED_TRAINING_V1'
            or set(admission.get('files',{}))!=files):
        raise PermissionError('CLOSED_SAME_GPU_LOSSLESS_NATIVE10_REQUIRED')
    root=Path(admission['qualification_root'])
    if not root.is_absolute()or root.is_symlink():raise PermissionError('NATIVE10_TRAINING_ROOT')
    values={}
    for name,digest in admission['files'].items():
        path=root/name
        if path.is_symlink()or not path.resolve().is_relative_to(root.resolve()):raise PermissionError('NATIVE10_TRAINING_PATH')
        body=path.read_bytes()
        if hashlib.sha256(body).hexdigest()!=digest:raise PermissionError('NATIVE10_TRAINING_CERTIFICATE_CHANGED:'+name)
        values[name]=json.loads(body)
    q,result,capacity,restore,copies,lease=(values[n]for n in('RUN_MANIFEST.json','RESULT.json','M32_CAPACITY_RESULT.json',
        'ACTUAL_DELTA_CHECKPOINT_RESTORE.json','ACTUAL_ACTIVATION_OFFLOAD.json','control/LEASE.json'))
    expected=dict(p,max_steps=10,TRAIN_visits=320,checkpoint_steps=[10],comparison_endpoint=10,
                  generation_queries=32,native_diagnostic_loss_forwards=32,capacity_qualification_only=True)
    if (q.get('GPU_uuid')!=manifest.get('GPU_uuid')or q.get('recipe')!=expected
            or any(q.get(k)!=manifest.get(k)for k in('checkpoint_manifest_sha256','delta_checkpoint_manifest_sha256','memory_execution_sha256'))
            or q.get('GPU_training_qualification')!=TRAINING_REQUEST
            or result.get('status')!='COMPLETE_BOOK5_FINITE_TARGET_LEARNING'or result.get('error')is not None
            or any(result.get(k)!=v for k,v in dict(model_load_attempts=1,optimizer_updates=10,TRAIN_visits=320,
                generation_attempts=32,completed_generations=32,physics_integrations=0,DEV_conditions=0).items())
            or capacity.get('capacity_pass')is not True or capacity.get('minimum_observed_free_bytes',0)<2*GIB
            or capacity.get('actual_optimizer_updates')!=10 or capacity.get('saved_and_restored')is not True
            or any(restore.get(k)!=v for k,v in dict(optimizer_updates=10,sampler_cursor=320,model_reconstruction=True,
                scheduler_restored=True,RNG_restored=True,frozen_parent_unchanged=True).items())
            or copies.get('policy')!=POLICY or copies.get('copied_tensors',0)<=0 or copies.get('restored_tensors',0)<=0
            or lease.get('status')!='CLOSED'or lease.get('manifest_sha256')!=admission['files']['RUN_MANIFEST.json']
            or lease.get('result_sha256')!=admission['files']['RESULT.json']):
        raise PermissionError('COMPLETE_MATCHED_CLOSED_GPU_NATIVE10_AND_TWO_GIB_MARGIN')
    qualified_training_entry(q)
    return required


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
    if manifest.get('GPU_training_qualification')is not None or manifest.get('GPU_training_admission')is not None:
        return qualified_training_entry(manifest)
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
