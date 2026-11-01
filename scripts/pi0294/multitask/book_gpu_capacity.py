"""Explicit entry for a finite small-GPU capacity experiment, never admission.

The normal worker retains its 24GiB entry rule. Only an identified ten-update
capacity run may measure the same model, images, batch and full AdamW restore
on a smaller device. Its measured peak/headroom determines future admission.
"""
GIB = 1024**3
REQUEST = dict(schema='N17_SMALL_GPU_NATIVE10_CAPACITY_V1', optimizer_updates=10,
    microbatch=1, accumulation=32, effective_batch=32, minimum_headroom_bytes=GIB,
    training_admission=False, automatic_retry=False)
INFERENCE_REQUEST = dict(schema='N17_SMALL_GPU_SAVED_TRAIN32_INFERENCE_V1',
    optimizer_updates=0, physics_integrations=0, generation_queries=96,
    minimum_headroom_bytes=GIB, training_admission=False, automatic_retry=False)


def entry_requirement(manifest):
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
