"""Extract the teacher's own patch-embed weights for its offline constructor.

The upstream constructor otherwise downloads DINO before the complete pinned
OmniVGGT state overwrites it. This asset uses only that same final checkpoint.
It constructs no model and never initializes CUDA.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha

WEIGHT_SHA = 'c9c3772b9bbfc648fa95ebffb3d9ff856f21e9b9712685eb0f98add53897969d'


def run(spec, counts):
    import torch
    from safetensors import safe_open
    if torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('TEACHER_PREINIT_EXTRACTION_IS_CPU_ONLY')
    manifest = spec['teacher_manifest']
    if sha(manifest['path']) != manifest['sha256']:
        raise ValueError('TEACHER_ASSET_MANIFEST_CHANGED')
    assets = json.loads(Path(manifest['path']).read_bytes())
    weight = Path(spec['teacher_weight'])
    if (assets['revision'] != '4a0917605798f1e8a645153ab8cdfb12aa453354'
            or not assets['all_files_verified']):
        raise ValueError('EXACT_ALREADY_PINNED_TEACHER_REQUIRED')
    counts['full_weight_hash_passes'] += 1
    if sha(weight) != WEIGHT_SHA:
        raise ValueError('EXACT_ALREADY_PINNED_TEACHER_REQUIRED')
    prefix = 'aggregator.patch_embed.'
    state, rows = {}, []
    counts['checkpoint_open_calls'] += 1
    with safe_open(str(weight), framework='pt', device='cpu') as checkpoint:
        keys = sorted(k for k in checkpoint.keys() if k.startswith(prefix))
        if not keys:
            raise ValueError('FULL_TEACHER_PATCH_EMBED_MISSING')
        for key in keys:
            value = checkpoint.get_tensor(key)
            if value.device.type != 'cpu' or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('FINITE_ORIGINAL_FP32_PATCH_EMBED_REQUIRED')
            name = key[len(prefix):]
            state[name] = value
            rows.append(dict(key=name, shape=list(value.shape), dtype=str(value.dtype),
                numel=value.numel(), sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest()))
    root = Path(spec['registered_root'])/'preinit'
    root.mkdir(exist_ok=False)
    path = root/'dinov2_vitl14_reg4_pretrain.pth'
    with path.open('xb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    counts['extracted_tensor_leaves'] = len(rows)
    counts['extracted_tensor_bytes'] = sum(v.numel()*v.element_size() for v in state.values())
    counts['preinit_roundtrip_loads'] += 1
    restored = torch.load(path, map_location='cpu', weights_only=True)
    if set(restored) != set(state) or not all(torch.equal(state[k], restored[k]) for k in state):
        raise ValueError('CPU_PATCH_ASSET_SERIALIZATION_NOT_BIT_EXACT')
    del restored, state
    counts['full_weight_hash_passes'] += 1
    if sha(weight) != WEIGHT_SHA or torch.cuda.is_initialized():
        raise ValueError('EXTRACTION_CHANGED_PARENT_OR_INITIALIZED_CUDA')
    value = dict(schema='SLA_TEACHER_OWN_PATCH_EMBED_PREINIT_ASSET_V1', passes=True,
        source_teacher_weight_sha256=WEIGHT_SHA, source_teacher_manifest=manifest,
        file=path.name, file_sha256=sha(path), file_bytes=path.stat().st_size, arrays=rows,
        external_DINO_weights_used=False, teacher_model_constructions=0, teacher_forwards=0,
        SLA_updates=0, teacher_updates=0, CUDA_initialized=False, asset_roundtrip_bit_exact=True)
    write(root/'MANIFEST.private.json', value)
    return dict(schema='SLA_TEACHER_PREINIT_EXTRACTION_ACTUAL_V1', passes=True,
        teacher_preinit=dict(path=str(root/'MANIFEST.private.json'), sha256=sha(root/'MANIFEST.private.json')),
        actual=counts, teacher_model_constructions=0, teacher_forwards=0,
        CUDA_initialized=False, optimizer_updates=0, physical_requests=0)


if __name__ == '__main__':
    spec = json.loads(Path(sys.argv[1]).read_bytes())
    counts = dict(full_weight_hash_passes=0, checkpoint_open_calls=0, preinit_roundtrip_loads=0,
                  extracted_tensor_leaves=0, extracted_tensor_bytes=0)
    try:
        result = run(spec, counts)
    except BaseException as exc:
        write(Path(spec['registered_root'])/'FAILED.private.json', dict(error=repr(exc), actual=counts,
            teacher_model_constructions=0, teacher_forwards=0, optimizer_updates=0, physical_requests=0))
        raise
    write(Path(spec['registered_root'])/'RESULT.private.json', result)
