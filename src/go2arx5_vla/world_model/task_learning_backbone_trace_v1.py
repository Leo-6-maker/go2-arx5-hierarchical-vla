"""First-query frozen-backbone boundaries inside the original fixed40 diagnostic."""
from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path

from .task_learning_tensor_trace_v1 import validate as validate_outer

MODE = 'FIXED40_FIRST1_FROZEN_BACKBONE_BOUNDARIES_V1'
MAX_BYTES = 2 * 1024**3


def validate(spec):
    extra = spec.get('backbone_trace')
    if extra is None:
        return None
    from .target_gr00t_profile_v1 import need
    need(extra == {'mode': MODE} and validate_outer(spec) is not None,
         'BACKBONE_TRACE_ONLY_INSIDE_ORIGINAL_COMPLETE40_DIAGNOSTIC')
    return extra


def selected_modules(model):
    """Version-pinned Qwen3-VL semantic boundaries, not replacement operators."""
    qwen = model.backbone.model
    visual, language = qwen.visual, qwen.language_model
    selected = {'backbone': model.backbone,
                'visual.patch_embed': visual.patch_embed,
                'visual.patch_embed.proj': visual.patch_embed.proj,
                'visual.rotary_pos_emb': visual.rotary_pos_emb,
                'visual.merger': visual.merger,
                'language.embed_tokens': language.embed_tokens,
                'language.rotary_emb': language.rotary_emb,
                'language.norm': language.norm}
    for group, layers in (('visual.blocks', visual.blocks),
                          ('visual.deepstack_merger_list', visual.deepstack_merger_list),
                          ('language.layers', language.layers)):
        for i, layer in enumerate(layers):
            selected[f'{group}.{i}'] = layer
    for group, layer, names in (
        ('visual.blocks.0', visual.blocks[0], ('norm1', 'attn', 'norm2', 'mlp')),
        ('language.layers.0', language.layers[0],
         ('input_layernorm', 'self_attn', 'post_attention_layernorm', 'mlp'))):
        for name in names:
            selected[group + '.' + name] = getattr(layer, name)
    return selected


def constants(model):
    qwen = model.backbone.model
    values = {'buffers': dict(model.backbone.named_buffers()),
              'vision_patch_weight': qwen.visual.patch_embed.proj.weight,
              'vision_position_weight': qwen.visual.pos_embed.weight,
              'text_original_inv_freq': qwen.language_model.rotary_emb.original_inv_freq}
    return values


def runtime_metadata(model, torch):
    import transformers
    qwen = model.backbone.model
    source = Path(inspect.getsourcefile(type(qwen.visual)))
    return dict(torch_version=torch.__version__, transformers_version=transformers.__version__,
                CUDA_version=torch.version.cuda, cuDNN_version=torch.backends.cudnn.version(),
                HF_Qwen_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                visual_config=qwen.visual.config.to_dict(),
                language_config=qwen.language_model.config.to_dict(),
                trainable_backbone_parameters=sum(p.numel() for p in model.backbone.parameters()
                                                  if p.requires_grad),
                trainable_visual_parameters=sum(p.numel() for p in qwen.visual.parameters() if p.requires_grad),
                trainable_language_parameters=sum(p.numel() for p in qwen.language_model.parameters()
                                                  if p.requires_grad))


@contextmanager
def trace(spec, output, model):
    if validate(spec) is None:
        yield
        return
    import numpy as np
    import torch
    from .storage import digest, write_json
    from .target_gr00t_profile_v1 import need
    folder = Path(output) / 'backbone-internal-trace'
    folder.mkdir(exist_ok=False)
    modules = selected_modules(model)
    metadata = runtime_metadata(model, torch)
    # Original Hopper retains requires_grad on its unused vocabulary head.
    # Preserve those loaded flags; the actual visual/language feature stacks are frozen.
    need(metadata['trainable_visual_parameters'] == metadata['trainable_language_parameters'] == 0,
         'FROZEN_BACKBONE_FEATURE_STACKS_DIAGNOSTIC_ONLY')
    need(metadata['HF_Qwen_source_sha256'] ==
         '685b10cbc48556343df4e159199c5884d02db9ad227640f0f4fbe75ddf0b2a99',
         'ACTUAL_PINNED_QWEN_IMPLEMENTATION_FOR_INTERNAL_TRACE')
    handles = []; rows = []; calls = {}; query = -1; total = 0; failure = None; complete = False

    def save(name, stage, value):
        nonlocal total
        if query != 0:
            return
        arrays = {}; structure = []

        def collect(item, path):
            if torch.is_tensor(item):
                cpu = item.detach().cpu()
                array = (cpu.float() if cpu.dtype == torch.bfloat16 else cpu).numpy().copy()
                need(array.dtype.kind != 'O' and array.nbytes <= 256 * 1024**2,
                     'BOUNDED_COMPLETE_BACKBONE_TENSOR')
                key = f'tensor{len(arrays):04d}'; arrays[key] = array
                structure.append(dict(key=key, path=path, dtype=str(item.dtype), shape=list(item.shape)))
            elif isinstance(item, Mapping):
                for key in sorted(item): collect(item[key], path + [str(key)])
            elif isinstance(item, (tuple, list)):
                for i, child in enumerate(item): collect(child, path + [str(i)])
            elif type(item).__module__.startswith('transformers.cache_utils'):
                # The first fresh query has no prior episode KV. Keep explicit
                # cache metadata, without duplicating every previous layer KV.
                structure.append(dict(path=path, cache_type=type(item).__name__,
                                      sequence_length=int(item.get_seq_length()),
                                      KV_tensors_recorded=False))
            else:
                need(item is None or type(item) in (str, int, float, bool),
                     'KNOWN_BACKBONE_TRACE_STRUCTURE')
                structure.append(dict(path=path, scalar=item))

        collect(value, [])
        size = sum(a.nbytes for a in arrays.values())
        need(total + size <= MAX_BYTES, 'BOUNDED_INTERNAL_TRACE_WITHOUT_TRUNCATION')
        total += size
        key = (name, stage); call = calls.get(key, 0); calls[key] = call + 1
        file = folder / f"{name.replace('.', '_')}-{stage}-{call:02d}.npz"
        with file.open('xb') as stream: np.savez(stream, **arrays)
        rows.append(dict(query_index=0, module=name, stage=stage, call=call,
                         file=file.name, sha256=digest(file), tensor_bytes=size,
                         structure=structure, active_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                         active_cuDNN_allow_tf32=torch.backends.cudnn.allow_tf32,
                         active_bf16_reduced_precision_reduction=
                         torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction))

    def pre(name):
        def hook(module, args, kwargs):
            nonlocal query
            if name == 'backbone':
                query += 1
                if query == 0: save('constants', 'load', constants(model))
            save(name, 'input', dict(args=args, kwargs=kwargs))
        return hook

    def post(name):
        def hook(module, args, value): save(name, 'output', value)
        return hook

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(pre(name), with_kwargs=True))
            handles.append(module.register_forward_hook(post(name)))
        yield
        need(query + 1 == 40 and calls.get(('constants', 'load')) == 1,
             'ORIGINAL40_QUERIES_AND_FIRST_QUERY_CONSTANTS')
        need(all(calls.get((name, 'input')) == calls.get((name, 'output')) == 1
                 for name in modules), 'COMPLETE_FIRST_QUERY_BACKBONE_MODULE_BOUNDARIES')
        complete = True
    except BaseException as error:
        failure = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        for handle in reversed(handles): handle.remove()
        write_json(folder / 'TRACE_MANIFEST.private.json', dict(mode=MODE,
            diagnostic_only=True, complete=complete, failure=failure,
            original_query_count=query + 1, traced_query_count=min(query + 1, 1),
            records=rows, tensor_bytes=total, maximum_bytes=MAX_BYTES,
            runtime=metadata, candidate_package_sha256=spec['candidate_package']['sha256'],
            numeric_flags_changed=False, tensors_mutated=False, package_qualified=False,
            cache_scope='Fresh first query; KV cache metadata only, no KV tensor duplication.'))
