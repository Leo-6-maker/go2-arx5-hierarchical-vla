"""Private intermediate tensors for the first four of the fixed40 queries.

The full original40-query inference still runs. The trace changes no tensors,
precision flags, weights or tolerances and never qualifies a native policy.
"""
from contextlib import contextmanager
from collections.abc import Mapping
import json
from pathlib import Path

MODE = 'MATCHED_FIXED40_FIRST4_INTERMEDIATE_TENSORS_V1'
QUERIES = 4
MAX_BYTES = 2 * 1024**3


def validate(spec):
    d = spec.get('tensor_trace')
    if d is None:
        return None
    from .target_gr00t_profile_v1 import need
    from .storage import digest
    routes = {
        'CORE_FUSION_NEURAL': ('wm02-task-learning-package-core-v1', 'TASK_LEARNING_PACKAGE_CORE_RECONSTRUCTION_V1'),
        'HOPPER_SINGLE_GPU': ('TASK_LEARNING_HOPPER_TENSOR_TRACE_V1', 'TASK_LEARNING_HOPPER_TENSOR_TRACE_V1'),
    }
    need(spec.get('diagnostic_only') is True and spec.get('numerical_diagnostic') is None
         and routes.get(spec.get('execution_site')) == (spec.get('profile'), spec.get('kind'))
         and spec.get('physics') is False and spec.get('device') == 'cuda'
         and spec.get('max_queries') == 40
         and all(spec.get(k) == 0 for k in ('GR00T_updates', 'WM_updates', 'Actor_updates')),
         'TRACE_ONLY_COMPLETE40_INFERENCE_WITHOUT_UPDATES_OR_OTHER_ABLATION')
    need(set(d) == {'mode', 'reference_result'} and d['mode'] == MODE,
         'FIXED_FIRST4_TRACE_PROTOCOL')
    pin = d['reference_result']; p = Path(pin['path'])
    need(digest(p) == pin['sha256'], 'PINNED_COMPLETE40_TRACE_REFERENCE')
    result = json.loads(p.read_bytes())
    need(result.get('actual_VLA_queries') == result.get('Processor_decode_calls') == 40
         and result.get('model_identity', {}).get('candidate_package', {}).get('sha256')
         == spec['candidate_package']['sha256'],
         'SAME_ACTUAL40_PACKAGE_TRACE_REFERENCE')
    return d


@contextmanager
def trace(spec, output, model):
    diagnostic = validate(spec)
    if diagnostic is None:
        yield
        return
    import numpy as np
    import torch
    from .storage import write_json, digest
    from .target_gr00t_profile_v1 import need
    need(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True,
         'TRACE_ORIGINAL_BF16_REDUCTION_SETTING')
    folder = Path(output) / 'intermediate-trace'
    folder.mkdir(exist_ok=False)
    head = model.action_head
    modules = dict(backbone=model.backbone, vlln=head.vlln,
                   state_encoder=head.state_encoder, action_encoder=head.action_encoder,
                   denoiser=head.model, action_decoder=head.action_decoder)
    if getattr(head, 'vl_self_attention', None) is not None:
        modules['vl_self_attention'] = head.vl_self_attention
    handles = []; rows = []; query = -1; total = 0; calls = {}; failure = None; complete = False

    def save(name, stage, value):
        nonlocal total
        if query >= QUERIES:
            return
        need(query >= 0, 'BACKBONE_STARTS_EACH_TRACED_QUERY')
        tensors = {}; structure = []

        def collect(v, path):
            if torch.is_tensor(v):
                cpu = v.detach().cpu()
                array = (cpu.float() if cpu.dtype == torch.bfloat16 else cpu).numpy().copy()
                need(array.dtype.kind != 'O' and array.nbytes <= 256 * 1024**2,
                     'COMPLETE_BOUNDED_PRIVATE_TRACE_TENSOR')
                key = 'tensor' + str(len(tensors)).zfill(4)
                tensors[key] = array
                structure.append(dict(key=key, path=path, dtype=str(v.dtype), shape=list(v.shape)))
            elif isinstance(v, Mapping):
                for k in sorted(v): collect(v[k], path + [str(k)])
            elif isinstance(v, (tuple, list)):
                for k, item in enumerate(v): collect(item, path + [str(k)])
            else:
                need(v is None or type(v) in (str, bool, int, float), 'KNOWN_TRACE_OUTPUT_STRUCTURE')
                structure.append(dict(path=path, scalar=v))

        collect(value, [])
        size = sum(v.nbytes for v in tensors.values())
        need(total + size <= MAX_BYTES, 'FINITE_TRACE_STORAGE_WITHOUT_TRUNCATION')
        total += size
        key = (query, name, stage); call = calls.get(key, 0); calls[key] = call + 1
        file = folder / f'q{query:02d}-{name}-{stage}-{call:02d}.npz'
        with file.open('xb') as stream: np.savez(stream, **tensors)
        rows.append(dict(query_index=query, module=name, stage=stage, call=call,
                         file=file.name, sha256=digest(file), tensor_bytes=size, structure=structure,
                         active_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                         active_allow_bf16_reduced_precision_reduction=
                         torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction))

    def pre(name):
        def hook(module, args, kwargs):
            nonlocal query
            if name == 'backbone': query += 1
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
        need(query + 1 == 40, 'ALL40_ORIGINAL_QUERIES_STILL_EXECUTED')
        for q in range(QUERIES):
            for name in modules:
                expected = 4 if name in ('action_encoder', 'denoiser', 'action_decoder') else 1
                need(calls.get((q, name, 'input')) == calls.get((q, name, 'output')) == expected,
                     'COMPLETE_FIRST4_FOUR_STEP_MODULE_BOUNDARIES')
        complete = True
    except BaseException as error:
        failure = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        for handle in reversed(handles): handle.remove()
        write_json(folder / 'TRACE_MANIFEST.private.json', dict(mode=MODE, diagnostic_only=True,
            complete=complete, failure=failure, original_query_count=query + 1,
            traced_query_count=min(query + 1, QUERIES), tensor_bytes=total, maximum_bytes=MAX_BYTES,
            reference_result=diagnostic['reference_result'], records=rows,
            numeric_flags_changed=False, tensors_mutated=False, package_qualified=False))
