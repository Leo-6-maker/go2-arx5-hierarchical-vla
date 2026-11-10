"""One-factor, inference-only diagnosis after a preserved Core40 mismatch.

This does not qualify a package or change any numerical acceptance threshold.
The original entry is unchanged unless the fresh spec declares this diagnosis.
"""
from contextlib import contextmanager
import json
from pathlib import Path

MODE = 'CORE_BF16_FULL_REDUCTION_ABLATION_V1'


def validate(spec):
    from .task_learning_core_inference_v1 import PROFILE, KIND, MODE as INFERENCE
    from .target_gr00t_profile_v1 import need
    from .storage import digest
    d = spec.get('numerical_diagnostic')
    if d is None:
        return None
    need(spec.get('profile') == PROFILE and spec.get('kind') == KIND
         and spec.get('target_inference_mode') == INFERENCE
         and spec.get('physics') is False and spec.get('device') == 'cuda'
         and spec.get('diagnostic_only') is True
         and spec.get('max_queries') == 40
         and all(spec.get(k) == 0 for k in ('GR00T_updates', 'WM_updates', 'Actor_updates')),
         'INFERENCE_ONLY_REDUCTION_DIAGNOSIS')
    need(set(d) == {'mode', 'reference_result'} and d['mode'] == MODE,
         'EXPLICIT_ONE_FACTOR_REDUCTION_DIAGNOSIS')
    pin = d['reference_result']
    p = Path(pin['path'])
    need(digest(p) == pin['sha256'], 'PINNED_PRECEDING_CORE40_RESULT')
    result = json.loads(p.read_bytes())
    need(result.get('status') == 'ACTUAL_CORE_CONTINUOUS_TASK_SEALED_TRAIN40_INFERENCE'
         and result.get('actual_VLA_queries') == result.get('Processor_decode_calls') == 40
         and result.get('model_identity', {}).get('candidate_package') == spec['candidate_package'],
         'SAME_PACKAGE_AND_COMPLETE_CORE40_REFERENCE')
    return d


@contextmanager
def execution(spec, output):
    diagnostic = validate(spec)
    if diagnostic is None:
        yield
        return
    import torch
    from .storage import write_json
    from .target_gr00t_profile_v1 import need
    backend = torch.backends.cuda.matmul
    before = backend.allow_bf16_reduced_precision_reduction
    need(before is True, 'REDUCTION_DIAGNOSIS_REQUIRES_ORIGINAL_ENABLED_DEFAULT')
    record = dict(mode=MODE, diagnostic_only=True, package_qualified=False,
                  only_changed_backend_flag='allow_bf16_reduced_precision_reduction',
                  original_value=before, diagnostic_value=False,
                  input_autocast_dtype_unchanged='torch.bfloat16',
                  parameters_unchanged=True, tolerances_relaxed=False,
                  reference_result=diagnostic['reference_result'], failure=None)
    target = Path(output) / 'REDUCTION_DIAGNOSTIC_ACTUAL.private.json'
    backend.allow_bf16_reduced_precision_reduction = False
    try:
        need(backend.allow_bf16_reduced_precision_reduction is False,
             'FULL_BF16_REDUCTION_ACTUALLY_APPLIED')
        write_json(Path(output) / 'REDUCTION_DIAGNOSTIC_STARTED.private.json', record)
        yield
    except BaseException as error:
        record['failure'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        backend.allow_bf16_reduced_precision_reduction = before
        record['original_flag_restored_on_exit'] = backend.allow_bf16_reduced_precision_reduction is before
        write_json(target, record)
