"""Explicit FP32 portability diagnostic around the original get_action method."""
from contextlib import contextmanager

MODE = 'FP32_NO_TF32_FIXED40_DIAGNOSTIC_V1'


def validate(spec):
    entry = spec.get('inference_precision_diagnostic')
    if entry is None:
        return None
    from .target_gr00t_profile_v1 import need
    from .task_learning_tensor_trace_v1 import validate as original_trace
    need(entry == {'mode': MODE} and spec.get('backbone_trace') is None
         and original_trace(spec) is not None,
         'FP32_ONLY_ORIGINAL_FIXED40_DIAGNOSTIC_NOT_NATIVE_OR_COMBINED_ABLATION')
    return entry


@contextmanager
def precision(spec, output, model):
    if validate(spec) is None:
        yield
        return
    import torch
    from pathlib import Path
    from .storage import write_json
    from .target_gr00t_profile_v1 import need
    need({str(p.dtype) for p in model.parameters()} == {'torch.float32'}
         and 'get_action' not in model.__dict__, 'ORIGINAL_FP32_WEIGHTS_AND_UNWRAPPED_GET_ACTION')
    original = model.get_action
    before = dict(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32)
    calls = 0; observations = []; failure = None
    write_json(Path(output)/'FP32_DIAGNOSTIC_STARTED.private.json',
               dict(mode=MODE, diagnostic_only=True, original_flags=before, weight_dtype_changed=False))

    def invoke(*args, **kwargs):
        nonlocal calls
        saved = dict(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            # ResidentModel still owns input preparation, inference_mode, seed,
            # original four-step generation and the single Processor decode.
            with torch.autocast('cuda', enabled=False):
                result = original(*args, **kwargs)
                need(not torch.is_autocast_enabled('cuda')
                     and result['action_pred'].dtype == torch.float32,
                     'ACTUAL_FP32_GET_ACTION_WITHOUT_AUTOCAST')
                observations.append(dict(query_index=calls, cuda_autocast_enabled=False,
                    action_prediction_dtype=str(result['action_pred'].dtype),
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))
                calls += 1
                return result
        finally:
            torch.backends.cuda.matmul.allow_tf32 = saved['matmul']
            torch.backends.cudnn.allow_tf32 = saved['cudnn']

    model.get_action = invoke
    try:
        yield
        need(calls == 40, 'ALL40_ORIGINAL_QUERIES_IN_FP32_DIAGNOSTIC')
    except BaseException as error:
        failure = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        delattr(model, 'get_action')
        after = dict(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32)
        write_json(Path(output)/'FP32_DIAGNOSTIC_ACTUAL.private.json',
            dict(mode=MODE, diagnostic_only=True, original_queries=calls, failure=failure,
                 observations=observations, original_flags=before, final_flags=after,
                 original_flags_restored=before == after, original_method_restored=model.get_action == original,
                 weights_changed=False, precision_changed=True, package_qualified=False))
