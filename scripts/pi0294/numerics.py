"""Explicit numerical requalification; no loss, precision or tolerance changes."""
import os

MODE='DETERMINISTIC_CUDA_MATH_SDP_V1'


def prepare_environment(mode):
    if mode is None:return
    if mode!=MODE:raise PermissionError('ONLY_DECLARED_NUMERICAL_REQUALIFICATION_MODE')
    current=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if current not in (None,':4096:8'):
        raise PermissionError('UNEXPECTED_CUBLAS_WORKSPACE_CONFIGURATION')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'


def configure_torch(torch,mode):
    if mode is None:return dict(mode='UNCHANGED_NATIVE_CUDA_DEFAULT')
    prepare_environment(mode)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if (not torch.are_deterministic_algorithms_enabled()
        or torch.backends.cuda.flash_sdp_enabled()
        or torch.backends.cuda.mem_efficient_sdp_enabled()
        or torch.backends.cuda.cudnn_sdp_enabled()
        or not torch.backends.cuda.math_sdp_enabled()):
        raise PermissionError('DETERMINISTIC_NUMERICAL_CONFIGURATION_NOT_APPLIED')
    return dict(mode=MODE,deterministic_algorithms=True,unsupported_ops_raise=True,
        CUBLAS_WORKSPACE_CONFIG=':4096:8',attention_backend='PYTORCH_MATH_SDP',
        bf16_unchanged=True,masked_flow_matching_unchanged=True,tolerances_relaxed=False)
