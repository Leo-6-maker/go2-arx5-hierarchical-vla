"""Bind the already validated site's declared cache before constructing a JIT.

No model import, numerical option change, cache deletion, directory creation,
GPU call or resource acquisition. Call only after the existing site validation.
Historical R1/R3 capsules are not modified or retrospectively qualified.
"""
from pathlib import Path


def bind_declared_cache(spec, config):
    directory=Path(spec['compilation_cache_dir']).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError('EXISTING_VALIDATED_CACHE_DIRECTORY_REQUIRED')
    before=getattr(config,'jax_compilation_cache_dir',None)
    config.update('jax_compilation_cache_dir',str(directory))
    actual=getattr(config,'jax_compilation_cache_dir',None)
    if actual!=str(directory):
        raise RuntimeError('DECLARED_COMPILATION_CACHE_BINDING_NOT_APPLIED')
    return dict(requested_directory=str(directory),effective_directory=actual,
        previous_directory=before,persistent_cache_enabled=getattr(config,'jax_enable_compilation_cache',None),
        configured_before_Provider_JIT=True,numerical_options_changed=False)
