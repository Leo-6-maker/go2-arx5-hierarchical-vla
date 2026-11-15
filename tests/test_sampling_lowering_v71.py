from pathlib import Path

import pytest

from scripts.sla.check_sampling_lowering_v71 import run


@pytest.mark.parametrize('platform,devices', [(None, ''), ('cuda', ''), ('cpu', '0')])
def test_full_graph_check_requires_explicit_cpu_before_jax_import(monkeypatch, platform, devices):
    if platform is None:
        monkeypatch.delenv('JAX_PLATFORMS', raising=False)
    else:
        monkeypatch.setenv('JAX_PLATFORMS', platform)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', devices)
    with pytest.raises(PermissionError, match='EXPLICIT_CPU_ONLY_ENVIRONMENT_REQUIRED'):
        run(Path('unread-helper'), [], [])
