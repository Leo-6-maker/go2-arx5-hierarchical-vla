"""Actual cold-stage capacity branch; fixtures are not native admission."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.pi0294.multitask import night_source_stage as stage
from scripts.pi0294.generalization import minimal_scope


@pytest.mark.parametrize('free_gib,retained_gib,passes',[(108,3,True),(106,3,False),(108,5,False)])
def test_v7_preserves_old_shortfalls_as_observations_but_requires_real_space(monkeypatch,free_gib,retained_gib,passes):
    GiB=1024**3
    old=dict(cap_changes_required={'local_CPU_s':{'before':48,'after':56}},
             dimensions={'Core_storage_bytes':dict(projected=120*GiB,used=20*GiB)})
    preserved=deepcopy(old);calls=[];outputs={}
    def forecast(g,l):
        assert(g,l)==(b'GLOBAL',b'LOCAL');calls.append(True);return deepcopy(old)
    monkeypatch.setattr(minimal_scope,'first512_forecast',forecast)
    code=stage.PROGRAM.split('if scope.V7_CARRY in account:\n from scripts.',1)[1]
    code='if scope.V7_CARRY in account:\n from scripts.'+code.split("put(root/'FULL14_PROOF.json'",1)[0]
    scope=SimpleNamespace(V7_CARRY='v7')
    ns=dict(scope=scope,account={'v7':{},'charged_upper':{'Core_peak_bytes':retained_gib*GiB}},
            gb=b'GLOBAL',lb=b'LOCAL',root=Path('/owned/source'),
            os=SimpleNamespace(statvfs=lambda p:SimpleNamespace(f_bavail=free_gib*GiB,f_frsize=1)),
            put=lambda p,v:outputs.update({p.name:v}),raw=lambda v:v)
    if passes:
        exec(compile(code,'actual_source_stage_capacity','exec'),ns)
        assert ns['budget']['original_forecast']==old
        assert outputs['ACTUAL_SOURCE_CAPACITY.json']['financial_pool_caps_unchanged']is True
    else:
        with pytest.raises(AssertionError):exec(compile(code,'actual_source_stage_capacity','exec'),ns)
    assert calls==[True]and old==preserved
