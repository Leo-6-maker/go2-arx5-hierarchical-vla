"""C/D transport uses its actually adopted window; old lease bytes stay old."""
from copy import deepcopy
import json
import pytest
from scripts.pi0294.pv1 import transport_fixture as f,mix_core_context as ctx,mix_core_accounting as c,successor_cd as s,mix_accounting as a
from scripts.pi0294.test_v5_core_stage_accounting import closed


def setup(tmp_path):
    root=tmp_path/'campaign/children/new';(root/'plans').mkdir(parents=True)
    g,l,_=closed();ledger=tmp_path/'LOCAL.json';ledger.write_bytes(a.raw(l))
    binding=root/ctx.LOCAL_BINDING;binding.write_bytes(a.raw(dict(canonical_LOCAL_ledger=str(ledger))))
    (root/'V5_CD_SCOPE_BINDING.json').write_bytes(a.raw(l[s.FIELD]))
    manifest=dict(scope=c.ROLE,platform_evaluation_version=c.VERSION,execution=dict(sealed_inputs={ctx.LOCAL_BINDING:a.sha(binding.read_bytes())}))
    registry=dict(execution_scope=s.TASK,execution_authority_release=s.TASK_SOURCE,scope_grant_sha256=s.GRANT_SHA,T_CD=l[s.FIELD]['T_CD'],execution_deadline_epoch=l[s.FIELD]['deadline_epoch'])
    old=root.parent.parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json';old.write_bytes(a.raw(dict(run_uuid='campaign',stop_new_physics_epoch=1)))
    return root,manifest,registry,old.read_bytes()


def test_actual_new_window_keeps_original_protocol_and_old_parent_clock(tmp_path):
    root,m,r,old=setup(tmp_path);v=f.fixture_window(root,m,r)
    assert v['stop_new_physics_epoch']==r['execution_deadline_epoch']and v['run_uuid']=='campaign'
    assert (root.parent.parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json').read_bytes()==old
    assert f.fixture_window(root,dict(scope='old-PV1'),{})['stop_new_physics_epoch']==1


@pytest.mark.parametrize('field,value',[('execution_scope','old'),('execution_authority_release','old'),('scope_grant_sha256','0'*64),('T_CD',999.),('execution_deadline_epoch',1)])
def test_old_ACK_or_adapted_clock_is_not_a_CD_execution_binding(tmp_path,field,value):
    root,m,r,_=setup(tmp_path);r[field]=value
    with pytest.raises(PermissionError):f.fixture_window(root,m,r)
