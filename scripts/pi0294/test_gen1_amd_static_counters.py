"""Real temporary original counter transactions; native binding is a fixture."""
import json
import pytest
from scripts.pi0294.generalization import amd_static as m


@pytest.fixture
def counter(tmp_path,monkeypatch):
    root=tmp_path/'gen1-amd-static24-01';root.mkdir()
    manifest=dict(limits=dict(m.LIMITS,cpus=[0],operation_deadline_epoch=1793537549.8133287))
    (root/'RUN_MANIFEST.json').write_text(json.dumps(manifest))
    value=dict(schema=m.SCHEMA,run_uuid=root.name,
        authority_release='5b5de6f5d26ac90a6002c83ebcb60005acf06462',
        unknown_effects=[],reservations={},live_and_recovery_integrations=0,
        shadow_integrations=0,engine_starts=0,main_starts=0,Core_CPU_s=0.)
    (root/'RESOURCE_LEDGER.json').write_text(json.dumps(value))
    monkeypatch.setattr(m,'binding',lambda r:(manifest,{}))
    return m.ledger(root)


def test_real_counter_charges_exact_constructor_upper_without_integrations(counter):
    counter.charge('engine_starts',44)
    with pytest.raises(PermissionError):counter.charge('engine_starts',1)
    value=json.loads(counter.path.read_bytes())
    assert value['engine_starts']==44 and value['live_and_recovery_integrations']==value['shadow_integrations']==0
    assert value['reservations']=={} and value['unknown_effects']==[]


def test_even_one_integration_reservation_refused_before_effect(counter):
    before=counter.path.read_bytes()
    with pytest.raises(PermissionError):
        with counter.steps('shadow_integrations',1,'prohibited'):
            pytest.fail('zero-integration phase must never reach native effect')
    assert counter.path.read_bytes()==before


def test_unknown_counter_retained_and_blocks_new_start(counter):
    value=json.loads(counter.path.read_bytes());value['unknown_effects']=['unresolved'];counter.path.write_text(json.dumps(value))
    before=counter.path.read_bytes()
    with pytest.raises(PermissionError):counter.charge('engine_starts',1)
    assert counter.path.read_bytes()==before
