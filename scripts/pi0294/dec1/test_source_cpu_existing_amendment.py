"""Existing adopted VAP1 caps are classified, never a new allowance or native GO."""
from copy import deepcopy
import pytest
from scripts.pi0294.dec1 import source_cpu_intake as unit
from scripts.pi0294.dec1.test_source_cpu_intake import packet,population,encode,sha,call


@pytest.fixture
def amended(packet,monkeypatch):
    p=packet;d=p['dec1']
    d.update(VAP1_authority_release=unit.VAP1_AUTHORITY,
        caps=dict(CoreLab_CPU_s=576000,local_CPU_s=57600),
        VAP1_cumulative_io_caps_bytes=dict(CoreLab=128*1024**3,transfer=128*1024**3))
    a=dict(schema='VAP1_ADOPTION_AND_DELTA_BUDGET_V1',authority_release=unit.VAP1_AUTHORITY,
        original_T0_epoch=d['original_T0_epoch'],original_hard_close_epoch=unit.HARD_CLOSE,
        old_clocks_and_counters_reset=False,cumulative_subcap_delta=dict(
            local_CPU_s=[28800,57600],CoreLab_CPU_s=[345600,576000],
            GPU_site_output_bytes=[68719476736,171798691840],
            CoreLab_output_bytes=[68719476736,137438953472],transfer_bytes=[68719476736,137438953472]))
    raw=encode(a);monkeypatch.setattr(unit,'VAP1_ADOPTION_SHA256',sha(raw))
    p['_amendment_bytes']=raw
    p['manifest']['authority']['amendments'].append(unit.VAP1_AUTHORITY)
    # Actual historical local costs can exceed old8h. They remain counted.
    p['parent']['CPU_s']['local_PC']=40000;p['budget']['DEC1_used']['local_CPU_s']=30000
    p['budget']['parent_remaining_after_spent_reserved_and_final']['local_CPU_s']=48*3600-40000
    p['budget']['site_remaining']['local_CPU_s']=48*3600-40000
    return p


def test_existing_exact_delta_avoids_old_cap_without_resetting_costs(amended):
    p=amended;before=deepcopy(p['dec1']);caps=unit.effective_caps(p['dec1'],p['_amendment_bytes'])
    assert caps['CoreLab_CPU_s']==160*3600 and caps['local_CPU_s']==16*3600
    assert caps['CoreLab_output_bytes']==caps['transfer_bytes']==128*1024**3
    assert caps['episode_starts']==136 and caps['physics_ticks']==5000000 and caps['constructors']==12000
    assert caps['local_output_bytes']==8*1024**3
    assert call(p)['native_dispatch_admitted'] is False
    assert p['dec1']==before and p['budget']['DEC1_used']['local_CPU_s']==30000


@pytest.mark.parametrize('fault',['missing_bytes','different_bytes','unknown_release','old_native_CPU_cap',
    'old_native_IO_cap','missing_manifest_amendment','unadopted_marker'])
def test_unproven_delta_never_expands_available_caps(amended,fault):
    p=amended
    if fault=='missing_bytes':p.pop('_amendment_bytes')
    elif fault=='different_bytes':p['_amendment_bytes']+=b' '
    elif fault=='unknown_release':p['dec1']['VAP1_authority_release']='f'*40
    elif fault=='old_native_CPU_cap':p['dec1']['caps']['local_CPU_s']=28800
    elif fault=='old_native_IO_cap':p['dec1']['VAP1_cumulative_io_caps_bytes']['CoreLab']=64*1024**3
    elif fault=='missing_manifest_amendment':p['manifest']['authority']['amendments']=[unit.AUTHORITY]
    elif fault=='unadopted_marker':p['dec1'].pop('VAP1_authority_release')
    with pytest.raises(PermissionError):call(p)


def test_legacy_without_adoption_retains_original_caps(packet):
    assert unit.effective_caps(packet['dec1'])==unit.CAPS
    assert call(packet)['native_dispatch_admitted'] is False


@pytest.mark.parametrize('fault',['none','missing_parent','different_parent','writable_parent','no_adoption'])
def test_actual_runtime_reads_the_native_parent_bytes_before_accepting_delta(tmp_path,fault):
    from scripts.pi0294.dec1 import source_cpu_runtime as runtime
    parent=tmp_path/'parent';root=parent/'children'/'component'
    source=root/unit.VAP1_PLAN;native=parent/'authority/VAP1/VAP1_ADOPTION_AND_DELTA_BUDGET.json'
    for p in (source,native):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'SYNTHETIC_HASH_VALIDATED_BY_INTAKE_SEPARATELY');p.chmod(0o400)
    pre=dict(VAP1_authority_release=unit.VAP1_AUTHORITY)
    if fault=='no_adoption':
        assert runtime._existing_amendment(root,parent,{}) is None;return
    if fault=='missing_parent':native.unlink()
    if fault=='different_parent':native.chmod(0o600);native.write_bytes(b'DIFFERENT');native.chmod(0o400)
    if fault=='writable_parent':native.chmod(0o600)
    if fault=='none':assert runtime._existing_amendment(root,parent,pre)==source.read_bytes()
    else:
        with pytest.raises((PermissionError,FileNotFoundError)):runtime._existing_amendment(root,parent,pre)
