"""Captured original accounting and real temporary onceIO, no native physics."""
import base64,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_adoption as m,amd_pilot_scope as p
from scripts.pi0294.dec1.source_cpu_once_io import commit,reconcile

W=Path(os.environ.get('PI0294_PRIVATE_WORKSPACE',str(Path(__file__).resolve().parents[4])))
B=W/'work/private/v5-execution-20261025'
R=Path(__file__).resolve().parents[2]

@pytest.fixture
def inputs():
    path=B/'C50_VANDA_EVAL_COLD_02_C50_AFFINITY_PREENTRY_FAILURE_1497143_ONCE_GLOBAL_SETTLEMENT.private.json'
    if not path.exists():pytest.skip('captured private native accounting unavailable')
    gb=base64.b64decode(json.loads(path.read_bytes())['value']['GLOBAL_base64'])
    lb=base64.b64decode(json.loads((B/'VANDA_COHOST_PREPARED.private.json').read_bytes())['historical_LOCAL_base64'])
    grant=(R/'reports/PI0294/design/parallel-science-20261022/GEN1_AMD_TEACHER_CORR1024_FOUR_CAP_AUTHORIZATION_20261027.json').read_bytes()
    return gb,lb,grant

def test_captured_exact_four_caps_history_and_window(inputs):
    gb,lb,grant=inputs;v=m.propose(gb,lb,grant,now_epoch=1793041400.)
    old=json.loads(gb);g=v['proposed_global']
    assert all(g[k]==x for k,x in old.items()if k!='revision')
    assert set(g)-set(old)=={m.FIELD} and not v['actual_LOCAL_changed']
    assert m.validate(g)['caps']==m.CAPS
    proof=p.full_forecast(m.raw(g),lb)
    assert proof['cap_changes_required']=={} and proof['dimensions']['H200_s']['projected']==421663
    assert proof['matrix3_and_final20GPUh40GiB_retained']

@pytest.mark.parametrize('bad',['grant','duplicate','clock','unknown','active','changed_cap'])
def test_rejects_unapproved_or_unsafe_adoption(inputs,bad):
    gb,lb,grant=inputs;g=json.loads(gb);now=1793041400.
    if bad=='grant':grant+=b' '
    elif bad=='duplicate':g[m.FIELD]={}
    elif bad=='clock':now=1793451149.8133287
    elif bad=='unknown':g['unknown_effects']=['unresolved']
    elif bad=='active':g['PBS_submissions'][-1]['status']='Q'
    elif bad=='changed_cap':
        v=m.propose(gb,lb,grant,now_epoch=now)['proposed_global'];v[m.FIELD]['caps']['PBS_submissions']=37
        with pytest.raises(PermissionError):m.validate(v)
        return
    with pytest.raises(PermissionError):m.propose(m.raw(g),lb,grant,now_epoch=now)

def test_real_temp_once_commit_lost_reply_readback_and_no_replay(inputs,tmp_path):
    gb,lb,grant=inputs;v=m.propose(gb,lb,grant,now_epoch=1793041400.)
    path=tmp_path/'GLOBAL.json';path.write_bytes(gb);intent=tmp_path/'INTENT.json';receipt=tmp_path/'RECEIPT.json';lock=tmp_path/'owner.lock'
    args=dict(expected_before_sha256=m.sha(gb),proposed_bytes=m.raw(v['proposed_global']),request_id='GEN1-EXACT-CAPS-01',intent_path=intent,receipt_path=receipt,lock_path=lock)
    commit(path,**args)
    assert reconcile(path,intent_path=intent,receipt_path=receipt)['status']=='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'
    with pytest.raises(FileExistsError):commit(path,**args)

def test_environment_only_allocation_is_once_bounded_and_not_teacher(inputs):
    gb,lb,grant=inputs;g=m.propose(gb,lb,grant,now_epoch=1793041400.)['proposed_global']
    v=m.environment_allocation(g,run_uuid='gen1-amd-environment-01',manifest_sha256='a'*64,now_epoch=1793041401.)
    assert v['row']['Core_CPU_upper_s']==1800 and v['row']['transfer_upper_bytes']==2*1024**3
    assert not any(v['row'][k]for k in ('constructors','physics_ticks','models','PBS'))
    assert all(v['proposed_global'][k]==x for k,x in g.items()if k not in(m.FIELD,'revision'))
    with pytest.raises(PermissionError):m.environment_allocation(v['proposed_global'],run_uuid='gen1-amd-environment-01',manifest_sha256='a'*64,now_epoch=1793041402.)
