"""Pure control tests; actual closed capture is optional and never native GO."""
from copy import deepcopy
import base64,gzip,json,os,subprocess,sys,time
from pathlib import Path
import pytest
from scripts.pi0294.generalization import minimal_execution as e,minimal_saved512_recovery_scope as s

ROOT=Path(__file__).resolve().parents[2]
CARD=ROOT/'reports/PI0294/design/parallel-science-20261022/ARX5_SAVED_E1_512_E2_RAW_FIXED512_RECOVERY_CONTROL_SCOPE_20261029.json'


def test_exact_scope_keeps_own_E1_512_and_only_remaining_E2_optimization():
    h=s.grant(CARD.read_bytes());assert sum(x['new_updates']for x in h['sessions'])==512
    assert sum(x['new_queries']for x in h['sessions'])==62
    assert sum(x['new_model_loads']for x in h['sessions'])==5
    assert h['sessions'][0]['new_updates']==0 and h['native_admission']is False
    assert h['control_partition']['independent_saved_state_audit_and_metadata_CPU_s']==600
    assert s.count_upper({},42)==42
    for key in('other_caps_changed','clocks_reset','native_admission'):
        bad=deepcopy(h);bad[key]=True
        with pytest.raises(PermissionError):s.grant(e.raw(bad))


def test_metadata_budget_and_export_comparison_import_without_torch():
    code="""import sys,importlib.abc
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch'or fullname.startswith('torch.'):raise AssertionError('BUDGET_MUST_NOT_IMPORT_TORCH')
sys.meta_path.insert(0,NoTorch())
from scripts.pi0294.generalization import minimal_saved512_recovery_scope,minimal_delivery_control,minimal_inference_transfer,minimal_inference_export
assert 'torch'not in sys.modules
"""
    env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(ROOT),str(ROOT/'src'))))
    run=subprocess.run([sys.executable,'-B','-c',code],env=env,capture_output=True,text=True,timeout=15)
    assert run.returncode==0,run.stderr


@pytest.fixture
def actual_capture():
    path=os.environ.get('ARX5_CURRENT_REMAINING_CAPTURE_DIR')
    if not path:pytest.skip('Private actual closed GLOBAL capture not provided')
    m=Path(path);closed=json.loads((m/'FIRST512_ENTRY_REPAIRED01_FAILED1505362_CLOSED_FULL_CANONICAL_INDEPENDENT.private.json').read_bytes())
    g=gzip.decompress(base64.b64decode(closed['GLOBAL_gzip_base64']));l=gzip.decompress(base64.b64decode(json.loads((m/'D25_SAME64_WIRE01_PREPARED.private.json').read_bytes())['LOCAL_gzip_base64']))
    assert e.sha(g)==closed['GLOBAL_sha256']and closed['all_complete_canonical_bytes_match']
    return g,l,closed


def test_real_closed_all14_proposal_history_and_budget_projection(actual_capture):
    gb,lb,closed=actual_capture;g=json.loads(gb);local=json.loads(lb);old_rows=deepcopy(g['PBS_submissions']);old_package=deepcopy(g[e.package.FIELD])if hasattr(e.package,'FIELD')else None
    # Clock is a fixture; the full native capture/rows/resources are real.
    now=time.time();proposal=s.propose(gb,lb,CARD.read_bytes(),closed_readback=dict(closed,observed_epoch=now),now_epoch=now);v=proposal['proposed_global'];p=proposal['full_future_proof']
    assert p['cap_changes_required']=={} and len(p['dimensions'])==14
    assert p['dimensions']['PBS_submissions']['projected']==43 and p['dimensions']['PBS_submissions']['matrix']==3
    assert p['dimensions']['H200_s']['final']==72000 and p['dimensions']['Hopper_storage_bytes']['final']==40*1024**3
    assert v['PBS_submissions']==old_rows and v['optimizer_updates']==g['optimizer_updates']and v['generation_queries']==g['generation_queries']
    assert v['CPU_s']==g['CPU_s']+3600 and v['new_storage_current_bytes']==g['new_storage_current_bytes']+8*1024**2
    assert v['new_storage_peak_bytes']>=g['new_storage_peak_bytes']and proposal['new_requests']==proposal['new_PBS']==proposal['new_model_queries_updates']==0
    assert proposal['native_admission']is False and s.validate(v)['saved_state_audit_complete_at_adoption']is False
    assert e.authority.effective_caps(v,{k:z['cap']for k,z in e.scope.first512_forecast(gb,lb)['dimensions'].items()})['PBS_submissions']==43
    projected=s.budget_LOCAL(v,local);assert s.budget_LOCAL(v,projected)==projected and local==json.loads(lb)
    assert projected['CPU_s']['local_PC']==local['CPU_s']['local_PC']+600
    with pytest.raises(PermissionError):s.propose(e.raw(v),lb,CARD.read_bytes(),closed_readback=dict(closed,observed_epoch=now),now_epoch=now)
    for key,bad in [('lease_sha256','0'*64),('GLOBAL_sha256','0'*64),('observed_epoch',now-121)]:
        witness=dict(closed,observed_epoch=now);witness[key]=bad
        with pytest.raises(PermissionError):s.propose(gb,lb,CARD.read_bytes(),closed_readback=witness,now_epoch=now)
