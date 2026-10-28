"""Actual captured nonzero terminal; fresh kernel observations are CPU fixtures."""
import base64,gzip,json,os,time
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import minimal_partial_completion_scope as n,minimal_execution as e
from scripts.pi0294.generalization import minimal_scope


@pytest.fixture(scope='module')
def captured():
    path=os.environ.get('ARX5_PARTIAL_COMPLETION_CAPTURE_DIR')
    if not path:pytest.skip('explicit actual1504095 closed full bytes required')
    root=Path(path);cap=json.loads((root/'FIRST512_REGISTERED_DISPATCH01_LATE_SETTLE_CAPTURE.private.json').read_bytes())
    gb=gzip.decompress(base64.b64decode(cap['GLOBAL_gzip_base64']))
    old=json.loads((root/'FIRST512_REGISTERED_DISPATCH_BOUND_GLOBAL_CORE_COLD_JOIN_1793196941512727319.private.json').read_bytes())
    lb=gzip.decompress(base64.b64decode(old['LOCAL_gzip_base64']))
    cb=(root/'FIRST512_REGISTERED_DISPATCH01_PARTIAL_BOTH_CLOSED_INDEPENDENT_READBACK.private.json').read_bytes()
    scope=(Path(__file__).parents[2]/'reports/PI0294/design/parallel-science-20261022/ARX5_FIRST512_PARTIAL_SAVED_PREFIX_ONE_COMPLETION_SCOPE_20261028.json').read_bytes()
    now=1793200200.;fresh=dict(observed_epoch=now,GLOBAL_sha256=e.sha(gb),closed_lease_sha256=n.LEASES,
        all_original_actors_absent=True,original_SETTLE_UNKNOWN_preserved=True,new_mutations=0,CPU_fixture_only=True)
    return gb,lb,cb,scope,fresh,now


@pytest.fixture(scope='module')
def proposed(captured):
    gb,lb,cb,scope,fresh,now=captured
    return n.propose(gb,lb,scope,closed_capture_bytes=cb,fresh_readback=fresh,now_epoch=now)


def test_control_only_proposal_preserves_all_real_costs_rows_and_nonzero_sessions(captured,proposed):
    gb,lb,cb,scope,fresh,now=captured;old=json.loads(gb);g=proposed['proposed_global']
    assert old['PBS_submissions']==g['PBS_submissions']and e.sha(e.raw(old))==e.sha(gb)
    assert (g['used']['optimizer_updates'],g['used']['neural_queries'])==(14788,5908)
    assert g['CPU_s']==old['CPU_s']+3600
    assert g['new_storage_current_bytes']==old['new_storage_current_bytes']+64*1024**2
    assert all(g[k]==v for k,v in old.items()if k not in('revision','CPU_s','new_storage_current_bytes','new_storage_peak_bytes'))
    assert proposed['new_requests']==proposed['new_PBS']==proposed['new_model_queries_updates']==0
    assert proposed['native_admission']is False
    assert n.validate(g)['prefix_independent_audit_complete_at_adoption']is False


def test_all14_future_dimensions_fit_and_matrix_final_not_borrowed(captured,proposed):
    f=proposed['full_future_proof'];assert f['cap_changes_required']=={}
    assert len(f['dimensions'])==14 and f['matrix3_and_final20GPUh40GiB_retained']
    assert f['dimensions']['PBS_submissions']['projected']==f['dimensions']['PBS_submissions']['cap']==42
    assert f['dimensions']['optimizer_updates']['projected']==15814
    assert f['dimensions']['H200_s']['projected']==517580.
    assert f['dimensions']['Hopper_storage_bytes']['shortfall']==0
    assert n.NATIVE['optimizer_updates']==1026 and n.NATIVE['neural_queries']==76
    assert n.NATIVE['Hopper_storage_bytes']==0 and n.MODEL_LOAD_LIMIT==7


def test_actual_canonical_local_is_not_claimed_and_projection_is_once(captured,proposed):
    _,lb,*_=captured;local=json.loads(lb);g=proposed['proposed_global'];view=n.budget_LOCAL(g,local)
    assert view['CPU_s']['local_PC']==local['CPU_s']['local_PC']+600
    assert view['network_transfer_accounted_bytes']==local['network_transfer_accounted_bytes']+256*1024
    assert n.budget_LOCAL(g,view)==view and json.loads(lb)==local
    assert view['ARX5_first512_saved_prefix_completion_control_projection']['canonical_LOCAL_write_proven']is False
    bad=deepcopy(view);bad['ARX5_first512_saved_prefix_completion_control_projection']['costs']['local_CPU_s']=0
    with pytest.raises(PermissionError):n.budget_LOCAL(g,bad)


@pytest.mark.parametrize('kind',['unknown','pending','wrong_lease','stale','changed_row','changed_grant','changed_capture'])
def test_missing_real_boundaries_reject_without_native_effects(captured,kind):
    gb,lb,cb,scope,original,now=captured;fresh=deepcopy(original);g=json.loads(gb)
    if kind=='unknown':g['unknown_effects']=['PRESERVED']
    elif kind=='pending':g['pending_updates']=[dict(unknown='PRESERVED')]
    elif kind=='wrong_lease':fresh['closed_lease_sha256']['science']='0'*64
    elif kind=='stale':fresh['observed_epoch']=now-121
    elif kind=='changed_row':g['PBS_submissions'][-1]['query_attempts']=0
    elif kind=='changed_grant':scope=scope.replace(b'"after": 42',b'"after": 43')
    elif kind=='changed_capture':cb=cb+b' '
    with pytest.raises((PermissionError,ValueError)):
        n.propose(e.raw(g),lb,scope,closed_capture_bytes=cb,fresh_readback=fresh,now_epoch=now)


def test_repeat_control_adoption_and_changed_paid_marker_are_refused(captured,proposed):
    _,lb,cb,scope,fresh,now=captured;g=proposed['proposed_global']
    with pytest.raises(PermissionError):n.propose(e.raw(g),lb,scope,closed_capture_bytes=cb,fresh_readback=fresh,now_epoch=now)
    bad=deepcopy(g);bad[n.FIELD]['control_paid_once']['transfer_bytes']=0
    with pytest.raises(PermissionError):n.validate(bad)


def test_original_seven_caps_remain_byte_equal_and_only_pbs_overlay_changes(captured,proposed):
    gb,*_=captured;old=json.loads(gb);g=proposed['proposed_global']
    caps=e.authority.effective_caps(old,e.authority.CAPS);new=e.authority.effective_caps(g,caps)
    assert {k:(caps[k],v)for k,v in new.items()if v!=caps[k]}=={'PBS_submissions':(41,42)}
    assert g[e.authority.FIELD]==old[e.authority.FIELD]
    assert e.authority.global_limits(g,{'PBS_submissions':41})=={'PBS_submissions':42}
