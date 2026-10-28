"""Future D25/cold Core-close values are CPU fixtures, never native GO."""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.test_minimal_execution import actual,request,closed_CPU_fixture
from scripts.pi0294.generalization import minimal_execution as e,minimal_concurrency as n
from scripts.pi0294.pv1 import mix_core_accounting as c,mix_accounting as a


@pytest.fixture
def bound(actual):
    g,cap=actual;g=closed_CPU_fixture(g);r=request(g,cap)
    v=e.register(a.raw(g),request_bytes=a.raw(r),Core_measurement_capture_bytes=cap,now_epoch=1793131100.)['proposed_global']
    return e.bind(v,r['request_id'],'12345.stdct-mgmt-02',now_epoch=1793131102.)


def D_fixture(g):
    return dict(evaluation_candidate='D25',native_platform='CORELAB_RTX4080',platform_evaluation_version=c.VERSION,
        source_sha='b'*40,created_epoch=1793131104.,**{n.KEY:n.binding(g)})


def test_exact_distinct_sites_and_new_D25_binding_do_not_remove_global_fences(bound):
    p=e.rows(bound);d=D_fixture(bound);b=deepcopy(bound)
    proof=n.validate_pair(bound,p,[d]);assert proof==n.binding(bound)
    assert bound==b and bound['pending_updates']==bound['unknown_effects']==[]
    assert n.permit_Core(bound,d,dict(authority={n.KEY:proof}))==proof
    with pytest.raises(PermissionError):n.permit_Core(bound,d,dict(authority={}))


@pytest.mark.parametrize('kind',['C50','legacy','new_scope_flag_only','wrong_main','old_source','two_trainers','two_Core','unknown','pending','unbound'])
def test_legacy_sharing_or_unknowns_cannot_use_cold_cross_scope_join(bound,kind):
    g=deepcopy(bound);p=e.rows(g);d=D_fixture(g);cores=[d]
    if kind=='C50':d['evaluation_candidate']='C50'
    elif kind=='legacy':d['platform_evaluation_version']='OLD_B24'
    elif kind=='new_scope_flag_only':d[n.KEY]=True
    elif kind=='wrong_main':d[n.KEY]['main_request_id']='ANOTHER_REQUEST'
    elif kind=='old_source':d['source_sha']='8fba44f38aa5c6858614d8bd4e34ad7f969dd40c'
    elif kind=='two_trainers':p=p+p
    elif kind=='two_Core':cores=cores+cores
    elif kind=='unknown':g['unknown_effects']=['PRESERVED']
    elif kind=='pending':g['pending_updates']=[{'preserved':True}]
    else:g['PBS_submissions'][-1].update(status=e.STATES[0],job_id=None)
    before=deepcopy(g)
    with pytest.raises(PermissionError):n.validate_pair(g,p,cores)
    assert g==before


def test_LOCAL_sync_is_only_one_original_control_and_inventory_proposal(bound):
    root=Path(os.environ['ARX5_FIRST512_MEASUREMENT_CAPTURE']).parent
    captured=json.loads((root/'CORE_RETRY12_CONTINUATION_INDEPENDENT_DUAL_READBACK.private.json').read_bytes())['local']
    local=json.loads(gzip.decompress(base64.b64decode(captured['LOCAL_gzip_base64'])))
    # Explicit future mirror fixture. Current native C12 remains running; this
    # test does not claim its real terminal/paired LOCAL was ever observed.
    local[c.FIELD]=deepcopy(bound[c.FIELD]);local[a.PACKAGE]=deepcopy(bound[a.PACKAGE])
    local['PBS_submissions']=deepcopy(bound['PBS_submissions'][:-1])
    before=deepcopy(local)
    with pytest.raises(PermissionError):n.validate_LOCAL(bound,local)
    out=n.propose_LOCAL(a.raw(bound),a.raw(local),now_epoch=1793131104.)
    result=out['proposed_LOCAL'];assert local==before
    assert result['CPU_s']['local_PC']==local['CPU_s']['local_PC']+630
    assert result['network_transfer_accounted_bytes']==local['network_transfer_accounted_bytes']+e.CONTROL['transfer_bytes']
    assert result['CPU_reserved_s']==local['CPU_reserved_s']and result['GPU_reserved_s']==local['GPU_reserved_s']
    assert result['PBS_submissions']==bound['PBS_submissions']and not out['native_admission']
    assert out['full_future_proof']['cap_changes_required']=={}
    assert n.validate_LOCAL(bound,result)==n.binding(bound)
    with pytest.raises(PermissionError):n.propose_LOCAL(a.raw(bound),a.raw(result),now_epoch=1793131105.)


def test_actual_closed_first512_can_mirror_without_reopening_native_job():
    path=os.environ.get('ARX5_TERMINAL_MIRROR_CAPTURE')
    if not path:pytest.skip('actual closed canonical pair not supplied')
    capture=json.loads(Path(path).read_bytes())
    gb=gzip.decompress(base64.b64decode(capture['GLOBAL_gzip_base64']))
    lb=gzip.decompress(base64.b64decode(capture['LOCAL_gzip_base64']))
    g=json.loads(gb);l=json.loads(lb);row=e.rows(g)[0]
    assert row['status']==e.STATES[2] and row['native_exit_status']!=0
    result=n.propose_LOCAL(gb,lb,now_epoch=capture['observed_epoch'])
    proposed=result['proposed_LOCAL'];c.paired(g,proposed)
    assert proposed['PBS_submissions']==g['PBS_submissions']
    assert result['full_future_proof']['cap_changes_required']=={}
    assert result['new_GPU']==result['new_PBS']==result['new_models']==0
    assert n.validate_LOCAL(g,proposed) is None
    with pytest.raises(PermissionError):n.propose_LOCAL(gb,a.raw(proposed),now_epoch=capture['observed_epoch'])
