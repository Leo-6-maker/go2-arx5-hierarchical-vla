"""Strict finite pending-self classification; native actors remain fixtures."""
import json,base64
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import minimal_registered_unsubmitted_resume as m
from scripts.pi0294.dec1 import vanda_campaign as q

def captured_shape():
    old=[dict(request_id='old-'+str(i),job_id=None,status='NATIVE_TERMINAL')for i in range(38)]
    row=dict(request_id=m.REQUEST,submission_nonce=m.NONCE,source_sha=m.SOURCE,request_sha256=m.REQUEST_SHA,
        run_manifest_sha256=m.MANIFEST,submit_count=39,status=m.e.STATES[0],job_id=None,
        T_CD=1792932749.8133287,stop_new_training_epoch=1793451149.8133287,deadline_epoch=1793537549.8133287)
    return dict(PBS_submissions=old+[row],used=dict(PBS_submissions=39),pending_updates=[],unknown_effects=[])

def empty_queue():
    return dict(schema=q.QUEUE,user='e1192548',observed_epoch=1793227000.,
        qselect=dict(argv=['/opt/pbs/bin/qselect','-u','e1192548'],returncode=0,stdout='',stderr=''),qstat=None)

def test_exact_self_gap_is_preserved_not_erased_from_canonical():
    g=captured_shape();before=deepcopy(g);v=m.predispatch(g,empty_queue(),now_epoch=1793227000.)
    assert v['exact_registered_self_gap']and len(v['original_classification_unknown'])==1
    assert g==before and len(g['PBS_submissions'])==39 and g['PBS_submissions'][-1]['job_id']is None

@pytest.mark.parametrize('kind',['wrong_request','wrong_nonce','wrong_source','bound','global_unknown','pending','another_unbound','bad_queue'])
def test_any_other_identity_effect_or_queue_gap_is_still_refused(kind):
    g=captured_shape();p=empty_queue();row=g['PBS_submissions'][-1]
    if kind=='wrong_request':row['request_id']='different'
    elif kind=='wrong_nonce':row['submission_nonce']='different'
    elif kind=='wrong_source':row['source_sha']='0'*40
    elif kind=='bound':row['job_id']='123.stdct-mgmt-02'
    elif kind=='global_unknown':g['unknown_effects']=['x']
    elif kind=='pending':g['pending_updates']=['x']
    elif kind=='another_unbound':g['PBS_submissions'][0]['status']=m.e.STATES[0]
    else:p['qselect']['returncode']=1
    with pytest.raises(PermissionError):m.predispatch(g,p,now_epoch=1793227000.)

def test_historical_projection_is_explicit_once_and_never_changes_input():
    g=captured_shape();body=(Path(__file__).parents[2]/'reports/PI0294/design/parallel-science-20261022/ARX5_EXACT_REGISTERED_UNSUBMITTED_CONTROL_CONTINUATION_V1_20261029.json').read_bytes()
    h=m.grant(body);g.update(CPU_s=130,new_storage_current_bytes=1024**2+3,new_storage_peak_bytes=1024**2+3)
    g[m.FIELD]=dict(schema='ARX5_EXACT_REGISTERED_UNSUBMITTED_CONTROL_ADOPTION_V1',grant_base64=base64.b64encode(body).decode(),
        grant_sha256=m.GRANT_SHA,control_paid_once=m.CONTROL,request_id=m.REQUEST,science_manifest_sha256=m.MANIFEST,
        new_request_or_GPU_authority=False,before=dict(PBS_submissions=39,CPU_s=10,new_storage_current_bytes=3,new_storage_peak_bytes=3),
        historical_LOCAL_sha256='a'*64,adopted_epoch=1793227000.,**{k:h[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch')})
    local=dict(CPU_s=dict(local_PC=123),network_transfer_accounted_bytes=456);before=deepcopy(local)
    out=m.budget_LOCAL(g,local);assert local==before
    assert out['CPU_s']['local_PC']==183 and out['network_transfer_accounted_bytes']==456+32768
    assert out[m.LOCAL_FIELD]['canonical_LOCAL_write_proven']is False and m.budget_LOCAL(g,out)==out
    bad=deepcopy(g);bad[m.FIELD]['control_paid_once']={}
    with pytest.raises(PermissionError):m.budget_LOCAL(bad,local)


def test_original_journal_confirmed_state_is_accepted_and_unknown_kept_blocked(tmp_path):
    from scripts.pi0294.dec1.owner_transport import Journal
    identity=dict(run_uuid='fixture-registered-unsubmitted',job_id='NO_GPU_YET',source_sha=m.SOURCE,request_sha256=m.e.authority.GRANT_SHA)
    j=Journal(tmp_path,identity);j.call('REGISTER_REQUEST','MUTATION',lambda intent,limit:dict(fixture=True))
    m.checked_setup_state(tmp_path,identity['run_uuid'])
    state=tmp_path/'RPC_EFFECT_STATE.json';v=json.loads(state.read_bytes())
    for status in('PENDING','UNKNOWN'):
        v['status']=status;state.write_text(json.dumps(v))
        with pytest.raises(PermissionError):m.checked_setup_state(tmp_path,identity['run_uuid'])
    v['status']='CONFIRMED';v['intent']['phase']='QSUB';state.write_text(json.dumps(v))
    with pytest.raises(PermissionError):m.checked_setup_state(tmp_path,identity['run_uuid'])


def test_original_watch_logic_reaches_remaining_session_method_without_mutating_module(tmp_path):
    from types import SimpleNamespace
    import inspect
    from scripts.pi0294.dec1 import cold_owner
    before=inspect.getsource(cold_owner.watch);calls=[];now=[100.]
    def read(path):
        return {'SOURCE_REQUEST.private.json':dict(request=dict(submit_count=39,platform_training_version=m.remaining.VERSION,profile=m.remaining.PROFILE),directory=str(tmp_path)),
                'SOURCE_JOB_BOUND.private.json':dict(job_id='123.stdct-mgmt-02'),
                'TRAIN_REGISTERED.private.json':dict(root=str(tmp_path))}[Path(path).name]
    def fresh(*a,**k):return dict(rows=dict(vanda=dict(jobs={'123.stdct-mgmt-02':dict(job_state='R',Job_Name='pi0294_dec1_a40_train',project='CFP04-CF-007')},receipts={})),view={})
    o=SimpleNamespace(native=SimpleNamespace(T0=0),cold_rpc_bound=True,BASE=tmp_path,read=read,fresh=fresh,
        record=lambda *x:calls.append(x),update=lambda *x:None,previous=SimpleNamespace(rpc=lambda *x:None),HEARTBEAT='fixture',
        request_template=dict(platform_training_version=m.remaining.VERSION,profile=m.remaining.PROFILE),
        release_science_session=lambda *a,**k:calls.append('remaining-science-method')or True)
    watch=m.compatible_watch();watch(o,'fixture-not-a-credential',clock=lambda:now[0],sleep=lambda n:now.__setitem__(0,1e9))
    assert 'remaining-science-method'in calls and inspect.getsource(cold_owner.watch)==before
