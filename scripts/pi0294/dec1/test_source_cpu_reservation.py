"""Draft CPU fixtures only. Pure proposals are never registration/evidence."""
import sys
_AUDIT={'active':True,'blocked':[]}
def _deny(event,args):
    if _AUDIT['active'] and (event.startswith(('subprocess.','socket.')) or event in (
            'os.system','os.fork','os.forkpty','os.posix_spawn','os.spawn','pty.spawn')):
        _AUDIT['blocked'].append(event);raise PermissionError('PURE_RESERVATION_EXTERNAL_EFFECT:'+event)
sys.addaudithook(_deny)
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.dec1 import source_cpu_reservation as unit
from scripts.pi0294.dec1.test_source_cpu_intake import packet,encode,sha,call as seal_packet
from scripts.pi0294.dec1.test_source_commissioning_plan import population
assert not _AUDIT['blocked'],'PROJECT_IMPORT_ATTEMPTED_EXTERNAL_EFFECT'
_AUDIT['active']=False


@pytest.mark.parametrize('lost_Core_write',[False,True])
def test_real_local_parent_then_Core_CAS_never_admits_a_partial_pair(facts,tmp_path,monkeypatch,lost_Core_write):
    from scripts.pi0294.dec1 import source_cpu_once_io as once,source_cpu_runtime as runtime
    f=facts;f['root']=str(tmp_path/f['current']['campaign_run_uuid']/'children'/f['m']['run_uuid'])
    args=f['bind'](child_root=f['root']);p=unit.propose_parent(parent_bytes=encode(f['p']['parent']),
        adoption_bytes=encode(f['p']['adoption']),**args)
    parent=tmp_path/'LOCAL_LEDGER.json';core=tmp_path/'CORE_LEDGER.json'
    parent.write_bytes(encode(f['p']['parent']));core.write_bytes(args['snapshot_bytes'])
    for path in (parent,core):path.chmod(0o400)
    root=Path(f['root']);root.mkdir(parents=True)
    (root/'RUN_MANIFEST.json').write_bytes(args['manifest_bytes']);(root/'RUN_MANIFEST.json').chmod(0o400)
    def commit(path,proposal):
        return once.commit(path,expected_before_sha256=proposal['before_snapshot_sha256'],
            proposed_bytes=encode(proposal['proposed_ledger']),request_id='fixture-cross-site-01',
            intent_path=path.with_suffix('.intent'),receipt_path=path.with_suffix('.receipt'),lock_path=path.with_suffix('.lock'))
    commit(parent,p)
    before_replace=once.os.replace
    if lost_Core_write:
        monkeypatch.setattr(once.os,'replace',lambda *_:(_ for _ in ()).throw(OSError('LOST_CORE_WRITE')))
        with pytest.raises(OSError):commit(core,p['Core_proposal'])
        monkeypatch.setattr(once.os,'replace',before_replace)
    else:commit(core,p['Core_proposal'])
    actual=unit.intake._object(core.read_bytes(),'actual Core fixture bytes')
    if lost_Core_write:
        with pytest.raises(PermissionError,match='ONCE_CHILD_RESERVATION_REQUIRED'):
            runtime._reservation(root,f['m'],f['p']['stage'],f['p']['budget'],actual,f['current'],
                closeout=False,now=f['p']['now_epoch'])
        assert once.reconcile(core,intent_path=core.with_suffix('.intent'),
            receipt_path=core.with_suffix('.receipt'))['status']=='CURRENT_OLD_BYTES_NO_RETRY_AUTHORIZATION'
    else:
        runtime._reservation(root,f['m'],f['p']['stage'],f['p']['budget'],actual,f['current'],
            closeout=False,now=f['p']['now_epoch'])
    local=unit.intake._object(parent.read_bytes(),'actual parent fixture bytes')
    assert local['CPU_s']==f['p']['parent']['CPU_s']  # No historic spend charged again.
    assert local['CPU_reserved_s']['CoreLab']==f['p']['parent']['CPU_reserved_s']['CoreLab']+f['p']['budget']['requested']['CoreLab_CPU_s']
    with pytest.raises(FileExistsError):commit(parent,p)

@pytest.fixture(autouse=True)
def no_external_effects():
    n=len(_AUDIT['blocked']);_AUDIT['active']=True
    try:yield
    finally:
        _AUDIT['active']=False
        assert len(_AUDIT['blocked'])==n


@pytest.fixture
def facts(packet):
    p=packet;seal_packet(p)
    current=deepcopy(p['dec1']);current['schema']='DEC1_NONADDITIVE_RESOURCE_CLASSIFICATION_V1'
    current['preserved_history']=dict(old_job='CLOSED_WITH_REAL_COST',old_updates=5891,old_CPU_s=123.5)
    current['source_cpu_current_classification']=deepcopy(p['budget']['DEC1_used'])
    m=p['manifest'];m.update(owner='EXISTING_LOCAL_PC',run_uuid='source-cpu-fixture')
    root='/registered/'+current['campaign_run_uuid']+'/children/'+m['run_uuid']
    seal=dict(source_sha=m['code']['vla_commit'],files={'synthetic-fixture.txt':'a'*64})
    projection=dict(schema='DEC1_SOURCE_CPU_NATIVE_RESOURCE_PROJECTION_V1',authority_release=unit.intake.AUTHORITY,
        campaign_run_uuid=current['campaign_run_uuid'],parent_projection_origin='EXISTING_LOCAL_PC',
        unknown_effects=[],native_safety_unknown_effects=[])
    def bind(*,child_root=None):
        raw=encode(current);p['budget']['DEC1_snapshot_sha256']=sha(raw)
        p['budget']['parent_snapshot_sha256']=sha(encode(p['parent']))
        p['budget']['adoption_sha256']=sha(encode(p['adoption']))
        for k in ('parent_snapshot_sha256','DEC1_snapshot_sha256','adoption_sha256','observed_epoch',
                'valid_until_epoch','DEC1_used','DEC1_reserved','parent_remaining_after_spent_reserved_and_final',
                'site_remaining','source_commissioning_used','source_commissioning_reserved','target_component_reserved'):
            projection[k]=deepcopy(p['budget'][k])
        projection_raw=encode(projection);p['budget']['native_projection_receipt_sha256']=sha(projection_raw)
        m['code']['source_seal_sha256']=sha(encode(seal))
        amendment=p.get('_amendment_bytes')
        m['authority']['amendments']=[unit.intake.AUTHORITY]+([unit.intake.VAP1_AUTHORITY] if amendment is not None else [])
        if amendment is None:m['execution']['sealed_inputs'].pop(unit.intake.VAP1_PLAN,None)
        else:m['execution']['sealed_inputs'][unit.intake.VAP1_PLAN]=sha(amendment)
        m['execution']['sealed_inputs'].update({unit.intake.PATHS['stage']:sha(encode(p['stage'])),
            unit.intake.PATHS['budget']:sha(encode(p['budget'])),unit.DEC1_SNAPSHOT:sha(raw),
            unit.PROJECTION:sha(projection_raw),unit.intake.PATHS['plan']:unit.intake.PLAN_SHA256,
            unit.intake.PATHS['adoption']:sha(encode(p['adoption'])),
            'plans/SOURCE_PARENT_RESOURCE_SNAPSHOT.private.json':sha(encode(p['parent']))})
        return dict(snapshot_bytes=raw,budget_bytes=encode(p['budget']),manifest_bytes=encode(m),
            stage_bytes=encode(p['stage']),projection_bytes=projection_raw,source_seal_bytes=encode(seal),
            child_root=root if child_root is None else child_root,now_epoch=p['now_epoch'],amendment_bytes=amendment)
    return dict(p=p,current=current,m=m,seal=seal,projection=projection,bind=bind,root=root)


def prior_row(f,*,request='prior',status='SETTLED',episodes=0,same_child=False):
    row=deepcopy(unit.propose(**f['bind']())['reservation'])
    row.update(request_id=request,status=status)
    if not same_child:row.update(child_run_uuid='old-child',child_root='/registered/old/children/old-child')
    row['requested']=dict.fromkeys(unit.intake.DIMS,0);row['requested']['episode_starts']=episodes
    row['remaining']=dict.fromkeys(unit.intake.DIMS,0)
    return row


def test_exact_all_eight_dims_copy_old_costs_and_never_mutate_inputs(facts):
    f=facts;args=f['bind']();before=deepcopy(args);old=deepcopy(f['current'])
    result=unit.propose(**args);request=f['p']['budget']['request_id'];after=result['proposed_ledger']
    assert result['status']=='PROPOSAL_ONLY_NOT_REGISTRATION_OR_AUTHORIZATION'
    assert result['native_dispatch_admitted'] is False
    assert result['original_parent_global_reservation_and_native_once_write_still_required'] is True
    assert args==before and f['current']==old
    for key,value in old.items():assert after[key]==value
    assert set(after['source_cpu_reserved'])==set(unit.intake.DIMS)
    assert after['source_cpu_reserved']==f['p']['budget']['requested']
    assert after['source_cpu_reservations'][request]==result['reservation']
    assert result['reservation']['child_manifest_sha256']==sha(args['manifest_bytes'])
    assert result['reservation']['pre_reservation_DEC1_sha256']==sha(args['snapshot_bytes'])
    assert result['reservation']['budget_sha256']==sha(args['budget_bytes'])
    assert result['source_seal_sha256']==sha(args['source_seal_bytes'])
    result['reservation']['remaining']['CoreLab_CPU_s']=0
    assert after['source_cpu_reservations'][request]['remaining']['CoreLab_CPU_s']>0


def test_first_legacy_snapshot_uses_complete_projection_not_zero_cost_guess(facts):
    f=facts;f['current'].pop('source_cpu_current_classification')
    result=unit.propose(**f['bind']())
    assert result['proposed_ledger']['CPU_s']==f['current']['CPU_s']
    assert result['proposed_ledger']['source_cpu_current_classification']==f['p']['budget']['DEC1_used']


@pytest.mark.parametrize('status',['RESERVED','RUNNING','SETTLED'])
def test_same_request_is_never_reused_even_if_settled(facts,status):
    f=facts;request=f['p']['budget']['request_id']
    row=prior_row(f,request=request,status=status)
    f['current'].update(source_cpu_reservations={request:row},source_cpu_reserved=dict.fromkeys(unit.intake.DIMS,0))
    before=deepcopy(f['current'])
    with pytest.raises(PermissionError,match='NEVER_REUSED'):unit.propose(**f['bind']())
    assert f['current']==before


@pytest.mark.parametrize('fault',['global_unknown','row_unknown','unknown_status','same_child','fifth_source',
    'aggregate','missing_aggregate','missing_dimension','bool_number','nan_number','negative_number',
    'lost_cost','cap','parent_remaining','late','future_observation','missing_closeout','missing_native_projection'])
def test_unknown_duplicate_budget_and_numeric_faults_fail_without_mutation(facts,fault):
    f=facts;p=f['p'];d=f['current'];b=p['budget']
    if fault=='global_unknown':d['unknown_effects']=['UNKNOWN_NATIVE_EFFECT']
    elif fault in ('row_unknown','unknown_status','same_child','fifth_source','aggregate','missing_aggregate'):
        row=prior_row(f,same_child=fault=='same_child',episodes=1 if fault=='fifth_source' else 0)
        if fault=='row_unknown':row['unknown_effects']=['UNKNOWN']
        if fault=='unknown_status':row['status']='UNKNOWN'
        d.update(source_cpu_reservations={'prior':row},source_cpu_reserved=dict.fromkeys(unit.intake.DIMS,0))
        if fault=='aggregate':d['source_cpu_reserved']['transfer_bytes']=1
        if fault=='missing_aggregate':d.pop('source_cpu_reserved')
    elif fault=='missing_dimension':b['requested'].pop('transfer_bytes')
    elif fault=='bool_number':b['components']['control']['local_CPU_s']=False
    elif fault=='nan_number':b['components']['control']['local_CPU_s']=float('nan')
    elif fault=='negative_number':b['components']['control']['local_CPU_s']=-1
    elif fault=='lost_cost':b['DEC1_used']['CoreLab_CPU_s']=0
    elif fault=='cap':b['DEC1_reserved']['CoreLab_CPU_s']=unit.intake.CAPS['CoreLab_CPU_s']
    elif fault=='parent_remaining':b['parent_remaining_after_spent_reserved_and_final']['transfer_bytes']=0;b['components']['control']['transfer_bytes']=1;b['requested']['transfer_bytes']=1
    elif fault=='late':p['now_epoch']=p['manifest']['limits']['operation_deadline_epoch']
    elif fault=='future_observation':b['observed_epoch']=p['now_epoch']+1
    elif fault=='missing_closeout':b['components'].pop('audit_closeout')
    elif fault=='missing_native_projection':f['projection'].pop('parent_projection_origin')
    before=deepcopy(d)
    if fault=='nan_number':
        # Encoding NaN is itself forbidden; exercise numeric validator directly.
        with pytest.raises(PermissionError):unit.intake._mapping(b['components']['control'],'bad')
    else:
        with pytest.raises(PermissionError):unit.propose(**f['bind']())
    assert d==before


@pytest.mark.parametrize('field',['snapshot_bytes','budget_bytes','manifest_bytes','stage_bytes','projection_bytes','source_seal_bytes'])
def test_bound_immutable_bytes_are_not_relabelled_or_replayed(facts,field):
    args=facts['bind']()
    if field=='manifest_bytes':
        changed=deepcopy(facts['m']);changed['data']['manifest_sha256']='0'*64;args[field]=encode(changed)
    else:args[field]+=b' '
    with pytest.raises(PermissionError):unit.propose(**args)


def adopted(f,monkeypatch):
    """Synthetic exact-delta fixture; patch SHA explicitly, never native evidence."""
    d=f['current'];p=f['p'];a=dict(schema='VAP1_ADOPTION_AND_DELTA_BUDGET_V1',
        authority_release=unit.intake.VAP1_AUTHORITY,original_T0_epoch=d['original_T0_epoch'],
        original_hard_close_epoch=unit.intake.HARD_CLOSE,old_clocks_and_counters_reset=False,
        cumulative_subcap_delta=dict(local_CPU_s=[28800,57600],CoreLab_CPU_s=[345600,576000],
            GPU_site_output_bytes=[68719476736,171798691840],
            CoreLab_output_bytes=[68719476736,137438953472],transfer_bytes=[68719476736,137438953472]))
    raw=encode(a);monkeypatch.setattr(unit.intake,'VAP1_ADOPTION_SHA256',sha(raw))
    d.update(VAP1_authority_release=unit.intake.VAP1_AUTHORITY,
        caps=dict(CoreLab_CPU_s=576000,local_CPU_s=57600),
        VAP1_cumulative_io_caps_bytes=dict(CoreLab=137438953472,transfer=137438953472))
    p['_amendment_bytes']=raw
    p['budget']['components']['main']['CoreLab_CPU_s']=350000
    p['budget']['requested']['CoreLab_CPU_s']=sum(c['CoreLab_CPU_s'] for c in p['budget']['components'].values())
    p['manifest']['limits']['CPU_s']=p['budget']['requested']['CoreLab_CPU_s']
    return raw


def parent_call(f):
    args=f['bind']()
    return unit.propose_parent(parent_bytes=encode(f['p']['parent']),
        adoption_bytes=encode(f['p']['adoption']),**args)


def test_already_adopted_exact_VAP1_caps_and_both_pure_proposals(facts,monkeypatch):
    f=facts;adopted(f,monkeypatch);old=deepcopy(f['current']);before=deepcopy(f['p']['parent'])
    result=parent_call(f)
    assert result['native_dispatch_admitted'] is False
    assert result['Core_proposal']['proposed_ledger']['CPU_s']==old['CPU_s']
    assert f['current']==old and f['p']['parent']==before
    assert result['reservation']['requested']['CoreLab_CPU_s']>unit.intake.CAPS['CoreLab_CPU_s']
    assert result['reservation']['requested']['CoreLab_CPU_s']<=576000


@pytest.mark.parametrize('fault',['missing_bytes','bad_bytes','bad_manifest_amendments','bad_manifest_seal','bad_native_caps','unadopted_bytes'])
def test_adopted_delta_and_final_manifest_cannot_be_mixed(facts,monkeypatch,fault):
    f=facts;adopted(f,monkeypatch);args=f['bind']()
    if fault=='missing_bytes':args['amendment_bytes']=None
    elif fault=='bad_bytes':args['amendment_bytes']+=b' '
    elif fault in ('bad_manifest_amendments','bad_manifest_seal'):
        changed=deepcopy(f['m'])
        if fault=='bad_manifest_amendments':changed['authority']['amendments']=[unit.intake.AUTHORITY]
        else:changed['execution']['sealed_inputs'][unit.intake.VAP1_PLAN]='0'*64
        args['manifest_bytes']=encode(changed)
    elif fault=='bad_native_caps':f['current']['caps']['CoreLab_CPU_s']=345600;args=f['bind']()
    elif fault=='unadopted_bytes':f['current'].pop('VAP1_authority_release');args=f['bind']()
    with pytest.raises(PermissionError):unit.propose(**args)


def test_parent_formal_reservations_are_added_once_and_history_inputs_stay_exact(facts):
    f=facts;parent=f['p']['parent'];before=deepcopy(parent);args=f['bind']()
    result=unit.propose_parent(parent_bytes=encode(parent),adoption_bytes=encode(f['p']['adoption']),**args)
    after=result['proposed_ledger'];req=f['p']['budget']['requested'];child=after['children'][-1]
    assert parent==before and result['before_snapshot_sha256']==sha(encode(parent))
    assert after['CPU_s']==before['CPU_s'] and after['final_reserve_inside_total']==before['final_reserve_inside_total']
    assert after['CPU_reserved_s']['CoreLab']==before['CPU_reserved_s']['CoreLab']+req['CoreLab_CPU_s']
    assert after['CPU_reserved_s']['local_PC']==before['CPU_reserved_s']['local_PC']+req['local_CPU_s']
    assert after['children'][:-1]==before['children'] and after['episode_starts']==before['episode_starts']
    assert after['physics_ticks']==before['physics_ticks'] and after['engine_starts']==before['engine_starts']
    assert child['CPU_reserved_s']==req['CoreLab_CPU_s'] and child['local_CPU_reserved_s']==req['local_CPU_s']
    assert child['episode_starts_reserved']==4 and child['physics_ticks_reserved']==req['physics_ticks']
    assert child['engine_starts_reserved']==req['constructors']
    assert set(after['source_cpu_reserved'])==set(unit.intake.DIMS)
    assert after['source_cpu_reservations'][f['p']['budget']['request_id']]==result['reservation']
    # Existing100 CoreCPU is held once by the top total and its existing child,
    # not charged a second time because the source8dim index is also present.
    assert sum(c['CPU_reserved_s'] for c in after['children'])==after['CPU_reserved_s']['CoreLab']


@pytest.mark.parametrize('fault',['same_request','same_request_settled','same_child','unknown','child_unknown',
    'caps','final_floor','epoch','global_reserved_doublecount','missing_IO_projection','stale_bytes'])
def test_parent_reuse_unknown_global_limits_and_complete_projection_reject(facts,fault):
    f=facts;p=f['p'];parent=p['parent']
    if fault in ('same_request','same_request_settled'):
        row=deepcopy(unit.propose(**f['bind']())['reservation'])
        if fault=='same_request_settled':row.update(status='SETTLED',remaining=dict.fromkeys(unit.intake.DIMS,0))
        parent['source_cpu_reservations']={row['request_id']:row}
        parent['source_cpu_reserved']=deepcopy(row['remaining'])
    elif fault=='same_child':parent['children'].append(dict(name=f['m']['run_uuid'],CPU_reserved_s=0))
    elif fault=='unknown':parent['unknown_effects']=['UNKNOWN']
    elif fault=='child_unknown':parent['children'][0]['unknown_effects']=['UNKNOWN']
    elif fault=='caps':parent['CPU_reserved_s']['CoreLab']=320*3600
    elif fault=='final_floor':parent['final_reserve_inside_total']['episode_starts']=0
    elif fault=='epoch':parent['T0_epoch']+=1
    elif fault=='global_reserved_doublecount':p['budget']['parent_remaining_after_spent_reserved_and_final']['CoreLab_CPU_s']-=100
    elif fault=='missing_IO_projection':p['budget']['parent_remaining_after_spent_reserved_and_final'].pop('transfer_bytes')
    args=f['bind']();raw=encode(parent)
    if fault=='stale_bytes':raw+=b' '
    old=deepcopy(parent)
    with pytest.raises(PermissionError):unit.propose_parent(parent_bytes=raw,adoption_bytes=encode(p['adoption']),**args)
    assert parent==old


def test_closed_external_data_spend_stays_global_and_is_not_science_episodes(facts,monkeypatch):
    """Synthetic ledger shapes: external data costs are not a newsource debit."""
    f=facts;adopted(f,monkeypatch);p=f['p'];parent=p['parent'];b=p['budget']
    parent['CPU_s']['CoreLab']=40000.0
    parent.update(episode_starts=48,physics_ticks=dict(live=747400,shadow=747438,reconstruction=0),
        engine_starts=4235,network_transfer_accounted_bytes=6575773950,Core_peak_bytes=13031636552)
    parent['children'].append(dict(name='DATA_SEPARATE_SUBCAP_CLOSED',status='CLOSED',CPU_reserved_s=0,
        episode_starts_reserved=0,physics_ticks_reserved=0,engine_starts_reserved=0,
        actual_data_episode_starts=48,unknown_effects=[]))
    b['DEC1_reserved'].update(CoreLab_CPU_s=31840.520213722,
        CoreLab_output_bytes=13031636552,transfer_bytes=6575773950)
    b['parent_remaining_after_spent_reserved_and_final'].update(
        CoreLab_CPU_s=320*3600-40000.0-100,episode_starts=900-48,
        physics_ticks=40000000-3200000-747400-747438,constructors=100000-4235,
        transfer_bytes=300*1024**3-6575773950)
    before=deepcopy(parent);result=parent_call(f);after=result['proposed_ledger']
    core=result['Core_proposal']['proposed_ledger'];request=b['requested']
    assert after['CPU_s']==before['CPU_s'] and after['episode_starts']==48
    assert after['physics_ticks']==before['physics_ticks'] and after['engine_starts']==4235
    assert after['network_transfer_accounted_bytes']==6575773950 and after['Core_peak_bytes']==13031636552
    assert core['CPU_s']==540 and core['episode_starts']==0 and core['physics_ticks']==0
    assert core['source_cpu_reserved']==request  # Independent holding is not added again here.
    assert after['source_cpu_reserved']==request and request['episode_starts']==4
    assert after['children'][-1]['episode_starts_reserved']==4
