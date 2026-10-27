"""Exact real failure plus temporary original heartbeat/effect IO; CPU only."""
import base64,json,time
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_training_bootstrap_failure as f,amd_training_runtime as runtime

B=Path(__file__).resolve().parents[4]/'work/private/v5-execution-20261025'
@pytest.fixture
def capture():
    p=B/'GEN1_INDEPENDENT_TRAIN_01_FAILED1497713_FULL_INPUTS_READONLY.private.json'
    if not p.exists():pytest.skip('private actual1497713 native final capture unavailable')
    return json.loads(p.read_bytes())

def decoded(c,name):return json.loads(base64.b64decode(c['files'][name]['base64']))

def test_actual_f_Q_R_causal_failure_keeps_real_allocation_and_missing_effects(capture):
    v=f.audit(capture)
    assert v['native']['conservative_GPU_s']==86 and v['native']['conservative_CPU_s']==122
    assert v['effects']['optimizer_updates_committed']==v['effects']['query_attempts']==v['effects']['model_load_attempts']==0
    assert v['effects']['original_native_effect_ledger_created']is False
    assert v['audit']['actual_neural_output_bytes']==0 and v['audit']['actual_retained_control_bytes']==16596195
    assert 'effects'not in decoded(capture,'PBS/PBS_PROCESS_FINAL.json')

@pytest.mark.parametrize('bad',['active','other_request','zero_allocated_GPU','owner_active','model_final','other_error','payload','state','source','unpaid_output'])
def test_no_generic_absence_or_other_phase_converted_to_known_zero(capture,bad):
    c=deepcopy(capture)
    if bad=='active':c['native'][f.JOB]['job_state']='R'
    elif bad=='other_request':c['native'][f.JOB]['Variable_List']['PV1_REQUEST_ID']='wrong'
    elif bad=='zero_allocated_GPU':c['native'][f.JOB]['Resource_List']['ngpus']=0
    elif bad=='owner_active':c['actors'][0]['actual_procstart']=c['actors'][0]['expected_procstart']
    elif bad=='payload':c['inventory']['training']=['training/checkpoint-512/model.safetensors']
    elif bad=='source':next(iter(c['modules'].values()))['sha256']='0'*64
    elif bad=='unpaid_output':c['retained']['logical_bytes']=64*1024**2+1
    else:
        name='control/OWNER_HEARTBEAT.json'if bad=='state'else'PBS/PBS_PROCESS_FINAL.json';x=decoded(c,name)
        if bad=='model_final':x['phase']='GEN1_INDEPENDENT_1024_TRAINING'
        elif bad=='other_error':x['error']['message']='some unrelated error'
        else:x['native_job_state']='R'
        b=f.a.raw(x);c['files'][name]=dict(bytes=len(b),sha256=f.a.sha(b),base64=base64.b64encode(b).decode())
    with pytest.raises((PermissionError,KeyError,ValueError)):f.audit(c)

def test_exact_prerelease_settlement_does_not_refund_paid_bytes_or_original_work(capture):
    before=json.loads(base64.b64decode(capture['GLOBAL_base64']));v=f.settle(before,capture);out=v['proposed_global']
    assert out['used']['H200_s']==150429 and out['used']['PBS_submissions']==33
    assert out['CPU_s']==before['CPU_s']+122 and out['CPU_reserved_s']==before['CPU_reserved_s']-54000
    assert out['new_storage_current_bytes']==before['new_storage_current_bytes']
    assert out['new_storage_peak_bytes']==before['new_storage_peak_bytes']
    assert out['storage_reserved_bytes']==before['storage_reserved_bytes']-70000000000
    assert out[f.a.original.PACKAGE]==before[f.a.original.PACKAGE]
    assert out['optimizer_updates']==13762 and out['generation_queries']==4482
    assert all(out['PBS_submissions'][i]==r for i,r in enumerate(before['PBS_submissions'][:-1]))
    row=f.a.rows(out)[0];assert row['neural_output_charge_bytes']==0 and row['additional_output_upper_bytes']==70000000000
    p=B/'GEN1_INDEPENDENT_TRAIN_01_PREPARED.private.json';prepared=json.loads(p.read_bytes())
    local=base64.b64decode(json.loads(base64.b64decode(prepared['small']['plans/GEN1_OWNER_RESOURCES.private.json']))['historical_LOCAL_base64'])
    proof=f.a.scope.full_forecast(f.a.raw(out),local)
    assert proof['cap_changes_required']=={'PBS_submissions':dict(before=36,after=37,after_GiB=None)}
    assert proof['matrix3_and_final20GPUh40GiB_retained']
    with pytest.raises(PermissionError):f.settle(out,capture)

@pytest.fixture
def live(tmp_path,capture,monkeypatch):
    root=tmp_path/'campaign/children/gen1-layout-v0-corr1024-01';(root/'control').mkdir(parents=True)
    g=base64.b64decode(capture['GLOBAL_base64']);(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json').write_bytes(g)
    lease=root/'lease.json';lease.write_text(json.dumps(dict(status='ACTIVE')))
    (root/'OUTPUT_BINDING.private.json').write_text(json.dumps(dict(lease=str(lease))))
    beat=decoded(capture,'control/OWNER_HEARTBEAT.json');(root/'control/OWNER_HEARTBEAT.json').write_text(json.dumps(beat))
    alloc=decoded(capture,'PBS/ALLOCATION_ADMISSION.json');monkeypatch.setattr(time,'time',lambda:alloc['allocation_start_epoch']+10)
    return root,alloc,beat

def test_actual_saved_Q_waits_then_same_identity_R_passes_without_releasing_Q(live):
    root,alloc,beat=live
    with pytest.raises(runtime.StartupOwnerStillQueued):runtime.guard(root,alloc,startup=True)
    with pytest.raises(PermissionError):runtime.guard(root,alloc)
    clock=[0.];polls=[]
    def sleep(seconds):
        polls.append(seconds);clock[0]+=seconds;beat['native_job_state']='R';(root/'control/OWNER_HEARTBEAT.json').write_text(json.dumps(beat))
    runtime.wait_initial_running_owner(root,alloc,clock=lambda:clock[0],sleep=sleep)
    assert polls==[.2]and not(root/'A40_NATIVE_EFFECT_LEDGER.json').exists()
    runtime.guard(root,alloc)

def test_initial_Q_wait_keeps180_deadline_and_never_accepts_stale_or_wrong_owner(live):
    root,alloc,beat=live;clock=[0.]
    def sleep(seconds):clock[0]+=181
    with pytest.raises(TimeoutError,match='ORIGINAL_180S'):
        runtime.wait_initial_running_owner(root,alloc,clock=lambda:clock[0],sleep=sleep)
    beat['source_sha']='0'*40;(root/'control/OWNER_HEARTBEAT.json').write_text(json.dumps(beat))
    with pytest.raises(PermissionError):runtime.wait_initial_running_owner(root,alloc,clock=lambda:0.,sleep=lambda _:pytest.fail('wrong owner must not wait'))
    beat['source_sha']=f.SOURCE;beat['observed_epoch']-=400;(root/'control/OWNER_HEARTBEAT.json').write_text(json.dumps(beat))
    from scripts.pi0294.mp1.owner_heartbeat import HeartbeatLost
    with pytest.raises(HeartbeatLost):runtime.wait_initial_running_owner(root,alloc,clock=lambda:0.,sleep=lambda _:pytest.fail('stale owner must not wait'))

def test_early_allocated_zero_effect_journal_has_original_summary_without_model_permission(live,capture):
    root,alloc,_=live;request=decoded(capture,'PBS/REQUEST.json');d=root/'pbs/alloc-33';d.mkdir(parents=True);(d/'REQUEST.json').write_text(json.dumps(request));(root/'plans').mkdir()
    prepared=json.loads((B/'GEN1_INDEPENDENT_TRAIN_01_PREPARED.private.json').read_bytes());recipe=base64.b64decode(prepared['small']['plans/PRIMARY_LEARNING_RECIPES.json']);(root/'plans/PRIMARY_LEARNING_RECIPES.json').write_bytes(recipe)
    ledger=runtime.Effects(root,alloc);ledger.initialize();v=ledger.summary()
    assert v['schema']==f.a.EFFECTS and v['model_load_attempts']==v['query_attempts']==v['optimizer_updates_committed']==0
    assert v['pending_updates']==v['unknown_query_intents']==[]
    with pytest.raises(FileExistsError):ledger.initialize()
    import ast,inspect
    tree=ast.parse(inspect.getsource(runtime.run));nodes=[n for n in ast.walk(tree)if isinstance(n,ast.Call)]
    init=next(n.lineno for n in nodes if isinstance(n.func,ast.Attribute)and n.func.attr=='initialize')
    wait=next(n.lineno for n in nodes if isinstance(n.func,ast.Name)and n.func.id=='wait_initial_running_owner')
    assert init<wait
