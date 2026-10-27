"""Real temporary original journals and CPU AdamW, never native model proof."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
from scripts.pi0294.test_minimal_execution import actual,request
from scripts.pi0294.test_pv1_recipe import old_recipe
from scripts.pi0294.generalization import minimal_execution as e,minimal_training as t
from scripts.pi0294.generalization.minimal_effects import Effects
from scripts.pi0294.e1.journal import Journal,MutationOutcomeUnknown
from scripts.pi0292.durable import read,write


@pytest.fixture
def effects(tmp_path,actual):
    g,cap=actual;r=request(g,cap);p=tmp_path/'pbs/alloc-35/REQUEST.json';p.parent.mkdir(parents=True)
    p.write_bytes(e.raw(r));p.chmod(0o400)
    allocation=dict(job_id='12345.stdct-mgmt-02',request_id=r['request_id'],
        request_sha256=e.sha(p.read_bytes()),source_sha=r['source_sha'])
    ledger=Effects(tmp_path,allocation,request_path=p,guard=lambda:None)
    ledger.initialize();(tmp_path/'plans').mkdir()
    write(tmp_path/'plans/FROZEN_SOURCE_PAIR_RECIPE.json',old_recipe())
    return ledger


def draw(ledger,identifier,*,fail=False):
    journal=Journal(ledger.root/'reservations/probe-generations',ledger.root.name)
    def operation():
        ledger.charge('generation_queries',1)
        if fail:raise RuntimeError('CPU_FIXTURE_LOST_REPLY_NOT_A_REAL_MODEL')
        return dict(CPU_journal_fixture=True)
    return journal.once(identifier,dict(input_fixture=identifier),operation)


def finish(ledger,identifier):
    spec=next(x for x in e.native_sessions()if x['id']==identifier)
    row=read(ledger.path)['sessions'][identifier];path=ledger.root/(identifier+'-fixture-terminal.json')
    write(path,dict(schema='ARX5_MINIMAL_RUNTIME_SESSION_TERMINAL_V1',session_id=identifier,arm=spec['arm'],
        request_sha256=ledger.allocation['request_sha256'],raw_parent_files_identity=t.RAW_PARENT_FILES_ID,
        statistics_sha256=e.R0,optimizer_updates=row['optimizer_updates'],query_attempts=row['generation_queries'],
        cursor=spec['sample_visits'],status='COMPLETE_SAVED_REQUIRES_INDEPENDENT_OWNER_AUDIT'))
    path.chmod(0o400);ledger.complete_session(identifier,path)


def E0(effects):
    sid=e.native_sessions()[0]['id'];effects.start_session(sid);effects.charge('model_loads',1)
    for i in range(39):draw(effects,'E0-CPU-'+str(i))
    finish(effects,sid)


def test_original_pre_effect_query_intent_unknown_survives_without_replay(effects):
    effects.start_session(e.native_sessions()[0]['id']);effects.charge('model_loads',1)
    with pytest.raises(RuntimeError):draw(effects,'E0-LOST-CPU',fail=True)
    assert effects.summary()['unknown_query_intents']==[0]
    before=effects.path.read_bytes()
    with pytest.raises(MutationOutcomeUnknown):draw(effects,'E0-LOST-CPU')
    assert effects.path.read_bytes()==before
    with pytest.raises(PermissionError):effects.charge('model_loads',1)
    with pytest.raises(PermissionError):effects.start_session(e.native_sessions()[1]['id'])
    assert effects.path.read_bytes()==before


def test_same_original_journal_counts_completed_E0_and_new_session_independently(effects):
    E0(effects);summary=effects.summary()
    assert summary['query_attempts']==39 and summary['model_load_attempts']==1
    assert summary['optimizer_updates_committed']==0 and not summary['all_five_sessions_runtime_complete']
    assert summary['independently_audited']is False
    with pytest.raises(PermissionError):effects.start_session(e.native_sessions()[0]['id'])
    with pytest.raises(PermissionError):effects.start_session(e.native_sessions()[2]['id'])
    effects.start_session(e.native_sessions()[1]['id'])
    assert effects.known_effects(e.native_sessions()[1]['id'])['optimizer_updates']==0
    with pytest.raises(PermissionError):effects.known_effects(e.native_sessions()[0]['id'])


def test_two_real_CPU_warmup_calls_are_charged_and_cannot_become_fresh_formal(effects):
    import torch
    from test_minimal_arx5_adapter_contract import fixture_pair
    from scripts.pi0294.e1.trainer import learning_rate_multiplier
    E0(effects);sid=e.native_sessions()[1]['id'];effects.start_session(sid);effects.charge('model_loads',1)
    recipe=t.recipe(old_recipe(),t.ARMS[0],raw_parent_manifest_sha256=t.RAW_PARENT_FILES_ID)
    with pytest.raises(PermissionError):effects.begin_update(sid,0,recipe)
    for i in range(15):draw(effects,'E1-CPU-ZERO-'+str(i))
    _,model,_,contract,_=fixture_pair();before={n:p.detach().clone()for n,p in model.named_parameters()}
    optimizer=torch.optim.AdamW(contract.parameters(),lr=1e-5)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda i:learning_rate_multiplier(i,1024,64))
    for step in range(2):
        effects.begin_update(sid,step,recipe)
        with pytest.raises(PermissionError):effects.begin_update(sid,step,recipe)
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(2,1,4),torch.tensor([10,10])).square().mean().backward()
        optimizer.step();scheduler.step();effects.commit_update(sid,step+1)
        if step==0:assert all(torch.equal(p,before[n])for n,p in model.named_parameters())
        with pytest.raises(PermissionError):effects.commit_update(sid,step+1)
    assert any(not torch.equal(p,before[n])for n,p in model.named_parameters()if p.requires_grad)
    assert effects.summary()['optimizer_updates_committed']==2
    assert effects.known_effects(sid)['optimizer_updates']==2
    with pytest.raises(PermissionError):effects.begin_update(sid,0,recipe)
    with pytest.raises(PermissionError):effects.begin_update(sid,2,recipe)
    for i in range(2):draw(effects,'E1-CPU-RESTORE-'+str(i))
    finish(effects,sid)
    assert effects.summary()['query_attempts']==56
    assert effects.summary()['sessions'][sid]['optimizer_updates']==2
    assert effects.summary()['sessions'][e.native_sessions()[3]['id']]['optimizer_updates']==0
    with pytest.raises(PermissionError):effects.start_session(e.native_sessions()[3]['id'])


@pytest.mark.parametrize('kind',['aggregate','pending','unknown','reorder','overquery','future_effect'])
def test_tampered_or_unknown_session_effects_never_clear_their_fence(effects,kind):
    v=read(effects.path);before=deepcopy(v);sid=e.native_sessions()[0]['id']
    if kind=='aggregate':v['generation_queries']=1
    elif kind=='pending':v['pending_updates']=[dict(request_id='OTHER')]
    elif kind=='unknown':v['unknown_effects']=['PRESERVED']
    elif kind=='reorder':v['active_session']=e.native_sessions()[2]['id'];v['sessions'][v['active_session']]['status']='ACTIVE'
    elif kind=='overquery':v['generation_queries']=v['sessions'][sid]['generation_queries']=40
    else:v['sessions'][e.native_sessions()[4]['id']]['model_loads']=1;v['model_loads']=1
    changed=deepcopy(v)
    with pytest.raises(PermissionError):effects.check(v)
    assert v==changed and read(effects.path)==before


def test_new_training_call_refuses_real_local_host_before_Trainer_or_model(effects,monkeypatch):
    from scripts.pi0294.generalization import minimal_train_session as runtime
    monkeypatch.delenv('PBS_JOBID',raising=False)
    with pytest.raises(PermissionError,match='NO_MODEL_ON_LOGIN'):
        runtime.run_training(effects.root,None,effects.allocation,effects,e.native_sessions()[1]['id'],
            guard=lambda:None,authorize_training=lambda _:pytest.fail('No release call on local PC'))
