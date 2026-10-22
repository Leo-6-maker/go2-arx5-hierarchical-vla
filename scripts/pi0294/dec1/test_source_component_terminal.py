"""Small synthetic supplied-bytes audit fixtures; helper never ran/imported.

No native or model evidence is produced. Tests exercise terminal integrity and
reuse the existing ACK CPU fixture for ONE complete prefix, not 80 fake episodes.
"""
from copy import deepcopy
from io import BytesIO
import hashlib
import json

import numpy as np
import pytest
from scripts.pi0292.durable import read,write
from scripts.pi0294.dec1 import source_component_terminal as unit
from scripts.pi0294.dec1 import source_component_anchor as anchor_unit
from scripts.pi0294.dec1.source_component_evidence import SourceComponentEvidence
from scripts.pi0294.dec1.source_component_evidence import canonical,SCHEMA as EVENT_SCHEMA,ZERO
from scripts.pi0294.dec1.test_source_commissioning_plan import population
from scripts.pi0294.dec1.test_source_component_ack import setup,completed,FakeAdapter
from types import SimpleNamespace


def encode(value): return unit._bytes(value)


def npz(**values):
    stream=BytesIO();np.savez_compressed(stream,**values);return stream.getvalue()


def chain(events,episode,executor):
    frames=[];previous=ZERO
    for sequence,event in enumerate(events):
        body=dict(schema=EVENT_SCHEMA,episode_id=episode,executor_id=executor,
            sequence=sequence,previous_sha256=previous,event=event)
        previous=hashlib.sha256(canonical(body)).hexdigest()
        frames.append(canonical(dict(body,sha256=previous))+b'\n')
    raw=b''.join(frames)
    return raw,dict(offset=len(raw),sequence=len(events),previous=previous)


@pytest.fixture
def fixture(population,monkeypatch):
    data,convert=population;population_raw=convert(data)
    plan=unit.plans.build(population_raw,owner_release='a'*40)
    plan_raw=(json.dumps(plan,sort_keys=True)+'\n').encode()
    monkeypatch.setattr(unit,'PLAN_SHA256',unit.sha(plan_raw));monkeypatch.setattr(unit,'PLAN_OWNER_RELEASE','a'*40)
    c=plan['conditions'][0];cid=c['condition_id'];executor='CPU_SOURCE'
    manifest=encode(dict(scope=unit.ROLE,code=dict(vla_commit='b'*40)))
    raw=[[.4,0.,.4,1.,0.,0.,0.,1.,0.,.5] for _ in range(40)]
    binding=dict(schema=unit.PLAN_SCHEMA,component_plan_sha256=unit.PLAN_SHA256,
        condition_id=cid,interface_mode=c['interface'],policy=unit.POLICY,purpose=unit.PURPOSE,
        episode_id=cid,executor_id=executor,sequence=0,tick_before=0,
        raw_chunk_id=cid+':ref0',raw_h40_sha256=unit.raw_h40_hash(raw),raw_H40_eef10=raw,
        frame=unit.FRAME,tcp=unit.TCP,offsets_s=[i/20 for i in range(40)],observation_timestamp=0.)
    owner=dict(component_plan_sha256=unit.PLAN_SHA256,plan_owner_release='a'*40,runtime_release='b'*40,
        condition_id=cid,card_sha256=c['card_sha256'],interface_mode=c['interface'],episode_id=cid,
        executor_id=executor,policy=unit.POLICY,purpose=unit.PURPOSE,execution_semantics=unit.SEMANTICS,
        student_neural_ACK_qualification=False,frame=unit.FRAME,tcp=unit.TCP)
    stop=dict(schema=unit.STOP_SCHEMA,status='LATCHED_READBACK_RECORD_ACK_REQUIRES_OWNER_RETURN',
        episode_id=cid,executor_id=executor,stopped=True,owned_live_dispatch_latched=True,
        native_frame_inflight=False,source_step_sha256=unit.ORIGINAL_13['scripts/pi0291_arx5_data/engine.py'],
        native_state_sha256='e'*64,native_steps=0,sim_time_s=0.)
    result=dict(id=cid,source_sha='b'*40,card_sha256=c['card_sha256'],fixture=False,
        status='FAILED_PRESERVED',physics_steps=0,rgb_pairs=1,score=dict(persistent_success=True))
    s1=dict(version=unit.ORIGINAL_S1_VERSION,execution_path='DEC1_SOURCE_SINGLE_POSE_MAPPED_PREFIX8',
        episode_id=cid,executor_id=executor,policy=unit.POLICY,purpose=unit.PURPOSE,interface_mode=c['interface'],
        runtime_release='b'*40,component_plan_sha256=unit.PLAN_SHA256,plan_owner_release='a'*40,
        engineering_unknown=False,events_readback_valid=True,terminal_stop_error=None,
        execution_state=dict(unknown=dict(shadow=False,live=False)),component_meter_unknown=False,
        component_ack_state=dict(unknown=False,sequence=0,pending_raw_plan=True),
        actual_stop_state=dict(stop_acknowledged=True,unknown=False,latched=True),
        shadow_integrations=0,native_live_integrations=0,live_integrations_lower_bound=0,
        shadow_and_live_finish_records=[],actual_completed_prefixes=[],actual_25tick_ACKs=[],
        pre_execution_rejections=0,known_partial_S1_rejections=0,live_native_failures=0,
        maximum_live_shadow_difference=None)
    row=dict(schema='DEC1_SOURCE_COMPONENT_ATTEMPT_V1',attempt_id=cid,condition_id=cid,
        interface=c['interface'],policy=unit.POLICY,component_plan_sha256=unit.PLAN_SHA256,
        native_STOP_return_and_owner_event_confirmed=True,student_neural_ACK_qualification=False,
        GR00T_queries=0,new_training_data=False,selected='NONE',deployment='NONE',main_integrations=0,
        shadow_integrations=0,completed_prefixes=0,RGB_pairs=1,status='KNOWN_COMPONENT_SAFETY_OR_REFERENCE_REJECT')
    h=dict(plan=plan_raw,population=population_raw,manifest=manifest,c=c,cid=cid,executor=executor,
        result=result,s1=s1,row=row,stop=stop,owner=owner,binding=binding,
        events=[dict(kind='SOURCE_FULL_H40_GATE',reference_id=cid+':ref0',decision=dict(decision='REJECT'))],
        files={},main=np.zeros((0,20)))
    return h


def seal(h):
    """Synthetic owner capture only; actual IO adapter must use live anchors."""
    files=dict(h['files']);files['SOURCE_COMPONENT_STOP.json']=encode(h['stop'])
    stop_event=dict(kind='SOURCE_STOP_ACK',episode_id=h['cid'],executor_id=h['executor'],
        receipt_id='SOURCE_COMPONENT_STOP.json#sha256='+unit.sha(files['SOURCE_COMPONENT_STOP.json']))
    events=h['events']+[stop_event];h['s1']['actual_stop_event']=stop_event
    h['s1']['full_H40_gate_results']=[e for e in events if e.get('kind')=='SOURCE_FULL_H40_GATE']
    raw,ea=chain(events,h['cid'],h['executor']);files['SOURCE_COMPONENT_EVENTS.jsonl']=raw
    files['RESULT.json']=encode(h['result']);files['S1_RESULT.json']=encode(h['s1'])
    h['row'].update(result_sha256=unit.sha(files['RESULT.json']),S1_sha256=unit.sha(files['S1_RESULT.json']))
    files[unit.PREFIX+'OWNER_COMPONENT_BINDING.json']=encode(h['owner'])
    files.setdefault(unit.PREFIX+'plans/000.json',encode(h['binding']))
    files['highrate.npz']=npz(rows=h['main'],nq=np.array(2),nv=np.array(2))
    anchor=encode(dict(schema=unit.ANCHOR_SCHEMA,condition_id=h['cid'],component_plan_sha256=unit.PLAN_SHA256,
        manifest_sha256=unit.sha(h['manifest']),runtime_release='b'*40,owner_closeout_known=True,
        engine_sha256=unit.ORIGINAL_13['scripts/pi0291_arx5_data/engine.py'],
        episode_id=h['cid'],executor_id=h['executor'],interface=h['c']['interface'],
        events=dict(ea,bytes_sha256=unit.sha(raw)),
        files={k:unit.sha(v) for k,v in files.items()}))
    h['row']['terminal_anchor_sha256']=unit.sha(anchor)
    files['DEC1_SOURCE_COMPONENT_ATTEMPT.json']=encode(h['row'])
    return files,anchor


def run(h,files,anchor,*,with_anchor=True):
    return unit.audit(h['plan'],h['population'],h['manifest'],manifest_sha256=unit.sha(h['manifest']),
        attempts={h['cid']:files},anchors={h['cid']:anchor} if with_anchor else {},
        anchor_sha256={h['cid']:unit.sha(anchor)} if with_anchor else {})


def test_four_slots_not_run_and_teacher_success_never_populates_student_cells(fixture):
    h=fixture
    empty=unit.audit(h['plan'],h['population'],h['manifest'],manifest_sha256=unit.sha(h['manifest']),
        attempts={},anchors={},anchor_sha256={})
    assert empty['component_counts']=={'NOT_RUN':4}
    files,anchor=seal(h);out=run(h,files,anchor)
    assert out['conditions'][0]['status']=='KNOWN_REFERENCE_OR_S1_REJECT'
    assert out['conditions'][0]['Teacher_task_success_diagnostic'] is True
    assert not out['student_neural_ACK_qualification'] and not out['target_qualification'] and out['core64_cells']==0


@pytest.mark.parametrize('fault',['anchor-missing','result-source','stop-return','interface','counter','unknown','fraud-complete','extra-ack'])
def test_missing_or_self_consistent_wrong_terminal_identity_cannot_pass(fixture,fault):
    h=fixture
    if fault=='result-source':h['result']['source_sha']='f'*40
    if fault=='stop-return':h['row']['native_STOP_return_and_owner_event_confirmed']=False
    if fault=='interface':h['s1']['interface_mode']='I_PROGRESS' if h['c']['interface']=='I_TIME' else 'I_TIME'
    if fault=='counter':h['s1']['native_live_integrations']=1
    if fault=='unknown':h['s1']['engineering_unknown']=True
    if fault=='fraud-complete':h['row']['status']='COMPONENT_COMPLETE_NOT_STUDENT_EVALUATION';h['result']['status']='VALID_COMPLETE'
    if fault=='extra-ack':h['files'][unit.PREFIX+'acks/000.json']=encode(dict(fake=True))
    files,anchor=seal(h);out=run(h,files,anchor,with_anchor=fault!='anchor-missing')
    assert out['conditions'][0]['status']=='ENGINEERING_INCOMPLETE_UNKNOWN'


def test_event_complete_line_truncation_cannot_derive_a_new_self_confirmed_tail(fixture):
    h=fixture;files,anchor=seal(h);a=unit.document(anchor);raw=files['SOURCE_COMPONENT_EVENTS.jsonl']
    trimmed=b'\n'.join(raw.splitlines()[:-1])+b'\n'
    with pytest.raises(ValueError):
        unit.verify_event_bytes(trimmed,episode_id=h['cid'],executor_id=h['executor'],maximum_bytes=128*1024**2,
            expected_offset=a['events']['offset'],expected_sequence=a['events']['sequence'],expected_previous=a['events']['previous'])


@pytest.mark.parametrize('kind',['s1','native'])
def test_known_partial_s1_and_actual_native_safety_flag_have_distinct_component_outcomes(fixture,kind):
    h=fixture;h['events'][0]['decision']['decision']='ACCEPT';ends=[]
    for domain,count,status in ([('shadow',37,'S1_REJECT')] if kind=='s1'
            else [('shadow',200,'PASS'),('live',1,'NATIVE_SAFETY_ERROR')]):
        token=domain+'-partial';h['events'].append(dict(kind='COMPONENT_PREFIX_RESERVED',token=token))
        h['events'].extend(dict(kind='COMPONENT_PREFIX_NATIVE_EFFECT',token=token,domain=domain,
            actual_increment=1,actual_lower_bound=i+1,unknown=False) for i in range(count))
        end=dict(kind='COMPONENT_PREFIX_FINISHED',token=token,domain=domain,known_ticks=count,
            reservation_retained=False,status=status);h['events'].append(end);ends.append(end)
    shadow=37 if kind=='s1' else 200;live=int(kind=='native')
    h['s1'].update(shadow_and_live_finish_records=ends,shadow_integrations=shadow,
        native_live_integrations=live,live_integrations_lower_bound=live,
        pre_execution_rejections=int(kind=='s1'),known_partial_S1_rejections=int(kind=='s1'),
        live_native_failures=int(kind=='native'))
    h['row'].update(main_integrations=live,shadow_integrations=shadow)
    if kind=='native':
        h['result'].update(status='SAFETY_STOP_PRESERVED',physics_steps=1)
        h['row']['status']='COMPONENT_NATIVE_SAFETY_FAIL';h['stop'].update(native_steps=1,sim_time_s=.002)
        h['main']=np.zeros((1,20));h['main'][0,0]=.002;h['main'][0,-1]=1
    files,anchor=seal(h);out=run(h,files,anchor)
    assert out['conditions'][0]['status']==('KNOWN_REFERENCE_OR_S1_REJECT' if kind=='s1' else 'NATIVE_SAFETY_FAIL')


def test_one_real_cpu_ack_contract_proof_and_npz_is_independently_linked_to_terminal(fixture,setup):
    h=fixture;adapter=FakeAdapter(h['c']['interface']);adapter.episode_id=h['cid']
    cpu=setup(episode_id=h['cid'],executor_id=h['executor'],adapter=adapter)
    cpu.adapter=adapter;cpu.engine.model.nu=8
    cpu.wrapped.plan(cpu.state);unused,proof=completed(cpu)
    events=read(cpu.events)
    gates=[dict(kind='SOURCE_FULL_H40_GATE',reference_id=proof['raw_chunk_id'],decision=dict(decision='ACCEPT')) for _ in range(17)]
    gates.extend(dict(kind='SOURCE_FULL_H40_GATE',reference_id=r['command_id'],decision=dict(decision='ACCEPT'))
        for r in proof['actual_reference_acks'] for _ in range(2))
    meters=[];finished=[]
    for domain in ('shadow','live'):
        token=domain+'0';meters.append(dict(kind='COMPONENT_PREFIX_RESERVED',token=token))
        meters.extend(dict(kind='COMPONENT_PREFIX_NATIVE_EFFECT',token=token,domain=domain,
            actual_increment=1,actual_lower_bound=i+1,unknown=False) for i in range(200))
        end=dict(kind='COMPONENT_PREFIX_FINISHED',token=token,domain=domain,known_ticks=200,
            reservation_retained=False,status='PASS');meters.append(end);finished.append(end)
    before=[json.loads(canonical(e)) for e in gates+meters+events];write(cpu.events,before)
    rows=np.zeros((200,20));rows[:,0]=np.arange(1,201)*.002
    receipt=cpu.owner.acknowledge(cpu.engine,rows,proof)
    returned=dict(kind='SOURCE_PROTOCOL_ACK_READBACK',episode_id=h['cid'],executor_id=h['executor'],
        nonce=proof['nonce'],receipt_id=receipt.receipt_id,
        acknowledgement_sha256=receipt.acknowledgement_sha256,physics_receipt_sha256=receipt.physics_receipt_sha256)
    cpu.state['time_s']=.4;cpu.wrapped.plan(cpu.state)
    h['files']={unit.PREFIX+p.relative_to(cpu.directory).as_posix():p.read_bytes() for p in cpu.directory.rglob('*') if p.is_file()}
    h['owner']=read(cpu.directory/'OWNER_COMPONENT_BINDING.json');h['main']=rows
    h['events']=before+[returned,dict(kind='SOURCE_FULL_H40_GATE',reference_id=h['cid']+':ref1',decision=dict(decision='REJECT'))]
    h['stop'].update(native_steps=200,sim_time_s=.4)
    h['result']['physics_steps']=200
    h['row'].update(main_integrations=200,shadow_integrations=200,completed_prefixes=1)
    h['s1'].update(shadow_integrations=200,native_live_integrations=200,live_integrations_lower_bound=200,
        shadow_and_live_finish_records=finished,actual_completed_prefixes=[dict(kind='SOURCE_PREFIX200_ACK',**proof)],
        actual_25tick_ACKs=[e for e in before if e['kind']=='SOURCE_ACTUAL_REFERENCE_ACK'],
        component_ack_state=cpu.owner.state(),maximum_live_shadow_difference=0.)
    files,anchor=seal(h);out=run(h,files,anchor)
    assert out['conditions'][0]['status']=='KNOWN_REFERENCE_OR_S1_REJECT'
    assert out['conditions'][0]['verified_completed_prefixes']==1
    broken=dict(files);broken[unit.PREFIX+'physics/000.npz']=npz(rows=rows+1,tick_before=0,tick_after=200)
    assert run(h,broken,anchor)['conditions'][0]['status']=='ENGINEERING_INCOMPLETE_UNKNOWN'


@pytest.fixture
def closed_owner(fixture,tmp_path):
    """Real CPU event writer state; explicit fake owner prerequisites, no native."""
    h=fixture;files,unused=seal(h);root=tmp_path/'owner';folder=root/'episodes'/h['cid'];folder.mkdir(parents=True)
    (root/'RUN_MANIFEST.json').write_bytes(h['manifest'])
    for relative,raw in files.items():
        if relative in ('SOURCE_COMPONENT_EVENTS.jsonl','DEC1_SOURCE_COMPONENT_ATTEMPT.json'):continue
        path=folder/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    # The actual ACK constructor creates all three directories before any
    # Teacher plan, including a zero-tick raw-gate rejection.
    for name in ('plans','physics','acks'):(folder/'source_component_ack'/name).mkdir(exist_ok=True)
    evidence=SourceComponentEvidence(folder/'SOURCE_COMPONENT_EVENTS.jsonl',episode_id=h['cid'],
        executor_id=h['executor'],maximum_bytes=128*1024**2)
    for event in h['events']+[h['s1']['actual_stop_event']]:evidence.record(event)
    evidence.close()
    f=SimpleNamespace(folder=folder,ledger=SimpleNamespace(root=root,guard=lambda **kw:None),
        _outer_reconciled=True,persist_confirmed=True,closed=True,unknown=False,evidence=evidence,
        actual_stop=SimpleNamespace(receipt_id=h['s1']['actual_stop_event']['receipt_id']),
        barrier=SimpleNamespace(state=lambda:dict(unknown=False,stop_acknowledged=True)),
        ack_options=dict(condition_id=h['cid'],runtime_release='b'*40,owner_release='a'*40,
            expected_plan_sha256=unit.PLAN_SHA256),adapter=SimpleNamespace(mode=h['c']['interface']),
        stop_options=dict(source_step_sha256=unit.ORIGINAL_13['scripts/pi0291_arx5_data/engine.py']),
        episode_id=h['cid'],executor_id=h['executor'])
    f._propagate_unknown=lambda:None
    f._mark_unknown=lambda:setattr(f,'unknown',True)
    return f,h


def test_anchor_uses_real_closed_reader_state_and_leaves_result_s1_stop_unchanged(closed_owner):
    f,h=closed_owner;original={p:(f.folder/p).read_bytes() for p in ('RESULT.json','S1_RESULT.json','SOURCE_COMPONENT_STOP.json')}
    descriptor=anchor_unit.capture_terminal_anchor(f,manifest_sha256=unit.sha(h['manifest']))
    captured=read(f.folder/descriptor['relative'])
    assert captured['events']['offset']==f.evidence.offset
    assert captured['events']['sequence']==f.evidence.sequence
    assert captured['events']['previous']==f.evidence.previous
    assert captured['events']['bytes_sha256']==unit.sha(f.evidence.path.read_bytes())
    assert descriptor['sha256']==unit.sha((f.folder/descriptor['relative']).read_bytes())
    assert all((f.folder/p).read_bytes()==raw for p,raw in original.items())
    assert 'DEC1_SOURCE_COMPONENT_ATTEMPT.json' not in captured['files']
    with pytest.raises(PermissionError,match='ALREADY_ATTEMPTED'):
        anchor_unit.capture_terminal_anchor(f,manifest_sha256=unit.sha(h['manifest']))


@pytest.mark.parametrize('fault',['not-closed','not-reconciled','truncated-after-close'])
def test_unknown_or_unconfirmed_owner_never_captures_success_anchor(closed_owner,fault):
    f,h=closed_owner
    if fault=='not-closed':f.closed=False
    if fault=='not-reconciled':f._outer_reconciled=False
    if fault=='truncated-after-close':f.evidence.path.write_bytes(f.evidence.path.read_bytes().splitlines(keepends=True)[0])
    with pytest.raises((ValueError,PermissionError)):
        anchor_unit.capture_terminal_anchor(f,manifest_sha256=unit.sha(h['manifest']))
    assert f.unknown and not (f.folder/'SOURCE_COMPONENT_TERMINAL_BYTE_ANCHOR.json').exists()


def test_persisted_anchor_reply_interruption_is_unknown_and_never_retried(closed_owner,monkeypatch):
    f,h=closed_owner;original=anchor_unit.write;calls=[]
    def interrupted(path,document):
        calls.append(path);original(path,document);raise KeyboardInterrupt('ANCHOR_WRITE_REPLY_UNKNOWN')
    monkeypatch.setattr(anchor_unit,'write',interrupted)
    with pytest.raises(KeyboardInterrupt):anchor_unit.capture_terminal_anchor(f,manifest_sha256=unit.sha(h['manifest']))
    assert f.unknown and (f.folder/'SOURCE_COMPONENT_TERMINAL_BYTE_ANCHOR.json').exists()
    with pytest.raises(PermissionError):anchor_unit.capture_terminal_anchor(f,manifest_sha256=unit.sha(h['manifest']))
    assert len(calls)==1
