"""Captured original script, real temporary wrapper files, mocked qsub only."""
import json,time,hashlib,subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.pi0294.generalization import minimal_native_launch as n,minimal_execution as e
from scripts.pi0294.test_minimal_native_admission import actual,packet,queue_limits


@pytest.fixture
def programs():
    path=Path('/home/liuyu/ISSAC-GR00T/work/private/v5-execution-20261025/D25_VANDA_FIRST_PREPARED.private.json')
    if not path.exists():pytest.skip('actual original closed D25 submit receipt required')
    v=json.loads(path.read_bytes())['programs'];assert e.sha(v['SUBMIT'].encode())==n.ORIGINAL_SUBMIT_SHA
    return v


def test_factory_uses_one_authoritative_original_resource_and_scientific_contract(packet):
    r=json.loads(packet['request_bytes']);marker=e.authority.validate(json.loads(packet['global_bytes']))
    template=n.template(source_sha=r['source_sha'],run_uuid=r['run_uuid'].lower(),request_id=r['request_id'],
        submission_nonce=r['submission_nonce'],scope_marker=marker,python='/CPU-fixture/venv/bin/python',
        raw_parent_audit_plan_sha256=r['raw_parent_audit_plan_sha256'],frozen_input_packet_sha256=r['frozen_input_packet_sha256'],
        Core_measurement_capture_sha256=r['Core_measurement_capture_sha256'])
    assert 'run_manifest_sha256'not in template and 'cold_input_audit_sha256'not in template
    assert all(template[k]==v for k,v in e.fixed_request_fields().items())
    # Explicit FUTURE hashes for CPU validation only, not a native cold audit.
    e.request(dict(template,run_manifest_sha256='a'*64,cold_input_audit_sha256='b'*64))


def fixture_root(tmp_path,packet):
    r=json.loads(packet['request_bytes']);r['python']='/CPU-fixture/venv/bin/python'
    root=tmp_path/'root';d=root/'pbs/alloc-35';d.mkdir(parents=True)
    (d/'REQUEST.json').write_bytes(e.raw(r));(d/'REQUEST.json').chmod(0o400)
    (root/'RUN_MANIFEST.json').write_bytes(e.raw(dict(code=dict(capsule_relative='capsules/CPU-fixture/code'))))
    return r,root,d,dict(TARGET=str(root),DIRECTORY=str(d),REQUEST=r,
        RESERVATION=dict(request_sha256=e.sha((d/'REQUEST.json').read_bytes()),observed_epoch=time.time()))


def test_actual_original_wrapper_keeps_one_qsub_no_mail_and_exact_new_profile(programs,packet,tmp_path,monkeypatch):
    original=programs['SUBMIT'];compiled=n.programs(programs)['SUBMIT'];r,root,d,values=fixture_root(tmp_path,packet);calls=[]
    def fake(argv,**kwargs):
        calls.append((argv,kwargs));assert argv[:2]==['/opt/pbs/bin/qsub','-v']and kwargs['timeout']==30
        return SimpleNamespace(returncode=0,stdout='12345.stdct-mgmt-02\n',stderr='')
    monkeypatch.setattr(subprocess,'run',fake);exec(compile(compiled,'<CPU-wrapper-only-not-native>','exec'),values)
    script=(d/'job.pbs').read_text()
    assert script.count('#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb')==1
    assert '#PBS -l walltime=16:30:00'in script and '#PBS -m n'in script and '#PBS -r n'in script
    assert 'PV1_RUN_MANIFEST='in script and e.PROFILE in script
    assert 'scripts.pi0294.dec1.a40_training_runtime'in script and len(calls)==1
    assert e.sha(programs['SUBMIT'].encode())==n.ORIGINAL_SUBMIT_SHA and programs['SUBMIT']==original
    assert (d/'QSUB_INTENT.json').stat().st_mode&0o777==0o400
    assert (d/'QSUB_REPLY.json').is_file()
    with pytest.raises(AssertionError):exec(compile(compiled,'<no-replay>','exec'),values)
    assert len(calls)==1


def test_original_qsub_lost_reply_preserves_intent_and_blocks_resubmission(programs,packet,tmp_path,monkeypatch):
    compiled=n.programs(programs)['SUBMIT'];r,root,d,values=fixture_root(tmp_path,packet);calls=[]
    def lost(argv,**kwargs):calls.append(argv);raise subprocess.TimeoutExpired(argv,30)
    monkeypatch.setattr(subprocess,'run',lost)
    with pytest.raises(subprocess.TimeoutExpired):exec(compile(compiled,'<CPU-qsub-lost>','exec'),values)
    assert (d/'QSUB_INTENT.json').exists()and not(d/'QSUB_REPLY.json').exists()
    with pytest.raises(AssertionError):exec(compile(compiled,'<CPU-no-replay>','exec'),values)
    assert len(calls)==1


def test_changed_original_program_is_not_an_implicit_new_launcher(programs):
    programs['SUBMIT']+='\n#changed\n'
    with pytest.raises(PermissionError,match='EXACT_PREVIOUS_SUCCESSFUL_D25'):n.programs(programs)


@pytest.mark.parametrize('change',[None,'expired','future','short_wall','wrong_queue'])
def test_stale_sealed_queue_is_rejected_before_any_registration_or_qsub(change):
    q=queue_limits(100.)
    if change=='expired':q['observed_epoch']=-21
    elif change=='future':q['observed_epoch']=101.
    elif change=='short_wall':q['maximum_wall_s']=59399
    elif change=='wrong_queue':q['destination']='another'
    if change:
        with pytest.raises(PermissionError):n.effective_queue(e.raw(q),created_epoch=100.,walltime_s=59400)
    else:assert n.effective_queue(e.raw(q),created_epoch=100.,walltime_s=59400)==q


def test_raw_queue_values_cannot_be_forged_or_change_after_request_binding():
    q=queue_limits(100.);body=e.raw(q);r={'effective_queue_proof_sha256':e.sha(body)}
    q['raw_queue_readbacks']['auto']['stdout']=json.dumps(dict(Queue={'auto':dict(enabled='True',started='True',route_destinations=['other'])}))
    with pytest.raises(PermissionError):n.effective_queue(e.raw(q),created_epoch=100.,walltime_s=59400)
    assert r['effective_queue_proof_sha256']!=e.sha(e.raw(q))


def test_collector_parses_real_shape_and_rejects_bad_native_route(monkeypatch):
    q=queue_limits(time.time());calls=[]
    def run(argv,**kwargs):
        calls.append(argv);row=q['raw_queue_readbacks'][argv[-1]]
        return SimpleNamespace(returncode=row['returncode'],stdout=row['stdout'],stderr=row['stderr'])
    monkeypatch.setattr(subprocess,'run',run);value=n.collect_effective_queue(walltime_s=59400)
    assert len(calls)==2 and value['maximum_wall_s']==604800
    q['raw_queue_readbacks']['auto']['stdout']=json.dumps(dict(Queue={'auto':dict(enabled='True',started='True',route_destinations=['elsewhere'])}))
    with pytest.raises(PermissionError):n.collect_effective_queue(walltime_s=59400)
