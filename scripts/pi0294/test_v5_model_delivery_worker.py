"""Actual local child/pipe/OS-exit mechanics; no HF, Core or neural qualification."""
import json,os,subprocess,time
from copy import deepcopy
import pytest
from scripts.pi0294.pv1 import mix_model_delivery_worker as w,mix_accounting as a


def test_existing_pipe_launch_detaches_and_records_real_OS_exit_without_credential_file(tmp_path,monkeypatch):
    root=tmp_path/'control';code=root/'code';path=code/'scripts/pi0294/pv1/mix_model_upload.py';path.parent.mkdir(parents=True)
    for parent in (path.parent,path.parent.parent,path.parent.parent.parent): (parent/'__init__.py').write_text('')
    path.write_text('import json,sys\np=json.loads(sys.stdin.read())\nassert isinstance(p.pop("token"),str)\n')
    (root/'CONTROL_RUN_MANIFEST.json').write_text(json.dumps(dict(source_sha='c'*40,model_transfer_only=True,worker_sha256=a.sha(path.read_bytes()))))
    payload=dict(control_root=str(root),source_sha='c'*40,candidate='C50',token='LOCAL_CPU_FAKE_TOKEN')
    monkeypatch.setattr(os,'getuid',lambda:22182)
    proc=[];original=subprocess.Popen
    def capture(*args,**kw):
        compile(args[0][-1],'<actual-detached-source-wrapper>','exec')
        p=original(*args,**kw);proc.append(p);return p
    monkeypatch.setattr(subprocess,'Popen',capture)
    exec(compile(w.START,'<unchanged-production-launch>','exec'),dict(P=deepcopy(payload)))
    assert proc[0].wait(timeout=10)==0
    exit_record=json.loads((root/'UPLOAD_OS_EXIT.json').read_bytes())
    start=json.loads((root/'UPLOAD_NATIVE_STARTED.json').read_bytes())
    assert exit_record['returncode']==0 and exit_record['pid']==start['pid']
    assert all(b'LOCAL_CPU_FAKE_TOKEN'not in p.read_bytes()for p in root.rglob('*')if p.is_file())
    exec(compile(w.POLL,'<unchanged-production-read>','exec'),dict(P=dict(root=str(root))))
    with pytest.raises(AssertionError):exec(compile(w.START,'<repeat-must-refuse>','exec'),dict(P=deepcopy(payload)))
    assert len(proc)==1


def test_failed_source_wrapper_preserves_actual_nonzero_OS_exit(tmp_path,monkeypatch):
    root=tmp_path/'control';p=root/'code/scripts/pi0294/pv1/mix_model_upload.py';p.parent.mkdir(parents=True);p.write_text('raise SystemExit(7)\n')
    for parent in (p.parent,p.parent.parent,p.parent.parent.parent): (parent/'__init__.py').write_text('')
    (root/'CONTROL_RUN_MANIFEST.json').write_text(json.dumps(dict(source_sha='c'*40,model_transfer_only=True,worker_sha256=a.sha(p.read_bytes()))))
    monkeypatch.setattr(os,'getuid',lambda:22182);proc=[];original=subprocess.Popen
    def capture(*args,**kw):
        q=original(*args,**kw);proc.append(q);return q
    monkeypatch.setattr(subprocess,'Popen',capture)
    exec(compile(w.START,'<unchanged-production-launch>','exec'),dict(P=dict(control_root=str(root),source_sha='c'*40,candidate='C50',token='FAKE')))
    assert proc[0].wait(timeout=10)==7
    assert json.loads((root/'UPLOAD_OS_EXIT.json').read_bytes())['returncode']==7


@pytest.mark.parametrize('arm',['E1_ADAPTER_ONLY','E2_SHARED_ACTION_HEAD'])
def test_same_detached_source_pipe_retains_exact_minimal_control_root(arm,tmp_path,monkeypatch):
    root=tmp_path/'control';p=root/'code/scripts/pi0294/pv1/mix_model_upload.py';p.parent.mkdir(parents=True)
    for parent in(p.parent,p.parent.parent,p.parent.parent.parent):(parent/'__init__.py').write_text('')
    p.write_text('import json,sys\nfrom pathlib import Path\np=json.loads(sys.stdin.read())\nassert p["candidate"]in("E1_ADAPTER_ONLY","E2_SHARED_ACTION_HEAD")\nassert Path(p["control_root"]).is_dir()\nassert p.pop("token")=="FAKE_PIPE_ONLY"\n')
    (root/'CONTROL_RUN_MANIFEST.json').write_text(json.dumps(dict(source_sha='c'*40,model_transfer_only=True,worker_sha256=a.sha(p.read_bytes()))))
    monkeypatch.setattr(os,'getuid',lambda:22182);processes=[];spawn=subprocess.Popen
    def capture(*args,**kw):
        proc=spawn(*args,**kw);processes.append(proc);return proc
    monkeypatch.setattr(subprocess,'Popen',capture)
    payload=dict(control_root=str(root),source_sha='c'*40,candidate=arm,token='FAKE_PIPE_ONLY')
    exec(compile(w.START,'<same-finite-source-only-pipe>','exec'),dict(P=deepcopy(payload)))
    assert processes[0].wait(timeout=10)==0
    assert json.loads((root/'UPLOAD_OS_EXIT.json').read_bytes())['returncode']==0
    assert all(b'FAKE_PIPE_ONLY'not in path.read_bytes()for path in root.rglob('*')if path.is_file()and path!=p)


def test_actual_original_Journal_directory_and_unknown_write_are_not_replayed(tmp_path):
    from scripts.pi0294.dec1.owner_transport import UnknownMutation
    cfg=dict(control_run_uuid='v5-c50-control-fixture',source_sha='c'*40,compiled_plan=dict(public_plan=dict(candidate='C50',checkpoint_step=1024)))
    journal=w.delivery_journal(tmp_path,cfg,b'CPU configuration fixture')
    counter=tmp_path/'applied-once'
    def apply_then_lose_reply(intent,limit):
        counter.write_text('one actual temporary-file effect')
        raise ConnectionError('simulated lost return after effect')
    with pytest.raises(UnknownMutation):journal.call('SOURCE_UPLOAD_ONCE','MUTATION',apply_then_lose_reply)
    with pytest.raises(UnknownMutation):journal.call('SOURCE_UPLOAD_ONCE','MUTATION',apply_then_lose_reply)
    assert counter.read_text()=='one actual temporary-file effect'
    assert json.loads((tmp_path/'rpc/RPC_EFFECT_STATE.json').read_bytes())['status']=='UNKNOWN'
    with pytest.raises(FileExistsError):w.delivery_journal(tmp_path,cfg,b'CPU configuration fixture')


@pytest.mark.parametrize('arm',['C50','D25'])
def test_model_delivery_uses_explicit_original_candidate(arm,tmp_path):
    cfg=dict(control_run_uuid='original-control',source_sha='c'*40,candidate=arm,compiled_plan=dict(public_plan=dict(candidate=arm,checkpoint_step=1024)))
    assert w.candidate(cfg)==arm
    journal=w.delivery_journal(tmp_path,cfg,b'CPU fixture')
    assert journal.identity['job_id']=='MODEL_ONLY_'+arm+'_NO_GPU'


@pytest.mark.parametrize('cfg',[dict(candidate='D25',compiled_plan=dict(public_plan=dict(candidate='C50',checkpoint_step=1024))),dict(compiled_plan=dict(public_plan=dict(candidate='D25',checkpoint_step=1024))),dict(candidate='GEN1',compiled_plan=dict(public_plan=dict(candidate='GEN1',checkpoint_step=1024))),dict(candidate='D25',compiled_plan=dict(public_plan=dict(candidate='D25',checkpoint_step=512)))])
def test_delivery_does_not_relabel_C50_or_invent_new_candidate(cfg):
    with pytest.raises(PermissionError):w.candidate(cfg)
