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
