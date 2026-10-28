"""Real temporary files/inodes only; no registered job, GR00T or model permission."""
import json,os
from pathlib import Path
import pytest
from types import SimpleNamespace
from scripts.pi0294.generalization import minimal_worker as worker
from scripts.pi0294.native_dataset import sha


def fixture(root,step=2):
    sid=worker.e.native_sessions()[1 if step==2 else 3]['id'];cp=root/'training'/sid/('checkpoint-'+str(step));cp.mkdir(parents=True)
    (cp/'model.safetensors').write_bytes(b'EXPLICIT_CPU_BYTES_NOT_A_MODEL')
    manifest=dict(global_step=step,training_identity=worker.e.training.ARMS[0],full_resume=True,qualification_only=step==2,files={'model.safetensors':sha(cp/'model.safetensors')})
    (cp/'CHECKPOINT_COMPLETE.json').write_text(json.dumps(manifest)+'\n');return sid,cp


@pytest.mark.parametrize('step',[2,512])
def test_complete_own_files_seal_without_restream_or_changing_bytes(tmp_path,step):
    sid,cp=fixture(tmp_path,step);pin=sha(cp/'CHECKPOINT_COMPLETE.json');before={p.name:p.read_bytes()for p in cp.iterdir()};calls=[]
    receipt=worker.seal_checkpoint(tmp_path,sid,step,pin,guard=lambda:calls.append('actual-caller-guard-fixture'))
    assert calls==['actual-caller-guard-fixture']*2 and receipt['tensor_bytes_restreamed']==0
    assert all(p.stat().st_mode&0o777==0o400 and p.read_bytes()==before[p.name]for p in cp.iterdir())


def test_alias_cannot_change_a_raw_input_inode(tmp_path):
    sid,cp=fixture(tmp_path);target=cp/'model.safetensors';raw=tmp_path/'RAW_INPUT';target.rename(raw);target.symlink_to(raw);before=raw.stat()
    with pytest.raises(PermissionError):worker.seal_checkpoint(tmp_path,sid,2,sha(cp/'CHECKPOINT_COMPLETE.json'),guard=lambda:None)
    after=raw.stat();assert (before.st_ino,before.st_ctime_ns,before.st_mode)==(after.st_ino,after.st_ctime_ns,after.st_mode)


def test_unlisted_or_wrong_arm_save_is_rejected_before_any_chmod(tmp_path):
    sid,cp=fixture(tmp_path);extra=cp/'unlisted';extra.write_bytes(b'not-accounted');before={p.name:p.stat().st_mode for p in cp.iterdir()}
    with pytest.raises(PermissionError):worker.seal_checkpoint(tmp_path,sid,2,sha(cp/'CHECKPOINT_COMPLETE.json'),guard=lambda:None)
    assert before=={p.name:p.stat().st_mode for p in cp.iterdir()}


def test_c50_probe_reuses_original_factory_not_a_new_dataset_arm(tmp_path,monkeypatch):
    from scripts.pi0294.pv1 import mix_training
    from scripts.pi0294.pv1.dataset import PV1Dataset
    with pytest.raises(PermissionError,match='PV1_FIXED_ROUTE_ARM_AND_SEED'):
        PV1Dataset(new_root='unused',new_raw_bindings={},route='COVERAGE_FIRST',arm='C50')
    inputs=dict(C1_dataset='old',new_dataset='old-pv1')
    values={'plans/PV1_TRAIN_INPUTS.private.json':inputs,
        'plans/MINIMAL_ORIGINAL_PROBE_READER.private.json':dict(reference_only_not_training=True,
            arm='C50',seed=20261019,route='COVERAGE_FIRST',inputs=inputs)}
    called=[];reader=object()
    monkeypatch.setattr(mix_training,'dataset',lambda root,arm,evidence:
        called.append((root,arm,evidence))or reader)
    caller=SimpleNamespace(root=tmp_path,evidence=tmp_path/'evidence',sealed=values.__getitem__)
    assert worker.Worker.original_probe_reader(caller)is reader
    assert called==[(tmp_path,'C50',tmp_path/'evidence/original-probe-reader')]
    values['plans/MINIMAL_ORIGINAL_PROBE_READER.private.json']['route']='COV1'
    with pytest.raises(PermissionError):worker.Worker.original_probe_reader(caller)


def test_twelve_parent_reader_uses_original_semantic_panel_hash(tmp_path,monkeypatch):
    from scripts.pi0294.generalization import minimal_train_session as session
    from test_minimal_arx5_training import actual_panel
    panel=actual_panel();plans=tmp_path/'plans';plans.mkdir()
    (plans/'MINIMAL_TWELVE_PANEL.json').write_text(json.dumps(panel,indent=2)+'\n')
    (tmp_path/'NEURAL_BINDINGS.private.json').write_text(json.dumps(dict(dataset='actual-original-role',
        source='original-source',cosmos='original-cosmos',raw_bindings=str(tmp_path/'raw.json'))))
    (tmp_path/'raw.json').write_text('{}')
    assert session.training.study.digest(panel)==session.PANEL_SHA
    assert session.execution.sha(session.execution.raw(panel))!=session.PANEL_SHA
    calls=[]
    class Reader:
        def __init__(self,*args,**kwargs):calls.append((args,kwargs))
        def initialize(self):calls.append('initialized')
    monkeypatch.setattr(session,'Dataset',Reader)
    assert isinstance(session.dataset(tmp_path,tmp_path/'evidence'),Reader)
    assert calls[-1]=='initialized'and calls[0][1]['panel']==panel
    panel['parents'][0]['sha256']='0'*64
    (plans/'MINIMAL_TWELVE_PANEL.json').write_text(json.dumps(panel))
    with pytest.raises(PermissionError):session.dataset(tmp_path,tmp_path/'bad')
