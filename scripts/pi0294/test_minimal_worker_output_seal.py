"""Real temporary files/inodes only; no registered job, GR00T or model permission."""
import json,os
from pathlib import Path
import pytest
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
