"""Real local files/NPZ + original saved-draw reader; all arrays CPU fixtures."""
import hashlib,os
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
from scripts.pi0294.generalization import minimal_remaining_inputs as inputs,minimal_remaining_contract as r
from scripts.pi0294.generalization import minimal_model as model
from scripts.pi0294.test_minimal_remaining_effects import remaining,actual_prefix


@pytest.fixture
def saved(tmp_path):
    value=inputs.Inputs.__new__(inputs.Inputs);value.science=tmp_path/'old-closed';value.science.mkdir()
    value.paths={};value.raw_checkpoint=tmp_path/'raw';value.raw_checkpoint.mkdir();value.guard=lambda:None
    files={}
    for name,pin in model.FILES.items():
        path=value.raw_checkpoint/name;path.write_bytes(b'CPU fixture: not GR00T weights');path.chmod(0o400)
        files[name]=dict(sha256=pin,stat=model.stat_identity(path))
    value.raw=dict(status='ACTUAL_FULL_RAW_PARENT_SEVEN_FILES_VERIFIED',revision=model.RAW_REVISION,
        raw_parent_files_identity=model.RAW_PARENT_FILES_ID,models_loaded=0,files=files,
        CPU_FIXTURE_NOT_NATIVE_STREAM=True)
    sid=r.sessions()[0]['id'];folder=value.science/'evidence/MINIMAL_FIVE_SESSIONS'/sid/'draws';folder.mkdir(parents=True)
    for i in range(15):
        path=folder/('zero-'+str(i).zfill(2)+'.npz')
        np.savez(path,physical_H40=np.zeros((40,10)),normalized=np.zeros((1,40,132)),noise_seed=29400+i,observation_fingerprint='CPU_SYNTHETIC_INPUT')
        path.chmod(0o400);s=path.stat();value.paths[str(path)]=dict(path=str(path),device=s.st_dev,inode=s.st_ino,
            bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return value


def test_original_draw_reader_reuses_fifteen_arrays_without_reading_any_tensor(saved,monkeypatch):
    original=Path.read_bytes;reads=[]
    def observed(p):
        assert not p.is_relative_to(saved.raw_checkpoint),'must reuse unchanged completed full-stream audit'
        reads.append(p);return original(p)
    monkeypatch.setattr(Path,'read_bytes',observed)
    before={str(p):p.stat()for p in saved.science.rglob('*')if p.is_file()}
    result=saved.E2_zeros(input_fingerprint='CPU_SYNTHETIC_INPUT')
    assert len(result['original_draws'])==len(reads)==15 and result['actual_saved_queries_reused']==15
    assert result['new_model_queries']==result['new_effect_query_links']==0
    assert all((p.stat().st_ino,p.stat().st_mtime_ns,p.stat().st_ctime_ns,p.stat().st_mode)==
        (before[str(p)].st_ino,before[str(p)].st_mtime_ns,before[str(p)].st_ctime_ns,before[str(p)].st_mode)for p in reads)


@pytest.mark.parametrize('kind',['changed_draw','changed_raw','symlink','unbound_path','wrong_input','writable'])
def test_changed_identity_or_unmatched_input_cannot_reuse_saved_query_credit(saved,kind):
    p=Path(next(iter(saved.paths)))
    if kind=='changed_draw':p.chmod(0o600);p.write_bytes(b'changed')
    elif kind=='changed_raw':(saved.raw_checkpoint/next(iter(model.FILES))).chmod(0o600)
    elif kind=='symlink':
        other=p.with_suffix('.original');p.rename(other);p.symlink_to(other)
    elif kind=='unbound_path':
        row=deepcopy(saved.paths[str(p)]);row['path']=str(saved.science.parent/'outside');saved.paths[row['path']]=row
    elif kind=='writable':p.chmod(0o600)
    with pytest.raises((PermissionError,FileNotFoundError,ValueError)):
        saved.E2_zeros(input_fingerprint='OTHER_INPUT'if kind=='wrong_input'else'CPU_SYNTHETIC_INPUT')


def test_remaining_worker_uses_original_arrays_and_does_not_call_draw_or_charge(remaining,saved,tmp_path,monkeypatch):
    from scripts.pi0294.generalization.minimal_remaining_worker import RemainingWorker
    from scripts.pi0294.e1 import model as fingerprint
    ledger,_=remaining;sid=r.sessions()[0]['id'];ledger.start_session(sid);ledger.charge('model_loads',1)
    worker=RemainingWorker.__new__(RemainingWorker);worker.saved=saved;worker.ledger=ledger
    worker.zero_payload=dict(CPU_FIXTURE='same saved observation',snapshot_uuid='not semantic input')
    observed=[]
    monkeypatch.setattr(fingerprint,'digest',lambda value:observed.append(value)or'CPU_SYNTHETIC_INPUT')
    worker.evidence=tmp_path/'new-only-evidence';(worker.evidence/sid).mkdir(parents=True)
    def refuse(*args,**kwargs):raise AssertionError('saved zero outputs must never call model generation')
    worker.draw=refuse;before=ledger.path.read_bytes()
    paths=worker.zero_draws(sid)
    assert len(paths)==15 and all(p.is_relative_to(saved.science)for p in paths)
    assert observed==[{'CPU_FIXTURE':'same saved observation'}]and ledger.path.read_bytes()==before
    assert(worker.evidence/sid/'SAVED_ZERO_INPUTS_READONLY.json').is_file()
