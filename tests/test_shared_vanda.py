import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import pytest
from scripts.pi0294.multitask import shared_vanda as vanda
from scripts.pi0294.night10 import native_relay as relay


def test_unknown_or_absent_pbs_never_loads_a_model(monkeypatch):
    monkeypatch.delenv('PBS_JOBID',raising=False)
    with pytest.raises(PermissionError,match='ALLOCATED_JOB'):vanda.job()
    monkeypatch.setenv('PBS_JOBID','1.server')
    def unknown(*a,**kw):raise subprocess.CalledProcessError(153,a)
    monkeypatch.setattr(vanda.subprocess,'run',unknown)
    with pytest.raises(subprocess.CalledProcessError):vanda.job()


def test_worker_copy_resolves_its_pinned_package_before_any_model(tmp_path):
    source=Path(__file__).resolve().parents[1]
    original=source/'scripts/pi0294/multitask/shared_policy.py'
    (tmp_path/'worker.py').write_bytes(original.read_bytes())
    manifest=dict(dependency_code=str(source),owner_uid=-1)
    body=json.dumps(manifest).encode()
    (tmp_path/'RUN_MANIFEST.json').write_bytes(body)
    (tmp_path/'NEURAL_BINDINGS.private.json').write_text('{}')
    q=subprocess.run([sys.executable,str(tmp_path/'worker.py'),'--root',str(tmp_path),
        '--manifest-sha256',hashlib.sha256(body).hexdigest()],capture_output=True,text=True,timeout=20)
    assert q.returncode!=0
    assert 'SHARED_EVAL_REGISTERED_NATIVE_OWNER'in q.stderr
    assert 'attempted relative import'not in q.stderr
    assert not(tmp_path/'MODEL_LOAD_INTENT.json').exists()


def archive(tmp_path,entries):
    file=tmp_path/'inputs.tar'
    with tarfile.open(file,'w')as stream:
        for name,body in entries.items():
            info=tarfile.TarInfo(name);info.size=len(body)
            stream.addfile(info,io.BytesIO(body))
    return file


def test_private_transfer_preserves_bytes_and_rejects_escape(tmp_path):
    pins=('manifest_sha256','statistics_sha256','population_sha256','task_DEV_sha256')
    names=('DATASET_MANIFEST.json','statistics.json','TRAIN_POPULATION.json','TASK_DEV_POPULATION.json')
    entries={'data/'+name:(name+'\n').encode()for name in names}
    file=archive(tmp_path,entries)
    c=dict(input_root=str(tmp_path/'ready'),input_archive=str(file),
        input_archive_sha256=relay.sha(file),input_archive_bytes=file.stat().st_size,
        shared_recipe_pins=dict(zip(pins,(hashlib.sha256(entries['data/'+name]).hexdigest()for name in names))))
    result=vanda.prepare_inputs(c,lambda:None)
    assert result['files']==4 and result['new_data_or_statistics']is False
    assert vanda.prepare_inputs(c,lambda:None)==result
    file=archive(tmp_path,{'../outside':b'bad'})
    c.update(input_root=str(tmp_path/'rejected'),input_archive_sha256=relay.sha(file))
    with pytest.raises(PermissionError,match='REGULAR_MEMBERS'):vanda.prepare_inputs(c,lambda:None)
    assert not(tmp_path/'outside').exists()


def test_a_vanda_stage_cannot_use_the_core_entry_or_expired_allocation(tmp_path,monkeypatch):
    monkeypatch.delenv('PBS_JOBID',raising=False)
    c=dict(execution_site='VANDA_A40',run_parent=str(tmp_path),site_context=dict(job_id='a'))
    with pytest.raises(PermissionError,match='EXACT_VANDA_ALLOCATION'):
        relay.stage(c,'new','unused',{}, {})
    assert not(tmp_path/'new').exists()


def test_native_time_gate_keeps8192_and_all_four_restores(tmp_path):
    native=tmp_path/'native';native.mkdir()
    (native/'OPTIMIZER_STEPS.jsonl').write_text(''.join(json.dumps(dict(step=i,epoch=100+i*8))+'\n'for i in range(1,11)))
    (native/'RESULT.json').write_text(json.dumps(dict(wall_s=250)))
    c=dict(root=str(tmp_path),formal_seconds=100000,export_and_close_reserve_s=3600,
        site_context=dict(allocation_deadline_epoch=100000))
    report=vanda.formal_time_admission(c,native,now=1000)
    assert report['passes'] and report['forecast_complete8192_s']>8192*8
    assert report['four_checkpoint_save_reload_pairs']
    other=tmp_path/'short';other.mkdir();c['root']=str(other)
    c['site_context']['allocation_deadline_epoch']=5000
    with pytest.raises(PermissionError,match='CANNOT_CONTAIN_FULL8192'):vanda.formal_time_admission(c,native,now=1000)
