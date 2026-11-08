"""Synthetic site/worker seam checks; never a GPU capacity qualification."""
import json,os,time
from pathlib import Path
import pytest
from go2arx5_vla.world_model import target_gr00t_profile_v1 as profile
from go2arx5_vla.world_model import target_gr00t_core_v1 as core
from go2arx5_vla.world_model.storage import digest


def spec(mode='CAPACITY',variant='INTERFACES_LAST2_DIT',endpoint=2,microbatch=1):
    return dict(recipe=profile.recipe('a'*64,'b'*64,mode=mode,variant=variant,endpoint=endpoint,microbatch=microbatch),
        profile=profile.PROFILE,kind=profile.KIND,execution_site='CORELAB_4080S',start_step=0,
        GR00T_updates=endpoint,WM_updates=0,Actor_updates=0,TASK_ADAPTER_updates=0,
        physics=False,WBC_changes=False,parent_manifest_sha256=profile.PARENT,resume=None,deadline_epoch=time.time()+3600)


def test_core_updates_are_internal_same_effective_batch_and_not_vanda_profile_relabel():
    p=profile.validate(spec());assert p['effective_batch']==32 and p['accumulation']==32
    assert p['visual_and_language_backbone_frozen']and p['external_adapter_updates']==0
    profile.validate(spec('FIT4',endpoint=200))


@pytest.mark.parametrize('args',[dict(mode='FORMAL',endpoint=2000),dict(variant='FULL_ACTION_SIDE'),
    dict(variant='INTERFACES_LAST6_DIT'),dict(microbatch=2)])
def test_core_first_qualification_cannot_silently_start_formal_or_larger_scope(args):
    with pytest.raises(PermissionError,match='CORE_FIRST'):profile.validate(spec(**args))


def test_core_failure_preserves_actual_updates_and_closes_without_fake_pbs(tmp_path,monkeypatch):
    from scripts.wm02 import train_target_gr00t_v1 as entry
    from go2arx5_vla.world_model import target_gr00t_training_v1 as training
    root=tmp_path/'child';root.mkdir();(root/'control').mkdir()
    s=dict(spec(),registered_root=str(root),source_sha='f'*40,allocation_request_id='own-core-scope')
    path=root/'EXECUTION_SPEC.json';path.write_text(json.dumps(s))
    (root/'control/LEASE.json').write_text(json.dumps(dict(status='ACTIVE',spec_sha256=digest(path),job_id='own-core-scope')))
    monkeypatch.setattr(os,'getuid',lambda:2009)
    monkeypatch.setattr(os,'uname',lambda:type('U',(),dict(nodename='corelab-4080s'))())
    monkeypatch.delenv('PBS_JOBID',raising=False)
    def fail(s):
        (root/'PROGRESS.json').write_text(json.dumps(dict(global_step=1,GR00T_updates=1)))
        raise RuntimeError('SYNTHETIC_AFTER_REAL_UPDATE_FAILURE')
    monkeypatch.setattr(training,'run',fail)
    with pytest.raises(RuntimeError,match='AFTER_REAL_UPDATE'):entry.main(['--spec',str(path)])
    actual=json.loads((root/'RESULT.json').read_bytes());lease=json.loads((root/'control/LEASE.json').read_bytes())
    assert actual['GR00T_updates']==1 and lease['status']=='CLOSED'and lease['worker_exit_code']==1
    assert json.loads((root/'WORKER_EXIT.private.json').read_bytes())['job_id']=='own-core-scope'


def test_core_worker_cannot_run_under_a_pbs_identity(tmp_path,monkeypatch):
    from scripts.wm02 import train_target_gr00t_v1 as entry
    root=tmp_path/'child';root.mkdir();(root/'control').mkdir()
    s=dict(spec(),registered_root=str(root),source_sha='f'*40,allocation_request_id='own-core-scope')
    path=root/'EXECUTION_SPEC.json';path.write_text(json.dumps(s))
    (root/'control/LEASE.json').write_text(json.dumps(dict(status='ACTIVE',spec_sha256=digest(path),job_id='own-core-scope')))
    monkeypatch.setattr(os,'getuid',lambda:2009);monkeypatch.setenv('PBS_JOBID','other-job')
    with pytest.raises(PermissionError,match='ENVELOPE'):entry.main(['--spec',str(path)])
    assert not(root/'WORKER_STARTED.private.json').exists()


def test_gpu_requires_original_uuid_mask_and_preserves_two_gib_margin(monkeypatch):
    from types import SimpleNamespace
    uuid='GPU-845348f1-1511-dc19-0877-cc97658e3edd';monkeypatch.setenv('CUDA_VISIBLE_DEVICES',uuid)
    monkeypatch.setattr(core.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=uuid+', NVIDIA GeForce RTX 4080 SUPER, 14719, 16376\n'))
    actual=core.gpu(dict(GPU_uuid=uuid));assert actual['free_bytes']==14719*1024**2
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    with pytest.raises(PermissionError,match='SINGLE_SHARED'):core.gpu(dict(GPU_uuid=uuid))
