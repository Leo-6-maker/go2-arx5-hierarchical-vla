"""Synthetic identity/whitelist/terminal tests; no GR00T model is constructed."""
import json,os
from pathlib import Path
import pytest
from go2arx5_vla.world_model import target_gr00t_profile_v1 as p
from go2arx5_vla.world_model import target_gr00t_training_v1 as training
from go2arx5_vla.world_model.storage import digest


def spec():
    recipe=p.recipe('a'*64,'b'*64,mode='FORMAL',variant='FULL_ACTION_SIDE',endpoint=2000)
    return dict(recipe=recipe,profile=p.PROFILE,kind=p.KIND,execution_site='VANDA_A40',
        start_step=0,GR00T_updates=2000,WM_updates=0,Actor_updates=0,TASK_ADAPTER_updates=0,
        physics=False,WBC_changes=False,parent_manifest_sha256=p.PARENT,resume=None,deadline_epoch=1794200000.)


@pytest.mark.parametrize('key,value',[
    ('GR00T_updates',0),('GR00T_updates',200),('TASK_ADAPTER_updates',2000),
    ('profile','wm02-pilot1-neural'),('execution_site','H200'),
    ('fusion_binding','old-zero-update-lease'),('physics',True),
    ('WBC_changes',True),('parent_manifest_sha256','c'*64)])
def test_nonzero_internal_gr00t_profile_cannot_alias_old_frozen_scope(key,value):
    s=spec();p.validate(s)
    with pytest.raises(PermissionError):p.validate(dict(s,**{key:value}))


def test_fixed_8000_schedule_and_continuation_are_explicit():
    s=spec();r=s['recipe']
    assert r['scheduler_horizon']==8000 and r['warmup_updates']==200 and r['effective_batch']==32
    assert r['Actor_updates']==r['external_adapter_updates']==r['WM_updates']==0
    next_recipe=p.recipe('a'*64,'b'*64,mode='FORMAL',variant='FULL_ACTION_SIDE',endpoint=4000)
    assert training.semantic_recipe(r)==training.semantic_recipe(next_recipe)
    with pytest.raises(PermissionError,match='CONTINUATION_NOT_FRESH'):
        p.validate(dict(s,recipe=next_recipe,GR00T_updates=4000))
    p.validate(dict(s,recipe=next_recipe,start_step=2000,GR00T_updates=2000,resume=dict(path='/sealed/cp2000',sha256='d'*64)))


@pytest.mark.parametrize('variant,count',[('FULL_ACTION_SIDE',32),('INTERFACES_LAST6_DIT',6),('INTERFACES_LAST2_DIT',2)])
def test_declared_action_whitelist_keeps_backbone_frozen_and_uses_internal_dit(variant,count):
    target=tuple(f'action_head.interface{i}.target_W'for i in range(14))
    names=target+tuple(f'action_head.model.transformer_blocks.{i}.weight'for i in range(32))+('action_head.projector.weight','backbone.visual.weight','backbone.language.weight')
    selected=p.whitelist(names,target,variant)
    assert set(target)<=set(selected)
    assert len([n for n in selected if 'transformer_blocks'in n])==count
    assert all(n.startswith('action_head.')for n in selected)
    assert ('action_head.projector.weight'in selected)==(count==32)


def test_site_rejection_closes_actual_worker_envelope_without_missing_result(tmp_path,monkeypatch):
    from scripts.wm02 import train_target_gr00t_v1 as entry
    root=tmp_path/'registered';root.mkdir();(root/'control').mkdir()
    s=dict(spec(),registered_root=str(root),source_sha='f'*40)
    path=root/'EXECUTION_SPEC.json';path.write_text(json.dumps(s))
    lease=dict(status='ACTIVE',spec_sha256=digest(path),job_id='1234.test')
    (root/'control/LEASE.json').write_text(json.dumps(lease))
    monkeypatch.setattr(os,'getuid',lambda:22182);monkeypatch.setenv('PBS_JOBID','1234.test')
    def reject(spec):raise PermissionError('SYNTHETIC_STORAGE_REJECTION_BEFORE_TORCH')
    monkeypatch.setattr(training,'run',reject)
    with pytest.raises(PermissionError,match='SYNTHETIC_STORAGE'):entry.main(['--spec',str(path)])
    result=json.loads((root/'RESULT.json').read_text());closed=json.loads((root/'control/LEASE.json').read_text())
    assert result['GR00T_updates']==0 and result['status']=='QUALIFICATION_FAILED'
    assert closed['status']=='CLOSED'and closed['worker_exit_code']==1
    assert (root/'WORKER_EXIT.private.json').exists()
