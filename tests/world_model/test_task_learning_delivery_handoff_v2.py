"""Formal terminal/inventory fixtures; no training or real model is run."""
import copy,json
from pathlib import Path
import pytest
from go2arx5_vla.world_model import task_learning_delivery_handoff_v2 as handoff
from go2arx5_vla.world_model.task_learning_probes_v1 import contract as probes,PHASES
from go2arx5_vla.world_model.storage import digest
from tests.world_model.test_task_learning_training_profile_v1 import specification


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return p


def fixture(tmp_path):
    r=tmp_path/'formal';s=specification();s.update(registered_root=str(r),generation_id='generation',job_id='fixture.pbs',source_sha='c'*40)
    write(r/'EXECUTION_SPEC.json',s);write(r/'ALLOCATION_CONFIG.json',dict(root=str(r),recipe=s['recipe'],source_sha=s['source_sha']))
    cp=r/'training/checkpoint-2000';write(cp/'CHECKPOINT_COMPLETE.json',dict(global_step=2000,generation_id='generation',recipe=s['recipe']))
    restore=dict(model_parameters_restored=True,optimizer_restored=True,scheduler_restored=True,RNG_restored=True,
        fixed_batch_generation_restored=True,sampler_cursor=64000,checkpoint_sha256=digest(cp/'CHECKPOINT_COMPLETE.json'))
    write(r/'RESTORE_ACTUAL.json',restore)
    package=r/'inference-package';weight=package/'target_parameters.pt';weight.parent.mkdir(parents=True);weight.write_bytes(b'fixture-never-loaded')
    manifest=dict(recipe=s['recipe'],generation_id='generation',global_step=2000,checkpoint_sha256=restore['checkpoint_sha256'],
        training_site='HOPPER_SINGLE_GPU',external_inference_adapter=False,source_sha=s['source_sha'],
        parent_manifest_sha256=s['parent_manifest_sha256'],files={'target_parameters.pt':digest(weight)})
    write(package/'PACKAGE_MANIFEST.json',manifest);pkg=dict(path=str(package),sha256=digest(package/'PACKAGE_MANIFEST.json'))
    proof=dict(status='PASS_ACTUAL_INDEPENDENT_PARENT_DELTA_RECONSTRUCTION',package=pkg,actual_matched_queries=40,
        absolute_component_tolerance=1e-6,max_abs_error=dict(physical_H40=0.,normalized=0.))
    write(r/'PACKAGE_RECONSTRUCTION_ACTUAL.json',proof)
    result=dict(status='CHECKPOINT_SEALED',profile=s['profile'],source_sha=s['source_sha'],global_step=2000,
        generation_id='generation',GR00T_updates=2000,sample_visits=64000,backbone_frozen=True,real_action_parameter_changed=True,
        external_adapter_updates=0,native_physics=0,checkpoint=str(cp),restore=restore,inference_package=pkg)
    write(r/'RESULT.json',result);write(r/'WORKER_EXIT.private.json',dict(exit_code=0,failure=None,job_id=s['job_id']))
    write(r/'control/LEASE.json',dict(status='CLOSED',worker_exit_code=0,job_id=s['job_id'],
        spec_sha256=digest(r/'EXECUTION_SPEC.json'),config_sha256=digest(r/'ALLOCATION_CONFIG.json'),result_sha256=digest(r/'RESULT.json')))
    for label,split in [('before','TRAIN'),('before-DEV','DEV'),('restored','TRAIN'),('restored-DEV','DEV'),('package-reconstructed','TRAIN')]:
        rows=[]
        for episode in range(8 if split=='TRAIN'else 4):
            for anchor,phase in enumerate(PHASES):
                file=r/label/f'e{episode}-a{anchor:02d}.npz';file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'inventory fixture, not model output')
                rows.append(dict(episode=f'e{episode}',anchor=anchor,phase_audit_only=phase,genuine_recovery_audit_only=episode>=4,
                    split=split,sample_file=file.name,sha256=digest(file)))
        write(r/label/'RESULT.json',dict(selection_contract=probes(),records=rows))
    job=dict(job_state='F',Exit_status=0,Job_Name='gr00t_task_sft',Job_Owner='e1192548@fixture')
    return r,job


def test_reads_the_production_RESULT_json_and_does_not_need_a_private_result_file(tmp_path):
    r,job=fixture(tmp_path)
    actual=handoff.audit_training_root(r,job,job_id='fixture.pbs')
    assert actual['global_step']==2000 and actual['status'].startswith('ACTUAL_FORMAL_ENDPOINT')
    assert not(r/'RESULT.private.json').exists()
    assert 'producer/RESULT.json'in actual['files']and 'package/target_parameters.pt'in actual['files']
    assert actual['parent_assets_copied']is actual['optimizer_archives_copied']is False
    assert not any('optimizer.pt'in v for v in actual['files'])


@pytest.mark.parametrize('fault',['queued','PBS_exit','job_id','worker','cursor','empty_reconstruction','foreign_checkpoint'])
def test_queued_incomplete_or_foreign_evidence_cannot_be_packaged(tmp_path,fault):
    r,job=fixture(tmp_path)
    if fault=='queued':job['job_state']='Q'
    elif fault=='PBS_exit':job['Exit_status']=1
    elif fault=='job_id':
        with pytest.raises(ValueError):handoff.audit_training_root(r,job,job_id='other.pbs')
        return
    elif fault=='worker':
        v=json.loads((r/'WORKER_EXIT.private.json').read_text());v['exit_code']=1;write(r/'WORKER_EXIT.private.json',v)
    else:
        if fault=='empty_reconstruction':
            p=r/'PACKAGE_RECONSTRUCTION_ACTUAL.json';v=json.loads(p.read_text());v['max_abs_error']={};write(p,v)
        else:
            p=r/'RESULT.json';v=json.loads(p.read_text())
            if fault=='cursor':v['restore']['sampler_cursor']=0
            else:v['checkpoint']=str(tmp_path/'other/checkpoint')
            write(p,v);lease=json.loads((r/'control/LEASE.json').read_text());lease['result_sha256']=digest(p);write(r/'control/LEASE.json',lease)
    with pytest.raises(ValueError):handoff.audit_training_root(r,job,job_id='fixture.pbs')
