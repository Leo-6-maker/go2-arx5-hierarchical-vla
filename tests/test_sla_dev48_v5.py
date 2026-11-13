"""DEV48 metadata and closed-prefix fixtures; no model or physics evidence."""
import copy
import json
from pathlib import Path

import pytest

from go2arx5_vla.world_model import sla_dev48_v5 as dev,sla_dense_native_v5 as dense
from go2arx5_vla.world_model import task_learning_grasp_v1 as task,task_learning_scene_v1 as scene
from go2arx5_vla.world_model import sla_population_v1 as old_sla
from go2arx5_vla.world_model.task_learning_expansion_v1 import OFFSETS as expansion
from go2arx5_vla.world_model.storage import digest


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    return dict(path=str(path),sha256=digest(path))


def panel(tmp_path):
    _,pin=dev.declaration();decl=dict(path=str(Path(dev.__file__).resolve().parents[3]/dev.DECLARATION),sha256=pin)
    proof=save(tmp_path/'disjoint.json',dict(schema=dev.DISJOINT_SCHEMA,passes=True,preregistration=decl,
        new_initial_inputs=[dict(layout=l,initial_geometry_sha256=old_sla.geometry_sha(scene.geometry(l)),
            model_sha256='b'*64,actual=old_sla.initial_geometry(scene.geometry(l)))for l in dev.OFFSETS],
        old_TEST_initial_inputs=20,original_physical_scenes_unchanged=True,old_TEST_scores_read=False,
        old_TEST_images_actions_trajectories_read=False,physics_integrations=0,model_calls=0,optimizer_updates=0))
    return dict(schema=dev.SCHEMA,task_learning_role=dense.PROTOCOL,protocol=dense.protocol(),
        population=dev.POPULATION,method=dense.NAMED_METHOD,B1_global_step=3000,source_sha='a'*40,
        policies={dense.NAMED_METHOD:dict(arm=dense.NAMED_METHOD)},preregistration=decl,scene_disjointness=proof,
        assignments=[dict(r,native_run=f'new-dev48-{r["layout"]}-r{r["restart"]}',method=dense.NAMED_METHOD,
            parent=scene.scene_id(r['layout']),split=dev.SPLIT,command=task.COMMAND,purpose=dev.EXPERIMENT,
            model_sha256='b'*64)for r in dev.inputs()])


def test_new_population_and_geometry_are_disjoint_from_old_inputs():
    old={**scene.TRAIN_OFFSETS,**scene.QUALIFICATION_OFFSETS,**scene.RECOVERY_OFFSETS,
         **scene.DEV_OFFSETS,**old_sla.DEV_OFFSETS,**expansion}
    assert not set(dev.OFFSETS.values())&set(old.values())
    rows=dev.inputs();assert len(rows)==48 and all(len(r['query_seeds'])==60 for r in rows)
    assert len({q for r in rows for q in r['query_seeds']})==48*60
    assert {scene.geometry(l)['role']for l in dev.OFFSETS}=={'SLA_V5_DEV48_NEW_GEOMETRY_NOT_TRAIN'}


def test_first_group_cannot_skip_or_select_the_population(tmp_path):
    p=panel(tmp_path);inp=dict(schema=dev.INPUT_SCHEMA,source_sha=p['source_sha'],assignment_slice=[0,8])
    assert len(dense.validate_panel(p))==48
    assert dense.selection(inp,p)==p['assignments'][:8]
    for span in ([1,9],[8,16],[0,9],[0,48]):
        with pytest.raises(ValueError):dense.selection(dict(inp,assignment_slice=span),p)
    old=json.loads((Path(dense.__file__).resolve().parents[3]/dense.DECLARATION).read_bytes())
    assert len(old['conditions'])==8


def test_readonly_capacity_wait_is_available_for_new_raw3k_without_new_physics_grant(tmp_path):
    from go2arx5_vla.world_model.sla_native_resource_continuation_v3 import configuration,WAIT_VERSION
    p=panel(tmp_path);p['registration']=dict(capacity_wait_before_pair_v3=dict(version=WAIT_VERSION,poll_s=60))
    assert configuration(p)==dict(version=WAIT_VERSION,poll_s=60)
    p['task_learning_role']=task.COLLECTION_ROLE
    with pytest.raises(ValueError):configuration(p)


def test_dispatch_capacity_uses_the_reconstructed_endpoint_card_not_old_base(tmp_path,monkeypatch):
    from go2arx5_vla.world_model.task_learning_student_registration_v2 import execution_gpu_base
    p=panel(tmp_path);q=dict(runtime=dict(python='/qualified',upstream_root='/upstream',JAX_memory_fraction=.8),
        GPU_uuid='qualified-card',parent_params='/base/model/params',parent_manifest=dict(path='/base/MANIFEST',sha256='c'*64),
        ordinary_parameter_bytes=10,allocator_limit_bytes=30,ordinary_RGB_reconstruction=dict(path='/actual/RGB',sha256='d'*64),
        schema=dense.NAMED_ADMISSION,neural_owner_lock='/qualified-card.lock')
    monkeypatch.setattr(dense,'policy_contract',lambda *a,**k:q)
    base=dict(GPU_uuid='old-card',neural_owner_lock='/old-card.lock')
    actual=execution_gpu_base(p,base)
    assert actual['GPU_uuid']=='qualified-card'and actual['neural_owner_lock']=='/qualified-card.lock'
    assert base['GPU_uuid']=='old-card'and actual['checkpoint_parameter_bytes']==10


@pytest.mark.parametrize('mutation',['split','seed','order','model','step500','old_input','proof_geometry'])
def test_dev_inputs_cannot_relabel_training_or_weaken_actual_geometry(tmp_path,mutation):
    p=panel(tmp_path)
    if mutation=='split':p['assignments'][0]['split']='TRAIN'
    elif mutation=='seed':p['assignments'][0]['query_seeds'][0]+=1
    elif mutation=='order':p['assignments'].reverse()
    elif mutation=='model':p['assignments'][0]['model_sha256']='c'*64
    elif mutation=='step500':p['B1_global_step']=500
    elif mutation=='old_input':p['schema']=dense.NAMED_SCHEMA
    else:
        proof=json.loads(Path(p['scene_disjointness']['path']).read_bytes());proof['new_initial_inputs'][0]['actual']['object_initial_xyz_m'][0]+=.001
        p['scene_disjointness']=save(Path(p['scene_disjointness']['path']),proof)
    with pytest.raises(ValueError):dense.validate_panel(p)


def prior_group(folder,pin,p,span,previous=None):
    inp=dict(schema=dev.INPUT_SCHEMA,source_sha=p['source_sha'],panel=pin['path'],panel_sha256=pin['sha256'],
        assignment_slice=span,population_predecessor=previous)
    inputpin=save(folder/'INPUT.private.json',inp)
    rows=[dict(r,technical_valid=True,hard_safety_stop=False,task_success=(i%2==0))
          for i,r in enumerate(p['assignments'][:span[1]])]
    save(folder/'OUTCOMES.private.json',rows)
    save(folder/'control/LEASE.json',dict(status='CLOSED',worker_exit_code=0,input_sha256=inputpin['sha256']))
    for r in rows[span[0]:]:
        for tail in ('','-gpu'):
            root=folder/'runs'/(r['native_run']+tail);spec=save(root/'EXECUTION_SPEC.json',dict(metadata_fixture=True))
            save(root/'control/LEASE.json',dict(status='CLOSED',worker_exit_code=0,spec_sha256=spec['sha256']))
            save(root/'WORKER_EXIT.private.json',dict(exit_code=0))
    return save(folder/'RESULT.private.json',dict(exit_code=0,source_sha=p['source_sha'],
        unresolved_pair=None,cumulative_completed=span[1],summary=dict(requested=48,complete=False)))


def test_two_actual_prefix_groups_retain_valid_failures_and_advance_once(tmp_path,monkeypatch):
    p=panel(tmp_path);pin=save(tmp_path/'PANEL.json',p)
    monkeypatch.setattr(dev.subprocess,'check_output',lambda *a,**k:'0\n')
    first=prior_group(tmp_path/'group0',pin,p,[0,8]);second=prior_group(tmp_path/'group8',pin,p,[8,16],first)
    inp=dict(schema=dev.INPUT_SCHEMA,source_sha=p['source_sha'],panel=pin['path'],panel_sha256=pin['sha256'],
        assignment_slice=[16,24],population_predecessor=second)
    assert len(dev.retained_prefix(inp,p))==16
    assert dev.selection(inp,p)==p['assignments'][16:24]
    lease=tmp_path/'group0/runs'/p['assignments'][0]['native_run']/'control/LEASE.json'
    value=json.loads(lease.read_bytes());value['status']='ACTIVE';save(lease,value)
    with pytest.raises(ValueError):dev.selection(inp,p)
