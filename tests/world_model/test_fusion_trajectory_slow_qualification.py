"""Synthetic lineage checks only; these are not a native qualification."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from go2arx5_vla.world_model import fusion_trajectory_qualification_contract as c
from go2arx5_vla.world_model.fusion_trajectory_adapter import TrajectoryAdaptation
from go2arx5_vla.world_model.storage import digest
from test_fusion_trajectory_qualification import recipe, episode, put, BASE


def slow_episode(scale, end):
    value=episode(scale,end)
    value.update(schema=c.SLOW_SCHEMA,phase=c.SLOW_PHASE,reference_rule=c.SLOW_RECIPE,
                 time_stretch_factor=2)
    return value


def completed_prior(spec, previous, inputs, *, scale, slow=False):
    mapped=np.linalg.inv(inputs['frame_map'])@inputs['query_tcp']
    p=TrajectoryAdaptation(inputs['raw'],inputs['query_tcp'],inputs['frame_map'],
        mapped,mapped,BASE,scale,time_stretch_factor=2 if slow else 1)
    e=slow_episode(scale,p.end_atom)if slow else episode(scale,p.end_atom)
    end=c.endpoint(e);atoms=(end-c.INIT_STEP)//25
    rows=[]
    for i in range(atoms):
        opening=.9;expiry=10.+i*.05
        rows.append(dict(native_start_step=c.INIT_STEP+25*i,native_end_step=c.INIT_STEP+25*(i+1),
            actual_ticks=25,solver_policies_with_exact_installed_target=[{'fixture':True}],
            gripper=dict(command_consumed=True,submitted=dict(sequence=i,opening=opening,expiry=expiry),
                consumed_wrapper=dict(status='ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY',
                    values=dict(sequence=i,steps=c.INIT_STEP+25*(i+1),opening=opening,expiry=expiry)))))
    report=dict(episode=e,all_40_knots_consumed=True,complete_task_atoms=atoms,
        solver_target_confirmed_atoms=atoms,gripper_confirmed_atoms=atoms,
        native_endpoint_step=end,windows=rows)
    previous=copy.deepcopy(previous)
    previous.update(phase=e['phase'],actual_target_integrations=end,reference_qualification=report)
    previous['trace_audit']['native_step_records']=end
    path=Path(spec['previous_stage']);put(path,previous)
    root=path.parent.parent
    put(root/'EXECUTION_SPEC.json',dict(phase=e['phase'],max_ticks=end))
    put(root/'control/LEASE.json',dict(status='CLOSED',worker_exit_code=0,
        spec_sha256=digest(root/'EXECUTION_SPEC.json')))
    spec['previous_stage_sha256']=digest(path)
    return previous


@pytest.fixture
def slow_recipe(recipe,monkeypatch,tmp_path):
    spec, previous, path, old_e, inputs=recipe
    previous=completed_prior(spec,previous,inputs,scale=.25)
    monkeypatch.setattr(c,'SLOW_FIRST_PREDECESSOR',spec['previous_stage_sha256'])
    mapped=np.linalg.inv(inputs['frame_map'])@inputs['query_tcp']
    p=TrajectoryAdaptation(inputs['raw'],inputs['query_tcp'],inputs['frame_map'],mapped,mapped,BASE,.5,
                           time_stretch_factor=2)
    e=slow_episode(.5,p.end_atom)
    put(Path(spec['episode_manifest']),e)
    spec.update(phase=c.SLOW_PHASE,wall_limit_s=1200,max_ticks=c.endpoint(e),
                episode_manifest_sha256=digest(spec['episode_manifest']))
    build_path=Path(spec['trajectory_trace_capacity_build']);build=json.loads(build_path.read_text())
    build.update(status='TRACE_ONLY_262144_CAPACITY_COMPILED_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED',
                 trace_capacity=262144)
    put(build_path,build);spec['trajectory_trace_capacity_build_sha256']=digest(build_path)
    failed=copy.deepcopy(previous)
    failed.update(status='NATIVE_STAGE_FAILED_PRESERVED',phase=c.PHASE,actual_target_integrations=10178,
                  failure='ValueError: ORIGINAL_JOINT_ACCELERATION_REJECT')
    failed['trace_audit'].update(native_step_records=10178,writer=dict(dropped=0))
    failed_root=tmp_path/'retained-failure';failed_path=failed_root/'native-01/RESULT.json'
    put(failed_path,failed);put(failed_root/'EXECUTION_SPEC.json',{'fixture':'old failed spec'})
    put(failed_root/'control/LEASE.json',dict(status='CLOSED',worker_exit_code=1,
        spec_sha256=digest(failed_root/'EXECUTION_SPEC.json')))
    put(failed_root/'WORKER_EXIT.private.json',dict(exit_code=1))
    spec.update(retiming_failed_result=str(failed_path),retiming_failed_result_sha256=digest(failed_path))
    monkeypatch.setattr(c,'SLOW_FAILED_RESULT',spec['retiming_failed_result_sha256'])
    put(path,spec)
    return spec,previous,path,e,inputs


def test_slow_half_uses_old_quarter_pass_and_preserves_old_half_failure(slow_recipe):
    spec,previous,_,e,_=slow_recipe
    actual,_=c.episode_config(spec)
    assert actual==e and c.episode_limit(spec,previous)==c.endpoint(e)
    assert c.validate_retiming_failure(spec)['status']=='NATIVE_STAGE_FAILED_PRESERVED'


@pytest.mark.parametrize('field,value', [('time_stretch_factor',True),('time_stretch_factor',1),
    ('reference_rule',c.RECIPE),('geometry_scale',.25),('planned_end_atom',2000)])
def test_slow_geometry_requires_new_explicit_timing_contract(slow_recipe,field,value):
    e=slow_recipe[3]
    with pytest.raises(ValueError):c.episode_contract(dict(e,**{field:value}))


@pytest.mark.parametrize('field,value', [('trace_capacity',131072),
    ('status','TRACE_ONLY_131072_CAPACITY_COMPILED_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED')])
def test_slow_cannot_reuse_old_trace_capacity_receipt(slow_recipe,field,value):
    spec=slow_recipe[0];p=Path(spec['trajectory_trace_capacity_build']);proof=json.loads(p.read_text())
    proof[field]=value;put(p,proof);spec['trajectory_trace_capacity_build_sha256']=digest(p)
    with pytest.raises(ValueError,match='ACTUAL_TRACE_ONLY_CAPACITY'):c.episode_config(spec)


def test_slow_full_requires_slow_half_complete_not_old_failed_half(slow_recipe):
    spec,previous,_,_,inputs=slow_recipe
    previous=completed_prior(spec,previous,inputs,scale=.5,slow=True)
    mapped=np.linalg.inv(inputs['frame_map'])@inputs['query_tcp']
    p=TrajectoryAdaptation(inputs['raw'],inputs['query_tcp'],inputs['frame_map'],mapped,mapped,BASE,1.,
                           time_stretch_factor=2)
    e=slow_episode(1.,p.end_atom);put(Path(spec['episode_manifest']),e)
    spec.update(max_ticks=c.endpoint(e),episode_manifest_sha256=digest(spec['episode_manifest']))
    assert c.episode_limit(spec,previous)==c.endpoint(e)
    # Rehashing a changed old-phase fixture cannot pass the family lineage.
    previous['phase']=c.PHASE;put(Path(spec['previous_stage']),previous)
    spec['previous_stage_sha256']=digest(spec['previous_stage'])
    with pytest.raises(ValueError,match='IMMEDIATELY_PRECEDING'):c.episode_limit(spec,previous)


def test_rehashed_failure_with_open_lease_is_still_rejected(slow_recipe):
    spec=slow_recipe[0];p=Path(spec['retiming_failed_result']).parent.parent/'control/LEASE.json'
    value=json.loads(p.read_text());value['status']='ACTIVE';put(p,value)
    with pytest.raises(ValueError,match='RETAINED_FAILURE_MUST_BE_TERMINAL'):c.validate_retiming_failure(spec)
