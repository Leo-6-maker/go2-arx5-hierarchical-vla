"""Graded recipe/gate tests use synthetic records, never native pass claims."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pi0292.eval_contract import matrix10
from go2arx5_vla.world_model import auto1_native, target_sequential
from go2arx5_vla.world_model import fusion_task_phase_initialization as original
from go2arx5_vla.world_model import fusion_task_phase_initialization_native as original_native
from go2arx5_vla.world_model import fusion_task_phase_bias_calibration_v2 as v2
from go2arx5_vla.world_model import fusion_task_phase_bias_calibration_contract_v2 as c2
from go2arx5_vla.world_model import fusion_task_phase_bias_continuation_v3 as v3
from go2arx5_vla.world_model import fusion_task_phase_bias_continuation_contract_v3 as contract
from go2arx5_vla.world_model import fusion_task_phase_bias_continuation_native_v3 as native
from go2arx5_vla.world_model.storage import digest
from scripts.wm02 import fusion_task_phase_bias_continuation_check_v3 as cli
from test_fusion_task_phase_initialization import setup as pure_setup, samples as hold_samples
from test_fusion_task_phase_bias_calibration_v2 import recipe as v2_recipe, BiasAdapter
from test_fusion_live_proposal_qualification_contract import prior_files, put, close
from test_fusion_trajectory_qualification import setup_flow
from test_fusion_reference_qualification_contract import reference


@pytest.fixture
def setup(monkeypatch):
    return pure_setup.__wrapped__(monkeypatch)


@pytest.mark.parametrize('grade,ramp,end', [(.1,256,10303), (.25,512,16703), (.5,768,23103), (1.,1024,29503)])
def test_all_registered_grades_keep_original_desired_recipe_and_same_frozen_bias(setup, grade, ramp, end):
    episode = v3.make_episode(run='/fixture/graded-init', geometry_scale=grade)
    p = v3.make_program(setup['inputs'], episode, setup['initial'], setup['decision'], setup['base'])
    old = original.make_program(setup['inputs'], v3.parent_recipe_view(episode),
        setup['initial'], setup['decision'], setup['base'])
    assert v3.endpoint(episode) == end and p.end_atom == ramp
    assert np.array_equal(p.desired_path, old.desired_path)
    assert np.array_equal(p.graded_goal_TCP_target, old.graded_goal_TCP_target)
    desired, refs = matrix10(p.desired_path), matrix10(p.reference_path)
    b = setup['decision'][:3,3]-setup['initial'][:3,3]
    left = setup['initial'][:3,:3] @ setup['decision'][:3,:3].T
    np.testing.assert_allclose(refs[:,:3,3], desired[:,:3,3]-b, rtol=0, atol=1e-14)
    np.testing.assert_allclose(refs[:,:3,:3], np.einsum('ij,njk->nik',left,desired[:,:3,:3]), rtol=0, atol=1e-14)
    proof = p.evidence()['frozen_predecision_reference_binding']
    assert proof['schema'] == 'WM02_FROZEN_PREDECISION_REFERENCE_BINDING_V3'
    assert proof['largest_measurement_native_step'] == 3703 and not proof['future_tracking_error_used']
    for offset in (0,7,ramp-1,ramp,ramp+7):
        value = v3.validate_install_preview(p.preview(offset), initial_tcp=setup['initial'], program=p,
            step=3703+25*offset, base=setup['base'], frozen_base=setup['base'], end_step=end)
        assert value['points_checked'] == 40 and not value['native_execution_authorized']
    with pytest.raises(ValueError): p.reference_path.setflags(write=True)


def test_old_v2_type_and_grade_identity_remain_strict_with_new_shared_math(setup):
    e2 = v2.make_episode(run='/fixture/old-v2',geometry_scale=.03)
    p2 = v2.make_program(setup['inputs'],e2,setup['initial'],setup['decision'],setup['base'])
    e3 = v3.make_episode(run='/fixture/new-v3',geometry_scale=.1)
    p3 = v3.make_program(setup['inputs'],e3,setup['initial'],setup['decision'],setup['base'])
    assert p2.evidence()['schema'] == v2.SCHEMA and p2.evidence()['recipe'] == v2.RECIPE
    assert p2.evidence()['frozen_predecision_reference_binding']['schema'].endswith('_V2')
    with pytest.raises(ValueError): v2.episode_contract(e3)
    with pytest.raises(ValueError): v3.episode_contract(e2)
    with pytest.raises(ValueError): v2.make_episode(run='/fixture/no-implicit-v2-expansion',geometry_scale=.1)
    with pytest.raises(ValueError): v3.make_episode(run='/fixture/no-repeat-v2-grade',geometry_scale=.03)
    with pytest.raises(ValueError):
        v2.validate_install_preview(p3.preview(),initial_tcp=setup['initial'],program=p3,
            step=3703,base=setup['base'],frozen_base=setup['base'],end_step=10303)
    with pytest.raises(ValueError):
        v3.validate_install_preview(p2.preview(),initial_tcp=setup['initial'],program=p2,
            step=3703,base=setup['base'],frozen_base=setup['base'],end_step=10303)
    changed = p3.preview(); changed[39,0] += .001
    with pytest.raises(ValueError):
        v3.validate_install_preview(changed,initial_tcp=setup['initial'],program=p3,
            step=3703,base=setup['base'],frozen_base=setup['base'],end_step=10303)


def qualified_prior(root, episode, spec, setup):
    is_first = episode['phase'] == v2.PHASE
    module = v2 if is_first else v3
    source = contract.FIRST_PREDECESSOR_SOURCE if is_first else spec['source_sha']
    previous, path, _, _ = prior_files(root, episode, spec, source=source)
    prior_spec = json.loads((root/'EXECUTION_SPEC.json').read_text())
    prior_spec.update(kind=c2.KIND if is_first else contract.KIND,
        source_root=spec['source_root'],source_files=copy.deepcopy(spec['source_files']))
    spec_pin = close(root,prior_spec)
    native_root = path.parent
    packet = native_root/'SYNTHETIC_DECISION_STATE.bin'
    packet.write_bytes(setup['args']['decision_packet_bytes'])
    measured_path = native_root/'SYNTHETIC_DECISION_MEASURED.json'
    put(measured_path,original_native._plain(setup['args']['measured_decision_state']))
    goal = dict(actual_spec_phase=episode['phase'],actual_spec_source_sha=source,
        episode_sha256=prior_spec['episode_manifest_sha256'],decision_step=3703,
        frozen_before_generated_reference_install=True,frozen_input=setup['inputs']['frozen'],
        saved_packet_path=str(packet),saved_packet_sha256=digest(packet),
        measured_state_path=str(measured_path),measured_state_sha256=digest(measured_path))
    rows = hold_samples(setup,episode['geometry_scale'])
    readiness = original.query_readiness(rows,module.parent_recipe_view(episode),setup['inputs'])
    assert all(value for key,value in readiness['predicates'].items() if key!='full_grade')
    binding = dict(experiment_role='NON_VLA_CALIBRATION',actual_VLA_commands=0,VLA_queries=0,
        object_and_goal_are_policy_inputs=False,readiness=readiness,
        readiness_sample_native_steps=readiness['actual_sample_steps'])
    for key,value in (('goal_binding',goal),('hold_samples',original_native._plain(rows)),('readiness',readiness)):
        artifact = native_root/('SYNTHETIC_'+key+'.json')
        binding.update({key+'_path':str(artifact),key+'_sha256':put(artifact,value)})
    previous['reference_qualification']['task_phase_initialization'] = binding
    result_pin = put(path,previous)
    return previous,path,result_pin,spec_pin


@pytest.fixture
def recipe(tmp_path,monkeypatch):
    spec,_,_,setup = v2_recipe.__wrapped__(tmp_path,monkeypatch)
    project,capsule = Path(__file__).resolve().parents[2],Path(spec['source_root'])
    for name in contract.SOURCE_MEMBERS:
        path=capsule/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((project/name).read_bytes());spec['source_files'][name]=digest(path)
    root=tmp_path/'SYNTHETIC_QUALIFIED_V2'
    prior_episode=v2.make_episode(run=str(root),geometry_scale=.03)
    previous,path,pin,spec_pin=qualified_prior(root,prior_episode,spec,setup)
    monkeypatch.setattr(contract,'FIRST_PREDECESSOR',pin)
    monkeypatch.setattr(contract,'FIRST_PREDECESSOR_SPEC',spec_pin)
    spec.update(phase=v3.PHASE,kind=contract.KIND,previous_stage=str(path),previous_stage_sha256=pin)
    episode=v3.make_episode(run=spec['registered_root'],geometry_scale=.1)
    spec['episode_manifest_sha256']=put(spec['episode_manifest'],episode);spec['max_ticks']=v3.endpoint(episode)
    return spec,previous,episode,setup


def test_first_point1_checks_exact_closed_v2_and_recomputes_nine_predicates_without_grant(recipe,tmp_path):
    spec,previous,episode,_=recipe
    assert contract.episode_limit(spec,previous)==10303
    path=tmp_path/'SPEC.json';put(path,spec)
    checked=cli.check(path)
    assert checked['prior_nine_non_full_grade_predicates_passed']
    assert checked['max_ticks']==10303 and not checked['native_execution_authorized']
    assert not checked['previous_state_reused_for_new_bias']
    assert checked['new_integrations']==checked['VLA_queries']==0


@pytest.mark.parametrize('predicate',sorted(contract.READINESS_PREDICATES-{'full_grade'}))
def test_each_failed_real_measurement_predicate_blocks_even_with_complete_native_pass_and_rehashed_artifacts(recipe,predicate,monkeypatch):
    spec,previous,_,setup=recipe
    binding=previous['reference_qualification']['task_phase_initialization']
    rows=json.loads(Path(binding['hold_samples_path']).read_text())
    for index,row in enumerate(rows):
        if predicate=='position':row['tcp'][2][3]+=.02
        elif predicate=='lateral':row['tcp'][0][3]+=.02
        elif predicate=='orientation':
            c,s=np.cos(.1),np.sin(.1);row['tcp']=np.asarray(row['tcp'])
            row['tcp'][:3,:3]=np.array([[c,-s,0],[s,c,0],[0,0,1]])@row['tcp'][:3,:3]
            row['tcp']=row['tcp'].tolist()
        elif predicate=='actual_cube_distance':row['object_pose']=copy.deepcopy(row['tcp'])
        elif predicate=='continuous_relative_hold':row['tcp'][0][3]+=index*.0005
        elif predicate=='actual_open':row['finger_q']=[.037,.037]
        elif predicate=='cube_still':row['object_pose'][0][3]+=.001
        elif predicate=='no_contact':row['robot_object_contact']=True
        elif predicate=='original_safety':row['safety_valid']=False
    prior=previous['reference_qualification']['episode']
    readiness=original.query_readiness(rows,v2.parent_recipe_view(prior),setup['inputs'])
    assert readiness['predicates'][predicate] is False
    binding['hold_samples_sha256']=put(binding['hold_samples_path'],rows)
    binding['readiness_sha256']=put(binding['readiness_path'],readiness);binding['readiness']=readiness
    spec['previous_stage_sha256']=put(spec['previous_stage'],previous)
    monkeypatch.setattr(contract,'FIRST_PREDECESSOR',spec['previous_stage_sha256'])
    with pytest.raises(ValueError,match='ALL_NINE_NON_FULL_GRADE'):
        contract.episode_limit(spec,previous)


@pytest.mark.parametrize('fault',['skip','repeat_v2','open','trace_drop','unconsumed','changed_saved_sample','lying_readiness','changed_packet'])
def test_prior_native_and_sample_evidence_are_not_replaced_by_pass_flags(recipe,fault,monkeypatch):
    spec,previous,_,_=recipe
    binding=previous['reference_qualification']['task_phase_initialization']
    if fault in('skip','repeat_v2'):
        episode=(v3.make_episode(run=spec['registered_root'],geometry_scale=.25)if fault=='skip'
                 else v2.make_episode(run=spec['registered_root'],geometry_scale=.03))
        spec['episode_manifest_sha256']=put(spec['episode_manifest'],episode);spec['max_ticks']=original.endpoint(episode)
    elif fault=='open':put(Path(spec['previous_stage']).parent.parent/'control/LEASE.json',{'status':'ACTIVE'})
    elif fault=='trace_drop':previous['trace_audit']['writer']['dropped']=1
    elif fault=='unconsumed':previous['reference_qualification']['windows'][-1]['gripper']['command_consumed']=False
    elif fault=='changed_saved_sample':Path(binding['hold_samples_path']).write_text('[]')
    elif fault=='lying_readiness':
        data=json.loads(Path(binding['readiness_path']).read_text());data['query_ready']=True
        binding['readiness_sha256']=put(binding['readiness_path'],data);binding['readiness']=data
    else:
        goal=json.loads(Path(binding['goal_binding_path']).read_text());Path(goal['saved_packet_path']).write_bytes(b'changed')
    if fault in('trace_drop','unconsumed','lying_readiness'):
        spec['previous_stage_sha256']=put(spec['previous_stage'],previous)
        monkeypatch.setattr(contract,'FIRST_PREDECESSOR',spec['previous_stage_sha256'])
    with pytest.raises((ValueError,PermissionError)):
        contract.episode_limit(spec,previous)


@pytest.mark.parametrize('previous_grade,next_grade',[(.1,.25),(.25,.5),(.5,1.)])
def test_subsequent_grades_require_immediate_same_v3_source_and_qualified_actual_hold(recipe,tmp_path,previous_grade,next_grade):
    spec,_,_,setup=recipe
    root=tmp_path/('SYNTHETIC_V3_PRIOR_'+str(previous_grade))
    previous,path,pin,_=qualified_prior(root,v3.make_episode(run=str(root),geometry_scale=previous_grade),spec,setup)
    spec.update(previous_stage=str(path),previous_stage_sha256=pin)
    episode=v3.make_episode(run=spec['registered_root'],geometry_scale=next_grade)
    spec['episode_manifest_sha256']=put(spec['episode_manifest'],episode);spec['max_ticks']=v3.endpoint(episode)
    assert contract.episode_limit(spec,previous)==v3.endpoint(episode)
    spec['source_sha']='ANOTHER_UNQUALIFIED_RESTART_SOURCE'
    with pytest.raises(ValueError,match='IMMEDIATELY_PRECEDING_SAME_V3'):
        contract.episode_limit(spec,previous)


class ContinuationAdapter(BiasAdapter):
    def install_reference(self,action):
        native.validate_adapter_install(self,action)
        if self.step>=3703:
            saved=json.loads((self.output/'TASK_PHASE_GOAL_BINDING.private.json').read_text())
            assert saved['actual_spec_phase']==v3.PHASE
            assert saved['frozen_predecision_reference_calibration']['schema'].endswith('_V3')
        self.active_reference=dict(reference(round(self.step*.002*1e9)/1e9),action_EEF10=action.tolist())
        self.installs.append((self.step,action.copy()));return self.active_reference


def test_first_continuation_shared_native_flow_keeps_current_restart_bias_and_real_samples(recipe,tmp_path,monkeypatch):
    spec,_,episode,setup=recipe
    output=tmp_path/'SYNTHETIC_V3_NATIVE';output.mkdir()
    adapter=ContinuationAdapter(spec,output,setup);setup_flow(monkeypatch,adapter)
    _,source=contract.episode_config(spec)
    monkeypatch.setattr(contract,'episode_config',lambda s:(episode,source))
    monkeypatch.setattr(original_native,'_verify_actual_fk',lambda *a:{'status':'SYNTHETIC_NO_NATIVE_CLAIM'})
    report=native.run(adapter)
    assert adapter.sealed==10303 and report['complete_task_atoms']==report['gripper_confirmed_atoms']==312
    assert report['all_40_knots_consumed'] and report['actual_trace_binding']['actual_trace_capacity']==524288
    binding=report['task_phase_initialization']
    assert binding['readiness_sample_native_steps']==list(range(10103,10304,25))
    assert binding['frozen_predecision_reference_calibration']['largest_measurement_native_step']==3703
    assert not report['historical_terminal_tracking_error_used']
    assert report['VLA_queries']==report['WM_updates']==report['Actor_updates']==0


def test_v3_dispatch_and_cli_are_original_worker_routes_without_backend_import(recipe,tmp_path,monkeypatch):
    spec,previous,_,_=recipe
    assert auto1_native.phase_limit(v3.PHASE,previous,spec)==10303
    assert target_sequential.native_task_bucket(v3.PHASE,True)=='64'
    adapter=target_sequential.TargetSequentialAdapter({'phase':v3.PHASE},tmp_path)
    token=object();monkeypatch.setattr(native,'run',lambda a:token)
    assert adapter.dynamic_diagnostic()is token
    path=tmp_path/'SPEC.json';put(path,spec);calls=[]
    monkeypatch.setitem(sys.modules,'scripts.wm02.pilot1_native_worker',SimpleNamespace(main=lambda a:calls.append(a) or 75))
    assert cli.main(['--spec',str(path),'--execute'])==75 and calls==[['--spec',str(path)]]
    code=('import sys;from go2arx5_vla.world_model import fusion_task_phase_bias_continuation_v3;'
          'from go2arx5_vla.world_model import fusion_task_phase_bias_continuation_native_v3;'
          'assert not {"mujoco","torch","rclpy","gr00t","transformers"}&sys.modules.keys()')
    subprocess.run([sys.executable,'-B','-c',code],check=True)
