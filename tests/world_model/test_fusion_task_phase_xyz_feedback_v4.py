"""Finite ordinary-feedback commissioning tests; synthetic evidence only."""
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
from go2arx5_vla.world_model import fusion_task_phase_bias_continuation_v3 as v3
from go2arx5_vla.world_model import fusion_trajectory_qualification_native as common_native
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_v4 as v4
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_contract_v4 as contract
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_native_v4 as native
from go2arx5_vla.world_model.storage import digest
from scripts.wm02 import fusion_task_phase_xyz_feedback_check_v4 as cli
from test_fusion_task_phase_initialization import setup as pure_setup, samples as hold_samples
from test_fusion_task_phase_bias_continuation_v3 import recipe as old_recipe, qualified_prior
from test_fusion_task_phase_bias_calibration_v2 import BiasAdapter
from test_fusion_live_proposal_qualification_contract import put
from test_fusion_trajectory_qualification import setup_flow
from test_fusion_reference_qualification_contract import reference


@pytest.fixture
def setup(monkeypatch):
    return pure_setup.__wrapped__(monkeypatch)


def program(setup):
    episode = v4.make_episode(run='/fixture/finite-xyz-feedback')
    return v4.make_program(setup['inputs'], episode, setup['initial'], setup['decision'], setup['base']), episode


def offset_actual(p, delta):
    actual = p.graded_goal_TCP_target.copy()
    actual[:3, 3] -= delta
    return actual


def test_exact_point1_clock_and_static_desired_baseline_remain_unchanged(setup):
    p, episode = program(setup)
    ancestor = original.make_program(setup['inputs'], v4.parent_recipe_view(episode),
        setup['initial'], setup['decision'], setup['base'])
    assert episode['planned_end_atom'] == p.end_atom == 368 and episode['ramp_atoms'] == 256
    assert v4.endpoint(episode) == 13103 and v4.FEEDBACK_STEPS == (10303,11003,11703,12403)
    np.testing.assert_array_equal(p.desired_path[:257], ancestor.desired_path)
    np.testing.assert_array_equal(p.desired_path[256:], np.repeat(ancestor.desired_path[-1][None],113,axis=0))
    proof = p.evidence()
    assert proof['baseline_condition'] == v4.BASELINE_CONDITION
    frozen = proof['frozen_predecision_reference_binding']
    assert frozen['baseline_condition'] == v4.BASELINE_CONDITION
    assert frozen['static_component_absolute_ancestor_condition'] == 'INIT_ABSOLUTE_V1'
    assert not frozen['future_online_feedback_path_known_at3703']
    assert frozen['largest_measurement_native_step'] == 3703
    for grade in (.03,.25,.5,1.,True):
        with pytest.raises(ValueError):v4.make_episode(run='/fixture/not-allowed',geometry_scale=grade)
    with pytest.raises(ValueError):v3.episode_contract(episode)
    with pytest.raises(ValueError):v4.episode_contract(dict(episode,planned_end_atom=369))


def test_prepare_save_commit_chronology_norm_cap_quintic_and_current_endpoint_metadata(setup):
    p,_ = program(setup)
    before, desired = p.reference_path.copy(), p.desired_path.copy()
    update = p.prepare_feedback(10303, offset_actual(p, np.array([.03,.04,0.])))
    np.testing.assert_array_equal(p.reference_path,before)
    assert p.feedback_count == 0
    np.testing.assert_allclose(update.evidence['applied_delta_world_m'],[.0024,.0032,0.],atol=1e-15)
    atom = 264
    np.testing.assert_allclose(update.reference_path[atom+10,:3]-before[atom,:3],[.0012,.0016,0.],atol=1e-15)
    np.testing.assert_array_equal(update.reference_path[:atom],before[:atom])
    np.testing.assert_array_equal(update.reference_path[:,3:],before[:,3:])
    assert update.evidence['new_H40_sha256'] == v2._sha(update.reference_path[np.minimum(atom+np.arange(40),368)])
    with pytest.raises(ValueError):p.commit_feedback(update,saved_decision_sha256='not-a-file-sha')
    p.commit_feedback(update,saved_decision_sha256='a'*64)
    assert p.feedback_count == 1
    np.testing.assert_array_equal(p.desired_path,desired)
    proof = p.evidence()
    np.testing.assert_array_equal(proof['reference_endpoint_TCP'],matrix10(p.reference_path[-1]))
    assert proof['reference_endpoint_TCP'] != proof['initial_static_reference_endpoint_TCP']
    assert proof['frozen_predecision_reference_binding']['reference_path_sha256'] == v2._sha(before)
    with pytest.raises(ValueError):p.commit_feedback(update,saved_decision_sha256='a'*64)
    with pytest.raises(ValueError):p.prepare_feedback(11703,offset_actual(p,[0.,0.,0.]))
    with pytest.raises(ValueError):p.reference_path.setflags(write=True)


def test_four_causal_decisions_have_finite_total_budget_and_no_extra_attempt(setup):
    p,_ = program(setup)
    for index,step in enumerate(v4.FEEDBACK_STEPS):
        update=p.prepare_feedback(step,offset_actual(p,[.05,0.,0.]))
        assert np.isclose(update.evidence['applied_delta_norm_m'],.004)
        p.commit_feedback(update,saved_decision_sha256=str(index)*64)
    assert np.isclose(p.evidence()['cumulative_feedback_reference_path_length_m'],.016)
    assert p.feedback_count==4
    with pytest.raises(ValueError):p.prepare_feedback(13103,offset_actual(p,[.05,0.,0.]))
    p.require_decisions_through(13103)


def test_deadband_commits_explicit_hold_and_unsaved_scheduled_preview_is_rejected(setup):
    p,_=program(setup);before=p.reference_path.copy()
    with pytest.raises(ValueError,match='NO_SCHEDULED_REFERENCE'):
        v4.validate_install_preview(p.preview(264),initial_tcp=setup['initial'],program=p,step=10303,
            base=setup['base'],frozen_base=setup['base'],end_step=13103)
    update=p.prepare_feedback(10303,offset_actual(p,[.003,0.,0.]))
    assert update.evidence['deadband_hold'] and update.evidence['applied_delta_norm_m']==0
    np.testing.assert_array_equal(update.reference_path,before)
    p.commit_feedback(update,saved_decision_sha256='c'*64)
    value=v4.validate_install_preview(p.preview(264),initial_tcp=setup['initial'],program=p,step=10303,
        base=setup['base'],frozen_base=setup['base'],end_step=13103)
    assert value['points_checked']==40
    wrong=p.preview(264);wrong[39,0]+=.001
    with pytest.raises(ValueError):
        v4.validate_install_preview(wrong,initial_tcp=setup['initial'],program=p,step=10303,
            base=setup['base'],frozen_base=setup['base'],end_step=13103)
    with pytest.raises(ValueError):
        v3.validate_install_preview(p.preview(),initial_tcp=setup['initial'],program=p,step=3703,
            base=setup['base'],frozen_base=setup['base'],end_step=13103)


def test_explicit_v4_readiness_endpoint_does_not_relax_public_v1_manifest_or_clock(setup):
    _,episode=program(setup)
    rows=hold_samples(setup,.1)
    for row in rows:row.update(native_step=row['native_step']+2800,time_s=(row['native_step']+2800)*.002)
    value=v4.query_readiness(rows,episode,setup['inputs'])
    assert value['actual_sample_steps']==list(range(12903,13104,25))
    assert value['schema'].endswith('_V4') and not value['native_execution_authorized']
    assert all(v for k,v in value['predicates'].items() if k!='full_grade')
    with pytest.raises(ValueError):original.query_readiness(rows,episode,setup['inputs'])
    with pytest.raises(ValueError):original.query_readiness(rows,v4.parent_recipe_view(episode),setup['inputs'])


@pytest.fixture
def recipe(tmp_path,monkeypatch):
    spec,_,_,setup=old_recipe.__wrapped__(tmp_path,monkeypatch)
    capsule,project=Path(spec['source_root']),Path(__file__).resolve().parents[2]
    for name in contract.SOURCE_MEMBERS:
        path=capsule/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((project/name).read_bytes());spec['source_files'][name]=digest(path)
    root=tmp_path/'SYNTHETIC_POINT1_NATIVE_COMPLETE_GOAL_FAILED'
    prior_spec=copy.deepcopy(spec);prior_spec['source_sha']=contract.FIRST_PREDECESSOR_SOURCE
    previous,path,_,spec_pin=qualified_prior(root,v3.make_episode(run=str(root),geometry_scale=.1),prior_spec,setup)
    binding=previous['reference_qualification']['task_phase_initialization']
    rows=json.loads(Path(binding['hold_samples_path']).read_text())
    for row in rows:row['tcp'][0][3]+=.013
    readiness=v3.query_readiness(rows,previous['reference_qualification']['episode'],setup['inputs'])
    assert not readiness['predicates']['position'] and not readiness['predicates']['lateral']
    binding['hold_samples_sha256']=put(binding['hold_samples_path'],rows)
    binding['readiness_sha256']=put(binding['readiness_path'],readiness);binding['readiness']=readiness
    pin=put(path,previous)
    monkeypatch.setattr(contract,'FIRST_PREDECESSOR',pin);monkeypatch.setattr(contract,'FIRST_PREDECESSOR_SPEC',spec_pin)
    spec.update(phase=v4.PHASE,kind=contract.KIND,previous_stage=str(path),previous_stage_sha256=pin,max_ticks=13103)
    episode=v4.make_episode(run=spec['registered_root'])
    spec['episode_manifest_sha256']=put(spec['episode_manifest'],episode)
    return spec,previous,episode,setup


def test_exact_native_pass_goal_failed_predecessor_does_not_become_fake_qualified_admission(recipe,tmp_path):
    spec,previous,episode,_=recipe
    assert contract.episode_limit(spec,previous)==13103
    path=tmp_path/'SPEC.json';put(path,spec)
    checked=cli.check(path)
    assert checked['prior_position_and_lateral_failure_retained']
    assert checked['prior_other_seven_non_full_grade_predicates_passed']
    assert not checked['native_execution_authorized'] and checked['new_integrations']==0
    assert checked['runtime_bias_goal_and_feedback_decisions']=='NOT_CREATED_BY_READ_ONLY_CHECK'
    with pytest.raises(ValueError):contract.episode_config(dict(spec,kind='FUSION_TASK_PHASE_BIAS_CONTINUATION_V3'))


@pytest.mark.parametrize('fault',['prior-position-success','prior-source','open','trace-drop','unconsumed','missing-sample','orientation','actual_open','no_contact','cube_still','original_safety','continuous_relative_hold','actual_cube_distance'])
def test_prior_goal_failure_is_not_permission_to_ignore_other_seven_predicates_or_native_proof(recipe,fault,monkeypatch):
    spec,previous,_,setup=recipe
    binding=previous['reference_qualification']['task_phase_initialization']
    rows=json.loads(Path(binding['hold_samples_path']).read_text())
    if fault=='prior-source':
        monkeypatch.setattr(contract,'FIRST_PREDECESSOR_SOURCE','unqualified-source')
    elif fault=='open':put(Path(spec['previous_stage']).parent.parent/'control/LEASE.json',{'status':'ACTIVE'})
    elif fault=='trace-drop':previous['trace_audit']['writer']['dropped']=1
    elif fault=='unconsumed':previous['reference_qualification']['windows'][-1]['gripper']['command_consumed']=False
    elif fault=='missing-sample':Path(binding['hold_samples_path']).write_text('[]')
    else:
        for index,row in enumerate(rows):
            if fault=='prior-position-success':row['tcp'][0][3]-=.013
            elif fault=='orientation':row['tcp'][0][0],row['tcp'][0][1],row['tcp'][1][0],row['tcp'][1][1]=0.,-1.,1.,0.
            elif fault=='actual_open':row['finger_q']=[.037,.037]
            elif fault=='no_contact':row['robot_object_contact']=True
            elif fault=='cube_still':row['object_pose'][0][3]+=.001
            elif fault=='original_safety':row['safety_valid']=False
            elif fault=='continuous_relative_hold':row['tcp'][0][3]+=index*.0005
            elif fault=='actual_cube_distance':row['object_pose']=copy.deepcopy(row['tcp'])
        readiness=v3.query_readiness(rows,previous['reference_qualification']['episode'],setup['inputs'])
        binding['hold_samples_sha256']=put(binding['hold_samples_path'],rows)
        binding['readiness_sha256']=put(binding['readiness_path'],readiness);binding['readiness']=readiness
    if fault not in ('prior-source','open','missing-sample'):
        spec['previous_stage_sha256']=put(spec['previous_stage'],previous)
        monkeypatch.setattr(contract,'FIRST_PREDECESSOR',spec['previous_stage_sha256'])
    with pytest.raises((ValueError,PermissionError)):contract.episode_limit(spec,previous)


class FeedbackAdapter(BiasAdapter):
    def install_reference(self,action):
        native.validate_adapter_install(self,action)
        if self.step in v4.FEEDBACK_STEPS:
            index=v4.FEEDBACK_STEPS.index(self.step)+1
            path=self.output/f'XYZ_FEEDBACK_{index:02d}_DECISION.private.json'
            saved=json.loads(path.read_text())
            assert saved['saved_before_program_commit_and_reference_install']
            assert self.trajectory_program.feedback_count==index
            assert digest(path)==self.trajectory_program.evidence()['feedback_decisions'][-1]['saved_decision_sha256']
        self.active_reference=dict(reference(round(self.step*.002*1e9)/1e9),action_EEF10=action.tolist())
        self.installs.append((self.step,action.copy()));return self.active_reference


def flow(recipe,tmp_path,monkeypatch):
    spec,_,episode,setup=recipe;output=tmp_path/'SYNTHETIC_FINITE_FEEDBACK';output.mkdir()
    adapter=FeedbackAdapter(spec,output,setup);setup_flow(monkeypatch,adapter)
    consumed=common_native.consumed_window
    monkeypatch.setattr(common_native,'consumed_window',lambda a,*args:dict(consumed(a,*args),reference_sha256=a.active_reference['wire_sha256']))
    _,source=contract.episode_config(spec);monkeypatch.setattr(contract,'episode_config',lambda s:(episode,source))
    fake_fk=lambda *a,**kw:{'status':'SYNTHETIC_ONLY_NO_NATIVE_CLAIM','expected_step':kw.get('expected_step',3703)}
    monkeypatch.setattr(original_native,'_verify_actual_fk',fake_fk);monkeypatch.setattr(native,'_verify_actual_fk',fake_fk)
    return adapter


def test_same_native_process_saves_every_causal_decision_before_forced_install_and_keeps_original_424_atom_lifecycle(recipe,tmp_path,monkeypatch):
    adapter=flow(recipe,tmp_path,monkeypatch);report=native.run(adapter)
    assert adapter.prepare_calls==1 and adapter.sealed==13103
    assert report['complete_task_atoms']==report['gripper_confirmed_atoms']==report['solver_target_confirmed_atoms']==424
    assert report['all_40_knots_consumed'] and len(adapter.grip_packets)==424
    assert [p['sequence'] for p in adapter.grip_packets]==list(range(424))
    assert set(v4.FEEDBACK_STEPS)<=set(step for step,_ in adapter.installs)
    feedback=report['finite_xyz_feedback'];assert feedback['actual_feedback_reference_updates']==4
    assert report['task_phase_initialization']['readiness_sample_native_steps']==list(range(12903,13104,25))
    for item in feedback['decisions']:
        value=json.loads(Path(item['decision_path']).read_text())
        assert digest(item['decision_path'])==item['decision_sha256']
        assert value['previous_complete_window']['native_end_step']==item['native_step']
        assert value['actual_fk_witness']['expected_step']==item['native_step']
        assert all(digest(row['path'])==row['sha256'] for row in value['artifact_files'].values())
    assert report['WM_updates']==report['Actor_updates']==report['VLA_queries']==0


@pytest.mark.parametrize('fault',['write-failure','readback-substitution','unconfirmed-window','packet-FK-mismatch','install-failure'])
def test_uncertain_feedback_evidence_or_install_failure_cannot_advance_or_silently_retry(recipe,tmp_path,monkeypatch,fault):
    adapter=flow(recipe,tmp_path,monkeypatch)
    if fault in ('write-failure','readback-substitution'):
        from scripts.pi0292 import durable
        original_write=durable.write
        def write(path,value):
            if 'XYZ_FEEDBACK_01_DECISION' in str(path):
                if fault=='write-failure':raise OSError('injected-save-failure')
                original_write(path,dict(value,corrupted=True));return
            original_write(path,value)
        monkeypatch.setattr(durable,'write',write)
    elif fault=='unconfirmed-window':
        consumed=common_native.consumed_window
        def consume(a,*args):
            value=consumed(a,*args)
            if value['native_end_step']==10303:value['solver_policies_with_exact_installed_target']=[]
            return value
        monkeypatch.setattr(common_native,'consumed_window',consume)
    elif fault=='packet-FK-mismatch':
        def fk(*a,**kw):
            if kw.get('expected_step')==10303:raise ValueError('injected-same-packet-FK-mismatch')
            return {'status':'SYNTHETIC_ONLY'}
        monkeypatch.setattr(native,'_verify_actual_fk',fk)
    else:
        install=adapter.install_reference
        def fail_install(action):
            if adapter.step==10303:raise RuntimeError('injected-install-failure')
            return install(action)
        adapter.install_reference=fail_install
    with pytest.raises((ValueError,OSError,RuntimeError)):native.run(adapter)
    assert adapter.step==10303 and len(adapter.grip_packets)==312
    assert not hasattr(adapter,'sealed')
    expected=1 if fault=='install-failure' else 0
    assert adapter.trajectory_program.feedback_count==expected
    assert not any(step>10303 for step,_ in adapter.installs)


def test_new_route_and_CLI_delegate_original_worker_without_backend_import(recipe,tmp_path,monkeypatch):
    spec,previous,_,_=recipe
    assert auto1_native.phase_limit(v4.PHASE,previous,spec)==13103
    assert target_sequential.native_task_bucket(v4.PHASE,True)=='64'
    adapter=target_sequential.TargetSequentialAdapter({'phase':v4.PHASE},tmp_path)
    token=object();monkeypatch.setattr(native,'run',lambda a:token)
    assert adapter.dynamic_diagnostic() is token
    path=tmp_path/'SPEC.json';put(path,spec);calls=[]
    monkeypatch.setitem(sys.modules,'scripts.wm02.pilot1_native_worker',SimpleNamespace(main=lambda a:calls.append(a) or 75))
    assert cli.main(['--spec',str(path),'--execute'])==75 and calls==[['--spec',str(path)]]
    code=('import sys;from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_v4;'
          'from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_native_v4;'
          'assert not {"mujoco","torch","rclpy","gr00t","transformers"}&sys.modules.keys()')
    subprocess.run([sys.executable,'-B','-c',code],check=True)
