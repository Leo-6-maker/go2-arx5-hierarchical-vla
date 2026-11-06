"""Finite pose-feedback contracts with explicitly synthetic execution records."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
from types import ModuleType,SimpleNamespace

import numpy as np
import pytest

from scripts.pi0292.eval_contract import matrix10
from go2arx5_vla.task_data.codec import pose10
from go2arx5_vla.world_model import auto1_native,target_sequential
from go2arx5_vla.world_model import fusion_task_phase_initialization as original
from go2arx5_vla.world_model import fusion_task_phase_initialization_native as original_native
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_v4 as v4
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_native_v4 as n4
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_continuation_v5 as v5
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_continuation_contract_v5 as c5
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_evidence as evidence
from go2arx5_vla.world_model import fusion_task_phase_pose_feedback_v6 as phase
from go2arx5_vla.world_model import fusion_task_phase_pose_feedback_native_v6 as native
from go2arx5_vla.world_model import fusion_task_phase_pose_feedback_contract_v6 as contract
from go2arx5_vla.world_model import fusion_trajectory_qualification_native as common_native
from go2arx5_vla.world_model.storage import digest
from scripts.wm02 import fusion_task_phase_pose_feedback_check_v6 as cli
from test_fusion_task_phase_initialization import setup as pure_setup,samples
from test_fusion_task_phase_xyz_feedback_continuation_v5 import recipe as prior_recipe,completed_native,QualifiedAdapter
from test_fusion_live_proposal_qualification_contract import prior_files,put,close
from test_fusion_reference_qualification_contract import reference
from test_fusion_trajectory_qualification import setup_flow


@pytest.fixture
def setup(monkeypatch):
    return pure_setup.__wrapped__(monkeypatch)


def program(setup,grade=.25):
    episode = phase.make_episode(run='/fixture/new-v6',geometry_scale=grade)
    return phase.make_program(setup['inputs'],episode,setup['initial'],setup['decision'],setup['base']),episode


@pytest.mark.parametrize('grade,ramp,end',[(.25,512,22303),(.5,768,28703),(1.,1024,35103)])
def test_static_own_identity_eight_finite_slots_and_unchanged_desired_algorithm(setup,grade,ramp,end):
    p,episode = program(setup,grade)
    ancestor = original.make_program(setup['inputs'],phase.parent_recipe_view(episode),
        setup['initial'],setup['decision'],setup['base'])
    assert phase.endpoint(episode) == end and p.end_atom == ramp+224
    assert episode['feedback_decision_native_steps'] == [3703+25*(ramp+8)+700*i for i in range(8)]
    np.testing.assert_array_equal(p.desired_path[:ramp+1],ancestor.desired_path)
    np.testing.assert_array_equal(p.desired_path[ramp:],np.repeat(ancestor.desired_path[-1][None],225,axis=0))
    assert episode['feedback_angular_error'] == 'LOG_R_DESIRED_GOAL_TIMES_R_ACTUAL_TRANSPOSE_TARGET_WORLD'
    assert episode['finite_native_endpoint_maximum'] == 35103
    assert p.evidence()['baseline_condition'] == v5.CONDITION
    for old in (original,v4,v5):
        with pytest.raises(ValueError):old.episode_contract(episode)
    rows = samples(setup,grade)
    for row,step in zip(rows,range(end-200,end+1,25)):
        row.update(native_step=step,time_s=step*.002)
    ready = phase.query_readiness(rows,episode,setup['inputs'])
    assert ready['actual_sample_steps'] == list(range(end-200,end+1,25))
    if grade == 1.:
        with pytest.raises(ValueError,match='FINITE_EXPLICIT'):
            original._query_readiness_at_endpoint(rows,phase.parent_recipe_view(episode),setup['inputs'],end=end)


@pytest.mark.parametrize('grade',[.03,.1,.2,True])
def test_no_extra_amplitudes_or_silent_dose_changes(grade):
    with pytest.raises(ValueError):phase.make_episode(run='/fixture/refuse',geometry_scale=grade)
    episode=phase.make_episode(run='/fixture/good',geometry_scale=.25)
    for field,value in [('feedback_angular_step_cap_rad',.02),('feedback_cumulative_path_length_cap_m',.033),
                        ('finite_feedback_decisions',9),('planned_end_atom',737)]:
        with pytest.raises(ValueError):phase.episode_contract(dict(episode,**{field:value}))


@pytest.mark.parametrize('axis',[np.array([1.,0.,0.]),np.array([.3,.4,.5])/np.sqrt(.5)])
@pytest.mark.parametrize('angle',[0.,1e-10,.02,np.pi-1e-8,np.pi])
def test_principal_so3_log_exp_has_zero_and_pi_limits(axis,angle):
    r=phase._exp_so3(axis*angle)
    np.testing.assert_allclose(phase._exp_so3(phase._log_so3(r)),r,rtol=0,atol=2e-14)


def test_world_left_error_noncommuting_frames_reference_endpoint_not_achieved_anchor(setup):
    p,episode=program(setup);before=p.reference_path.copy();desired=p.desired_path.copy()
    world_error=np.array([.03,-.04,.02]);actual=p.graded_goal_TCP_target.copy()
    actual[:3,3]-=[.03,.04,0.]
    actual[:3,:3]=phase._exp_so3(-world_error)@actual[:3,:3]
    step=phase.feedback_steps(episode)[0];atom=(step-3703)//25
    update=p.prepare_feedback(step,actual)
    np.testing.assert_allclose(update.evidence['angular_error_target_world_rotvec_rad'],world_error,atol=2e-15)
    applied=world_error/np.linalg.norm(world_error)*.01
    np.testing.assert_allclose(update.evidence['applied_rotation_target_world_rad'],applied,atol=2e-15)
    initial_ref=matrix10(before[atom]);new_end=matrix10(update.reference_path[-1])
    np.testing.assert_allclose(new_end[:3,:3],phase._exp_so3(applied)@initial_ref[:3,:3],atol=2e-14)
    assert not np.allclose(new_end[:3,:3],initial_ref[:3,:3]@phase._exp_so3(applied),atol=1e-7)
    np.testing.assert_allclose(update.reference_path[atom+10,:3]-before[atom,:3],[.0012,.0016,0.],atol=1e-15)
    np.testing.assert_allclose(matrix10(update.reference_path[atom+10])[:3,:3],
        phase._exp_so3(.5*applied)@initial_ref[:3,:3],atol=2e-14)
    np.testing.assert_array_equal(update.reference_path[:atom+1],before[:atom+1])
    np.testing.assert_array_equal(update.reference_path[:,9],before[:,9])
    np.testing.assert_array_equal(p.reference_path,before)
    assert p.feedback_count==0
    p.commit_feedback(update,saved_decision_sha256='a'*64)
    np.testing.assert_array_equal(p.desired_path,desired)
    assert p.evidence()['current_reference_goal_TCP_target'] == new_end.tolist()
    assert p.evidence()['frozen_predecision_reference_binding']['reference_path_sha256'] == phase.fixed._sha(before)


def test_eight_doses_use_sum_norms_independent_deadbands_and_reject_extra_or_unsaved_revision(setup):
    p,episode=program(setup);desired=p.desired_path.copy()
    for index,step in enumerate(phase.feedback_steps(episode)):
        actual=p.graded_goal_TCP_target.copy();actual[:3,3]-=[(-1)**index*.05,0.,0.]
        actual[:3,:3]=phase._exp_so3([0.,(-1)**index*.05,0.])@actual[:3,:3]
        update=p.prepare_feedback(step,actual)
        assert np.isclose(update.evidence['applied_delta_norm_m'],.004)
        assert np.isclose(update.evidence['applied_rotation_norm_rad'],.01)
        with pytest.raises(ValueError,match='NO_SCHEDULED_REFERENCE'):
            phase.validate_install_preview(p.preview((step-3703)//25),initial_tcp=setup['initial'],program=p,
                step=step,base=setup['base'],frozen_base=setup['base'],end_step=phase.endpoint(episode))
        p.commit_feedback(update,saved_decision_sha256=str(index)*64)
        phase.validate_install_preview(p.preview((step-3703)//25),initial_tcp=setup['initial'],program=p,
            step=step,base=setup['base'],frozen_base=setup['base'],end_step=phase.endpoint(episode))
    assert np.isclose(p.evidence()['cumulative_feedback_reference_path_length_m'],.032)
    assert np.isclose(p.evidence()['cumulative_feedback_reference_rotation_path_rad'],.08)
    np.testing.assert_array_equal(p.desired_path,desired)
    with pytest.raises(ValueError):p.prepare_feedback(phase.endpoint(episode),actual)
    q,episode=program(setup);before=q.reference_path.copy();actual=q.graded_goal_TCP_target.copy()
    actual[:3,3]-=[.003,0.,0.]
    update=q.prepare_feedback(phase.feedback_steps(episode)[0],actual)
    assert update.evidence['deadband_hold'] and update.evidence['angular_deadband_hold']
    np.testing.assert_array_equal(update.reference_path,before)
    actual[:3,:3]=phase._exp_so3([.02,0.,0.])@actual[:3,:3]
    rotation_only=q.prepare_feedback(phase.feedback_steps(episode)[0],actual)
    assert rotation_only.evidence['applied_delta_norm_m']==0 and rotation_only.evidence['applied_rotation_norm_rad']>0


class V6QualifiedAdapter(QualifiedAdapter):
    def install_reference(self,action):
        native.validate_adapter_install(self,action)
        value=reference(round(self.step*.002*1e9)/1e9)
        wire=json.dumps(dict(synthetic_not_native_command=True,native_step=self.step,H40=action.tolist()))
        pin=hashlib.sha256(wire.encode()).hexdigest();value.update(wire=wire,wire_sha256=pin)
        for row in value['receiver_events']:
            row['ref_id']=pin
            if row['kind']=='accept':row['raw']=wire
        self.active_reference=dict(value,action_EEF10=action.tolist())
        self.installs.append((self.step,action.copy()));return self.active_reference


def completed_v6(root,spec,setup,grade,monkeypatch,adapter_type=V6QualifiedAdapter):
    candidate=copy.deepcopy(spec);candidate.update(registered_root=str(root),phase=phase.PHASE,kind=contract.KIND)
    episode=phase.make_episode(run=str(root),geometry_scale=grade)
    outer,path,_,_=prior_files(root,episode,candidate,source=candidate['source_sha'])
    candidate.update(episode_manifest=str(root/'EPISODE_MANIFEST.private.json'),
        episode_manifest_sha256=digest(root/'EPISODE_MANIFEST.private.json'),max_ticks=phase.endpoint(episode))
    adapter=adapter_type(candidate,root/'native-01',setup)
    with pytest.MonkeyPatch.context() as local:
        setup_flow(local,adapter);consumed=common_native.consumed_window
        local.setattr(common_native,'consumed_window',lambda a,*args:dict(consumed(a,*args),reference_sha256=a.active_reference['wire_sha256']))
        def fk(a,packet,measured,**kwargs):
            return dict(status='SYNTHETIC_NOT_NATIVE_OR_FK',native_step=packet['step'],native_time_s=packet['time_s'],
                model_sha256=original.SCENE,reader_sha256=a.spec['source_files']['scripts/pi0294/task_state.py'],
                packet_qpos_qvel_ctrl_exact=True,original_reader_instance_retained=True)
        local.setattr(original_native,'_verify_actual_fk',fk);local.setattr(n4,'_verify_actual_fk',fk)
        cfg=dict(source_observation_bytes=setup['args']['source_observation_bytes'],
            T_source_from_target=setup['args']['T_source_from_target'],scene_sha256=original.SCENE)
        local.setattr(contract,'episode_config',lambda s:(episode,cfg))
        report=native.run(adapter)
    outer['reference_qualification']=report;pin=put(path,outer);specpin=close(root,candidate)
    return candidate,outer,path,pin,specpin,adapter


@pytest.fixture
def recipe(tmp_path,monkeypatch):
    spec,_,_,setup,_,_=prior_recipe.__wrapped__(tmp_path,monkeypatch)
    capsule,project=Path(spec['source_root']),Path(__file__).resolve().parents[2]
    for name in contract.SOURCE_MEMBERS:
        path=capsule/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((project/name).read_bytes());spec['source_files'][name]=digest(path)
    spec['_fixture_grade']=.25
    prior,previous,path,pin,specpin,_=completed_native(tmp_path/'SYNTHETIC_V5_FAILED_POINT25',spec,setup,v5,monkeypatch)
    spec.pop('_fixture_grade');prior['source_sha']=contract.FIRST_PREDECESSOR_SOURCE
    # Keep the saved goal/decision identities exact for this synthetic source.
    report=previous['reference_qualification'];previous['source_sha']=prior['source_sha']
    goalpath=Path(report['task_phase_initialization']['goal_binding_path'])
    goal=json.loads(goalpath.read_bytes());goal['actual_spec_source_sha']=prior['source_sha']
    report['task_phase_initialization']['goal_binding_sha256']=put(goalpath,goal)
    for receipt in report['finite_xyz_feedback']['decisions']:
        data=json.loads(Path(receipt['decision_path']).read_bytes());data['actual_spec_source_sha']=prior['source_sha']
        data['original_goal_binding_sha256']=digest(goalpath)
        receipt['decision_sha256']=put(receipt['decision_path'],data)
    # Recompute adaptation with the new saved-decision SHA identities.
    proof=report['adaptation']
    for row,receipt in zip(proof['feedback_decisions'],report['finite_xyz_feedback']['decisions']):
        row['saved_decision_sha256']=receipt['decision_sha256']
    binding=report['task_phase_initialization'];rows=json.loads(Path(binding['hold_samples_path']).read_bytes())
    for row in rows:
        tcp=np.asarray(row['tcp']);tcp[0,3]+=.02;tcp[:3,:3]=phase._exp_so3([0.,0.,.06])@tcp[:3,:3]
        row['tcp']=tcp.tolist()
        next(w for w in report['windows'] if w['native_end_step']==row['native_step'])['actual_TCP']=copy.deepcopy(row['tcp'])
    report['actual_endpoint_TCP']=copy.deepcopy(rows[-1]['tcp'])
    inputs=original.load_inputs(source_observation_bytes=setup['args']['source_observation_bytes'],
        decision_packet_bytes=Path(goal['saved_packet_path']).read_bytes(),
        measured_decision_state=json.loads(Path(goal['measured_state_path']).read_bytes()),
        T_source_from_target=setup['args']['T_source_from_target'],scene_sha256=original.SCENE)
    readiness=v5.query_readiness(rows,report['episode'],inputs)
    binding['hold_samples_sha256']=put(binding['hold_samples_path'],rows)
    binding['readiness_sha256']=put(binding['readiness_path'],readiness);binding['readiness']=readiness
    pin=put(path,previous);specpin=close(path.parent.parent,prior)
    monkeypatch.setattr(contract,'FIRST_PREDECESSOR',pin);monkeypatch.setattr(contract,'FIRST_PREDECESSOR_SPEC',specpin)
    spec.update(phase=phase.PHASE,kind=contract.KIND,previous_stage=str(path),previous_stage_sha256=pin,max_ticks=22303)
    episode=phase.make_episode(run=spec['registered_root'],geometry_scale=.25)
    spec['episode_manifest_sha256']=put(spec['episode_manifest'],episode)
    return spec,previous,episode,setup,prior


def test_exact_same_failed_point25_repair_gate_and_cli_do_not_grant_native_execution(recipe,tmp_path):
    spec,previous,_,_,prior=recipe
    checked=evidence.audit_completed_feedback(prior,previous,recipe=v5)
    assert not checked['all_nine_non_full_grade_predicates_passed']
    assert contract.episode_limit(spec,previous)==22303
    path=tmp_path/'CHECK.json';put(path,spec);value=cli.check(path)
    assert value['first_point25_repairs_exact_three_failed_predicates'] and not value['native_execution_authorized']


@pytest.mark.parametrize('prior_grade,next_grade',[(.25,.5),(.5,1.)])
def test_only_adjacent_same_v6_source_complete_eight_decisions_and_all_nine_pass_advance(recipe,tmp_path,monkeypatch,prior_grade,next_grade):
    spec,_,_,setup,_=recipe
    prior,previous,path,pin,_,adapter=completed_v6(tmp_path/f'SYNTHETIC_V6_{prior_grade}',spec,setup,prior_grade,monkeypatch)
    checked=evidence.audit_completed_feedback(prior,previous,recipe=phase)
    assert checked['actual_feedback_decisions']==8 and checked['angular_deadband_HOLD_count']==8
    assert checked['sum_applied_SO3_delta_norms_rad']==0
    spec.update(previous_stage=str(path),previous_stage_sha256=pin)
    ep=phase.make_episode(run=spec['registered_root'],geometry_scale=next_grade)
    spec['episode_manifest_sha256']=put(spec['episode_manifest'],ep);spec['max_ticks']=phase.endpoint(ep)
    assert contract.episode_limit(spec,previous)==phase.endpoint(ep)
    assert adapter.sealed==phase.endpoint(previous['reference_qualification']['episode'])
    spec['source_sha']='OTHER_SOURCE'
    with pytest.raises(ValueError,match='IMMEDIATE_SAME_V6_SOURCE'):contract.episode_limit(spec,previous)


@pytest.mark.parametrize('fault',['skip-grade','false-safety','wrong-three-failures','changed-first-pin','open-lease'])
def test_repair_requires_exact_failed_baseline_and_all_other_six_measurements(recipe,monkeypatch,fault):
    spec,previous,_,_,prior=recipe
    if fault=='skip-grade':
        ep=phase.make_episode(run=spec['registered_root'],geometry_scale=.5)
        spec['episode_manifest_sha256']=put(spec['episode_manifest'],ep);spec['max_ticks']=28703
    elif fault=='false-safety':previous['trace_audit']['writer']['dropped']=1
    elif fault=='wrong-three-failures':previous['reference_qualification']['task_phase_initialization']['readiness']['predicates']['orientation']=True
    elif fault=='changed-first-pin':monkeypatch.setattr(contract,'FIRST_PREDECESSOR_SPEC','0'*64)
    else:put(Path(spec['previous_stage']).parent.parent/'control/LEASE.json',{'status':'ACTIVE'})
    if fault not in ('changed-first-pin','skip-grade','open-lease'):
        spec['previous_stage_sha256']=put(spec['previous_stage'],previous)
        monkeypatch.setattr(contract,'FIRST_PREDECESSOR',spec['previous_stage_sha256'])
    with pytest.raises((ValueError,PermissionError,KeyError)):contract.episode_limit(spec,previous)


def test_new_routes_and_deferred_worker_without_backend_import_or_new_scheduler(recipe,tmp_path,monkeypatch):
    spec,previous,_,_,_=recipe
    assert auto1_native.phase_limit(phase.PHASE,previous,spec)==22303
    assert target_sequential.native_task_bucket(phase.PHASE,True)=='96'
    adapter=target_sequential.TargetSequentialAdapter({'phase':phase.PHASE},tmp_path)
    token=object();monkeypatch.setattr(native,'run',lambda a:token)
    assert adapter.dynamic_diagnostic() is token
    path=tmp_path/'SPEC.json';put(path,spec);calls=[]
    monkeypatch.setitem(sys.modules,'scripts.wm02.pilot1_native_worker',SimpleNamespace(main=lambda a:calls.append(a) or 75))
    assert cli.main(['--spec',str(path),'--execute'])==75 and calls==[['--spec',str(path)]]
    code=('import sys;from go2arx5_vla.world_model import fusion_task_phase_pose_feedback_v6;'
          'from go2arx5_vla.world_model import fusion_task_phase_pose_feedback_native_v6;'
          'assert not {"torch","mujoco","rclpy","gr00t","transformers"}&sys.modules.keys()')
    subprocess.run([sys.executable,'-B','-c',code],check=True)


class V6ResidualAdapter(V6QualifiedAdapter):
    """Finite deterministic residuals, explicitly not a physics model."""
    def _sample(self):
        value=super()._sample()
        if self.step>3703 and getattr(self,'trajectory_program',None) is not None:
            value['tcp'][0,3]+=.008
            value['tcp'][:3,:3]=phase._exp_so3([0.,.02,0.])@value['tcp'][:3,:3]
        return value


@pytest.mark.parametrize('fault',[None,'angular-dose','angular-law','old-wire-after-decision','wrong-H40'])
def test_native_eight_nonzero_pose_decisions_saved_before_install_and_every_rolling_revision_rejoins(recipe,tmp_path,monkeypatch,fault):
    spec,_,_,setup,_=recipe
    prior,previous,_,_,_,adapter=completed_v6(tmp_path/'SYNTHETIC_NONZERO_POSE',spec,setup,.25,monkeypatch,V6ResidualAdapter)
    report=previous['reference_qualification'];feedback=report['finite_xyz_feedback']
    assert feedback['nonzero_XYZ_corrections']==feedback['nonzero_SO3_corrections']==8
    if fault=='angular-dose':feedback['sum_applied_SO3_delta_norms_rad']+=.001
    elif fault=='angular-law':
        item=feedback['decisions'][0];data=json.loads(Path(item['decision_path']).read_bytes())
        data['decision']['angular_error_target_world_rotvec_rad'][0]+=.001
        item['decision_sha256']=put(item['decision_path'],data)
    elif fault=='old-wire-after-decision':
        first=(phase.feedback_steps(report['episode'])[0]-2503)//25
        report['windows'][first]['reference_sha256']=report['windows'][first-1]['reference_sha256']
    elif fault=='wrong-H40':
        data=json.loads(Path(feedback['decisions'][0]['decision_path']).read_bytes())
        item=data['artifact_files']['H40.private.json'];p=Path(item['path']);rows=json.loads(p.read_bytes())
        rows[39][4]+=.001;p.write_text(json.dumps(rows))
    if fault:
        with pytest.raises((ValueError,KeyError)):evidence.audit_completed_feedback(prior,previous,recipe=phase)
    else:
        checked=evidence.audit_completed_feedback(prior,previous,recipe=phase)
        assert checked['all_nine_non_full_grade_predicates_passed']
        assert checked['actual_feedback_decisions']==8 and checked['actual_reference_renewals']>90
        assert np.isclose(checked['sum_applied_SO3_delta_norms_rad'],.08)
        assert adapter.sealed==22303
        assert feedback['rotation_error_frame']=='ACTUAL_TARGET_WORLD_LEFT_LOG_R_GOAL_R_ACTUAL_TRANSPOSE'


@pytest.mark.skipif(not shutil.which('g++'),reason='C++ compiler required for unchanged native ABI')
def test_unchanged_cpp_quota_header_rejects_full1_bucket64_and_accepts_exact_v6_clocks_bucket96(tmp_path):
    root=Path(__file__).resolve().parents[2];header=root/'scripts/pi0294/task_quota_r1.hpp'
    before=header.read_bytes();pin=hashlib.sha256(before).hexdigest()
    frozen=subprocess.check_output(['git','show','e5272f8fb90de2d910bb52c0575771b7a2125ee7:scripts/pi0294/task_quota_r1.hpp'],cwd=root)
    assert before==frozen
    source=tmp_path/'quota.cpp';source.write_text('#include "task_quota_r1.hpp"\nint main(){uint64_t n=0;return gr00t_quota_r1::read_bound(n)?0:1;}\n')
    executable=tmp_path/'quota'
    subprocess.run(['g++','-std=c++17','-I'+str(header.parent),str(source),'-o',str(executable)],check=True,capture_output=True)
    env=dict(os.environ,GR00T_TASK_RUN_UUID='SYNTHETIC_V6',GR00T_TASK_CARD_SHA256='fixture-card',GR00T_TASK_RESERVATION_ID='fixture-reservation')
    binding=tmp_path/'binding';env['GR00T_TASK_QUOTA_BINDING']=str(binding)
    for end,bucket,expected in [(35103,64,1),(22303,96,0),(28703,96,0),(35103,96,0)]:
        binding.write_text(f'GRTQUOTA2\nSYNTHETIC_V6\nfixture-card\nfixture-reservation\n{end}\n5\n{bucket}\n')
        assert subprocess.run([str(executable)],env=env,check=False).returncode==expected
    assert target_sequential.native_task_bucket(phase.PHASE,True)=='96'
    assert target_sequential.native_task_bucket(v4.PHASE,True)==target_sequential.native_task_bucket(v5.PHASE,True)=='64'
    assert hashlib.sha256(header.read_bytes()).hexdigest()==pin


@pytest.mark.parametrize('old_recipe',[v4,v5])
def test_frozen_e527_shared_native_engine_and_current_default_produce_identical_old_artifact_bytes(recipe,tmp_path,monkeypatch,old_recipe):
    spec,_,_,setup,_=recipe;spec=copy.deepcopy(spec);spec['_fixture_grade']=.25
    project=Path(__file__).resolve().parents[2]
    member='src/go2arx5_vla/world_model/fusion_task_phase_xyz_feedback_native_v4.py'
    raw=subprocess.check_output(['git','show','e5272f8fb90de2d910bb52c0575771b7a2125ee7:'+member],cwd=project)
    frozen=ModuleType('go2arx5_vla.world_model._v6_frozen_e527_native')
    frozen.__package__='go2arx5_vla.world_model'
    monkeypatch.setitem(sys.modules,frozen.__name__,frozen)
    exec(compile(raw,'<readonly-e527-native-engine>','exec'),frozen.__dict__)
    # The frozen module delegates only the explicit synthetic witness fixture.
    monkeypatch.setattr(frozen,'_verify_actual_fk',lambda *a,**kw:n4._verify_actual_fk(*a,**kw))
    root=tmp_path/'SYNTHETIC_OLD_RECIPE_SAME_PATH'
    current=completed_native(root,spec,setup,old_recipe,monkeypatch)[1]
    original_files={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    shutil.rmtree(root)
    with pytest.MonkeyPatch.context() as local:
        local.setattr(n4,'run_feedback_program',frozen.run_feedback_program)
        archived=completed_native(root,spec,setup,old_recipe,monkeypatch)[1]
    replay_files={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert current==archived
    assert original_files==replay_files
