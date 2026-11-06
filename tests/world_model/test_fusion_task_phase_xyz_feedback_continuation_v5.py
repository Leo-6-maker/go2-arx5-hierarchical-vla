"""Adjacent ordinary-feedback gate tests use explicit synthetic records only."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from go2arx5_vla.world_model import auto1_native, target_sequential
from go2arx5_vla.world_model import fusion_task_phase_initialization as original
from go2arx5_vla.world_model import fusion_task_phase_initialization_native as original_native
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_v4 as v4
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_native_v4 as n4
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_contract_v4 as c4
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_continuation_v5 as v5
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_continuation_native_v5 as native
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_continuation_contract_v5 as contract
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_evidence as evidence
from go2arx5_vla.world_model import fusion_trajectory_qualification_native as common_native
from go2arx5_vla.world_model.storage import digest
from scripts.wm02 import fusion_task_phase_xyz_feedback_continuation_check_v5 as cli
from test_fusion_task_phase_initialization import setup as pure_setup
from test_fusion_task_phase_xyz_feedback_v4 import recipe as old_recipe
from test_fusion_task_phase_bias_calibration_v2 import BiasAdapter
from test_fusion_live_proposal_qualification_contract import prior_files, put, close
from test_fusion_reference_qualification_contract import reference
from test_fusion_trajectory_qualification import setup_flow


@pytest.fixture
def setup(monkeypatch):
    return pure_setup.__wrapped__(monkeypatch)


@pytest.mark.parametrize('grade,ramp,end',[(.25,512,19503),(.5,768,25903),(1.,1024,32303)])
def test_exact_adjacent_grade_clocks_same_law_and_old_v4_typed_guard(setup,grade,ramp,end):
    episode=v5.make_episode(run='/fixture/fresh-v5',geometry_scale=grade)
    p=v5.make_program(setup['inputs'],episode,setup['initial'],setup['decision'],setup['base'])
    ancestor=original.make_program(setup['inputs'],v5.parent_recipe_view(episode),setup['initial'],setup['decision'],setup['base'])
    assert v5.endpoint(episode)==end and p.end_atom==ramp+112
    assert episode['feedback_decision_native_steps']==[3703+25*(ramp+8)+700*i for i in range(4)]
    np.testing.assert_array_equal(p.desired_path[:ramp+1],ancestor.desired_path)
    before=p.desired_path.copy()
    for index,step in enumerate(v5.feedback_steps(episode)):
        actual=p.graded_goal_TCP_target.copy();actual[0,3]-=.02
        update=p.prepare_feedback(step,actual)
        assert np.isclose(update.evidence['applied_delta_norm_m'],.004)
        assert update.evidence['schema'].endswith('_V5')
        p.commit_feedback(update,saved_decision_sha256=str(index)*64)
        preview=p.preview((step-3703)//25)
        assert v5.validate_install_preview(preview,initial_tcp=setup['initial'],program=p,step=step,
            base=setup['base'],frozen_base=setup['base'],end_step=end)['points_checked']==40
    np.testing.assert_array_equal(p.desired_path,before)
    assert np.isclose(p.evidence()['cumulative_feedback_reference_path_length_m'],.016)
    with pytest.raises(ValueError):
        v4.validate_install_preview(p.preview(),initial_tcp=setup['initial'],program=p,step=3703,
            base=setup['base'],frozen_base=setup['base'],end_step=end)
    with pytest.raises(ValueError):v4.episode_contract(episode)


@pytest.mark.parametrize('grade',[.03,.1,.2,True])
def test_no_repeat_point1_or_arbitrary_amplitude_or_law(grade):
    with pytest.raises(ValueError):v5.make_episode(run='/fixture/refuse',geometry_scale=grade)
    episode=v5.make_episode(run='/fixture/good',geometry_scale=.25)
    with pytest.raises(ValueError):v5.episode_contract(dict(episode,feedback_gain=.6))
    with pytest.raises(ValueError):v5.episode_contract(dict(episode,feedback_step_norm_cap_m=.005))


class QualifiedAdapter(BiasAdapter):
    """Perfect desired-tracking fixture, never native/FK evidence."""
    def _sample(self):
        value=super()._sample()
        if self.step>3703 and getattr(self,'trajectory_program',None) is not None:
            from scripts.pi0292.eval_contract import matrix10
            value['tcp']=matrix10(self.trajectory_program.desired_at((self.step-3703)//25))
        return value

    def install_reference(self,action):
        module=n4 if self.spec['phase']==v4.PHASE else native
        module.validate_adapter_install(self,action)
        value=reference(round(self.step*.002*1e9)/1e9)
        wire=json.dumps(dict(synthetic_not_native_command=True,native_step=self.step,H40=action.tolist()))
        pin=hashlib.sha256(wire.encode()).hexdigest()
        value.update(wire=wire,wire_sha256=pin)
        for row in value['receiver_events']:
            row['ref_id']=pin
            if row['kind']=='accept':row['raw']=wire
        self.active_reference=dict(value,action_EEF10=action.tolist())
        self.installs.append((self.step,action.copy()));return self.active_reference


def completed_native(root,spec,setup,recipe_module,monkeypatch):
    module=n4 if recipe_module is v4 else native
    candidate=copy.deepcopy(spec)
    candidate.update(registered_root=str(root),source_sha=contract.FIRST_PREDECESSOR_SOURCE if recipe_module is v4 else spec['source_sha'],
        phase=recipe_module.PHASE,kind=c4.KIND if recipe_module is v4 else contract.KIND)
    episode=(v4.make_episode(run=str(root)) if recipe_module is v4 else
             v5.make_episode(run=str(root),geometry_scale=spec['_fixture_grade']))
    outer,path,_,_=prior_files(root,episode,candidate,source=candidate['source_sha'])
    candidate.update(episode_manifest=str(root/'EPISODE_MANIFEST.private.json'),
        episode_manifest_sha256=digest(root/'EPISODE_MANIFEST.private.json'),max_ticks=recipe_module.endpoint(episode))
    candidate.pop('_fixture_grade',None)
    adapter=QualifiedAdapter(candidate,root/'native-01',setup)
    with pytest.MonkeyPatch.context() as local:
        setup_flow(local,adapter)
        consumed=common_native.consumed_window
        local.setattr(common_native,'consumed_window',lambda a,*args:dict(consumed(a,*args),reference_sha256=a.active_reference['wire_sha256']))
        def fk(a,packet,measured,**kw):
            return dict(status='SYNTHETIC_NOT_FK_OR_NATIVE_EVIDENCE',native_step=packet['step'],
                native_time_s=packet['time_s'],model_sha256=original.SCENE,
                reader_sha256=a.spec['source_files']['scripts/pi0294/task_state.py'],
                packet_qpos_qvel_ctrl_exact=True,original_reader_instance_retained=True)
        local.setattr(original_native,'_verify_actual_fk',fk);local.setattr(n4,'_verify_actual_fk',fk)
        cfg={'source_observation_bytes':setup['args']['source_observation_bytes'],
             'T_source_from_target':setup['args']['T_source_from_target'],'scene_sha256':original.SCENE}
        local.setattr(c4 if recipe_module is v4 else contract,'episode_config',lambda s:(episode,cfg))
        report=module.run(adapter)
    outer['reference_qualification']=report
    pin=put(path,outer);spec_pin=close(root,candidate)
    return candidate,outer,path,pin,spec_pin,adapter


@pytest.fixture
def recipe(tmp_path,monkeypatch):
    spec,_,_,setup=old_recipe.__wrapped__(tmp_path,monkeypatch)
    capsule,project=Path(spec['source_root']),Path(__file__).resolve().parents[2]
    for name in contract.SOURCE_MEMBERS:
        path=capsule/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((project/name).read_bytes());spec['source_files'][name]=digest(path)
    spec['python']=sys.executable
    spec['native_environment']={key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
    for key in spec['native_environment']:monkeypatch.setenv(key,'1')
    prior,previous,path,pin,spec_pin,adapter=completed_native(tmp_path/'SYNTHETIC_V4_QUALIFIED',spec,setup,v4,monkeypatch)
    assert all(value for key,value in previous['reference_qualification']['task_phase_initialization']['readiness']['predicates'].items() if key!='full_grade')
    monkeypatch.setattr(contract,'FIRST_PREDECESSOR',pin);monkeypatch.setattr(contract,'FIRST_PREDECESSOR_SPEC',spec_pin)
    spec.update(phase=v5.PHASE,kind=contract.KIND,previous_stage=str(path),previous_stage_sha256=pin,max_ticks=19503)
    episode=v5.make_episode(run=spec['registered_root'],geometry_scale=.25)
    spec['episode_manifest_sha256']=put(spec['episode_manifest'],episode)
    return spec,previous,episode,setup,prior,adapter


def test_first_point25_reconstructs_every_real_synthetic_decision_and_rolling_wire_with_no_backend_call(recipe,tmp_path,monkeypatch):
    spec,previous,_,_,prior,_=recipe
    checked=evidence.audit_completed_feedback(prior,previous,recipe=v4)
    assert checked['all_nine_non_full_grade_predicates_passed']
    assert checked['actual_feedback_decisions']==4 and checked['deadband_HOLD_count']==4
    assert checked['complete_task_atoms']==424 and checked['actual_reference_renewals']>50
    assert not checked['native_execution_authorized'] and checked['new_integrations']==checked['model_loads']==0
    assert contract.episode_limit(spec,previous)==19503
    path=tmp_path/'CLI_SPEC.json';put(path,spec)
    value=cli.check(path)
    assert value['max_ticks']==19503 and value['prior_saved_decisions_and_every_rolling_reference_consumption_rejoined']
    assert not value['native_execution_authorized']


@pytest.mark.parametrize('fault',['rolling-suffix','old-wire-after-decision','decision-delta','remaining-path','packet','dose-summary','ACK','grip','trace','hold-actual-TCP','wrong-native-python','wrong-BLAS','skip-grade','open-lease'])
def test_no_counter_or_pass_boolean_replaces_causal_measurement_and_actual_rolling_consumption(recipe,fault,monkeypatch):
    spec,previous,episode,_,prior,_=recipe
    report=previous['reference_qualification'];feedback=report['finite_xyz_feedback']
    if fault=='rolling-suffix':report['windows'][330]['full_final_installed_H40'][39][0]+=.001
    elif fault=='old-wire-after-decision':report['windows'][312]['reference_sha256']=report['windows'][311]['reference_sha256']
    elif fault=='decision-delta':
        row=feedback['decisions'][0];data=json.loads(Path(row['decision_path']).read_text())
        data['decision']['applied_delta_norm_m']=.004;row['decision_sha256']=put(row['decision_path'],data)
    elif fault in ('remaining-path','packet'):
        data=json.loads(Path(feedback['decisions'][0]['decision_path']).read_text())
        key='REMAINING_PATH.private.json' if fault=='remaining-path' else 'STATE.bin'
        Path(data['artifact_files'][key]['path']).write_bytes(b'changed')
    elif fault=='dose-summary':feedback['sum_applied_XYZ_delta_norms_m']+=.001
    elif fault=='ACK':report['original_install_ACKs']=[]
    elif fault=='grip':report['windows'][320]['gripper']['command_consumed']=False
    elif fault=='trace':previous['trace_audit']['writer']['dropped']=1
    elif fault=='hold-actual-TCP':report['windows'][-1]['actual_TCP'][0][3]+=.001
    elif fault=='wrong-native-python':
        prior['python']='/fixture/not-original-python';close(Path(spec['previous_stage']).parent.parent,prior)
        monkeypatch.setattr(contract,'FIRST_PREDECESSOR_SPEC',digest(Path(spec['previous_stage']).parent.parent/'EXECUTION_SPEC.json'))
    elif fault=='wrong-BLAS':monkeypatch.setenv('OPENBLAS_NUM_THREADS','2')
    elif fault=='skip-grade':
        ep=v5.make_episode(run=spec['registered_root'],geometry_scale=.5)
        spec['episode_manifest_sha256']=put(spec['episode_manifest'],ep);spec['max_ticks']=25903
    else:put(Path(spec['previous_stage']).parent.parent/'control/LEASE.json',{'status':'ACTIVE'})
    spec['previous_stage_sha256']=put(spec['previous_stage'],previous)
    monkeypatch.setattr(contract,'FIRST_PREDECESSOR',spec['previous_stage_sha256'])
    with pytest.raises((ValueError,PermissionError,KeyError)):contract.episode_limit(spec,previous)


def test_goal_measurement_failure_blocks_grade_even_with_complete_feedback_and_native_pass(recipe,monkeypatch):
    spec,previous,_,setup,prior,_=recipe
    # Construct self-consistent failing actual end samples/windows; all original
    # feedback and native consumption still rejoin. The amplitude gate must fail.
    binding=previous['reference_qualification']['task_phase_initialization']
    rows=json.loads(Path(binding['hold_samples_path']).read_text())
    for row in rows:row['tcp'][0][3]+=.02
    inputs=original.load_inputs(source_observation_bytes=setup['args']['source_observation_bytes'],
        decision_packet_bytes=Path(json.loads(Path(binding['goal_binding_path']).read_text())['saved_packet_path']).read_bytes(),
        measured_decision_state=json.loads(Path(json.loads(Path(binding['goal_binding_path']).read_text())['measured_state_path']).read_text()),
        T_source_from_target=setup['args']['T_source_from_target'],scene_sha256=original.SCENE)
    ep=previous['reference_qualification']['episode'];readiness=v4.query_readiness(rows,ep,inputs)
    binding['hold_samples_sha256']=put(binding['hold_samples_path'],rows)
    binding['readiness_sha256']=put(binding['readiness_path'],readiness);binding['readiness']=readiness
    for row in rows:
        selected=next(w for w in previous['reference_qualification']['windows'] if w['native_end_step']==row['native_step'])
        selected['actual_TCP']=copy.deepcopy(row['tcp'])
    previous['reference_qualification']['actual_endpoint_TCP']=copy.deepcopy(rows[-1]['tcp'])
    spec['previous_stage_sha256']=put(spec['previous_stage'],previous)
    monkeypatch.setattr(contract,'FIRST_PREDECESSOR',spec['previous_stage_sha256'])
    with pytest.raises(ValueError,match='ALL_NINE_ACTUAL_MEASUREMENT'):contract.episode_limit(spec,previous)


@pytest.mark.parametrize('prior_grade,next_grade',[(.25,.5),(.5,1.)])
def test_only_immediate_same_source_v5_complete_qualified_predecessor_advances(recipe,tmp_path,monkeypatch,prior_grade,next_grade):
    spec,_,_,setup,_,_=recipe;spec['_fixture_grade']=prior_grade
    prior,previous,path,pin,_,adapter=completed_native(tmp_path/f'SYNTHETIC_V5_PRIOR_{prior_grade}',spec,setup,v5,monkeypatch)
    spec.pop('_fixture_grade');spec.update(previous_stage=str(path),previous_stage_sha256=pin)
    ep=v5.make_episode(run=spec['registered_root'],geometry_scale=next_grade)
    spec['episode_manifest_sha256']=put(spec['episode_manifest'],ep);spec['max_ticks']=v5.endpoint(ep)
    assert contract.episode_limit(spec,previous)==v5.endpoint(ep)
    assert adapter.sealed==v5.endpoint(previous['reference_qualification']['episode'])
    spec['source_sha']='DIFFERENT_UNQUALIFIED_SOURCE'
    with pytest.raises(ValueError,match='IMMEDIATE_SAME_V5_SOURCE'):contract.episode_limit(spec,previous)


def test_original_phase_and_CLI_routes_do_not_construct_new_backend_or_dispatch(recipe,tmp_path,monkeypatch):
    spec,previous,_,_,_,_=recipe
    assert auto1_native.phase_limit(v5.PHASE,previous,spec)==19503
    assert target_sequential.native_task_bucket(v5.PHASE,True)=='64'
    adapter=target_sequential.TargetSequentialAdapter({'phase':v5.PHASE},tmp_path)
    token=object();monkeypatch.setattr(native,'run',lambda a:token)
    assert adapter.dynamic_diagnostic() is token
    path=tmp_path/'SPEC.json';put(path,spec);calls=[]
    monkeypatch.setitem(sys.modules,'scripts.wm02.pilot1_native_worker',SimpleNamespace(main=lambda a:calls.append(a) or 75))
    assert cli.main(['--spec',str(path),'--execute'])==75 and calls==[['--spec',str(path)]]
    code=('import sys;from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_continuation_v5;'
        'from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_continuation_native_v5;'
        'assert not {"torch","mujoco","rclpy","gr00t","transformers"}&sys.modules.keys()')
    subprocess.run([sys.executable,'-B','-c',code],check=True)


def test_private_shared_native_engine_has_only_known_explicit_versioned_callers():
    with pytest.raises(ValueError,match='EXACT_KNOWN_FINITE_FEEDBACK'):
        n4.run_feedback_program(None,{},None,actual_phase='UNDECLARED_PHASE')
    with pytest.raises(ValueError,match='EXACT_TYPED_VERSIONED_FEEDBACK'):
        n4.run_feedback_program(None,{},None,recipe_module=v5,actual_phase=v4.PHASE)


def test_different_venv_path_is_rejected_even_when_base_python_binary_is_identical(tmp_path):
    alias=tmp_path/'other-venv-python';alias.symlink_to(sys.executable)
    assert alias.resolve()==Path(sys.executable).resolve()
    with pytest.raises(ValueError,match='RECORDED_ORIGINAL_NATIVE_PYTHON'):
        evidence.audit_completed_feedback(dict(phase=v4.PHASE,python=str(alias)),{},recipe=v4)
