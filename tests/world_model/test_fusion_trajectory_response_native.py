"""Offline response-entry tests. Every clock/exit fixture is non-native evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from go2arx5_vla.world_model import fusion_trajectory_response_contract as p
from go2arx5_vla.world_model import fusion_trajectory_response_native_contract as c
from go2arx5_vla.world_model import fusion_trajectory_response_native as n
from go2arx5_vla.world_model import fusion_trajectory_qualification_contract as g
from go2arx5_vla.world_model import fusion_trajectory_qualification_native as gn
from go2arx5_vla.world_model.fusion_trajectory_adapter import TrajectoryAdaptation
from go2arx5_vla.world_model.auto1_native import phase_limit
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter, native_display_command
from scripts.wm02 import fusion_trajectory_response_check as cli
from test_infer_fusion_observation import put
from test_fusion_reference_qualification_contract import reference
from test_fusion_trajectory_qualification import (
    recipe as geometry_recipe, FakeAdapter as GeometryFake, setup_flow as geometry_flow, BASE)


def panel(source_sha):
    return p.make_panel(physical_parents={'TRAIN_PARENT':'TRAIN','TR2':'TRAIN','TR3':'TRAIN','TR4':'TRAIN',
                          'DV1':'DEV','DV2':'DEV'}, source_sha=source_sha,
        query_pins={k:v for k,v in g.PINS.items() if k != 'first_predecessor'})


@pytest.fixture
def response_recipe(geometry_recipe, tmp_path, monkeypatch):
    spec, previous, _, old_episode, inputs = geometry_recipe
    # Generate the previous report through the existing complete fake collector,
    # rather than replacing actual native qualification in production.
    anchor=np.linalg.inv(inputs['frame_map']) @ inputs['query_tcp']
    full=TrajectoryAdaptation(inputs['raw'],inputs['query_tcp'],inputs['frame_map'],anchor,anchor,BASE,1.,time_stretch_factor=2)
    old_episode.update(schema=g.SLOW_SCHEMA,phase=g.SLOW_PHASE,reference_rule=g.SLOW_RECIPE,
                       time_stretch_factor=2,geometry_scale=1.,planned_end_atom=full.end_atom)
    put(Path(spec['episode_manifest']),old_episode)
    spec.update(phase=g.SLOW_PHASE,episode_manifest_sha256=digest(spec['episode_manifest']), max_ticks=g.endpoint(old_episode))
    output=tmp_path/'fake-full1';output.mkdir()
    class SlowFake(GeometryFake):
        def prepare_dynamic(self):
            from go2arx5_vla.world_model.init_trace_reader import RECORD
            (self.output/'init.trace').write_bytes(struct.pack('<8sII',b'WM02IT01',RECORD.size,262144))
    old=SlowFake(spec,output)
    with monkeypatch.context() as temporary:
        geometry_flow(temporary,old)
        temporary.setattr(gn,'episode_config',lambda spec:(old_episode,inputs))
        previous_report=gn.run(old)
    previous.update(phase=g.SLOW_PHASE, actual_target_integrations=spec['max_ticks'],
                    reference_qualification=previous_report)
    previous['trace_audit']['native_step_records']=spec['max_ticks']
    prior=Path(spec['previous_stage']);put(prior,previous)
    prior_root=prior.parent.parent
    put(prior_root/'EXECUTION_SPEC.json',dict(phase=g.SLOW_PHASE,max_ticks=spec['max_ticks']))
    put(prior_root/'control/LEASE.json',dict(status='CLOSED',worker_exit_code=0,
        spec_sha256=digest(prior_root/'EXECUTION_SPEC.json')))
    spec['previous_stage_sha256']=digest(prior)
    project=Path(__file__).resolve().parents[2]
    for name in c.SOURCE_MEMBERS:
        member=Path(spec['source_root'])/name;member.parent.mkdir(parents=True,exist_ok=True)
        member.write_bytes((project/name).read_bytes());spec['source_files'][name]=digest(member)
    spec.update(source_sha='a'*40,phase=p.PHASE,trajectory_response_collection='PILOT12',
                trajectory_response_pilot_parent='TRAIN_PARENT')
    build_path=Path(spec['trajectory_trace_capacity_build']);build=json.loads(build_path.read_text())
    build.update(status='TRACE_ONLY_262144_CAPACITY_COMPILED_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED',trace_capacity=262144)
    put(build_path,build);spec['trajectory_trace_capacity_build_sha256']=digest(build_path)
    plan=panel(spec['source_sha']);plan_path=tmp_path/'response-panel.json';put(plan_path,plan)
    ep=dict(schema=p.EPISODE_SCHEMA,phase=p.PHASE,purpose='TRAJECTORY_RESPONSE_IDENTIFICATION',
        reference_rule=p.RECIPE,panel_sha256=p.canonical_sha256(plan),geometry_scale=1.,
        query_pins=plan['query_pins'],run=spec['registered_root'],scene_id='TRAIN_SCENE',
        parent='TRAIN_PARENT',split='TRAIN',context_id='C0',theta_level=0,restart_index=0,
        planned_end_atom=full.end_atom)
    put(Path(spec['episode_manifest']),ep)
    spec.update(episode_manifest_sha256=digest(spec['episode_manifest']),
        trajectory_response_panel=str(plan_path),trajectory_response_panel_sha256=digest(plan_path),
        max_ticks=c.endpoint(ep))
    path=tmp_path/'response-spec.json';put(path,spec)
    return spec,previous,path,ep,plan,inputs


def set_episode(recipe, **changes):
    spec,_,path,ep,_,_=recipe;ep.update(changes)
    put(Path(spec['episode_manifest']),ep)
    spec['episode_manifest_sha256']=digest(spec['episode_manifest']);put(path,spec)


def test_real_readonly_capture_query_and_full_geometry_chain_contract(response_recipe):
    spec,previous,path,ep,plan,inputs=response_recipe
    actual,actual_panel,loaded=c.episode_config(spec)
    assert actual==ep and actual_panel==plan
    np.testing.assert_array_equal(loaded['raw'],inputs['raw'])
    assert c.episode_limit(spec,previous)==spec['max_ticks']
    assert phase_limit(p.PHASE,previous,spec)==spec['max_ticks']
    checked=cli.check(path)
    assert not checked['native_execution_authorized'] and checked['VLA_queries']==0
    assert checked['pilot_state_theta_identifiability']=='REQUIRES_ACTUAL_DATA_NOT_ESTABLISHED_BY_RECIPE'


@pytest.mark.parametrize('change',['half-scale','failed','open','missing-window','wrong-grip','trace-gap'])
def test_smaller_failed_or_incomplete_geometry_never_admits_response(response_recipe,change):
    spec,previous,*_=response_recipe
    if change=='half-scale': previous['reference_qualification']['episode']['geometry_scale']=.5
    elif change=='failed': previous['status']='NATIVE_STAGE_FAILED_PRESERVED'
    elif change=='open':
        root=Path(spec['previous_stage']).parent.parent
        lease=json.loads((root/'control/LEASE.json').read_text());lease['status']='ACTIVE';put(root/'control/LEASE.json',lease)
    elif change=='missing-window':previous['reference_qualification']['windows'].pop()
    elif change=='wrong-grip':previous['reference_qualification']['windows'][-1]['gripper']['consumed_wrapper']['values']['sequence']+=1
    else:previous['trace_audit']['missing']=['STEP_AFTER']
    put(Path(spec['previous_stage']),previous);spec['previous_stage_sha256']=digest(spec['previous_stage'])
    with pytest.raises(ValueError):c.episode_limit(spec,previous)


@pytest.mark.parametrize('change',[dict(physics_dt_s=.004),dict(WBC_changes=True),
    dict(fusion_binding='/oldXY'),dict(parallel_batch_spec='/batch'),
    dict(trajectory_response_pilot_parent='TR2'),dict(trajectory_response_collection='UNDECLARED')])
def test_no_old_domain_or_safety_or_pilot_parent_bypass(response_recipe,change):
    spec,*_=response_recipe
    with pytest.raises(ValueError):c.episode_config(dict(spec,**change))


def test_changed_source_or_panel_or_episode_is_rejected(response_recipe):
    spec,_,_,ep,plan,_=response_recipe
    path=Path(spec['trajectory_response_panel']);bad=copy.deepcopy(plan);bad['epsilon_m']=.001
    put(path,bad);spec['trajectory_response_panel_sha256']=digest(path)
    with pytest.raises(ValueError):c.episode_config(spec)
    put(path,plan);spec['trajectory_response_panel_sha256']=digest(path)
    set_episode(response_recipe,planned_end_atom=ep['planned_end_atom']+1)
    spec['max_ticks']+=25
    with pytest.raises(ValueError,match='COMMON_CLOCK'):c.episode_config(spec)


class ResponseFake(GeometryFake):
    def prepare_dynamic(self):
        from go2arx5_vla.world_model.init_trace_reader import RECORD
        (self.output/'init.trace').write_bytes(struct.pack('<8sII',b'WM02IT01',RECORD.size,262144))
    def install_reference(self,action):
        n.validate_adapter_install(self,action)
        self.active_reference=dict(reference(round(self.step*.002*1e9)/1e9),action_EEF10=action.tolist())
        self.installs.append((self.step,action.copy()));return self.active_reference


def setup_flow(monkeypatch,adapter):
    geometry_flow(monkeypatch,adapter)
    monkeypatch.setattr(n,'consumed_window',lambda a,start,before,after,clock:dict(
        native_start_step=before['step'],native_end_step=after['step'],actual_ticks=25,clock=clock,
        solver_policies_with_exact_installed_target=[dict(fake_only=True)]))


@pytest.mark.parametrize('context',['C0','C1'])
@pytest.mark.parametrize('theta',[-1,0,1])
def test_complete_fake_context_pulse_tail_and_actual_history(response_recipe,tmp_path,monkeypatch,context,theta):
    set_episode(response_recipe,context_id=context,theta_level=theta)
    spec,_,_,ep,plan,_=response_recipe
    output=tmp_path/'response-flow';output.mkdir();adapter=ResponseFake(spec,output);setup_flow(monkeypatch,adapter)
    report=n.run(adapter);count=(spec['max_ticks']-2503)//25
    assert report['complete_task_atoms']==report['solver_target_confirmed_atoms']==report['gripper_confirmed_atoms']==count
    assert report['all_40_knots_consumed'] and len(adapter.grip_packets)==count
    assert [x['sequence'] for x in adapter.grip_packets]==list(range(count))
    assert adapter.sealed==spec['max_ticks'] and report['VLA_queries']==0
    context_program=p.ContextProgram(np.eye(4),BASE,context)
    for step,action in adapter.installs:
        expected=(context_program.preview((step-2503)//25) if step<3703 else
                  adapter.trajectory_response_program.preview((step-3703)//25))
        np.testing.assert_array_equal(action,expected)
    decision=json.loads((output/'TRAJECTORY_RESPONSE_DECISION.private.json').read_text())
    history=decision['actual_history']
    assert history['actual_state_history_shape']==[9,106]
    assert np.asarray(history['past_consumed_EEF10_BASE7']).shape==(8,17)
    np.testing.assert_array_equal(decision['planned_context_endpoint_TCP'],np.eye(4))
    assert not decision['future_theta_available_in_predecision_state_features']
    assert decision['assignment_before_install']==dict(native_step=3703,theta_level=theta,future_response_observed=False)
    post=[w for w in report['windows'] if w['probe_elapsed_atom'] is not None]
    assert post[0]['actual_history_before_atom']==history
    assert len(post)>=40 and len(post[0]['full_nominal_H40'])==len(post[0]['full_corrected_H40'])==40
    assert post[0]['nominal_EEF10']==post[0]['corrected_submitted_EEF10'] # b(0)=0.
    assert not report['exact_state_fork'] and report['WM_benefit']=='NOT_TESTED_BY_PROBE_COLLECTION'
    assert report['theta_response_identifiable']=='REQUIRES_ACTUAL_PILOT_ANALYSIS'
    assert report['decision_record']['sha256']==digest(output/'TRAJECTORY_RESPONSE_DECISION.private.json')
    # The fake actual state remains identical across contexts: no gain or state
    # separation can be invented from the C0/C1 labels alone.
    assert np.max(np.abs(np.asarray(history['actual_state_history'])[:,0]))==0


def test_context_preview_cannot_depend_on_future_theta_or_be_published_early(response_recipe):
    spec,_,_,ep,plan,inputs=response_recipe
    context=p.ContextProgram(np.eye(4),BASE,'C1')
    programs=[c.make_program(inputs,dict(ep,theta_level=t),plan,np.eye(4),np.eye(4),BASE) for t in (-1,0,1)]
    for t,program in zip((-1,0,1),programs):
        action=context.preview(40)
        with pytest.raises(ValueError,match='NO_THETA_PROGRAM'):
            c.validate_install_preview(action,context=context,program=program,step=3503,base=BASE,frozen_base=BASE,end_step=spec['max_ticks'])
        assert c.validate_install_preview(action,context=context,program=None,step=3503,base=BASE,frozen_base=BASE,end_step=spec['max_ticks'])['points_checked']==40
    assert len({program.end_atom for program in programs})==1
    np.testing.assert_array_equal(programs[0].preview(0)[0],programs[2].preview(0)[0])


def test_default_check_and_execute_keep_original_worker_and_no_native_import(response_recipe,monkeypatch):
    _,_,path,*_=response_recipe;calls=[]
    monkeypatch.setitem(sys.modules,'scripts.wm02.pilot1_native_worker',SimpleNamespace(main=lambda argv:calls.append(argv) or 75))
    assert cli.main(['--spec',str(path)])==0 and calls==[]
    assert cli.main(['--spec',str(path),'--execute'])==75 and calls==[['--spec',str(path)]]
    code=('import sys;import scripts.wm02.fusion_trajectory_response_check;'
          'import go2arx5_vla.world_model.fusion_trajectory_response_native_contract;'
          'assert not {"torch","rclpy","mujoco"}&sys.modules.keys()')
    subprocess.run([sys.executable,'-B','-c',code],check=True)


def test_only_new_phase_hook_and_trace_first_order(tmp_path,monkeypatch):
    adapter=TargetSequentialAdapter(dict(phase=p.PHASE),tmp_path)
    monkeypatch.setattr(n,'run',lambda a:('response-only',a))
    assert adapter.dynamic_diagnostic()==('response-only',adapter)
    spec=dict(phase=p.PHASE,trajectory_trace_capacity_build='/proof',xvfb_run='/xvfb',native_binary='/native',
              trace_library='/trace',outer_library='/outer',observer_library='/observer')
    argv,env=native_display_command(spec,tmp_path,dict(LD_PRELOAD='unsafe'))
    assert 'LD_PRELOAD=/trace:/outer:/observer' in argv and 'LD_PRELOAD' not in env
    assert phase_limit('cold-1',None)==50
    with pytest.raises(PermissionError):phase_limit('collect-target-unknown',None)


@pytest.mark.parametrize('phase',[p.PHASE,g.SLOW_PHASE])
def test_new_phases_defer_pose_install_grip_write_without_affecting_old_phase(tmp_path,monkeypatch,phase):
    from scripts.pi0294 import task_state
    calls=[];monkeypatch.setattr(task_state,'write_gripper',lambda *args:calls.append(args))
    monkeypatch.setattr(n,'validate_adapter_install',lambda *args:dict(checked=True))
    monkeypatch.setattr(gn,'validate_adapter_install',lambda *args:dict(checked=True))
    def prepared(actual_phase):
        a=TargetSequentialAdapter(dict(phase=actual_phase),tmp_path)
        a.reference_sequence=-1;a.latest=dict(ee=SimpleNamespace(time=5.006))
        a.wait_source=lambda:None;a.observe=lambda:dict(step=2503,time_s=5.006)
        a.command_event=lambda event:None;a.refs=object();rows=[]
        def publish(topic,wire):
            ref=hashlib.sha256(wire.encode()).hexdigest()
            rows.extend(dict(kind=k,ref_id=ref) for k in ('accept','installed'))
        a.publish=publish;a.reference_events=lambda:rows;a.spin=lambda seconds,predicate:predicate()
        return a
    action=g.hold_preview(np.eye(4))
    prepared(phase).install_reference(action);assert calls==[]
    prepared('g2-standing-eef-y-v2').install_reference(action);assert len(calls)==1


def test_original_unchanged_cpp_quota_accepts_new64_bucket_and_rejects32(tmp_path):
    from go2arx5_vla.world_model.target_sequential import native_task_bucket
    root=Path(__file__).resolve().parents[2]
    header=root/'scripts/pi0294/task_quota_r1.hpp'
    frozen=subprocess.check_output(['git','show','d699145e847d20bc1c9261112e74f1c3c2605786:scripts/pi0294/task_quota_r1.hpp'],cwd=root)
    assert header.read_bytes()==frozen # Original header/control ABI is unchanged.
    code=tmp_path/'quota.cpp';code.write_text('#include "task_quota_r1.hpp"\nint main(){uint64_t n=0;return gr00t_quota_r1::read_bound(n)?0:1;}\n')
    exe=tmp_path/'quota'
    subprocess.run(['g++','-std=c++17','-I'+str(header.parent),str(code),'-o',str(exe)],check=True,capture_output=True)
    env=dict(os.environ,GR00T_TASK_QUOTA_BINDING=str(tmp_path/'binding'),GR00T_TASK_RUN_UUID='fixture',
             GR00T_TASK_CARD_SHA256='card',GR00T_TASK_RESERVATION_ID='owner')
    for phase,dynamic,bound,expected in ((g.SLOW_PHASE,True,33000,0),(p.PHASE,True,33000,0),
        (g.PHASE,True,33000,1),(g.PHASE,True,17000,0),('cold-1',False,2503,0)):
        bucket=native_task_bucket(phase,dynamic)
        (tmp_path/'binding').write_text(f'GRTQUOTA2\nfixture\ncard\nowner\n{bound}\n5\n{bucket}\n')
        assert subprocess.run([str(exe)],env=env).returncode==expected


def test_only_new_phases_have_finite1200_scope_wall(tmp_path):
    from go2arx5_vla.world_model.auto1_native import validate_scope
    binding=tmp_path/'binding.json';put(binding,dict(fixture=True))
    spec=dict(campaign_root=str(tmp_path),campaign_binding=str(binding),domain='WM02_AUTO1_TARGET_COMMISSION_V1',
        physics=True,site_profile='4080-simulation',owner='EXISTING_LOCAL_PC',host='corelab-4080s',uid=2009,
        phase=p.PHASE,wall_limit_s=1200)
    lease=dict(status='ACTIVE',scope='WM02_AUTO1_SIM_ONLY',owner=spec['owner'],run_uuid=tmp_path.name,
               binding_sha256=digest(binding),deadline_epoch=100.)
    registry=dict(entries=[dict(scope=lease['scope'],owner=spec['owner'],path=str(tmp_path),local_lease=str(tmp_path/'control/LEASE.json'))])
    for phase in (p.PHASE,g.SLOW_PHASE):
        validate_scope(dict(spec,phase=phase),lease,registry,host=spec['host'],uid=2009,now=1.)
    for phase in (g.PHASE,'collect-target-fusion-reference-qualification-v1','cold-1'):
        with pytest.raises(PermissionError):validate_scope(dict(spec,phase=phase),lease,registry,host=spec['host'],uid=2009,now=1.)


@pytest.mark.parametrize('phase',[p.PHASE,g.SLOW_PHASE])
def test_new_phases_reject_actual_old_trace_header_at2503(response_recipe,tmp_path,phase):
    spec,*_=response_recipe;spec=dict(spec,phase=phase)
    adapter=ResponseFake(spec,tmp_path);adapter.prepare_dynamic()
    assert gn.actual_trace_binding(adapter)['actual_trace_capacity']==262144
    raw=bytearray((tmp_path/'init.trace').read_bytes());struct.pack_into('<I',raw,12,131072)
    (tmp_path/'init.trace').write_bytes(raw)
    with pytest.raises(ValueError,match='CAPACITY_262144'):gn.actual_trace_binding(adapter)


def test_original_outer_commission_accepts_response_report_without_old_policy_fields(response_recipe,tmp_path,monkeypatch):
    from scripts.wm02 import commission_target
    from go2arx5_vla.world_model import init_trace_reader,native_replay,target_sequential
    spec,*_=response_recipe;spec['candidate_source_sha']='b'*40
    output=tmp_path/'mock-outer';output.mkdir()
    class Outer(ResponseFake):
        def __init__(self,spec,output):
            super().__init__(spec,output);setup_flow(monkeypatch,self)
            self.node=None;self.exit_records=[];self.trace_sealed=False
        def start(self):(self.output/'state.bin.journal').write_bytes(b'non-native fixture')
        def dynamic_phase(self):return True
        def reference_phase(self):return True
        def dynamic_diagnostic(self):return n.run(self)
        def close(self):self.exit_records=[dict(name=name,exit_code=0) for name in ('sim','controller','solver','planner')]
    monkeypatch.setattr(target_sequential,'TargetSequentialAdapter',Outer)
    monkeypatch.setattr(init_trace_reader,'audit_trace',lambda *a,**k:dict(status='RECORDED_TRACE_CHECKS_PASS',fake_only=True))
    monkeypatch.setattr(native_replay,'packets',lambda path:[None]*spec['max_ticks'])
    assert commission_target.run(spec,output)==0
    result=json.loads((output/'RESULT.json').read_text())
    assert result['actual_target_integrations']==spec['max_ticks'] and result['shutdown_clean']
    assert result['reference_qualification']['all_40_knots_consumed']
    assert result['reference_qualification']['WM_benefit']=='NOT_TESTED_BY_PROBE_COLLECTION'
