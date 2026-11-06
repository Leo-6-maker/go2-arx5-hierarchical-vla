"""Known-posture contract fixtures; no native qualification, query or physics."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.pi0292.eval_contract import FRAME,TCP,matrix10
from scripts.pi0292.relay_protocol import sealed
from go2arx5_vla.world_model import fusion_source_posture_qualification_contract as c
from go2arx5_vla.world_model.fusion_trajectory_adapter import TrajectoryAdaptation
from go2arx5_vla.world_model.storage import digest
from tests.world_model.test_fusion_live_proposal_qualification_contract import (
    recipe as prior_recipe,put,prior_files,BASE)
from tests.world_model.test_fusion_observation import png


def episode(root,inputs,scale):
    program=TrajectoryAdaptation(inputs['raw'],inputs['query_tcp'],inputs['frame_map'],
        np.eye(4),np.eye(4),BASE,scale,time_stretch_factor=2)
    return dict(c._fixed_episode(),run=str(root),geometry_scale=scale,planned_end_atom=program.end_atom)


@pytest.fixture
def recipe(tmp_path,monkeypatch):
    spec,previous,_,archived,calls=prior_recipe.__wrapped__(tmp_path,monkeypatch)
    project=Path(__file__).resolve().parents[2];capsule=Path(spec['source_root'])
    for name in c.SOURCE_MEMBERS:
        target=capsule/name;target.parent.mkdir(parents=True,exist_ok=True)
        original=project/name
        target.write_bytes(original.read_bytes()if original.exists()else b'SYNTHETIC_UNEXECUTED_NATIVE_CLI_MEMBER')
        spec['source_files'][name]=digest(target)
    import base64,hashlib
    theta=np.deg2rad(80.)
    goal=[.25,-.10,.08,np.cos(theta),-np.sin(theta),0.,np.sin(theta),np.cos(theta),0.,1.]
    images=[]
    for camera,color in [('external_rgb',20),('wrist_rgb',60)]:
        image=png(color)
        images.append(dict(camera=camera,png_base64=base64.b64encode(image).decode(),sha256=hashlib.sha256(image).hexdigest()))
    payload=dict(snapshot_uuid='7cd0c1ea-e3bb-47cd-aec4-6bd7fa0be120',language='Original source full task prompt.',
        state_eef10=goal,frame=FRAME,tcp=TCP,images=images)
    source_record=sealed(dict(protocol='PI0292_PAUSED_RELAY_ACK_CHAIN_V2',kind='observation',run_uuid='source-confirm-01',
        episode_uuid='008b845c-d14a-4aeb-9b87-9532bceed38f',sequence=0,tick=0,payload=payload))
    path=tmp_path/'SYNTHETIC_ORIGINAL_SOURCE_OBSERVATION.json';pin=put(path,source_record)
    monkeypatch.setattr(c,'POSTURE_OBSERVATION',pin);monkeypatch.setattr(c,'POSTURE_BODY',source_record['sha256'])
    monkeypatch.setattr(c,'TRACE_BUILD',spec['trajectory_trace_capacity_build_sha256'])
    monkeypatch.setattr(c,'FIRST_PREDECESSOR',spec['previous_stage_sha256'])
    first_spec=Path(spec['previous_stage']).parent.parent/'EXECUTION_SPEC.json'
    monkeypatch.setattr(c,'FIRST_PREDECESSOR_SPEC',digest(first_spec))
    archived['payload'].update(frame=FRAME,tcp=TCP)
    def read(native,gpu,*,pins):
        calls.append((native,gpu,pins));assert pins==c.PINS;return archived
    monkeypatch.setattr(c,'read_archive',read)
    spec.update(phase=c.PHASE,posture_source_observation=str(path),posture_source_observation_sha256=pin,
        source_posture_query_pins=copy.deepcopy(c.PINS),
        source_posture_native_spec=spec.pop('live_proposal_native_spec'),
        source_posture_gpu_spec=spec.pop('live_proposal_gpu_spec'))
    spec.pop('live_proposal_query_pins')
    inputs=c.load_query(spec)
    ep=episode(Path(spec['registered_root']),inputs,.03)
    spec['episode_manifest_sha256']=put(spec['episode_manifest'],ep);spec['max_ticks']=c.endpoint(ep)
    return spec,previous,ep,inputs,archived,calls


def rewrite(spec,ep):
    spec['episode_manifest_sha256']=put(spec['episode_manifest'],ep);spec['max_ticks']=c.endpoint(ep)


def test_source_observed_posture_is_generated_calibration_goal_never_the_vla_proposal_or_expert_label(recipe):
    spec,previous,ep,inputs,archived,_=recipe
    actual,loaded=c.episode_config(spec)
    assert actual==ep and c.episode_limit(spec,previous)==c.endpoint(ep)
    np.testing.assert_array_equal(loaded['raw'],np.repeat(np.asarray(loaded['source_observation_payload']['state_eef10'])[None],40,0))
    assert not np.array_equal(loaded['raw'],archived['physical_H40'])
    np.testing.assert_array_equal(loaded['query_tcp'],matrix10(archived['payload']['state_eef10']))
    np.testing.assert_array_equal(loaded['source_goal_TCP_source'],matrix10(loaded['source_observation_payload']['state_eef10']))
    assert loaded['raw_is_generated_calibration_reference'] and loaded['raw_generated_H40']
    assert loaded['provenance']['desired_outcome_provenance']=='CALIBRATION_INTENT'
    assert not loaded['provenance']['query02_VLA_action_used_as_program']
    assert not loaded['provenance']['source_observed_pose_is_expert_action_label']
    assert loaded['provenance']['VLA_action_installations']==loaded['provenance']['WM_updates']==0
    with pytest.raises(ValueError):loaded['raw'].setflags(write=True)


def test_independent_restart_has_explicit_rigid_relocation_with_invariant_clock_and_fixed_init_reference(recipe):
    spec,_,ep,inputs,*_=recipe
    baseline=c.make_program(inputs,ep,np.eye(4),np.eye(4),BASE)
    theta=.6;r=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    decision=np.eye(4);decision[:3,:3]=r;decision[:3,3]=[.02,-.01,.015]
    initial=np.eye(4);initial[:3,3]=[-.01,0,.005]
    restarted=c.make_program(inputs,ep,initial,decision,BASE)
    assert baseline.end_atom==restarted.end_atom==ep['planned_end_atom']
    assert restarted.time_stretch_factor==2
    np.testing.assert_array_equal(restarted.query_TCP_source,inputs['query_tcp'])
    np.testing.assert_allclose(restarted.desired_path[0,:3],decision[:3,3])
    np.testing.assert_allclose(restarted.reference_path[0,:3],initial[:3,3])
    np.testing.assert_array_equal(restarted.decoded_H40,inputs['raw'])


@pytest.mark.parametrize('scale',c.SCALES)
def test_every_grade_preserves_40_generated_knots_full_preview_and_bounded_original_clock(recipe,scale):
    spec,_,_,inputs,*_=recipe
    ep=episode(Path(spec['registered_root']),inputs,scale)
    p=c.make_program(inputs,ep,np.eye(4),np.eye(4),BASE)
    assert c.endpoint(ep)==3703+25*(p.end_atom+8)<=33000
    assert len(p.knot_indices)==40 and p.preview().shape==(40,10)
    np.testing.assert_allclose(p.preview(p.end_atom+7),np.repeat(p.reference_path[-1:,:],40,axis=0))


@pytest.mark.parametrize('key,value',[
    ('phase','collect-target-fusion-live-proposal-qualification-v3'),('scope','WM02_FUSION_REFERENCE_SIM_ONLY'),
    ('fusion_binding','/OLD-FUSION'),('next2_binding','/OLD-NEXT2'),('physics_dt_s',.004),
    ('WBC_changes',True),('wall_limit_s',1201),('wall_limit_s',True),('trajectory_trace_capacity',262144)])
def test_old_scope_native_or_clock_changes_cannot_bypass_new_family(recipe,key,value):
    spec,*_=recipe
    with pytest.raises(ValueError):c.episode_config(dict(spec,**{key:value}))


@pytest.mark.parametrize('change',[
    dict(purpose='FROZEN_VLA_TRAJECTORY_REPLAY_EXECUTION_DIAGNOSTIC'),dict(experiment_role='EXPERT'),
    dict(desired_outcome_provenance='INTENT_POSE'),dict(raw_generated_H40=False),
    dict(source_observed_pose_is_expert_action_label=True),dict(VLA_action_installations=1),
    dict(geometry_scale=.75),dict(time_stretch_factor=1),dict(split='TEST'),dict(WM_updates=1),
    dict(full_task_success_claim=True),dict(online_closed_loop_claim=True)])
def test_commission_cannot_be_relabelled_as_vla_training_or_task_success(recipe,change):
    _,_,ep,*_=recipe
    with pytest.raises(ValueError):c.episode_contract(dict(ep,**change))


@pytest.mark.parametrize('fault',['source_bytes','source_pin','source_body','frame','query_pin','pair_failed','goal_changed','use_VLA_raw','clock','capsule'])
def test_source_goal_archive_and_frozen_capsule_bindings_fail_closed(recipe,fault):
    spec,_,ep,inputs,archived,_=recipe
    if fault=='source_bytes':Path(spec['posture_source_observation']).write_text('{}')
    elif fault=='source_pin':spec['posture_source_observation_sha256']='e'*64
    elif fault=='source_body':
        source=json.loads(Path(spec['posture_source_observation']).read_text());source['sha256']='e'*64
        # Even an owner-updated file pin cannot silently replace the frozen goal.
        put(spec['posture_source_observation'],source)
    elif fault=='frame':archived['payload']['frame']='UNRELATED_FRAME'
    elif fault=='query_pin':spec['source_posture_query_pins']['action_sha256']='e'*64
    elif fault=='pair_failed':archived['paired_return']['established']=False
    elif fault=='goal_changed':
        changed=dict(inputs,raw=inputs['raw'].copy());changed['raw'][4,0]+=.001
        with pytest.raises(ValueError):c.make_program(changed,ep,np.eye(4),np.eye(4),BASE)
        return
    elif fault=='use_VLA_raw':
        with pytest.raises(ValueError):c.make_program(dict(inputs,raw=archived['physical_H40']),ep,np.eye(4),np.eye(4),BASE)
        return
    elif fault=='clock':rewrite(spec,dict(ep,planned_end_atom=ep['planned_end_atom']+1))
    else:(Path(spec['source_root'])/c.SOURCE_MEMBERS[-1]).write_text('CHANGED')
    with pytest.raises(ValueError):c.episode_config(spec)


def test_second_grade_requires_complete_same_family_first_grade_not_bootstrap_or_skip(recipe,tmp_path):
    spec,bootstrap,_,inputs,*_=recipe
    second=episode(Path(spec['registered_root']),inputs,.1);rewrite(spec,second)
    with pytest.raises(ValueError):c.episode_limit(spec,bootstrap)
    first_root=tmp_path/'SYNTHETIC_POSTURE_FIRST';first=episode(first_root,inputs,.03)
    previous,path,pin,_=prior_files(first_root,first,spec)
    spec.update(previous_stage=str(path),previous_stage_sha256=pin)
    assert c.episode_limit(spec,previous)==c.endpoint(second)
    rewrite(spec,episode(Path(spec['registered_root']),inputs,.25))
    with pytest.raises(ValueError):c.episode_limit(spec,previous)


@pytest.mark.parametrize('fault',['trace_drop','unconsumed_grip','truncated','wrong_family','wrong_source_goal'])
def test_successor_requires_actual_complete_native_predecessor_not_only_pass_bool(recipe,tmp_path,fault):
    spec,_,_,inputs,*_=recipe
    rewrite(spec,episode(Path(spec['registered_root']),inputs,.1))
    root=tmp_path/'SYNTHETIC_POSTURE_FIRST';first=episode(root,inputs,.03)
    previous,path,pin,_=prior_files(root,first,spec);spec.update(previous_stage=str(path),previous_stage_sha256=pin)
    if fault=='trace_drop':previous['trace_audit']['writer']['dropped']=1
    elif fault=='unconsumed_grip':previous['reference_qualification']['windows'][-1]['gripper']['command_consumed']=False
    elif fault=='truncated':previous['reference_qualification']['windows'].pop()
    elif fault=='wrong_family':previous['phase']='collect-target-fusion-live-proposal-qualification-v3'
    else:previous['reference_qualification']['episode']['posture_source_observation_sha256']='e'*64
    spec['previous_stage_sha256']=put(path,previous)
    with pytest.raises(ValueError):c.episode_limit(spec,previous)


def test_import_has_no_model_renderer_native_or_live_query_backend():
    code=('import sys;from go2arx5_vla.world_model import fusion_source_posture_qualification_contract;'
          'assert not {"torch","mujoco","rclpy","transformers","gr00t"}&sys.modules.keys()')
    subprocess.run([sys.executable,'-B','-c',code],check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
