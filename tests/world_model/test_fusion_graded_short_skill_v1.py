"""Actual-sample arithmetic fixtures, never a native qualification."""
import copy
import numpy as np
import pytest

from go2arx5_vla.world_model import fusion_short_skill as old
from go2arx5_vla.world_model import fusion_graded_short_skill_v1 as new
from test_fusion_short_skill import measured


def graded_samples():
    rows=measured()
    for row in rows:
        row['native_step']+=new.DECISION_STEP-3703
        row['time_s']=row['native_step']*.002
    return rows


def test_original_contract_and_task_identity_unchanged():
    assert old.document_hash(old.score_contract())=='b4f5fe077de25403dcca240ddfd007bd2039f4f53b4d3148de46a9e7f074c74b'
    assert old.document_hash(old.task_document('a'*64))=='769ed7192b86be925ceea8e62ba923a35dc07ac8c21c326815682e54c8a8a70e'
    expected=dict(old.score_contract(),schema='WM02_FUSION_GRADED_TARGET_APPROACH_STEP_SCORE_V1',
        task_id=new.TASK_ID,decision_native_step=34303,
        score_role='NEW_GRADED_TARGET_SHORT_SKILL_NOT_OLD_3703_OR_PICK_PLACE')
    assert new.score_contract()==expected
    task=new.task_document('a'*64)
    assert task['language']==old.LANGUAGE and task['task_id']!=old.TASK_ID
    assert task['score_contract_sha256']==old.document_hash(expected)
    from go2arx5_vla.world_model.fusion_paused_observation import _task
    assert _task(task,old.document_hash(task),'a'*64)==task


def test_only_the_preregistered_actual_clock_can_be_scored():
    value=new.score(graded_samples(),new.score_contract())
    assert value['approach_step_success'] and value['decision_native_step']==34303
    assert value['task_id']==new.TASK_ID and value['task_id']!=old.TASK_ID
    assert not value['full_task_success'] and not value['native_execution_authorized']
    assert not value['old_3703_score_relabelled'] and not value['source_task_phase_equivalence']
    assert value['metrics']==old.score(measured(),old.score_contract())['metrics']
    for entry,rows,contract in ((old,graded_samples(),old.score_contract()),
                               (new,measured(),new.score_contract()),
                               (new,graded_samples(),old.score_contract()),
                               (old,measured(),new.score_contract())):
        with pytest.raises(ValueError):entry.score(rows,contract)


@pytest.mark.parametrize('change',['early-close','no-progress','contact','object','clock','BASE','desired-only','looser-score'])
def test_negative_short_skill_outcomes_stay_negative(change):
    rows=graded_samples();contract=new.score_contract()
    if change=='early-close':rows[3]['finger_q'][0]=.0036
    if change in('no-progress','desired-only'):
        for row in rows:
            row['desired_TCP']=copy.deepcopy(row['tcp']);row['tcp'][0,3]=0.
    if change=='contact':rows[4]['robot_object_contact']=True
    if change=='object':rows[6]['object_pose'][0,3]+=.001
    if change=='BASE':rows[7]['qpos'][0]=.003
    if change=='clock':rows[3]['native_step']+=25
    if change=='looser-score':contract['minimum_signed_progress_m']=0.
    if change in('clock','looser-score'):
        with pytest.raises(ValueError):new.score(rows,contract)
    else:assert not new.score(rows,contract)['approach_step_success']


def proposal(change=None):
    from go2arx5_vla.world_model.fusion_trajectory_adapter import TrajectoryAdaptation
    anchor=np.eye(4);anchor[2,3]=1.
    cube=anchor.copy();cube[0,3]=.3
    raw=np.tile([0.,0.,1.,1.,0.,0.,0.,1.,0.,1.],(40,1))
    raw[:,0]=.003*np.minimum(np.arange(40)/8,1.)
    if change=='late-close':raw[-1,9]=0.
    if change=='large-middle':raw[20,0]=.08
    if change=='late-lateral':raw[-1,1]=.005
    if change=='away':raw[:,0]*=-1
    if change=='near-cube':cube[0,3]=.13
    scale=.1 if change=='scaled'else 1.
    return TrajectoryAdaptation(raw,anchor,np.eye(4),anchor,anchor,
        np.array([0.,0.,.4,1.,0.,0.,0.]),scale,time_stretch_factor=1),cube


def test_complete_proposal_screen_is_separate_from_measured_outcome_and_native_safety():
    program,cube=proposal();before=program.evidence()
    value=new.screen_proposal(program,cube)
    assert value['qualified'] and value['raw_knots']==40
    assert value['desired_points_checked']==len(program.desired_path)
    assert not value['actual_task_success_established'] and not value['native_safety_established']
    assert not value['native_execution_authorized'] and program.evidence()==before


@pytest.mark.parametrize('change',['late-close','large-middle','late-lateral','away','near-cube','scaled'])
def test_screen_never_selects_a_favourable_prefix_or_crops_geometry(change):
    program,cube=proposal(change)
    if change=='scaled':
        with pytest.raises(ValueError,match='FULL_UNCLIPPED'):new.screen_proposal(program,cube)
    else:assert not new.screen_proposal(program,cube)['qualified']


def test_tracking_diagnostic_retains_lateral_failure_and_unchanged_strict_score():
    program,cube=proposal('late-lateral');before=program.evidence()
    value=new.tracking_diagnostic_screen(program,cube)
    assert value['eligible_for_original_native_certificate']
    assert value['lateral_task_failure_retained'] and not value['strict_short_proposal_screen']['qualified']
    assert not value['strict_short_task_success'] and not value['native_execution_authorized']
    assert program.evidence()==before
    assert old.document_hash(new.score_contract())=='755139841bbd17949c4a93fa91369b9375ab738915d063a08b3728265cc6ff2a'


@pytest.mark.parametrize('change',['late-close','large-middle','away','near-cube'])
def test_tracking_diagnostic_still_refuses_closed_away_unbounded_or_contact_domain(change):
    program,cube=proposal(change)
    assert not new.tracking_diagnostic_screen(program,cube)['eligible_for_original_native_certificate']
