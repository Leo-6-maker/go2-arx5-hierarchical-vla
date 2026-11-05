"""One paused KEEP/RELEASE choice; original WBC and safety remain in charge."""
import json
from pathlib import Path
import numpy as np
from .storage import digest
from .pilot2_supported_decision import observed_history, features, PROGRAMS, DECISION_ATOM

SCHEMA='WM02_PILOT2_SUPPORTED_POLICY_EPISODE_V1'
METHODS=('TRAINED_GOAL_POLICY','GOAL_FSM','CONSTANT_KEEP','CONSTANT_RELEASE','TRAIN_MEAN_RETURN')


def effective(episode, selected=0):
    return dict(episode,schema='WM02_PILOT2_SUPPORTED_EPISODE_V1',command_program=PROGRAMS[selected])


def contract(episode):
    from .pilot2_supported import contract as base_contract
    if (episode.get('schema')!=SCHEMA or episode.get('command_program')!='SELECT_KEEP_RELEASE_AFTER_SUPPORTED_HOLD'
            or episode.get('close_start_atom')!=56 or episode.get('policy_decision_atom')!=112
            or episode.get('method') not in METHODS or episode.get('purpose')!='SUPPORTED_POLICY_EVALUATION'):
        raise PermissionError('EXACT_QUALIFIED_COMMON_PREFIX_POLICY_CONTRACT_REQUIRED')
    base_contract(effective(episode)); return episode


def validate_qualifications(spec,episode):
    proofs=spec['supported_program_qualifications'];found=set()
    if len(proofs)!=2:raise PermissionError('BOTH_ACTUAL_SINGLE_PROGRAM_QUALIFICATIONS_REQUIRED')
    for proof in proofs:
        p=Path(proof['path']);n=json.loads(p.read_text());score=p.parent/'SUPPORTED_CLOSE_RESULT.private.json'
        epath=p.parent.parent/'EPISODE_MANIFEST.private.json';e=json.loads(epath.read_text());s=json.loads(score.read_text())
        if (digest(p)!=proof['sha256'] or digest(score)!=proof['score_sha256'] or digest(epath)!=proof['episode_sha256']
                or n['status']!='NATIVE_STAGE_PASSED' or not n['shutdown_clean'] or n['physics_stops']
                or n['actual_target_integrations']!=6703 or n['scene_sha256']!=digest(spec['model_path'])
                or n['native_binary_sha256']!=digest(spec['native_binary'])
                or e.get('schema')!='WM02_PILOT2_SUPPORTED_EPISODE_V1' or e['parent']!=episode['parent']
                or e['split']!=episode['split'] or e['close_start_atom']!=56 or e['command_program'] not in PROGRAMS
                or not s['goal_success'] or not s['qualified_supported_grasp_hold']):
            raise PermissionError('EXACT_SCENE_AND_PROGRAM_ACTUAL_SAFE_QUALIFICATION_REQUIRED')
        found.add(e['command_program'])
    if found!=set(PROGRAMS):raise PermissionError('KEEP56_AND_RELEASE56_BOTH_REQUIRED')
    if episode['method']=='TRAINED_GOAL_POLICY':
        for name in ('supported_policy_numpy','supported_policy_training_result'):
            if digest(spec[name])!=spec[name+'_sha256']:raise PermissionError('FROZEN_SUPPORTED_POLICY_CHANGED')
        result=json.loads(Path(spec['supported_policy_training_result']).read_text())
        if (result['status']!='SUPPORTED_GOAL_POLICY_TRAINED_NATIVE_EVALUATION_PENDING'
                or result['portable_policy_sha256']!=spec['supported_policy_numpy_sha256']
                or result['complete_restore_error']!=0 or result['endpoint_reload_error']!=0):
            raise PermissionError('REAL_TRAINED_RESTORED_SUPPORTED_POLICY_REQUIRED')


def choose(adapter,episode,rows):
    from .pilot2_supported import GOALS
    from .pilot1_contact_policy_numpy import logits
    from scripts.pi0292.durable import write
    contract(episode);step=adapter.observe()['step']
    if step!=2503+25*DECISION_ATOM or len(rows)!=DECISION_ATOM:
        raise ValueError('EXACT_PAUSED_COMMON_SUPPORTED_PREFIX_REQUIRED')
    history=observed_history(rows);x=features(history,episode['task_goal'])
    ticks=[e for r in rows[-20:] for e in r['native_contact_events']]
    eligible=len(ticks)==500 and all(e['native_grasp'] and e['table_support'] and e['stage']=='SUPPORTED_HOLD' for e in ticks)
    goal=GOALS.index(episode['task_goal']);method=episode['method'];scores=None;model_sha=None
    if not eligible:selected=1  # Safe original opening, recorded as invalid decision prefix.
    elif method=='TRAINED_GOAL_POLICY':
        s=adapter.spec;model_sha=digest(s['supported_policy_numpy'])
        if model_sha!=s['supported_policy_numpy_sha256']:raise PermissionError('SUPPORTED_POLICY_CHANGED_BEFORE_CHOICE')
        with np.load(s['supported_policy_numpy'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        scores=logits(arrays,x);selected=int(scores.argmax())
    elif method=='GOAL_FSM':selected=goal
    elif method=='CONSTANT_KEEP':selected=0
    elif method=='CONSTANT_RELEASE':selected=1
    else:
        s=adapter.spec;p=Path(s['supported_train_return'])
        if digest(p)!=s['supported_train_return_sha256']:raise PermissionError('TRAIN_ONLY_RETURN_BASELINE_CHANGED')
        baseline=json.loads(p.read_text())
        if baseline['fit_split']!='TRAIN' or baseline['exact_fork_oracle'] is not False:
            raise PermissionError('ACTUAL_TRAIN_ONLY_RESTART_RETURN_BASELINE_REQUIRED')
        selected=int(np.asarray(baseline['expected_return_by_goal'])[goal].argmax())
    value=dict(schema='WM02_PILOT2_SUPPORTED_POLICY_DECISION_V1',source_sha=adapter.spec['source_sha'],
        native_step=step,method=method,task_goal=episode['task_goal'],selected_action=selected,
        selected_program=PROGRAMS[selected],decision_prefix_eligible=eligible,
        invalid_prefix_reason=None if eligible else'NO_500_CONTINUOUS_SUPPORTED_HOLD_TICKS',
        past_features=x.tolist(),logits=None if scores is None else scores.tolist(),policy_sha256=model_sha,
        future_response_observed=False,program_known_from_anchor=111,full_hidden_state_restore=False)
    write(adapter.output/'SUPPORTED_POLICY_DECISION.private.json',value)
    if adapter.observe()['step']!=step:raise RuntimeError('PHYSICS_ADVANCED_DURING_SUPPORTED_POLICY_CHOICE')
    adapter.command_event(dict(phase='supported-policy-choice-before-install',native_step=step,
        decision_sha256=digest(adapter.output/'SUPPORTED_POLICY_DECISION.private.json'),selected_action=selected,
        future_response_observed=False))
    return effective(episode,selected)
