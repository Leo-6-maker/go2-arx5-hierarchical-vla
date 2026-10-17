"""Committed CORR visits with observed gripper events, not invented phases."""
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.pi0294.e1.correction import window_key
from scripts.pi0294.native_dataset import sha
from .event_exposure import episode_events,summarize_visits
from .exposure import actual_exposure


def correction_inventory(root,row):
    if (row['split']!='TRAIN' or row['task']!='T1' or not row.get('source_student_episode')
            or row.get('execution_actor')!='ORIGINAL_QUALIFIED_SOURCE_EXPERT'):
        raise PermissionError('ORIGINAL_QUALIFIED_TRAIN_T1_CORRECTION_ONLY')
    root=Path(root);p=root/row['file']
    if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
        raise PermissionError('NO_CORRECTION_NUMERIC_PATH_ESCAPE')
    if sha(p)!=row['sha256']:raise ValueError('FROZEN_CORRECTION_NUMERIC_HASH')
    with np.load(p,allow_pickle=False) as z:
        commands,states=[z[k].copy() for k in ['committed_absolute10','actual_state10']]
        valid=z['valid_window'].copy();rgb=z['rgb_sha256'].copy()
        representation=z['label_representation'].item()
    n=len(commands);anchors=row['legal_anchors']
    if (commands.shape!=(n,10) or states.shape!=(n,10) or n<40 or n!=row['expert_commands']
            or n!=8*row['actual_ACKs'] or valid.shape!=(n,) or valid.dtype!=np.bool_
            or anchors!=np.flatnonzero(valid).tolist() or row['windows']!=len(anchors)
            or not anchors or any(type(a) is not int or not 0<=a<=n-40 for a in anchors)
            or rgb.shape!=(n,2) or any(len(str(h))!=64 for h in rgb.flat)
            or representation!='ABSOLUTE_COMMITTED_EEF10'
            or not np.isfinite(commands).all() or not np.isfinite(states).all()
            or np.any((commands[:,9]<0)|(commands[:,9]>1))):
        raise ValueError('EXACT_ACKED_CORRECTION_COMMANDS_STATES_AND_LEGAL_H40_ANCHORS')
    grip=np.r_[states[0,9],commands[:,9]]
    events=dict(close=np.flatnonzero((grip[1:]<.5)&(grip[:-1]>=.5)).tolist(),
                open=np.flatnonzero((grip[1:]>=.5)&(grip[:-1]<.5)).tolist())
    keys={str(a):window_key(states[a],commands[a:a+40],rgb[a]) for a in anchors}
    if sha(p)!=row['sha256']:raise ValueError('CORRECTION_NUMERIC_CHANGED_DURING_AUDIT')
    return dict(id=row['id'],task=row['task'],origin=row['origin'],
        source_student_episode=row['source_student_episode'],source_anchor_tick=row['anchor_tick'],
        actual_commands=n,actual_ACKs=row['actual_ACKs'],legal_anchors=anchors,events=events,
        qualified_window_fingerprints=keys,numeric_sha256=row['sha256'],
        source_result_sha256=row['source_result_sha256'],
        full_recovery=row['full_recovery'],full_recovery_success=row['full_recovery_success'],
        stage_labels='NOT_RECORDED_NOT_INFERRED_FROM_GEOMETRY',
        detach_and_withdraw_events='UNKNOWN_NOT_ZERO',
        gripper_event_definition='THRESHOLD_0_5_CROSSING_OF_FROZEN_ACTUAL_COMMITTED_COMMANDS')


def summarize_correction_visits(inventories,visits):
    if set(inventories)!=set(visits):raise ValueError('EXACT_CORRECTION_EPISODE_LEFT_JOIN')
    episodes={};actors=Counter();parents=Counter();origins=Counter()
    totals={name:Counter() for name in ['close','open']}
    for identifier,inv in inventories.items():
        counts=visits[identifier];legal=set(inv['legal_anchors'])
        if any(type(a) is not int or a not in legal or type(n) is not int or n<=0 for a,n in counts.items()):
            raise ValueError('ONLY_ACTUALLY_CONSUMED_LEGAL_CORRECTION_ANCHORS')
        events={}
        for name,indices in inv['events'].items():
            rows=[]
            for i in indices:
                prefix=sum(n for a,n in counts.items() if a<=i<a+8)
                late=sum(n for a,n in counts.items() if a+8<=i<a+40)
                rows.append(dict(command_index=i,prefix8_visit_hits=prefix,late_H40_visit_hits=late))
            summary=dict(segment_event_occurrences=len(rows),occurrences=rows,
                prefix8_visit_hits=sum(r['prefix8_visit_hits'] for r in rows),
                late_H40_visit_hits=sum(r['late_H40_visit_hits'] for r in rows),
                occurrences_uncovered_in_sampled_prefix8=sum(not r['prefix8_visit_hits'] for r in rows),
                occurrences_uncovered_in_sampled_H40=sum(not r['prefix8_visit_hits'] and not r['late_H40_visit_hits'] for r in rows))
            events[name]=summary;totals[name].update({k:v for k,v in summary.items() if k!='occurrences'})
        student=counts.get(0,0);expert=sum(counts.values())-student
        actors.update(STUDENT_AT_CORRECTION_ANCHOR=student,EXPERT_AFTER_CORRECTION_ANCHOR=expert)
        parents[inv['source_student_episode']]+=sum(counts.values());origins[inv['origin']]+=sum(counts.values())
        episodes[identifier]=dict(source_student_episode=inv['source_student_episode'],origin=inv['origin'],
            sample_visits=sum(counts.values()),unique_consumed_anchors=len(counts),
            student_anchor_visits=student,subsequent_expert_state_visits=expert,events=events,
            numeric_sha256=inv['numeric_sha256'],stage_labels='NOT_RECORDED',
            detach_and_withdraw_events='UNKNOWN_NOT_ZERO')
    return dict(episodes=episodes,segment_count=len(episodes),independent_student_parents=len(parents),
        sample_visits=sum(parents.values()),actor_visits=dict(actors),parent_visits=dict(parents),
        origin_visits=dict(origins),gripper_event_totals=totals,
        events_are_segment_occurrences_not_independent_parent_episodes=True,
        hits_count_repeated_sample_visits_not_independent_demonstrations=True,
        detached_or_withdrawn_from_XYZ_motion_inferred=False)


def audit(dataset,trace,stop,guard=lambda:None):
    exposure=actual_exposure(dataset,trace,0,stop)
    if dataset.correction is None or not exposure['correction_sample_visits']:
        raise PermissionError('ACTUAL_FIXED_CORR_TRACE_REQUIRED')
    original_rows={};original_visits={};corrected_rows={};corrected_visits={}
    for actual in trace:
        row,anchor=dataset.rows[actual['index']]
        rows,visits=(corrected_rows,corrected_visits) if actual['index']>=dataset.original_rows else (original_rows,original_visits)
        rows[row['id']]=row;visits.setdefault(row['id'],Counter())[anchor]+=1
    original={};corrected={}
    for identifier,row in sorted(original_rows.items()):
        guard();original[identifier]=episode_events(dataset.root,Path(dataset.raw_bindings.get(identifier,row['raw_root'])),row)
    for identifier,row in sorted(corrected_rows.items()):
        guard();corrected[identifier]=correction_inventory(dataset.correction.root,row)
    result=dict(status='PASS_ACTUAL_CORR_VISITS_JOIN_ORIGINAL_RAW_AND_C1_COMMITTED_GRIPPER_EVENTS',
        exposure=exposure,original_expert=summarize_visits(original,original_visits),
        correction=summarize_correction_visits(corrected,corrected_visits),
        original_inventory=original,correction_inventory=corrected,
        C1_detach_withdraw_or_teacher_stage_labels_fabricated=False,
        new_queries=0,new_physics=0,sampler_or_training_modified=False)
    if ({k:v for k,v in result['correction']['actor_visits'].items() if v}
            !={k:v for k,v in exposure['current_state_actor_visits'].items() if k!='ORIGINAL_SUCCESS_EXPERT_DEMONSTRATION' and v}):
        raise ValueError('ACTUAL_CORRECTION_ACTORS_EQUAL_INDEPENDENT_TRACE_ACCOUNTING')
    return result
