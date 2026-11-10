"""Side-specific event strata from the sealed 6ms contact transitions.

This reuses the audited transitions rather than rereading100 native traces.
Contact-point counts are not durations, grasp certificates or force totals.
"""
import json
from collections import Counter
from pathlib import Path

from .storage import digest

DT=.002
RECORDED_STRIDE=3
LAST_SAMPLE=21202
LABELS={'LEFT':'task_left_pad','RIGHT':'task_right_pad','SUPPORT':'task_table'}


def active_sides(pairs):
    found=set()
    for pair in pairs:
        names=set(pair.split(' / '))
        if len(names)!=2 or 'task_cube_geom'not in names:raise ValueError('Known audited cube pair required')
        other=next(iter(names-{'task_cube_geom'}))
        side=next((label for label,name in LABELS.items()if name==other),None)
        if side is None:raise ValueError('Unknown contact geometry must not be guessed')
        found.add(side)
    return found


def strata(sides):
    pads=sides&{'LEFT','RIGHT'}
    kind='BILATERAL'if len(pads)==2 else 'LEFT_ONLY'if pads=={'LEFT'}else 'RIGHT_ONLY'if pads=={'RIGHT'}else 'NO_PAD'
    return kind,('SUPPORTED'if 'SUPPORT'in sides else'SUPPORT_CONTACT_ABSENT')


def episode_strata(row):
    transitions=row['transitions'];intervals=[];events=[];previous=set();durations=Counter()
    if not transitions or transitions[0]['native_step']!=5704:raise ValueError('Original first post-boundary sample required')
    for i,event in enumerate(transitions):
        start=event['native_step'];stop=transitions[i+1]['native_step']if i+1<len(transitions)else LAST_SAMPLE+RECORDED_STRIDE
        if not(start<stop and (stop-start)%RECORDED_STRIDE==0):raise ValueError('Monotonic recorded 6ms transitions')
        sides=active_sides(event['pairs']);kind,support=strata(sides)
        interval=dict(start_native_step=start,stop_native_step_exclusive=stop,duration_s=(stop-start)*DT,
            pad_state=kind,support_contact_state=support)
        intervals.append(interval);durations[kind+'__'+support]+=(stop-start)*DT
        if i:
            appeared=sorted(sides-previous);disappeared=sorted(previous-sides)
            events.append(dict(native_step=start,onset_interval_native_steps=[start-RECORDED_STRIDE,start],
                appeared=appeared,disappeared=disappeared,before_pad_state=strata(previous)[0],after_pad_state=kind,
                before_support_contact_state=strata(previous)[1],after_support_contact_state=support))
        previous=sides
    first={label:next((v['native_step']for v in events if label in v['appeared']),None)for label in ('LEFT','RIGHT')}
    support_transitions=[v for v in events if 'SUPPORT'in v['disappeared']]
    longest=max((v['duration_s']for v in intervals if v['pad_state']=='BILATERAL'and
                 v['support_contact_state']=='SUPPORT_CONTACT_ABSENT'),default=0.)
    return dict(episode=row['episode'],source_trace_sha256=row['trace_sha256'],intervals=intervals,events=events,
        duration_s_by_contact_state=dict(durations),first_pad_contact_native_steps=first,
        first_pad_asymmetry_s=(abs(first['LEFT']-first['RIGHT'])*DT if all(v is not None for v in first.values())else None),
        support_contact_disappearance_events=len(support_transitions),
        maximum_recorded_uninterrupted_bilateral_without_support_s=longest,
        native_grasp_or_hold_certificate=False)


def run(spec,output,guard):
    from scripts.pi0292.durable import write
    pin=spec['contact_audit'];p=Path(pin['path'])
    if digest(p)!=pin['sha256']:raise ValueError('Pinned contact audit required')
    source=json.loads(p.read_bytes())
    if source['audited_episodes']!=100 or source['distinct_supervised_windows']!=1450:raise ValueError('Original TRAIN denominator required')
    rows=[];windows=[]
    for original in source['rows']:
        guard();row=episode_strata(original);rows.append(row)
        for old in source['windows']:
            if old['episode']!=row['episode']:continue
            start=5703+1000*old['anchor'];stop=start+1000
            overlap=[v for v in row['intervals']if max(start,v['start_native_step'])<min(stop,v['stop_native_step_exclusive'])]
            events=[v for v in row['events']if start<=v['native_step']<stop]
            windows.append(dict(episode=old['episode'],anchor=old['anchor'],phase_audit_only=old['phase_audit_only'],
                present_contact_states=sorted({v['pad_state']+'__'+v['support_contact_state']for v in overlap}),
                left_onsets=sum('LEFT'in v['appeared']for v in events),right_onsets=sum('RIGHT'in v['appeared']for v in events),
                bilateral_onsets=sum(v['after_pad_state']=='BILATERAL'and v['before_pad_state']!='BILATERAL'for v in events),
                support_contact_disappearances=sum('SUPPORT'in v['disappeared']for v in events),
                events=events,actual_contact_metadata_not_model_input=True))
    if len(rows)!=100 or len(windows)!=1450:raise ValueError('Complete original TRAIN100/1450 strata required')
    result=dict(schema='TASK_LEARNING_TRAIN100_SIDE_SPECIFIC_CONTACT_STRATA_V3',source_sha=spec['source_sha'],
        contact_audit=pin,dataset=source['dataset'],requested_episodes=100,audited_episodes=100,
        distinct_TRAIN_windows=1450,recorded_contact_period_s=.006,rows=rows,windows=windows,
        summary=dict(episodes_with_both_pad_onsets=sum(all(x is not None for x in r['first_pad_contact_native_steps'].values())for r in rows),
            episodes_with_recorded_bilateral_no_support_at_least1s=sum(r['maximum_recorded_uninterrupted_bilateral_without_support_s']>=1. for r in rows),
            windows_by_event={k:sum(w[k]>0 for w in windows)for k in ('left_onsets','right_onsets','bilateral_onsets','support_contact_disappearances')},
            phase_event_windows={phase:{k:sum(w[k]>0 for w in windows if w['phase_audit_only']==phase)for k in
                ('left_onsets','right_onsets','bilateral_onsets','support_contact_disappearances')}for phase in sorted({w['phase_audit_only']for w in windows})}),
        original_contact_force_threshold_N=.05,force_values_or_unrecorded_contacts_not_reconstructed=True,
        contact_point_counts_not_used_as_duration=True,final_interval_uses_one_recorded_sample_bin=True,
        TeacherV8_qualification_requests=0,RecoveryOracle_calls=0,DAgger_labels=0,model_calls=0,optimizer_updates=0,
        physics_integrations=0,original_data_changed=False,TEST_read=False,
        meaning='Temporal contact strata only; neither a new Teacher qualification nor independent physical grasp/hold success')
    output=Path(output);output.mkdir();write(output/'RESULT.private.json',result);return result
