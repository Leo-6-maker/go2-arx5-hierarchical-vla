"""Original TRAIN30 event/window coverage; no DEV, RGB, simulator or model load."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import time
from types import SimpleNamespace

import numpy as np

from scripts.pi0292.durable import read,write
from scripts.pi0294.bank import PHASES,STAGE_PHASE
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.teacher import PUSH_STAGES
from scripts.pi0294.e1.scope import DATA,STATS
from .dataset import SourceSampler
from .scope import validate,charge_cpu


def window_counts(stages,anchors):
    stages=list(stages);anchors=list(anchors)
    if any(s not in PUSH_STAGES for s in stages) or any(type(a) is not int or a<0 or a+40>len(stages) for a in anchors):
        raise ValueError('EXACT_KNOWN_T2_STAGE_AND_COMPLETE_H40_ONLY')
    count=Counter(stages[a] for a in anchors);touches=Counter();pure=Counter();transitions=0
    for a in anchors:
        values=stages[a:a+40]
        touches.update(set(values))
        if len(set(values))==1:pure[values[0]]+=1
        else:transitions+=1
    return dict(anchor_stage_counts={s:count[s] for s in PUSH_STAGES},
        windows_containing_any_stage={s:touches[s] for s in PUSH_STAGES},
        windows_fully_within_one_stage={s:pure[s] for s in PUSH_STAGES},
        multi_stage_windows=transitions,windows=len(anchors),
        containing_any_stage_counts_overlap=True,window_counts_are_not_independent_episodes=True)


def analyze_episode(row,packet,samples,events,card):
    if row['split']!='TRAIN' or row['task']!='T2' or card['id']!=row['id'] or card['split']!='TRAIN' or card['task']!='T2':
        raise PermissionError('ONLY_ORIGINAL_TRAIN_T2_ROWS')
    anchors=[int(x) for x in np.flatnonzero(packet['valid_window'])]
    if anchors!=list(range(row['windows'])) or packet['label_representation'].item()!='ABSOLUTE_COMMITTED_EEF10' or packet['committed_absolute10'].shape!=(640,10):
        raise ValueError('ORIGINAL_COMPLETE_COMMITTED_H40_WINDOW_CONTRACT')
    if len(samples)!=641 or samples[-1]['supervision'] is not None:
        raise ValueError('ORIGINAL_640_ACTIONS_PLUS_UNLABELED_TERMINAL_OBSERVATION_REQUIRED')
    stages=[s['privileged']['teacher_stage'] for s in samples[:-1]]
    if len(anchors)!=601 or not np.allclose(packet['anchor_time_s'],np.arange(601)/20,atol=1e-8,rtol=0):
        raise ValueError('ORIGINAL_601_ANCHORS_20HZ_REQUIRED')
    expected={p:[a for a in anchors if STAGE_PHASE.get(stages[a])==p] for p in PHASES}
    if expected!=row['stage_anchors'] or row.get('unmapped_stages'):
        raise ValueError('ORIGINAL_MANIFEST_PHASE_ANCHORS_DIFFER_FROM_RAW_STAGE')
    if set().union(*(set(x) for x in expected.values()))!=set(anchors):raise ValueError('EVERY_ANCHOR_HAS_ONE_PHASE')
    if any(not s['supervision']['valid'] or not s['supervision']['actually_committed'] for s in samples[:-1]):
        raise ValueError('ONLY_ACTUALLY_COMMITTED_TRAIN_ACTIONS')
    # Events may contain repeated stages; preserve every transition, never dedupe it.
    previous='approach';last=-1.
    for e in events:
        if e['previous']!=previous or e['stage'] not in PUSH_STAGES or not last<e['time_s']<=32:
            raise ValueError('INVALID_ORDERED_TEACHER_EVENT_HISTORY')
        previous=e['stage'];last=e['time_s']
    if previous!='hold':raise ValueError('ORIGINAL_SUCCESS_DEMO_MUST_HAVE_HOLD_EVENT')
    geometry=dict(scene_group=row['group'],family=row['family'],seed=row['seed'],initial_state_sha256=card['initial_state_sha256'],
        source_xy_m=card['source_xy'],goal_xy_m=card['goal_xy'],cube_yaw_rad=card['cube_yaw'],
        source_goal_distance_m=float(np.linalg.norm(np.asarray(card['source_xy'])-card['goal_xy'])))
    return dict(id=row['id'],numeric_sha256=row['sha256'],samples_sha256=row['samples_sha256'],geometry=geometry,
        exact_stage_windows=window_counts(stages,anchors),phase_anchor_counts={p:len(v) for p,v in expected.items()},
        ordered_events=events,first_detach_s=next(e['time_s'] for e in events if e['stage']=='detach'),
        first_withdraw_s=next(e['time_s'] for e in events if e['stage']=='withdraw'),
        first_hold_s=next(e['time_s'] for e in events if e['stage']=='hold'),
        terminal_observation_supervision_absent_not_a_training_action=True,
        task_success_is_existing_raw_audit_not_new_evaluation=True),stages


def forecast_exposure(manifest,stages_by_episode):
    originals=[(r,a) for r in manifest['episodes'] if r['split']=='TRAIN' for a in range(r['windows'])]
    dataset=SimpleNamespace(rows=originals,original_rows=len(originals),manifest_sha=DATA)
    sampler=SourceSampler(dataset,stop=2048*32);counts=Counter();episodes=Counter()
    for cursor in range(1,2048*32,2):
        index=sampler.index_at(cursor);row,anchor=dataset.rows[index]
        if row['task']!='T2' or row['id'] not in stages_by_episode:raise ValueError('T2_FIXED_SOURCE_SAMPLER_ONLY')
        counts[stages_by_episode[row['id']][anchor]]+=1;episodes[row['id']]+=1
    return dict(status='DETERMINISTIC_FORECAST_NOT_ACTUAL_TRAIN_CONSUMPTION',seed=20261017,optimizer_updates_simulated=0,
        planned_updates_per_branch=2048,planned_sample_visits_per_branch=65536,T2_sample_visits=32768,
        anchor_stage_visits={s:counts[s] for s in PUSH_STAGES},episode_visits=dict(sorted(episodes.items())),
        same_T2_sampling_for_S_BC_and_S_CORR=True,task_ratio='1:1',T2_original_only=True,
        stage_balanced_sampling=False,episode_uniform_then_window_uniform=True)


def run(root,attempt=1):
    if attempt not in [1,2]:raise PermissionError('AT_MOST_ONE_SETTLED_CPU_CONTRACT_REPAIR')
    root=Path(root);m=validate(root,'4080-cpu-data');started=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    binding=root/'plans/T2_TRAIN30_COVERAGE_BINDING.private.json'
    if m['execution'].get('t2_coverage_binding_sha256')!=sha(binding):raise PermissionError('SEALED_TRAIN30_COVERAGE_REQUEST')
    q=read(binding);bank=Path(q['bank'])
    if q['maximum_CPU_s']!=600 or q['new_physics'] or q['new_neural'] or sha(bank/'DATASET_MANIFEST.json')!=DATA or sha(bank/'statistics.json')!=STATS:
        raise PermissionError('FROZEN_SOURCE_STATS_CPU_ONLY')
    manifest=read(bank/'DATASET_MANIFEST.json');selected=[r for r in manifest['episodes'] if r['task']=='T2' and r['split']=='TRAIN']
    if len(selected)!=30 or set(q['episodes'])!={r['id'] for r in selected}:raise PermissionError('EXACT_ORIGINAL_T2_TRAIN30_ONLY')
    output=root/'evidence'/('T2_TRAIN30_EVENT_COVERAGE_'+str(attempt).zfill(2));output.mkdir(mode=0o700);rows=[];stages={}
    try:
        for row in selected:
            if time.process_time()-started>=500:raise TimeoutError('TRAIN30_CPU_BOUNDARY')
            path=bank/row['file'];raw=Path(row['raw_root'])
            if not path.resolve().is_relative_to(bank.resolve()) or sha(path)!=row['sha256']:raise ValueError('ORIGINAL_NUMERIC_CHANGED')
            if sha(raw/'samples.json')!=row['samples_sha256'] or sha(raw/'FILE_MANIFEST.json')!=row['raw_FILE_MANIFEST_sha256']:
                raise ValueError('ORIGINAL_RAW_MANIFEST_OR_SAMPLES_CHANGED')
            for name,digest in q['episodes'][row['id']].items():
                if Path(name).is_absolute() or '..' in Path(name).parts or sha(raw/name)!=digest:raise ValueError('SEALED_RAW_EVENT_INPUT_CHANGED')
            audit=read(raw/'AUDIT.json');score=read(raw/'SCORE.json')
            if audit['status']!='PASS_INDEPENDENT_RAW_AUDIT' or not audit['admitted'] or not score['persistent_success']:
                raise PermissionError('ONLY_EXISTING_ADMITTED_SUCCESS_DEMONSTRATIONS')
            with np.load(path,allow_pickle=False) as z:packet={k:z[k].copy() for k in z.files}
            result,raw_stages=analyze_episode(row,packet,read(raw/'samples.json'),read(raw/'TEACHER_EVENTS.json'),read(raw/'CARD.json'))
            result['input_hashes']=q['episodes'][row['id']];rows.append(result);stages[row['id']]=raw_stages
        totals=Counter();present=Counter();phase=Counter()
        for row in rows:
            totals.update(row['exact_stage_windows']['anchor_stage_counts']);phase.update(row['phase_anchor_counts'])
            present.update(k for k,v in row['exact_stage_windows']['anchor_stage_counts'].items() if v)
        value=dict(schema='MP1_ORIGINAL_T2_TRAIN30_EVENT_COVERAGE_V1',status='PASS_ACTUAL_TRAIN30_NUMERIC_RAW_STAGE_AND_EVENT_AUDIT',
            run_uuid=root.name,source_sha=m['code']['vla_commit'],source_manifest_sha256=DATA,statistics_sha256=STATS,
            original_TRAIN_episodes=126,selected_T2_TRAIN_episodes=30,independent_scene_groups=len({r['geometry']['scene_group'] for r in rows}),
            distinct_initial_state_hashes=len({r['geometry']['initial_state_sha256'] for r in rows}),family_episodes=dict(Counter(r['geometry']['family'] for r in rows)),
            total_H40_windows=sum(r['exact_stage_windows']['windows'] for r in rows),anchor_stage_counts=dict(totals),phase_anchor_counts=dict(phase),
            independent_episodes_with_stage_anchor={s:present[s] for s in PUSH_STAGES},rows=rows,
            next_source_pair_T2_forecast=forecast_exposure(manifest,stages),new_neural=0,new_physics=0,new_optimizer_updates=0,
            DEV_metadata_only_not_numeric_RGB_or_labels=True,RGB_bytes_loaded=0,
            interpretation='Coverage is measured, not proof of sufficient student-state coverage or a unique causal bottleneck.',
            selected='NONE',deployment='NONE')
        write(output/'RESULT.json',value);return value
    finally:
        after=resource.getrusage(resource.RUSAGE_CHILDREN);charge_cpu(root,time.process_time()-started+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--attempt',type=int,default=1);a=p.parse_args()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));resource.setrlimit(resource.RLIMIT_CPU,(600,600))
    result=run(a.root,a.attempt);print(json.dumps(dict(status=result['status'],episodes=len(result['rows']))),flush=True)
