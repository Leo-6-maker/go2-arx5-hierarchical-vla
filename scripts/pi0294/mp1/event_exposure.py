"""Read-only committed-action events, never a replacement sampling policy."""
from collections import Counter
from pathlib import Path

import numpy as np

from go2arx5_vla.task_data.codec import pose10
from scripts.pi0292.durable import read
from scripts.pi0292.eval_contract import error_metrics, summarize_records
from scripts.pi0294.native_dataset import sha
from .exposure import actual_exposure

EVENTS = ('close', 'open', 'detach', 'withdraw')


def window_events(events, anchor):
    if type(anchor) is not int or anchor < 0:
        raise ValueError('EXACT_NONNEGATIVE_ACTION_ANCHOR')
    return {name:dict(prefix8=[i-anchor for i in events[name] if anchor <= i < anchor+8],
                      late_H40=[i-anchor for i in events[name] if anchor+8 <= i < anchor+40])
            for name in EVENTS}


def episode_events(dataset_root, raw_root, row):
    """Check numeric exports against the original installed-command timeline."""
    if row['split'] != 'TRAIN':
        raise PermissionError('TRAIN_ONLY_ACTUAL_EVENT_EXPOSURE')
    dataset_root, raw_root = Path(dataset_root), Path(raw_root)
    numeric = dataset_root/row['file']
    if numeric.is_symlink() or not numeric.resolve().is_relative_to(dataset_root.resolve()):
        raise PermissionError('NO_NUMERIC_EVENT_PATH_ESCAPE')
    if raw_root.is_symlink():
        raise PermissionError('EXACT_EXISTING_RAW_EPISODE')
    required = {'FILE_MANIFEST.json':row['raw_FILE_MANIFEST_sha256'],
                'samples.json':row['samples_sha256']}
    for name, expected in required.items():
        if (raw_root/name).is_symlink() or sha(raw_root/name) != expected:
            raise ValueError('ACTUAL_EVENT_SOURCE_MANIFEST_OR_SAMPLES_CHANGED')
    members = read(raw_root/'FILE_MANIFEST.json')
    hashes = {m['path']:m['sha256'] for m in members}
    if len(hashes) != len(members):
        raise ValueError('UNIQUE_RAW_EVENT_MANIFEST_MEMBERS')
    for name in ['samples.json', 'TEACHER_EVENTS.json']:
        if (raw_root/name).is_symlink() or sha(raw_root/name) != hashes[name]:
            raise ValueError('RAW_COMMAND_OR_TEACHER_EVENT_BYTES_CHANGED')
    if sha(numeric) != row['sha256']:
        raise ValueError('FROZEN_NUMERIC_EVENT_EXPORT_CHANGED')
    samples, transitions = read(raw_root/'samples.json'), read(raw_root/'TEACHER_EVENTS.json')
    with np.load(numeric, allow_pickle=False) as z:
        commands, state = z['committed_absolute10'].copy(), z['actual_state10'].copy()
        valid = z['valid_window'].copy()
        representation = z['label_representation'].item()
    n = len(commands)
    if (commands.shape != (n,10) or state.shape != (n+1,10) or len(samples) != n+1
            or n < 40 or row['windows'] != n-39 or int(valid.sum()) != row['windows']
            or not np.asarray(valid[:row['windows']]).all()
            or not np.isfinite(commands).all() or not np.isfinite(state).all()
            or representation != 'ABSOLUTE_COMMITTED_EEF10'):
        raise ValueError('COMPLETE_ACTUAL_COMMAND_STATE_AND_H40_WINDOW_COUNTS')
    raw_commands, raw_states, stages = [], [], []
    for i, sample in enumerate(samples):
        obs = sample['policy_obs'];supervision = sample['supervision']
        if abs(obs['timestamp_s']-i/20) > 1e-6:
            raise ValueError('EXACT_ORIGINAL_20HZ_EVENT_CLOCK')
        raw_states.append(pose10(np.asarray(obs['actual_tcp']), obs['actual_gripper']))
        stages.append(sample['privileged']['teacher_stage'])
        if i == n:
            if supervision is not None:
                raise ValueError('NO_COMMAND_AT_TERMINAL_OBSERVATION')
            continue
        if (supervision['sequence'] != i or not supervision['valid']
                or not supervision['actually_committed']
                or abs(supervision['command_time_s']-i/20) > 1e-6
                or not np.array_equal(supervision['committed'],supervision['installed'])
                or supervision['committed_gripper'] != supervision['desired_gripper']):
            raise ValueError('RAW_INSTALLED_COMMAND_AND_CONTINUOUS_SEQUENCE')
        raw_commands.append(pose10(np.asarray(supervision['committed']),supervision['committed_gripper']))
    if (np.max(np.abs(np.asarray(raw_commands)-commands)) > 1e-12
            or np.max(np.abs(np.asarray(raw_states)-state)) > 1e-12):
        raise ValueError('NUMERIC_LABELS_AND_STATES_EQUAL_ACTUAL_ORIGINAL_TIMELINE')
    grip = np.r_[state[0,9],commands[:,9]]
    events = dict(close=np.flatnonzero((grip[1:]<.5)&(grip[:-1]>=.5)).tolist(),
                  open=np.flatnonzero((grip[1:]>=.5)&(grip[:-1]<.5)).tolist(),detach=[],withdraw=[])
    for event in transitions:
        index = int(round(event['time_s']*20))
        if (not 0 <= index < n or abs(event['time_s']-index/20) > 1e-6
                or stages[index] != event['stage']
                or index and stages[index-1] != event['previous']):
            raise ValueError('TEACHER_EVENT_EQUALS_RECORDED_STATE_STAGE_TRANSITION')
        if event['stage'] in ['detach','withdraw']:
            events[event['stage']].append(index)
    for name in ['detach','withdraw']:
        actual = [i for i in range(n) if stages[i] == name and (i == 0 or stages[i-1] != name)]
        if events[name] != actual:
            raise ValueError('NO_MISSING_OR_DUPLICATE_RAW_STAGE_EVENT')
    if any(sha(raw_root/name) != expected for name,expected in required.items()) or sha(numeric) != row['sha256']:
        raise ValueError('EVENT_SOURCE_CHANGED_DURING_READ')
    return dict(id=row['id'],task=row['task'],origin=row.get('origin','legacy'),windows=row['windows'],
        events=events,raw_stages=stages,committed_commands=n,
        evidence_sha256=dict(numeric=row['sha256'],raw_manifest=row['raw_FILE_MANIFEST_sha256'],
            samples=row['samples_sha256'],teacher_events=hashes['TEACHER_EVENTS.json']),
        stage_definition='ACTUAL_SAVED_TEACHER_STAGE_NOT_COARSE_MANIFEST_PHASE')


def summarize_visits(inventories, visits):
    if set(inventories) != set(visits):
        raise ValueError('NO_EPISODE_JOIN_LOSS_OR_EXTRA_VISITS')
    episodes, totals = {}, {}
    for identifier, inv in inventories.items():
        counts = visits[identifier]
        if any(type(a) is not int or not 0 <= a < inv['windows'] or type(c) is not int or c <= 0 for a,c in counts.items()):
            raise ValueError('VALID_COMMITTED_WINDOW_VISITS_ONLY')
        summary = dict(task=inv['task'],origin=inv['origin'],sample_visits=sum(counts.values()),
            unique_anchors=len(counts),raw_stage_visits=dict(Counter()),prefix8_all_hold_visits=0,
            H40_all_hold_visits=0,H40_cross_stage_visits=0,events={})
        phase_counts=Counter()
        for anchor, count in counts.items():
            phase_counts[inv['raw_stages'][anchor]]+=count
            summary['prefix8_all_hold_visits']+=count*all(p=='hold' for p in inv['raw_stages'][anchor:anchor+8])
            summary['H40_all_hold_visits']+=count*all(p=='hold' for p in inv['raw_stages'][anchor:anchor+40])
            summary['H40_cross_stage_visits']+=count*(len(set(inv['raw_stages'][anchor:anchor+40]))>1)
        summary['raw_stage_visits']=dict(phase_counts)
        for name,indices in inv['events'].items():
            rows=[]
            for index in indices:
                prefix=sum(count for a,count in counts.items() if a<=index<a+8)
                late=sum(count for a,count in counts.items() if a+8<=index<a+40)
                rows.append(dict(command_index=index,prefix8_visit_hits=prefix,late_H40_visit_hits=late,
                    covered_in_sampled_prefix8=prefix>0,covered_in_any_sampled_H40=prefix+late>0))
            summary['events'][name]=dict(event_occurrences=len(rows),occurrences=rows,
                prefix8_visit_hits=sum(x['prefix8_visit_hits'] for x in rows),
                late_H40_visit_hits=sum(x['late_H40_visit_hits'] for x in rows),
                uncovered_in_sampled_prefix8=sum(not x['covered_in_sampled_prefix8'] for x in rows),
                uncovered_in_any_sampled_H40=sum(not x['covered_in_any_sampled_H40'] for x in rows))
        episodes[identifier]=summary
        key=inv['task']+':'+inv['origin']
        group=totals.setdefault(key,dict(episodes=0,sample_visits=0,raw_stage_visits=Counter(),
            prefix8_all_hold_visits=0,H40_all_hold_visits=0,H40_cross_stage_visits=0,
            events={name:Counter() for name in EVENTS}))
        group['episodes']+=1;group['sample_visits']+=summary['sample_visits']
        group['raw_stage_visits'].update(phase_counts)
        for field in ['prefix8_all_hold_visits','H40_all_hold_visits','H40_cross_stage_visits']:
            group[field]+=summary[field]
        for name in EVENTS:
            group['events'][name].update({k:v for k,v in summary['events'][name].items() if k!='occurrences'})
    return dict(episodes=episodes,task_origin_groups=totals,
        event_hits_count_repeated_sample_visits_not_independent_demonstrations=True,
        prefix_indices='0..7',late_indices='8..39',
        close_open_definition='THRESHOLD_0.5_CROSSING_IN_GLOBAL_ACTUAL_COMMITTED_COMMAND_SEQUENCE',
        detach_withdraw_definition='RECORDED_TEACHER_EVENT_AND_RAW_STATE_STAGE_AGREE',
        future_H40_event_is_current_executed_event=False)


def audit(dataset, trace, stop, guard=lambda:None):
    exposure=actual_exposure(dataset,trace,0,stop)
    if exposure['correction_sample_visits']:
        raise PermissionError('THIS_BOUND_READONLY_AUDIT_IS_ORIGINAL_S_BC_ONLY')
    visits={};rows={}
    for actual in trace:
        row,anchor=dataset.rows[actual['index']]
        rows[row['id']]=row;visits.setdefault(row['id'],Counter())[anchor]+=1
    inventories={}
    for identifier,row in sorted(rows.items()):
        guard();raw=Path(dataset.raw_bindings.get(identifier,row['raw_root']))
        inventories[identifier]=episode_events(dataset.root,raw,row)
    result=summarize_visits(inventories,visits)
    result.update(status='PASS_ACTUAL_COMMITTED_VISITS_JOIN_RAW_INSTALLED_EVENTS',
        exposure=exposure,inventory= inventories,source_episodes=len(inventories),
        new_queries=0,new_physics=0,sampler_or_training_modified=False,
        correction_event_exposure='NOT_YET_RUN_NO_C1_TRACE_IN_THIS_BOUND_BC_CHECKPOINT')
    return result


def probe_strata(folder, inventories, manifest, independent_audit):
    """Stratify already independently qualified draws, not generate new ones."""
    folder=Path(folder);path=folder/'ON_EXPERT_RESULTS.json';saved=read(path)
    if (independent_audit['status']!='PASS_SAVED_CANDIDATE_TRAIN_PROBE_METRICS_INPUTS_NOISE_AND_WEIGHT_IDENTITY'
            or independent_audit['candidate_result_sha256']!=sha(path)
            or saved['complete'] is not True or len(saved['records'])!=96):
        raise PermissionError('EXACT_PRIOR_INDEPENDENTLY_AUDITED_96_DRAW_RESULT')
    fixed={r['id']:r for r in manifest['windows']}
    if (len(fixed)!=96 or len({r['id'] for r in saved['records']})!=96
            or {r['id'] for r in saved['records']}!=set(fixed)):
        raise ValueError('EXACT_FIXED_PROBE_TO_RAW_EPISODE_LEFT_JOIN')
    groups={};signed={};annotations=[]
    for row in saved['records']:
        w=fixed[row['id']]
        if any(row[k]!=w[k] for k in ['episode_id','task','anchor','split']) or row['split']!='TRAIN':
            raise PermissionError('FIXED_TRAIN_PROBE_EVENT_CONTEXT')
        inv=inventories[row['episode_id']];anchor=row['anchor']
        if not 0<=anchor<inv['windows'] or row['task']!=inv['task']:
            raise ValueError('PROBE_EVENT_ANCHOR_BELONGS_TO_TRUE_EPISODE')
        p=folder/(row['id']+'.npz')
        if p.is_symlink() or sha(p)!=row['arrays_sha256']:
            raise ValueError('QUALIFIED_PROBE_ARRAY_CHANGED')
        with np.load(p,allow_pickle=False) as z:
            prediction,expert,state=[z[k].copy() for k in ['physical_H40','expert_absolute10','state']]
        errors=error_metrics(prediction,expert,state)
        if errors!=row['errors']:
            raise ValueError('PROBE_POINTWISE_ERRORS_STILL_RECOMPUTE')
        event=window_events(inv['events'],anchor);stage=inv['raw_stages'][anchor]
        labels=[row['task']+':ALL',row['task']+':RAW_STAGE:'+stage]
        for name,value in event.items():
            for horizon in ['prefix8','late_H40']:
                if value[horizon]:labels.append(row['task']+':'+horizon+':'+name)
        record=dict(id=row['episode_id'],split='TRAIN',gate=row['gate'],errors=errors)
        delta=np.mean(prediction[:8,:3]-expert[:8,:3],axis=0)
        for label in labels:
            groups.setdefault(label,[]).append(record)
            signed.setdefault(label,{}).setdefault(row['episode_id'],[]).append(delta)
        annotations.append(dict(id=row['id'],episode_id=row['episode_id'],task=row['task'],anchor=anchor,
            raw_anchor_stage=stage,events=event,trajectory_gate=row['gate']['decision'],
            original_S1='NOT_RUN',geometric_minimum_clearance='NOT_MEASURED'))
    summaries={}
    for label,rows in groups.items():
        value=summarize_records(rows)['TRAIN']
        summaries[label]=dict(queries=value['queries'],independent_episodes=len(value['episodes']),
            episode_macro=value['episode_macro'],trajectory_gate_only=value['decisions'],
            signed_XYZ_prefix8_episode_macro_mean_m=np.mean([np.mean(v,axis=0) for v in signed[label].values()],axis=0).tolist())
    return dict(status='PASS_ALREADY_QUALIFIED_EXPERT_PROBES_JOIN_ACTUAL_RAW_EVENT_STRATA',
        draws=96,distinct_episodes=len({r['episode_id'] for r in saved['records']}),
        raw_stage_and_event_groups=summaries,annotations=annotations,
        pointwise_errors_or_units_mixed=False,new_queries=0,new_physics=0,
        S1_and_clearance_not_imputed=True,current_student_state_stratification='PENDING_NOT_SUBSTITUTED_BY_EXPERT_STATES')
