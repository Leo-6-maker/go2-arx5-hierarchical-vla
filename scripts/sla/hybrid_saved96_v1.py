"""Zero model calls: fixed FM EEF plus each of two predeclared gripper sources."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scripts.pi0292.durable import write
from scripts.sla.export_mfree_c20_v1 import _reference
from scripts.sla.close_goal_g1_v1 import width_metrics
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.hybrid_eef_grip_v1 import compose, coordination
from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze


def read(path): return json.loads(Path(path).read_bytes())


def summarize(rows):
    events = [r for r in rows if r['width']['expert_crossing_in_window'] is not None]
    phases = {p:dict(count=sum(r['phase']==p for r in rows),
        passed=sum(r['phase']==p and r['envelope']['original_full_H40_pass'] for r in rows))
        for p in ('PREGRASP','DESCEND','CLOSE','LIFT','HOLD')}
    close = phases['CLOSE']; other = [r for r in rows if r['phase']!='CLOSE']
    space_bad = sum(any(row['envelope']['paths'][path]['channels'][channel]['exceedance_indices']
        for path in ('nominal','reference') for channel in ('linear_velocity','angular_velocity','linear_acceleration','angular_acceleration'))
        for row in rows)
    seams = [abs(r['coordination']['previous_actual_command_to_first_speed_m_s']) for r in rows]
    misses = sum(e['width']['crossing_error_s'] is None for e in events)
    timely = sum(e['width']['crossing_error_s'] is not None and e['width']['crossing_error_s']<=.100000001 for e in events)
    terminal = max(r['width']['terminal_width_error_m'] for r in rows)
    return dict(predictions=len(rows), phases=phases, all_pass=sum(r['envelope']['original_full_H40_pass'] for r in rows),
        CLOSE_pass=close['passed'], CLOSE_count=close['count'], nonCLOSE_pass=sum(r['envelope']['original_full_H40_pass'] for r in other),
        nonCLOSE_count=len(other), full_spatial_channel_failures=space_bad, annotated_events=len(events),
        crossing_within100ms=timely, missed_crossings=misses, maximum_terminal_width_error_m=terminal,
        command_seam_peak_m_s=max(seams), command_seam_over40mm_s=sum(x>.04*(1+1e-10) for x in seams),
        offline_candidate_admitted=None if len(rows)!=96 else (close['passed']>=18 and len(other)==76
            and all(r['envelope']['original_full_H40_pass'] for r in other) and space_bad==0
            and len(events)==20 and timely==20 and terminal<=.001 and max(seams)<=.04*(1+1e-10)),
        coordination_is_report_only=True, physical_admission=False)


def run(spec):
    root=Path(spec['registered_root']); dataset=Path(spec['dataset']); manifest=read(dataset/'MANIFEST.private.json')
    if sha(dataset/'MANIFEST.private.json')!=spec['dense_manifest_sha256']:raise ValueError('FIXED_C20_DATASET')
    bank=[]; live={}; frames={}
    for ep in manifest['episodes']:
        p=dataset/ep['labels']['file']
        if sha(p)!=ep['labels']['sha256']:raise ValueError('ORIGINAL_C20_LABELS_CHANGED')
        with np.load(p,allow_pickle=False)as z:
            bank.extend((ep,anchor,z['state_eef10'][anchor].copy(),row.copy())for anchor,row in enumerate(z['nominal_policy_H40']))
        cap=read(ep['original_live_capture']['path']);live[ep['id']]=read(cap['labels']['path'])
        ns=read(Path(ep['native_result']['path']).parent/'EXECUTION_SPEC.json');frames[ep['id']]=np.asarray(ns['policy_from_world'])
    if sha(spec['FM_predictions']['path'])!=spec['FM_predictions']['sha256']:raise ValueError('PINNED_FM_PREDICTIONS_CHANGED')
    fm=read(spec['FM_predictions']['path']); completed={}; fm_spatial=[]
    for label,pin in spec['grip_predictions'].items():
        if sha(pin['path'])!=pin['sha256']:raise ValueError('PINNED_GRIP_PREDICTIONS_CHANGED')
        grip=read(pin['path']);rows=[];folder=root/label;folder.mkdir(exist_ok=False)
        if len(fm)!=96 or len(grip)!=96:raise ValueError('FIXED48_NOISE2_COMPLETE_PREDICTIONS')
        for a,b in zip(fm,grip):
            if(a['index'],a['noise'],a['phase'])!=(b['index'],b['noise'],b['phase']):raise ValueError('SAME_OBSERVATION_NOISE_ONLY')
            arrays=[]
            for r in (a,b):
                if sha(r['output']['path'])!=r['output']['sha256']:raise ValueError('SAVED_PREDICTION_CHANGED')
                with np.load(r['output']['path'],allow_pickle=False)as z:arrays.append(z['physical_H40'].copy())
            ep,anchor,state,expert=bank[a['index']]
            if (ep['id'],anchor)!=(a['episode'],a['anchor']):raise ValueError('EXACT_ORIGINAL_WINDOW_IDENTITY')
            action=compose(*arrays);frame=frames[ep['id']];reference=_reference(live[ep['id']][anchor])
            envelope=analyze(action,state,frame,reference)
            # This function evaluates every spatial channel before any compiler short circuit.
            if label==next(iter(spec['grip_predictions'])):
                original=analyze(arrays[0],state,frame,reference)
                fm_spatial.append(dict(index=a['index'],noise=a['noise'],paths=original['paths']))
            f=folder/f"{a['index']:04d}-{a['noise']}.npz"
            with f.open('xb')as handle:np.savez(handle,physical_H40=action)
            rows.append(dict(index=a['index'],noise=a['noise'],episode=ep['id'],family=ep['family'],anchor=anchor,phase=a['phase'],
                output=dict(path=str(f),sha256=sha(f)),width=width_metrics(action,expert),envelope=envelope,
                coordination=coordination(action,expert,state,b['previous_reference'],frame),
                EEF_source='FM500',grip_source=label,EEF_bit_exact=True,grip_bit_exact=True))
        write(folder/'RECORDS.private.json',rows);completed[label]=dict(summary=summarize(rows),
            records=dict(path=str(folder/'RECORDS.private.json'),sha256=sha(folder/'RECORDS.private.json')))
    write(root/'FM_ALL_SPATIAL_CHANNELS.private.json',fm_spatial)
    selected=next((name for name in ('FM_EEF_TIME_GRIP','FM_EEF_SHIFT7_GRIP')if completed[name]['summary']['offline_candidate_admitted']),None)
    result=dict(schema='MFREE_HYBRID_SAVED96_ACTUAL_V1',exit_code=0,completed=completed,selected_global_candidate=selected,
        selection_rule='TIME_PRIMARY_THEN_SHIFT7_SECONDARY_GLOBAL_TRAIN_ONLY_NO_PER_QUERY_SWITCH',
        new_composite_H40_checks=192,original_FM_full_spatial_checks=len(fm_spatial),new_model_calls=0,new_updates=0,new_native_requests=0,new_integrations=0,
        physical_admission=False,goal_met=False)
    write(root/'RESULT.private.json',result);return result
