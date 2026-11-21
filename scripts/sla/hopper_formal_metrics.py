"""Summarize one sealed formal panel with original clocks, budgets and physical output rows."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

PANEL='3136404c06f890a9dd830b2cf5f40e487ef436ede16e805e14ffc0cb06b5eefd'


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def population(rows):
    return {key:[r for r in rows if r['kind']==key] for key in ('target','target_phase','robodojo','libero')}


def targets(rows,folder,panel):
    speeds=[]
    accs=[]
    joint=0
    reasons=Counter()
    phase=Counter()
    close_errors=[]
    for row in rows:
        m=row['metrics']
        ctx=panel['target_contexts'][str(row['index'])]
        prior=ctx['previous_command']
        if abs(prior['query_native_s']-prior['source_native_s']-.050)>1e-8:
            raise ValueError('Expected original 50ms previous-command interval')
        p=folder/row['file']
        if p.name!=f"{row['kind']}-{row['ordinal']:03d}.npz" or sha(p)!=row['file_sha256']:
            raise ValueError('Changed endpoint output')
        with np.load(p,allow_pickle=False) as f:
            h=f['decoded']
            if h.shape!=(40,10) or hashlib.sha256(np.ascontiguousarray(h).tobytes()).hexdigest()!=row['decoded_sha256']:
                raise ValueError('Invalid physical H40')
            widths=np.r_[prior['width_m'], .002+.040*h[:10,9]]
        velocity=np.diff(widths)/.050
        speeds.append(float(np.abs(velocity).max()))
        accs.append(float(np.abs(np.diff(velocity)/.050).max()))
        ok=bool(m['envelope']['original_full_H40_pass'] and not m['gripper_seam']['original_execution_speed_exceeded'])
        if ok!=m['H40_and_seam_pass']:
            raise ValueError('Saved and recomputed gate identity differs')
        if ok:
            joint+=1
        else:
            first=m['envelope']['original_short_circuit_component']
            reasons[first['channel'] if first else 'actual_command_gripper_speed']+=1
        if 'phase' in m:
            d=m['phase']
            if d['expert_P10_due_close']:
                phase['expert_due']+=1
                phase['predicted_due_on_expert_due']+=int(d['predicted_P10_due_close'])
                phase['timely_and_gates']+=int(d['timely_P10_close_and_original_gates'])
                if d['first22mm_crossing_error_s'] is not None:
                    close_errors.append(d['first22mm_crossing_error_s'])
    return dict(count=len(rows),H40_and_seam_pass=joint,first_gate_failures=dict(reasons),
                phase_timing=dict(phase),mean_absolute_crossing_error_s_if_both_cross=float(np.mean(close_errors)) if close_errors else None,
                paired_crossing_count=len(close_errors),per_window_peak_P10_speed_p95_m_s=float(np.quantile(speeds,.95)),
                per_window_peak_P10_acceleration_p95_m_s2=float(np.quantile(accs,.95)),
                mean_xyz_error_m=float(np.mean([r['metrics']['pose_error']['xyz_m'] for r in rows])))


def summarize(run,panel_file):
    original=read(run/'formal/RESULT.json')
    if original['status']!='SEALED' or original['attempted']!=original['generated'] or sha(panel_file)!=PANEL or original['panel']!=PANEL:
        raise ValueError('Ordinary output or fixed panel incomplete')
    panel=read(panel_file)
    groups=population(original['rows'])
    if {k:len(v) for k,v in groups.items()}!=dict(target=96,target_phase=88,robodojo=96,libero=96):
        raise ValueError('Missing formal population')
    for k,rows in groups.items():
        expected=panel['phase_train'] if k=='target_phase' else panel['formal'][k]
        if any(r['ordinal']!=i or r['noise_id']!=e['noise_id'] or r['index']!=e['index'] for i,(r,e) in enumerate(zip(rows,expected,strict=True))):
            raise ValueError('Output identities or sample order differ')
    phase=groups['target_phase']
    report=dict(phase=original['phase'],checkpoint_sha256=original['checkpoint'],panel_sha256=PANEL,
                ordinary_receipt_sha256=sha(run/'formal/RESULT.json'),ordinary_source_target_counts=original['generated'],
                target_train=targets(groups['target'],run/'formal',panel),phase_train=targets(phase,run/'formal',panel),
                phase_buckets={k:targets([r for r in phase if r['phase_bucket']==k],run/'formal',panel) for k in sorted({r['phase_bucket'] for r in phase})},
                source_normalized_active_MAE={k:float(np.mean([r['metrics']['normalized_active_mae'] for r in groups[k]])) for k in ('robodojo','libero')},
                restore=None)
    restored=run/'formal-restore/RESULT.json'
    if restored.is_file():
        r=read(restored)
        if r['checkpoint']!=original['checkpoint'] or r['panel']!=PANEL:
            raise ValueError('Mismatched restore endpoint')
        report['restore']=dict(status=r['status'],exact=r['exact'],generated=r['generated'],error=r['error'],result_sha256=sha(restored))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('panel',type=Path)
    a=p.parse_args()
    print(json.dumps(summarize(a.run,a.panel),allow_nan=False))
