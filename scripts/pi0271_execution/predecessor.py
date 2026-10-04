"""Saved-only PI0270 transition/subgroup analysis, after its fixed CPU replay."""
import ast
import csv
import json
import math
import os
import statistics
from pathlib import Path
from support import BASE, OLD, ROOT, read, sha, write

RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')


def table(path):
    with path.open() as f:return list(csv.DictReader(f))


def summary(rows):
    result=dict(rows=len(rows),REJECT=sum(r.get('gate')=='REJECT' for r in rows))
    for key in ('xyz_rmse_m','so3_rmse_rad','prefix_xyz_rmse_m','prefix_so3_rmse_rad',
                'suffix_xyz_rmse_m','suffix_so3_rmse_rad','first_xyz_error_m','prefix_endpoint_error_m'):
        values=[]
        for row in rows:
            try:value=float(row.get(key,''))
            except (ValueError,TypeError):continue
            if math.isfinite(value):values.append(value)
        result['finite_'+key]=len(values);result['median_'+key]=statistics.median(values) if values else None
    return result


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));os.nice(19)
    import numpy as np
    import sys
    sys.path.insert(0,str(BASE/'PI0250/runtime/reference-317a/src'))
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
    delivery=RAW/'delivery';verdict=read(delivery/'PRIMARY_VERDICT.json')
    assert verdict['status']=='PRIMARY_EVALUATION_COMPLETE' and verdict['virtual_episodes']==192
    assert read(delivery/'CPU_REPLAY.json')['status']=='PASS_SAVED_ARRAY_REPLAY'
    source=BASE/'PI0269/code/final_analysis.py'
    assert sha(source)=='91ef396a5fdddf09dd243c38c3450d911cc1c256e46701455876cd82908bcd6b'
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('median','reliability_summary')]
    scope=dict(json=json,statistics=statistics);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
    virtual=table(delivery/'VIRTUAL192.csv');offline=table(delivery/'OFFLINE2904.csv');newfit=table(delivery/'NEW_FIT384.csv')
    assert (len(virtual),len(offline),len(newfit))==(192,2904,384)
    groups=[]
    for point in ('ROT2048','PREFIX4096','AGG4096'):
        for task in ('M1','M3'):
            for backend in ('E0','E1'):
                own=[r for r in virtual if (r['point'],r['task'],r['backend'])==(point,task,backend)];assert len(own)==16
                groups.append(dict(point=point,task=task,backend=backend,command='ALL',**scope['reliability_summary'](own)))
                for command in ('0','1'):
                    cell=[r for r in own if r['command']==command];assert len(cell)==8
                    groups.append(dict(point=point,task=task,backend=backend,command=command,**scope['reliability_summary'](cell)))
    fit_card=read(OLD/'design/NEW_FIT192.json');buckets={r['input_id']:r['source_mapping']['bucket'] for r in fit_card['rows']}
    assert set(buckets.values())=={'ORIGINAL_Q0','ORIGINAL_Q1'}
    panels=[]
    for point in ('ROT2048','PREFIX4096','AGG4096'):
        for seen in ('True','False'):
            own=[r for r in offline if r['point']==point and r['domain']=='fit' and r['actual_AGG_seen']==seen]
            panels.append(dict(point=point,domain='OLD_FIT',actually_AGG_seen=seen,**summary(own)))
    for point in ('PREFIX4096','AGG4096'):
        for task in ('M1','M3'):
            for bucket in ('ORIGINAL_Q0','ORIGINAL_Q1'):
                own=[r for r in newfit if r['point']==point and r['task']==task and buckets[r['uid']]==bucket];assert len(own)==48
                panels.append(dict(point=point,domain='NEW_FIT',task=task,bucket=bucket,**summary(own)))
    anchor=np.asarray(read(BASE/'PI0267/design/SCENE_CARD.json')['policy_anchor_world_m']);transitions=[];m3=[]
    for path in sorted((RAW/'virtual/AGG4096').glob('*/result.json')):
        rec=read(path);assert rec['arrays_sha256']==sha(path.parent/'trajectory.npz')
        if rec['task']=='M3':
            s=rec['scorer'];m3.append(dict(case=path.parent.name,family=rec['family'],command=rec['command'],backend=rec['backend'],
                original_success=s['success'],entry_index=s['entry_index'],entry_time_s=s['entry_time_s'],ordered_bins=s['ordered_bins'],
                reverse_violation=s['reverse'],max_backward_excursion_m=s['max_backward_excursion_m'],
                total_negative_variation_m=s['total_negative_variation_m'],reason=s['reason'],
                gap_shortcut=s['gap_shortcut'],original_V3_and_V2_shared=True))
        with np.load(path.parent/'trajectory.npz',allow_pickle=False) as z:
            times=z['time'].copy();actual=z['actual_p'].copy();installed=z['installed_p'].copy();spec=json.loads(str(z['spec_json']))
        goal=np.asarray(spec['goal']);gates={g['replan']:g['result'] for g in rec['gates']}
        for q in rec['queries']:
            request=q['query'];prediction=Path(request['output']);assert sha(prediction)==q['prediction_sha256']
            with np.load(request['input'],allow_pickle=False) as z:current=z['position_world'].copy();now=float(z['timestamp'])
            with np.load(prediction,allow_pickle=False) as z:decoded=z['decoded'][0].copy()
            row=dict(case=path.parent.name,family=rec['family'],task=rec['task'],command=rec['command'],backend=rec['backend'],
                     query_index=q['replan'],timestamp_s=now,prediction_sha256=sha(prediction),current_target_xyz_m=float(np.linalg.norm(current-goal)),
                     gate_decision=gates.get(q['replan'],{}).get('decision','NOT_INSTALLED'),
                     actual_prefix_executed=False,predicted_reference_is_not_achieved=True)
            if np.isfinite(decoded).all():
                try:rot6d_rows_to_matrix(decoded[:,3:9].astype(float))
                except ValueError:row['prediction_status']='INVALID_ROTATION'
                else:
                    endpoint=decoded[8,:3].astype(float)+anchor
                    row.update(prediction_status='FINITE_VALID_ROTATION',original_model_reference_0p4_target_xyz_m=float(np.linalg.norm(endpoint-goal)),
                               original_model_reference_first_point_bias_m=float(np.linalg.norm(decoded[0,:3]+anchor-current)))
            else:row['prediction_status']='NONFINITE'
            endpoint_tick=np.flatnonzero(np.abs(times-(now+.4))<=1e-9)
            if row['gate_decision'] in ('ACCEPT','WARN') and len(endpoint_tick)==1:
                k=int(endpoint_tick[0]);row.update(actual_prefix_executed=True,
                    installed_reference_same_tick_target_xyz_m=float(np.linalg.norm(installed[k]-goal)),
                    achieved_same_tick_target_xyz_m=float(np.linalg.norm(actual[k]-goal)),
                    same_tick_tracking_xyz_m=float(np.linalg.norm(actual[k]-installed[k])),
                    achieved_target_error_change_m=float(np.linalg.norm(actual[k]-goal)-np.linalg.norm(current-goal)))
            else:row['execution_missingness']='GATE_NOT_INSTALLED' if row['gate_decision']=='REJECT' else 'NO_COMPLETE_0P4_INTERVAL'
            transitions.append(row)
    out=ROOT/'evidence/predecessor-analysis';out.mkdir(parents=True,exist_ok=True)
    with (out/'SAME_TICK_REFERENCE_INSTALLED_ACHIEVED.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in transitions for k in r)));writer.writeheader();writer.writerows(transitions)
    sources=[dict(path=str(p),sha256=sha(p)) for p in [delivery/n for n in ('PRIMARY_VERDICT.json','CPU_REPLAY.json','VIRTUAL192.csv','OFFLINE2904.csv','NEW_FIT384.csv','COMPONENT96.csv','PAIRED_OFFLINE_CHANGES.csv','PAIRED_NEW_FIT_CHANGES.csv')]+[source,Path(__file__)]]
    write(out/'PREDECESSOR_SUBGROUP_AND_TRANSITION_ANALYSIS.json',dict(status='PASS_COMPLETE_SAVED_PI0270_SUBGROUP_TRANSITION_ANALYSIS',
          main_groups=groups,old_seen_and_NEWFIT_buckets=panels,M3_entry_and_coverage=m3,original_qualifications=verdict['ORIGINAL_VIRTUAL_QUALIFICATION'],
          reliable_qualifications=verdict['RELIABLE_VIRTUAL_QUALIFICATION'],legacy=verdict['legacy_retention_vs_original_ROT2048'],
          transitions=len(transitions),transition_table_sha256=sha(out/'SAME_TICK_REFERENCE_INSTALLED_ACHIEVED.csv'),source_files=sources,
          shared_original_V3_scorer=True,existing_numeric_CPU_replay_reused=True,finite_REJECTs_retained=True,
          no_new_neural_or_teacher_or_render_calls=True,A0_requires_predecessor_terminal_and_archives=True,
          selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('PASS_COMPLETE_SAVED_PI0270_SUBGROUP_TRANSITION_ANALYSIS',len(transitions),flush=True)


if __name__=='__main__':main()
