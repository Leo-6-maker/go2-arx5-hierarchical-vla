"""Summarize complete saved tables, with no model, renderer or score changes."""
import ast
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parents[2]/'worktree/reports/PI0273/final'
def table(name):
    with (root/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))
def number(r,k):
    try:v=float(r[k]);return v if np.isfinite(v) else None
    except (ValueError,KeyError):return None
def metrics(rows):
    result=dict(rows=len(rows),REJECT=sum(r.get('gate')=='REJECT' for r in rows))
    for k in ('prefix_xyz_rmse_m','prefix_so3_rmse_rad','xyz_rmse_m','so3_rmse_rad','suffix_xyz_rmse_m','suffix_so3_rmse_rad','prefix_xyz_mse_m2','teacher_displacement_m','predicted_displacement_m','predicted_residual_motion_m','signed_teacher_direction_progress_m','direction_cosine','first_xyz_error_m','prefix_endpoint_error_m','first_so3_error_rad','prefix_endpoint_so3_error_rad'):
        vs=[v for r in rows if (v:=number(r,k)) is not None]
        result[k]=dict(defined=len(vs),median=float(np.median(vs)) if vs else None,negative=sum(v<0 for v in vs),zero=sum(v==0 for v in vs),positive=sum(v>0 for v in vs))
    return result
early=table('EARLY_CONTEXT768.csv');late=table('LATE_CONTEXT2368.csv');means=table('LATE_PER_CONDITION1184.csv')
assert (len(early),len(late),len(means))==(768,2368,1184)
out=dict(status='SAVED_TABLE_DESCRIPTIVE_ANALYSIS_COMPLETE_NOT_NEW_QUALIFICATION',model_queries=0,renders=0,physics=0,
    support=[],paired=[],main_family=[],components=[],diagnosis=[],source_tables={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.csv')})
for kind,rows,keys in [('early',early,('point','task')),('late',late,('point','task','bucket')),('late_condition_mean',means,('point','task','bucket'))]:
    for key in sorted({tuple(r[k] for k in keys) for r in rows}):
        own=[r for r in rows if tuple(r[k] for k in keys)==key]
        out['support'].append(dict(panel=kind,**dict(zip(keys,key)),**metrics(own)))
for name in ('PAIRED_EARLY_CHANGES.csv','PAIRED_LATE_CONDITION_CHANGES.csv'):
    rows=table(name);keys=('before','after','task')+(('bucket',) if 'LATE' in name else ())
    for key in sorted({tuple(r[k] for k in keys) for r in rows}):
        own=[r for r in rows if tuple(r[k] for k in keys)==key];values={}
        for m in ('prefix_xyz_rmse_m','prefix_so3_rmse_rad','xyz_rmse_m','so3_rmse_rad','prefix_xyz_mse_m2','suffix_xyz_rmse_m','suffix_so3_rmse_rad','first_xyz_error_m','first_so3_error_rad','prefix_endpoint_error_m','prefix_endpoint_so3_error_rad'):
            vs=[v for r in own if (v:=number(r,'delta_'+m)) is not None]
            values[m]=dict(defined=len(vs),median=float(np.median(vs)) if vs else None,better=sum(v<0 for v in vs),equal=sum(v==0 for v in vs),worse=sum(v>0 for v in vs))
        out['paired'].append(dict(table=name,**dict(zip(keys,key)),rows=len(own),metrics=values))
rows=table('VIRTUAL192.csv');assert len(rows)==192
out['main_pairs']=[]
casekeys=('family','task','command','backend')
by={p:{tuple(r[k] for k in casekeys):r for r in rows if r['point']==p} for p in ('AGG4096','BROAD5120','TARGET5120')}
for before,after in [('AGG4096','BROAD5120'),('AGG4096','TARGET5120'),('BROAD5120','TARGET5120')]:
    assert by[before].keys()==by[after].keys() and len(by[before])==64
    for metric in ('success','reliable'):
        out['main_pairs'].append(dict(before=before,after=after,metric=metric,paired_cases=64,transitions=dict(Counter(str((by[before][k][metric]=='True',by[after][k][metric]=='True')) for k in by[before]))))
out['early_strata']=[]
for keys in [('point','task','family'),('point','task','command'),('point','task','j')]:
    if not all(k in early[0] for k in keys):continue
    for key in sorted({tuple(r[k] for k in keys) for r in early}):out['early_strata'].append(dict(**dict(zip(keys,key)),**metrics([r for r in early if tuple(r[k] for k in keys)==key])))
for point,family in sorted({(r['point'],r['family']) for r in rows}):
    own=[r for r in rows if (r['point'],r['family'])==(point,family)]
    assert len(own)==8
    out['main_family'].append(dict(point=point,family=family,success=sum(r['success']=='True' for r in own),reliable=sum(r['reliable']=='True' for r in own)))
components=table('COMPONENT96.csv')
for point in sorted({r['point'] for r in components}):
    own=[r for r in components if r['point']==point]
    out['components'].append(dict(point=point,proposals=len(own),validity=dict(Counter(r['validity'] for r in own)),diagnostic_V3_success=sum(bool(ast.literal_eval(r['original_V3'])['success']) for r in own if r['original_V3'])))
diag=table('FAILURE_AND_POST_SUCCESS_DIAGNOSIS128.csv')
for point,task in sorted({(r['point'],r['task']) for r in diag}):
    own=[r for r in diag if (r['point'],r['task'])==(point,task)]
    out['diagnosis'].append(dict(point=point,task=task,rows=len(own),flags=dict(Counter(f for r in own for f in ast.literal_eval(r['flags'])))))
fit=table('OFFLINE2904.csv')
out['actual_fit_consumption']={p:dict(Counter(r['actual_current_arm_seen'] for r in fit if r['point']==p and r['domain']=='fit')) for p in ('BROAD5120','TARGET5120')}
out['same_tick']=[]
ticks=table('SAME_TICK_REFERENCE_INSTALLED_ACHIEVED.csv');assert len(ticks)==12368
for keys in [('point','task','backend')]:
    for key in sorted({tuple(r[k] for k in keys) for r in ticks}):
        own=[r for r in ticks if tuple(r[k] for k in keys)==key];summary={}
        for k in ('original_model_reference_first_point_bias_m','original_model_reference_first_point_bias_rad','original_model_reference_0p4_target_xyz_m','original_model_reference_0p4_target_so3_rad','installed_reference_same_tick_target_xyz_m','installed_reference_same_tick_target_so3_rad','achieved_same_tick_target_xyz_m','achieved_same_tick_target_so3_rad','same_tick_tracking_xyz_m','same_tick_tracking_so3_rad','achieved_target_error_change_m','achieved_target_error_change_rad'):
            vs=[v for r in own if (v:=number(r,k)) is not None]
            summary[k]=dict(defined=len(vs),median=float(np.median(vs)) if vs else None,p95=float(np.quantile(vs,.95)) if vs else None,max=float(max(vs)) if vs else None)
        out['same_tick'].append(dict(**dict(zip(keys,key)),rows=len(own),metrics=summary))
(root/'DESCRIPTIVE_ANALYSIS.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8')
for r in out['support']:
    if r['panel']=='early' or r.get('bucket')=='M3_TERMINAL':print(r['panel'],r['point'],r['task'],r.get('bucket'),r['rows'],r['prefix_xyz_rmse_m']['median'],r['prefix_so3_rmse_rad']['median'],r['direction_cosine']['negative'])
print('DIAGNOSIS',json.dumps(out['diagnosis']))
print('COMPONENTS',json.dumps(out['components']))
print('FIT_CONSUMPTION',out['actual_fit_consumption'])
