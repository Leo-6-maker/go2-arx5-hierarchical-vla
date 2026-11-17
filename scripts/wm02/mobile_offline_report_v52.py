"""Read sealed V5.2 arrays on Core; derive fixed gates and requested metrics."""
import ast
import json
from pathlib import Path
import numpy as np
from go2arx5_vla.sla_adapter.mobile_offline_audit_v52 import audit
from go2arx5_vla.world_model.storage import digest
from scripts.wm02.mobile_d0_v52 import boundary


def read(path):return json.loads(Path(path).read_bytes())


def pinned(pin):
    if digest(pin['path'])!=pin['sha256']:raise ValueError('SEALED_OFFLINE_REPORT_INPUT_CHANGED')
    return read(pin['path'])


def action(pin):
    if digest(pin['path'])!=pin['sha256']:raise ValueError('SEALED_PREDICTION_CHANGED')
    with np.load(pin['path'],allow_pickle=False)as z:value=z['physical_H40'].copy()
    if value.shape!=(40,10)or not np.isfinite(value).all():raise ValueError('FULL_FINITE_H40_REPORT_ONLY')
    return value


def enrich(row,output,reference,*,key,target=None,endpoint=None,baseline=None):
    value=action(output);width=.002+.040*value[:,9]
    result=dict(row,id=key,endpoint=boundary(value)if endpoint is None else endpoint,
        previous_command_seam_speed_m_s=float((width[0]-reference['width_m'])/.05))
    if target is not None:
        result['EEF_errors_m']=np.linalg.norm(value[:,:3]-target[:,:3],axis=1).tolist()
        result['terminal_intent_matches']=bool((width[-1]>.022)==(.002+.040*target[-1,9]>.022))
    elif baseline is not None:result['EEF_errors_m']=np.linalg.norm(value[:,:3]-baseline[:,:3],axis=1).tolist()
    return result


def run(followup,*,include_successors=True):
    root=Path(followup['registered_root']);base_spec=pinned(followup['E1_specs']['B1']);dataset=Path(base_spec['dataset'])
    manifest=read(dataset/'MANIFEST.private.json')
    if digest(dataset/'MANIFEST.private.json')!=base_spec['dense_manifest_sha256']:raise ValueError('SAME_FROZEN_C20_REPORT_TARGETS')
    labels=[]
    for ep in manifest['episodes']:
        path=dataset/ep['labels']['file']
        if digest(path)!=ep['labels']['sha256']:raise ValueError('C20_REPORT_LABEL_CHANGED')
        with np.load(path,allow_pickle=False)as z:labels.extend(z['nominal_policy_H40'].copy())
    if len(labels)!=1376:raise ValueError('ALL_C20_TARGET_IDENTITIES')
    bank=pinned(followup['CN38_bank']);entries={r['index']:r for r in bank['records']};roll=pinned(followup['ROLL96'])
    d0=read(root.parent/'sla-mobile-f2-v52-d0-20261117-01/RESULT.private.json');saved=pinned(d0['artifacts']['ALL_QUERY_BOUNDARIES.private.json'])
    old={(r['case'],r['query']):r for r in saved if'action_sha256'in r};required=[r['case']+'/q%d'%r['query']for r in saved if r.get('consumption')and not r['consumption']['full_H40_compiled']]
    old_cn=pinned(followup['B0_R1_records']);trains={'B0':followup['B0_training_result']};results={};inference_rows=[]
    if include_successors:
        terminal=read(root/'offline/RESULT.private.json')
        if terminal['exit_code']or terminal['new_generations']!=600:raise ValueError('COMPLETE_OFFLINE_E1_EVIDENCE_REQUIRED')
        inference_rows=[json.loads(line)for line in(root/'offline/RECORDS.private.jsonl').read_text().splitlines()]
        if len(inference_rows)!=828:raise ValueError('ALL_THREE_MATCHED_SAVED_INPUT_PANELS_REQUIRED')
        trains.update({a:dict(path=str(Path(p['path']).parent/'RESULT.private.json'),sha256=digest(Path(p['path']).parent/'RESULT.private.json'))for a,p in followup['E1_specs'].items()})
    for arm,training_pin in trains.items():
        training=pinned(training_pin);path=Path(training_pin['path']).parent/'TRAIN48_PREDICTIONS.private.json';train=read(path)
        if len(train)!=96 or len({(r['index'],r['noise'])for r in train})!=96:raise ValueError('EXACT_TRAIN96_CONDITIONS')
        train=[enrich(r,r['output'],r['previous_reference'],key='%d/%d'%(r['index'],r['noise']),target=labels[r['index']])for r in train]
        source=old_cn if arm=='B0'else[r for r in inference_rows if r['arm']==arm and r['bank']=='CN38'];cn=[]
        for row in source:
            entry_index=row['entry']if arm=='B0'else ast.literal_eval(row['key'])[0];entry=entries[entry_index]
            if row['input_sha256']!=entry['input']['sha256']:raise ValueError('SAME_CN38_REPORT_INPUT')
            if digest(entry['truth']['path'])!=entry['truth']['sha256']:raise ValueError('SAME_EXPERT_REPORT_TARGET')
            with np.load(entry['truth']['path'],allow_pickle=False)as z:target=z['physical_H40'].copy()
            cn.append(enrich(row,row['output'],entry['previous_reference'],key='%d/%d'%(entry_index,row['noise']),target=target))
        mapping={(r['case'],r['query']):r for r in inference_rows if r['arm']==arm and r['bank']=='ROLL96'};rolling=[]
        for entry in roll:
            previous=old[(entry['case'],entry['query'])];key=entry['case']+'/q%d'%entry['query'];baseline=action(entry['B0_saved_output'])
            if previous['action_sha256']!=entry['B0_saved_output']['sha256']:raise ValueError('EXACT_B0_ROLL_REUSE_IDENTITY')
            row=mapping[(entry['case'],entry['query'])]if arm!='B0'else dict(envelope=dict(original_full_H40_pass=previous['consumption']['full_H40_compiled'],paths=dict(nominal=dict(endpoint_progress_m=float(np.linalg.norm(baseline[-1,:3]-baseline[0,:3]))))))
            output=row['output']if arm!='B0'else entry['B0_saved_output']
            rolling.append(enrich(row,output,entry['previous_reference'],key=key,baseline=baseline,endpoint=previous if arm=='B0'else row['endpoint']))
        report=audit(train,cn,rolling,saved_reject_ids=required)
        report.update(arm=arm,package=training['package'],training_result=training_pin,TRAIN96_sha256=digest(path))
        results[arm]=report
    return dict(schema='SLA_V52_OFFLINE_REPORT_ACTUAL',by_arm=results,new_generations=0,new_updates=0,new_native_requests=0,new_integrations=0,TEST_outputs_read=False)
