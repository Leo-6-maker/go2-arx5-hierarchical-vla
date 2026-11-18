"""CPU-only final target377 accounting and exact predeclared promotion gates."""
import ast,json
from pathlib import Path
import numpy as np
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.mobile_offline_audit_v52 import audit,bank_summary
from scripts.wm02.mobile_offline_report_v52 import action,enrich,pinned,read
from go2arx5_vla.sla_adapter.mobile_selection_v52 import summarize


def run(spec):
    root=Path(spec['registered_root']);terminal=read(root/'RESULT.private.json')
    if terminal['exit_code']or terminal['new_generations']!=615:raise ValueError('ACTUAL_COMPLETE615_REQUIRED')
    rows=[json.loads(line)for line in(root/'RECORDS.private.jsonl').read_text().splitlines()]
    if len(rows)!=843:raise ValueError('ALL_THREE_FIXED281_SAVED_INPUT_CONDITIONS_REQUIRED')
    dataset=Path(spec['dataset']);manifest=read(dataset/'MANIFEST.private.json');labels=[]
    if sha(dataset/'MANIFEST.private.json')!=spec['dense_manifest_sha256']:raise ValueError('FROZEN_C20_TARGET_REQUIRED')
    for episode in manifest['episodes']:
        path=dataset/episode['labels']['file']
        if sha(path)!=episode['labels']['sha256']:raise ValueError('FROZEN_C20_LABELS_REQUIRED')
        with np.load(path,allow_pickle=False)as z:labels.extend(z['nominal_policy_H40'].copy())
    bank=pinned(spec['CN38_bank']);cn={r['index']:r for r in bank['records']};roll=pinned(spec['ROLL96']);roll_by={(r['case'],r['query']):r for r in roll}
    required=spec['historical_reject3_ids'];results={}
    for arm in('T-B1','T-FM','T-H40'):
        training_pin=spec['training_results'][arm];training=pinned(training_pin);path=Path(training_pin['path']).parent/'TRAIN48_PREDICTIONS.private.json';train=read(path)
        if len(train)!=96 or len({(r['index'],r['noise'])for r in train})!=96:raise ValueError('EXACT_EXISTING_TRAIN96_REQUIRED_NO_REGENERATION')
        if arm!='T-B1'and sha(path)!=training['TRAIN96_predictions']['sha256']:raise ValueError('SEALED_NEW_ENDPOINT_TRAIN96_REQUIRED')
        train=[enrich(r,r['output'],r['previous_reference'],key='%d/%d'%(r['index'],r['noise']),target=labels[r['index']])for r in train]
        cn_rows=[];roll_rows=[];new_rejects=[]
        for row in rows:
            if row['arm']!=arm:continue
            if row['bank']=='CN38':
                entry=cn[ast.literal_eval(row['key'])[0]]
                if row['input_sha256']!=entry['input']['sha256']:raise ValueError('EXACT_CN_REPORT_INPUT_REQUIRED')
                if sha(entry['truth']['path'])!=entry['truth']['sha256']:raise ValueError('EXACT_CN_REPORT_TARGET_REQUIRED')
                with np.load(entry['truth']['path'],allow_pickle=False)as z:truth=z['physical_H40'].copy()
                cn_rows.append(enrich(row,row['output'],row['previous_reference'],key=row['key'],target=truth))
            elif row['bank']=='ROLL96':
                entry=roll_by[row['case'],row['query']];roll_rows.append(enrich(row,row['output'],row['previous_reference'],key=row['case']+'/q%d'%row['query'],baseline=action(entry['B0_saved_output'])))
            else:new_rejects.append(row)
        result=audit(train,cn_rows,roll_rows,saved_reject_ids=required)
        all5=len(new_rejects)==5 and all(r['envelope']['original_full_H40_pass']for r in new_rejects)
        result['E2_reject5']=dict(count=len(new_rejects),passes=sum(r['envelope']['original_full_H40_pass']for r in new_rejects),failed=[r['key']for r in new_rejects if not r['envelope']['original_full_H40_pass']])
        result['promotion_offline_gates']['old3_and_E2reject5_all']=result['promotion_offline_gates']['saved_reject3_all']and all5
        result['all_offline_promotion_gates']=all(result['promotion_offline_gates'].values())
        result.update(arm=arm,package=training['package'],training_result=training_pin,TRAIN96_sha256=sha(path),fixed_conditions=377)
        results[arm]=result
    return dict(schema='SLA_V6_TARGET377_ACTUAL_OFFLINE_REPORT',by_arm=results,new_generations=0,new_updates=0,new_native_requests=0,TEST_outputs_read=False)


def decision(rows,offline):
    students=[r for r in rows if r['role']=='STUDENT'];order=('T-B1','T-FM','T-H40')
    if len(students)!=24 or any(sum(r['arm']==a for r in students)!=8 for a in order):raise ValueError('COMPLETE_NEW24_REQUIRED_NO_PREFIX_WINNER')
    gates={a:dict(offline['by_arm'][a]['promotion_offline_gates'])for a in order}
    for a in order:gates[a]['saved_reject3_all']=gates[a]['old3_and_E2reject5_all']
    result=summarize(students,gates,order=order);candidate=result['unique_confirmation_candidate']
    result.update(schema='SLA_V6_TARGET_COMPLETE_SCREEN_DECISION',qualified=candidate is not None,selected_arm=candidate,
        native_screen_requests=24,confirmation_budget=8 if candidate else 0,large_DEV_TEST_authorized=False,all8_rejects_required=True)
    return result
