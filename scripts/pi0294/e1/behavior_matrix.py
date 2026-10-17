"""Saved-evidence matrix retaining all failures, interruptions and model contexts."""
import argparse
import json
from pathlib import Path
import time

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha
from .analysis import audited_outcome,autonomous_summary,canonical_model,probe_summary

MODELS=['M1-1024-CANONICAL','M2-BC','M2-CORR']


def conditions(rows):
    cells=[]
    for model in MODELS:
        for family in ['F0','F1']:
            for task in ['T1','T2']:
                layout='PI0294/TRAIN/'+family+'/000'
                attempts=[r for r in rows if r['canonical_reporting_model']==model and r['scene_group']==layout and r['task']==task
                    and r['model_autonomous_evaluation'] and not r['excluded_scope_deviation']]
                valid=[r for r in attempts if r['pipeline_trustworthy'] and r['audit_verified'] and r['status'] in ['SUCCESS','VALID_TASK_FAIL']]
                if len(valid)>1:raise ValueError('NO_BEST_OF_MULTIPLE_VALID_CONDITION_OUTCOMES')
                status=(valid[0]['status'] if valid else 'NOT_RUN' if not attempts else
                    'ENGINEERING_INCOMPLETE' if all(r['status']=='ENGINEERING_INCOMPLETE' for r in attempts) else 'UNVERIFIED_POLICY_OUTCOME')
                cells.append(dict(model=model,scene_group=layout,task=task,status=status,
                    valid_policy_outcome=bool(valid),actual_attempt_ids=[r['attempt_id'] for r in attempts],
                    actual_attempts=len(attempts),evidence_accounted_attempts=sum(r['evidence_accounted'] for r in attempts)))
    return cells


def build(root):
    root=Path(root);rows=[];provenance={};missing=[]
    deviation=root/'evidence/PROTOCOL_SCOPE_DEVIATION_01.json'
    excluded=read(deviation)['attempt']['attempt_id'] if deviation.exists() else None
    if deviation.exists():provenance[str(deviation.relative_to(root))]=sha(deviation)
    index_path=root/'evidence/MAIN_EVIDENCE_AUDIT_INDEX.private.json'
    index=read(index_path);indexed={r['attempt_id']:r for r in index['rows']}
    if len(indexed)!=len(index['rows']):raise ValueError('DUPLICATE_ACTUAL_AUDIT_ATTEMPT_ID')
    provenance[str(index_path.relative_to(root))]=sha(index_path)
    for folder in sorted((root/'episodes').iterdir()):
        if not folder.is_dir():continue
        adjudication=folder/'E1_ADJUDICATION.json'
        if not adjudication.exists():missing.append(dict(attempt_id=folder.name,missing='ACTUAL_ADJUDICATION'));continue
        a=read(adjudication);card_path=folder/'CARD.json';card=read(card_path);audit=indexed.get(folder.name)
        if a['attempt_id']!=folder.name or card['id']!=folder.name or a['task']!=card['task'] or a['scene_group']!=card['scene_group'] or card['split']!='TRAIN':
            raise ValueError('ACTUAL_E1_CARD_ADJUDICATION_POPULATION_MISMATCH')
        row=audited_outcome(a,audit,autonomous=not card.get('e1_intervention_collection',False),excluded_scope_deviation=folder.name==excluded)
        row['canonical_reporting_model']=canonical_model(row['model'])
        row['video']=None if audit is None else audit.get('video')
        zero_accounted=bool(audit and audit.get('status')=='PASS_ZERO_INTEGRATION_ENGINEERING_EVIDENCE'
            and audit.get('errors')==[] and audit.get('checks') and all(v is True for v in audit['checks'].values())
            and audit.get('physical_integrations')==0 and audit.get('score') is None
            and audit.get('video',{}).get('initial_frame_only') is True
            and audit.get('scientific_policy_outcome_valid') is False and a['status']=='ENGINEERING_INCOMPLETE' and a['main_integrations']==0)
        row['evidence_accounted']=row['audit_verified'] or zero_accounted
        if not row['evidence_accounted']:missing.append(dict(attempt_id=folder.name,missing='PASS_PHYSICAL_OR_ZERO_STEP_EVIDENCE_AUDIT'))
        rows.append(row);provenance[str(adjudication.relative_to(root))]=sha(adjudication);provenance[str(card_path.relative_to(root))]=sha(card_path)
    summary=autonomous_summary(rows)
    fixed={label:{task:summary.get(label,{}).get(task,dict(complete_predeclared_panel=False,valid_policy_outcomes=0,task_successes=0,conditions_attempted=0,actual_attempts=0)) for task in ['T1','T2']} for label in MODELS}
    probes={};probe_root=root/'datasets/received-fixed-probes/native/evidence'
    if probe_root.exists():
        for path in sorted(probe_root.glob('*/probes/ON_EXPERT_RESULTS.json')):
            label=path.parents[1].name;value=read(path);records=value.get('records',[])
            probes[label]=dict(complete_fixed_96=value.get('complete') is True and len(records)==96,
                precision_context_kept_separate=True,actual_fixed_expert_windows=len(records),on_expert=probe_summary(records))
            provenance[str(path.relative_to(root))]=sha(path)
    s1_path=root/'evidence/PROBE_PREFIX_S1_RESULTS.private.json';s1=read(s1_path) if s1_path.exists() else None
    s1_summary={}
    if s1 is not None:
        provenance[str(s1_path.relative_to(root))]=sha(s1_path)
        for label in s1['labels']:
            s1_summary[label]={}
            for kind in ['on_expert','on_student']:
                s1_summary[label][kind]={}
                for task in ['T1','T2']:
                    selected=[r for r in s1['rows'] if r['model']==label and r['population']==kind and r['task']==task]
                    s1_summary[label][kind][task]=dict(anchors=len(selected),original_prefix_executable=sum(r['prefix_executable'] is True for r in selected),
                        reference_reject=sum(r['reference_decision']=='REJECT' for r in selected),shadow_not_live=True)
    return dict(schema='E1_MODEL_BEHAVIOR_MATRIX_V1',run_uuid=root.name,observed_epoch=time.time(),
        status='FIXED_TERMINAL_MODELS_AUTONOMOUS_PANELS_AUDITED' if not missing and all(fixed[m][t]['complete_predeclared_panel'] for m in MODELS for t in ['T1','T2']) else 'PARTIAL_WITH_EXPLICIT_EVIDENCE_GAPS',
        fixed_terminal_models=fixed,predeclared_condition_matrix=conditions(rows),
        all_contexts_autonomous_summary=summary,actual_attempt_rows=rows,audit_gaps=missing,
        zero_step_accounted_evidence_is_not_policy_outcome=True,
        original_model_labels_preserved=True,collection_is_not_autonomous_evaluation=True,valid_policy_outcomes_not_rerun_to_success=True,
        fixed_probe_contexts=probes,fixed_probe_prefix_S1=s1_summary,probe_S1_available=s1 is not None,
        probe_errors_are_episode_macro_not_independent_windows=True,model_training_and_resource_receipts_separate=True,
        on_student_oracle_errors_in_private_probe_result='AVAILABLE' if s1 is not None else 'NOT_YET_AVAILABLE',
        evidence_sha256=provenance,precision_gap_and_scope_deviation_preserved=True,
        DEV_generalization_proven=False,Go2_WBC_or_real_robot_proven=False,selected='NONE',deployment='NONE')


def run(root):
    from .scope import validate,Ledger
    root=Path(root);validate(root,'4080-simulation');ledger=Ledger(root,'core');ledger.guard(closeout=True)
    if read(ledger.path)['reservations']:raise PermissionError('E1_MATRIX_ONLY_AFTER_PHYSICAL_RESERVATIONS_SETTLED')
    started=time.process_time()
    try:
        value=build(root);write(root/'evidence/MODEL_BEHAVIOR_MATRIX.json',value);return value
    finally:ledger.cpu(time.process_time()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();v=run(a.root);print(json.dumps(dict(status=v['status'],actual_attempts=len(v['actual_attempt_rows']),audit_gaps=len(v['audit_gaps']))))
