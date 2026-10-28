"""Reconcile saved audit metadata, not raw simulations or current native state."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from .analysis import audited_outcome,autonomous_summary


def validate_saved_report(path,expected_sha256):
    path=Path(path);raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!=expected_sha256:raise ValueError('FIXED_PUBLISHED_REPORT_BYTES_REQUIRED')
    source=json.loads(raw);audits=source['independent_T1_actual_evidence_audits']
    if source['model']!='M1-1024-CANONICAL' or len(audits)!=2 or source['Teacher_takeover'] is not False:
        raise ValueError('EXACT_PREVIOUSLY_REPORTED_TWO_TRAIN_T1_OUTCOMES')
    rows=[audited_outcome(a['adjudication'],a,autonomous=True) for a in audits]
    if any(r['task']!='T1' or r['model']!=source['model'] for r in rows):raise ValueError('EXACT_SAVED_SOURCE_MODEL_AND_TASK')
    summary=autonomous_summary(rows)[source['model']]['T1']
    if not summary['source_task_competence_observed'] or summary['task_successes']!=source['T1_success'] or source['T1_planned']!=2:
        raise ValueError('SAVED_AUDIT_METADATA_DOES_NOT_SUPPORT_PUBLISHED_T1_CLAIM')
    return dict(schema='E1_OFFLINE_REPORTING_VALIDATION_V1',status='SAVED_AUDIT_METADATA_RECONCILED_NOT_NEW_EXPERIMENT',
        confidence='SHARE_WITH_CAVEATS_NOT_CURRENT_NATIVE_READBACK',observed_UTC=datetime.now(timezone.utc).isoformat(),
        input_report_sha256=digest,implementation_sha256=hashlib.sha256(Path(__file__).with_name('analysis.py').read_bytes()).hexdigest(),
        source_model=source['model'],actual_saved_T1_attempt_ids=[r['attempt_id'] for r in rows],T1_summary=summary,
        checks=dict(fixed_input_bytes=True,independent_saved_audit_statuses_and_checks=True,reported_T1_counts_recomputed=True,
            two_seen_TRAIN_layouts_only=True,other_T2_and_DEV_results_not_reclassified=True),
        new_model_queries=0,new_physics_integrations=0,new_server_jobs=0,raw_images_or_physics_reaudited=False,
        current_native_or_Core_terminal_states_verified=False,E1_goal_complete=False,selected='NONE',deployment='NONE',
        caveats=['This rechecks prior published audit metadata, not current raw files or server processes.',
            'T1 two seen TRAIN layouts are not DEV, complete SMALL8, Go2/WBC or real-robot generalization.',
            'The complete E1 matrix still requires actual remaining outcomes, independent audits and terminal resource readback.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--output')
    a=p.parse_args();v=validate_saved_report(a.input,a.expected_sha256)
    if a.output:
        with Path(a.output).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(v,allow_nan=False))
