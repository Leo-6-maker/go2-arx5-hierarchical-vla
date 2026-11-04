"""Read delivered evidence only; never inference, training, native construction or admission."""
import argparse,json,hashlib
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report-dir',type=Path,default=Path('reports/WM02/autonomous-recovery'));a=p.parse_args(argv);root=a.report_dir.resolve();manifest=json.loads((root/'EVIDENCE_MANIFEST.json').read_text())
    if manifest['schema']!='WM02_AUTONOMOUS_EVIDENCE_V1':raise ValueError('Unknown evidence schema')
    for name,pin in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root)or hashlib.sha256(path.read_bytes()).hexdigest()!=pin:raise ValueError('Evidence bytes differ:'+name)
    models={};main=probe=0
    for name in manifest['model_candidates']:
        r=json.loads((root/(name+'_TRAINING.json')).read_text());g=json.loads((root/(name+'_DEV_GATES.json')).read_text());main+=r['main_updates'];probe+=r['continuation_probe_updates'];models[name]=dict(checkpoint_sha256=r['checkpoint_sha256'],development_gate=g['all_development_gates_pass'])
    target=json.loads((root/'TARGET_SCOPE_REQUEST.json').read_text())
    resource=json.loads((root/'RESOURCE_AND_PROVENANCE.json').read_text())
    if target['status']!='REQUEST_ONLY_NOT_GRANTED':raise ValueError('A request must not be interpreted as permission')
    result=dict(status='EVIDENCE_VERIFIED',files=len(manifest['files']),models=models,WM_main_updates=main,continuation_probe_updates=probe,total_actual_optimizer_updates=main+probe,new_S_transitions=resource['source_actual_transitions'],new_T_integrations=resource['target_integrations'],Actor_updates=resource['Actor_updates'],complete_new_tasks=resource['full_tasks'],GPU_worker_wall_hours=resource['GPU_worker_wall_hours'],measured_CPU_core_hours=resource['measured_CPU_core_hours'],target='OWNER_BLOCKED',scope_status=target['status'])
    if 'TRACE_EVIDENCE_V2_ACTUAL.json'in manifest['files']:
        trace=json.loads((root/'TRACE_EVIDENCE_V2_ACTUAL.json').read_text());request=json.loads((root/'TARGET_SCOPE_REQUEST_TRACE_V2.json').read_text())
        if request['status']!='REQUEST_ONLY_NOT_GRANTED'or request['candidate_source_sha']!=trace['source_sha']or request['candidate_binary_sha256']!=trace['candidate']['binary_sha256']:raise ValueError('Trace candidate/request identity differs')
        result['trace_evidence_v2']=dict(source_sha=trace['source_sha'],binary_sha256=trace['candidate']['binary_sha256'],build_exit_code=trace['build']['exit_code'],loader_audit_status=trace['loader_audits']['task_actuation_r1']['status'],scope_status=request['status'],target_qualification='NOT_ESTABLISHED')
        result['measured_CPU_core_hours']+=trace['cost']['additional_measured_CPU_s']/3600
    if 'POSTHOC_RELEASE_RANK.json'in manifest['files']:
        release=json.loads((root/'POSTHOC_RELEASE_RANK.json').read_text())
        result['posthoc_release_ranking']=dict(source_sha=release['actual_run_source_sha'],positive_regret_parent_counts=release['positive_regret_parent_counts'],qualification='PRIVILEGED_EXPLANATION_ONLY_NO_NEW_METHOD_VALIDATION')
        result['measured_CPU_core_hours']+=release['CPU_s']/3600
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
