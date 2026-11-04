"""Verify published AUTO1 evidence; no environment, physics, model or permission effects."""
import argparse,json,hashlib
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report-dir',type=Path,default=Path('reports/WM02/auto1-target-entry'));a=p.parse_args(argv);root=a.report_dir.resolve()
    manifest=json.loads((root/'EVIDENCE_MANIFEST.json').read_text())
    for name,pin in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root)or hashlib.sha256(path.read_bytes()).hexdigest()!=pin:raise ValueError('Evidence bytes differ:'+name)
    v=json.loads((root/'G1_ACTUAL.json').read_text());ticks=0
    if v['registration']['registration']!='REGISTERED':raise ValueError('Actual registration missing')
    for name,limit in(('cold-1',50),('cold-2',50),('hold-2s-1',1000),('hold-2s-2',1000),('hold-5s-1',2500),('hold-5s-2',2500)):
        s=v['phases'][name];r=s['result'];ticks+=r['actual_target_integrations']
        if(r['status']!='NATIVE_STAGE_PASSED'or r['actual_target_integrations']!=limit or not r['shutdown_clean']or any(p['exit_code']!=0 for p in r['owned_process_exits'])or r['physics_stops']or r['original_joint_or_warning_samples']or r['trace_audit']['status']!='RECORDED_TRACE_CHECKS_PASS'or r['trace_audit']['writer']['dropped']or s['worker_lease_status']!='CLOSED'):raise ValueError('Incomplete native stage:'+name)
    if ticks!=7100 or v['counts']['all_new_target_integrations']!=7350:raise ValueError('Actual integration count differs')
    print(json.dumps(dict(status='DELIVERED_G1_EVIDENCE_VERIFIED_NOT_NEW_EXECUTION',files=len(manifest['files']),actual_source_sha=v['actual_run_source_sha'],G1_ticks=ticks,counts=v['counts'],safety_STOP_actual=v['safety_STOP_actual'],target_WM_entry=v['target_WM_entry'],scope_status=v['scope_lease_status_at_export'],scope_is_live_daemon=False),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
