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
    if target['status']!='REQUEST_ONLY_NOT_GRANTED':raise ValueError('A request must not be interpreted as permission')
    print(json.dumps(dict(status='EVIDENCE_VERIFIED',files=len(manifest['files']),models=models,WM_main_updates=main,continuation_probe_updates=probe,total_actual_optimizer_updates=main+probe,new_S_transitions=0,new_T_integrations=0,Actor_updates=0,complete_new_tasks=0,target='OWNER_BLOCKED',scope_status=target['status']),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
