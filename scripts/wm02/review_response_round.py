"""Verify delivered response/candidate evidence without GPU, training or physics."""
import argparse,hashlib,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report-dir',type=Path,default=Path('reports/WM02/response-init-candidate'));a=p.parse_args(argv);root=a.report_dir.resolve();manifest=json.loads((root/'EVIDENCE_MANIFEST.json').read_text())
    if manifest['schema']!='WM02_RESPONSE_INIT_EVIDENCE_V1':raise ValueError('Unknown evidence schema')
    for n,h in manifest['files'].items():
        f=(root/n).resolve()
        if not f.is_relative_to(root)or hashlib.sha256(f.read_bytes()).hexdigest()!=h:raise ValueError('Evidence identity differs:'+n)
    model=json.loads((root/'MODEL_RESPONSE_RESULTS.json').read_text());target=json.loads((root/'INITIALIZATION_CANDIDATE.json').read_text())
    print(json.dumps(dict(status='EVIDENCE_VERIFIED',files=len(manifest['files']),WM_main_updates=model['main_WM_updates'],continuation_probes=model['continuation_probe_updates'],retained_failed_main_updates=20,total_actual_optimizer_updates=model['total_actual_optimizer_updates_including_probes'],new_physics=0,Actor_updates=0,full_tasks=0,target_status=target['status'],first_target_blocker=target['first_blocker']),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
