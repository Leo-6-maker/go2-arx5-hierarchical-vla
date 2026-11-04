"""Verify delivered time-response evidence without inference, training or physics."""
import argparse,json,hashlib
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report-dir',type=Path,default=Path('reports/WM02/temporal-response'));a=p.parse_args(argv);root=a.report_dir.resolve();manifest=json.loads((root/'EVIDENCE_MANIFEST.json').read_text())
    if manifest['schema']!='WM02_TEMPORAL_EVIDENCE_V1':raise ValueError('Unknown evidence schema')
    for name,pin in manifest['files'].items():
        f=(root/name).resolve()
        if not f.is_relative_to(root)or hashlib.sha256(f.read_bytes()).hexdigest()!=pin:raise ValueError('Evidence bytes differ:'+name)
    m=json.loads((root/'M3_TRAINING.json').read_text());g=json.loads((root/'DEV_GATES.json').read_text());t=json.loads((root/'TARGET_STATUS.json').read_text())
    print(json.dumps(dict(status='EVIDENCE_VERIFIED',files=len(manifest['files']),WM_main_updates=m['main_updates'],continuation_probe_updates=m['continuation_probe_updates'],total_actual_optimizer_updates=m['main_updates']+m['continuation_probe_updates'],new_physics=0,Actor_updates=0,Critic_updates=0,full_tasks=0,all_development_gates_pass=g['all_development_gates_pass'],new_holdout_sampling=g['new_holdout_sampling'],target_cold_starts=t['cold_starts'],first_target_blocker=t['first_blocker']),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
