"""Retire only this study's explicitly discarded, compared canary payloads."""
import argparse
import json
import subprocess
from pathlib import Path
ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0272').resolve()

def main(arm):
    receipt=json.loads((ROOT/arm/'canary_compare.json').read_text())
    assert receipt['status']=='PASS_PI0272_REAL_EXACT_RESUME_CANARY'
    assert receipt['saved_rng_scheduler_exact']
    inventory=ROOT/arm/'DISCARDED_CANARY_PAYLOADS.json'
    if inventory.exists():
        previous=json.loads(inventory.read_text())
        assert previous['status']=='DISCARDED_CANARY_PAYLOADS_RETIRED', 'resume explicit incomplete deletion inventory before continuing'
        return
    candidates=[]
    for phase,step in [('continuous',8),('split',4),('split',8)]:
        original=ROOT/arm/phase/'training'/f'checkpoint-{step}'
        assert not original.is_symlink()
        cp=original.resolve();assert cp.is_relative_to(ROOT/arm)
        state=json.loads((cp/'trainer_state.json').read_text());assert state['global_step']==step
        index=json.loads((cp/'model.safetensors.index.json').read_text())
        for name in sorted(set(index['weight_map'].values())|{'optimizer.pt'}):
            p=cp/name;assert p.parent==cp and not p.is_symlink() and p.resolve().is_relative_to(cp)
            stat=p.stat();assert stat.st_nlink==1 and p.is_file()
            candidates.append(dict(path=str(p),bytes=stat.st_size,device=stat.st_dev,inode=stat.st_ino,reason='AUTHORIZED_DISCARDED_PASSED_CANARY_ONLY'))
    free=lambda:int(subprocess.check_output(['df','-B1','--output=avail','/scratch/e1192548'],text=True).splitlines()[-1])
    report=dict(status='INVENTORIED',arm=arm,rows=candidates,before_free=free(),canary_receipt=receipt)
    inventory.write_text(json.dumps(report,indent=2)+'\n')
    for row in candidates:
        p=Path(row['path']);stat=p.stat();assert (stat.st_dev,stat.st_ino,stat.st_size)==(row['device'],row['inode'],row['bytes'])
        p.unlink();row['deleted']=True
        inventory.write_text(json.dumps(report,indent=2)+'\n')
    report.update(status='DISCARDED_CANARY_PAYLOADS_RETIRED',after_free=free(),logical_deleted_bytes=sum(r['bytes'] for r in candidates))
    inventory.write_text(json.dumps(report,indent=2)+'\n');print(report['status'],arm,report['logical_deleted_bytes'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=('S4FIT','DIRECT'));main(p.parse_args().arm)
