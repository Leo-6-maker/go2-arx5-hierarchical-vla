"""Empirical deterministic-position floor under the frozen 6000 training visits."""
import json
import sqlite3

import numpy as np

from pi0260_r2 import ROOT,read,sha
from pi0260_r3_condition_audit import raw_key,summarize
from pi0260_r3_eval_freeze import source


def main():
    root=ROOT/'R3'
    visits=read(root/'train_visits.json')['visits']
    assert len(visits)==6000
    db=sqlite3.connect(root/'condition_cache.sqlite')
    keys=dict(db.execute('select raw_key,processed_key from processed'))
    rows=[]
    for visit in visits:
        seq=visit['sequence'];rec,z=source(seq);j=next(i for i,r in enumerate(rec['rows']) if r['index']==visit['index'])
        assert rec['sequence']['split']=='TRAIN' and rec['rows'][j]['usable']
        with np.load(ROOT/visit['frame']) as frame:
            raw=raw_key(frame['body_rgb'],frame['wrist_rgb'],z['state_v1'][j],rec['sequence']['instruction'])
        rows.append(dict(key=keys[raw],raw_key=raw,source=seq+':'+str(visit['index']),task=visit['task'],
                         command=visit['command'],language=rec['sequence']['instruction'],time=visit['time_s'],label=z['raw_action'][j]))
    db.close()
    summary,_=summarize(rows)
    result=dict(status='R3_FROZEN_EXPOSURE_AMBIGUITY_FLOOR',source_schedule_sha256=sha(root/'train_visits.json'),
                condition_audit_sha256=sha(root/'condition_audit.json'),weights='one uniform weight per frozen training visit; duplicate visits retain multiplicity',
                updates=6000,condition_count=summary['conditions'],aliased_visits=summary['aliased_windows'],
                empirical_mse_m2=summary['empirical_mse_m2'],empirical_rmse_m=summary['empirical_rmse_m'],
                by_task={task:{k:v for k,v in t.items() if k in ('windows','conditions','aliased_windows','empirical_mse_m2','empirical_rmse_m')} for task,t in summary['by_task'].items()},
                interpretation='Position mean-square lower bound for deterministic condition-only prediction under these empirical exposures; not median/p95/model quality.',
                new_statistics_fit=0,model_loads=0,model_queries=0)
    p=root/'exposure_floor.json';assert not p.exists()
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
